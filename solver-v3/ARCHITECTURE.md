# Solver V3 — Architecture & Implementation Notes

## Pipeline Overview

```
run.py
  1. detect_a4_area / warp_a4_region      → a4_rectified.png
  2. detect_pieces                         → pieces_detected.png, threshold.png
  3. detect_corners (per piece)            → corners_detected.png
  4. classify_piece (per piece)
       └─ detect_outside_segments          (enriches all_segments in place)
       └─ detect_frame_corners             (reads enriched segments)
                                           → classified.png, debug.png, outside_segments.png
```

## Module Structure

| File | Responsibility |
|------|---------------|
| `puzzle_finder/` | A4 detection & perspective rectification |
| `puzzle_analyzer/piece_detector.py` | Otsu threshold, contour filtering by physical area |
| `puzzle_analyzer/corner_detector.py` | PCA straight segment detection, corner classification |
| `puzzle_analyzer/outside_line_detector.py` | Detect which straight segments are on the outer frame boundary |
| `puzzle_analyzer/frame_corner_detector.py` | 4-criterion frame corner detection |
| `puzzle_analyzer/piece_classifier.py` | classify_piece, draw_classification, draw_debug |

---

## Key Decisions

### A4 orientation — landscape
The A4 sheet is always in **landscape** orientation:
- Width  = **297 mm**
- Height = **210 mm**

Used in `piece_detector.py` (min area threshold) and `piece_classifier.py` (mm length labels).

### Minimum piece area
5 cm² physical minimum, computed from image pixel dimensions vs A4 mm dimensions.
Aborts if piece count is not 4 or 6.

### Straight segment detection — PCA
`_is_straight_pca` in `corner_detector.py`:
threshold = `max(6px, segment_length * 0.05)`
Approx polygon epsilon = `0.012 * perimeter`.

---

## Segment Data Flow

`detect_corners` builds `all_segments` — a list of dicts, one per approx polygon edge:
```python
{
  'approx_index': int,
  'p1': (x, y),   # start point
  'p2': (x, y),   # end point
  'length': float, # pixel length
  'is_straight': bool,
}
```

`detect_outside_segments` **mutates these dicts in place**, adding:
```python
{
  'is_outside':   bool,
  'zone1':        np.ndarray | None,  # forbidden zone at p1 (far end of prev direction)
  'zone2':        np.ndarray | None,  # forbidden zone at p2 (far end of next direction)
  'violations1':  list,
  'violations2':  list,
}
```

This enrichment happens inside `classify_piece` before `detect_frame_corners` is called,
so criterion 4 can read the pre-computed zones instead of recomputing them.

---

## Forbidden Zone Construction (`_build_forbidden_zone`)

Lives in `frame_corner_detector.py`, imported by `outside_line_detector.py`.

```
_build_forbidden_zone(corner, far_endpoint, bisector)
```

- Direction: `far_endpoint - corner`
- Tilt: 10° toward `bisector`
- Perpendicular offset: `zone_offset=20px` **away** from bisector (outside the angle)
- Extends: `zone_backward=50px` toward corner, `zone_forward=200px` past far_endpoint
- Width: `zone_width=50px`

### Outside segment zones
`inward` = perpendicular to segment pointing toward centroid (acts as bisector).
- Zone at p1: `_build_forbidden_zone(p2, p1, inward)`
- Zone at p2: `_build_forbidden_zone(p1, p2, inward)`

### Criterion 4 reuse
Criterion 4 reads the pre-computed zones from segments:
- `prev_seg['zone1']` / `prev_seg['violations1']` → far-endpoint zone of prev segment
- `next_seg['zone2']` / `next_seg['violations2']` → far-endpoint zone of next segment

---

## Frame Corner Detection — 4 Criteria

| # | Name | Check |
|---|------|-------|
| 1 | Connection | Corner sits between two straight segments in approx polygon |
| 2 | Angle | Angle between vectors to far endpoints is 90° ± 15° |
| 3 | Inward arrow | Centroid vector has positive dot product with both edge vectors |
| 4 | Forbidden zones | Far-endpoint zones of both segments are clear of contour points |

---

## Piece Classification

| Type | Condition |
|------|-----------|
| `corner` | ≥ 1 confirmed frame corner |
| `edge` | 0 frame corners, ≥ 1 straight segment |
| `inner` | 0 straight segments |

---

## Output Images

| File | Contents |
|------|---------|
| `a4_rectified.png` | Warped A4 region |
| `threshold.png` | Otsu threshold mask |
| `pieces_detected.png` | Contours + indices |
| `corners_detected.png` | Straight edges (yellow), outer corners (orange), inner (green) |
| `classified.png` | Type-colored contours; green=frame corner segs, yellow=outside segs, magenta=other straight |
| `debug.png` | Full debug: segment IDs + mm lengths, zones, bisectors, F-markers |
| `outside_segments.png` | Outside segment detection zones (green=outside, cyan=not outside) |

---

## segment prev/next orientation (important!)

In `_build_all_segments`, segment `i` goes from `approx_flat[i]` (p1) to `approx_flat[i+1]` (p2).

In criterion 1, for a corner at `corner_idx`:
- `prev_seg = all_segments[corner_idx - 1]` → **p1 is the far end**, p2 is at the corner
- `next_seg = all_segments[corner_idx]`     → p1 is at the corner, **p2 is the far end**

So for criterion 4 / outside zones:
- Far endpoint of `prev_seg` = `prev_seg['p1']` → use `zone1` / `violations1`
- Far endpoint of `next_seg` = `next_seg['p2']` → use `zone2` / `violations2`
