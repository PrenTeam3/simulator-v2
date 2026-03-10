# Solver V2 — Development Progress

## Overview

The solver lives in `puzzle_solverv2/` and is integrated into `run.py`.
It solves puzzle piece placement into a fixed **190 × 128 mm** frame.

---

## Modules Implemented

### Step 1 — `frame.py`
- Constants: `PUZZLE_WIDTH_MM = 190.0`, `PUZZLE_HEIGHT_MM = 128.0`, `A4_WIDTH_MM = 297.0`
- `PuzzleFrame` frozen dataclass with `width_mm`, `height_mm`, `px_per_mm`, `.px_to_mm()`, `.mm_to_px()`
- `build_frame(a4_image_width_px)` factory: `px_per_mm = a4_width_px / 297.0`

### Step 2 — `border_info.py`
- `PieceBorderInfo` dataclass: `piece_idx`, `outside_segments`, `corner_pairs`, `straight_edges`, `centroid_px`, `px_per_mm`
- Collects `is_outside=True` segments from the analyzer, converts lengths via `frame.px_to_mm()`
- Builds `corner_pairs` from `frame_corners` (two outside segments meeting at ~90°)
- Segment IDs use format `P{piece_idx}S{local_seg_idx}` — consistent with labels in `debug.png`

### Step 3 — `variants.py`
- `Variant` dataclass: `piece_idx`, `type` (`'corner'` / `'edge'`), `edges`, `total_length_mm`
- `PieceVariants` dataclass: `piece_idx`, `variants`
- Corner variants come from `corner_pairs` (2 edges each)
- Edge variants come from individual `straight_edges` (1 edge each)

### Step 4 — `similarity.py`
- Pairwise duplicate detection: area diff (±5%) → Hu Moments (`cv2.matchShapes`, threshold 0.10)
- `SimilarityResult` dataclass with `likely_duplicate` flag

### Step 5 — `constraints.py`
- `PlacedPiece` dataclass used during tree search
- 5 constraints (cheapest first):
  - **C1** — exactly 4 corners, one per position (TL/TR/BR/BL)
  - **C2** — each side within ±15 mm of target
  - **C3** — each piece used exactly once
  - **C4** — each piece has at least one frame edge assigned
  - **C5** — corner piece centroid inside frame after rotation (margin 10 mm)
- `_transform_centroid()` helper applies rotation matrix to check C5

### Step 6 — `tree_search.py` *(currently excluded from `run.py`)*
- Two start configs (TL and BL) each with clockwise DFS traversal:
  - **TL start**: top → right → bottom → left closure
  - **BL start**: bottom → right → top → left closure
- `_START_CONFIGS` list drives the side sequence per start position
- BL start swaps `edges[0] ↔ edges[1]` so horiz goes right along bottom, vert goes up along left
- `_search()` recursive DFS with backtracking (`append`/`pop`, `add`/`remove` on `used` set)
- `_close_left()` handles left-side closure: `start_vert_len + edge_pieces + end_vert_len ≈ 128 mm`
- Full debug output: `_d()` indented logger, `_side_summary()` showing all 4 sides' running totals

---

## Step 6 Visual Debug — `tree_search_v2.py`

Renamed from `placement_debug.py`. This is the active visual debug tool used to verify placement geometry before re-enabling the full tree search.

### Key geometry constants

```python
_PUZZLE_CORNERS_MM = {
    'TL': (0,   0),    'TR': (190, 0),
    'BR': (190, 128),  'BL': (0,   128),
}

_EXPECTED_HORIZ = {
    'TL': (+1, 0),   # going right along top
    'TR': (-1, 0),   # going left along top
    'BR': (-1, 0),   # going left along bottom
    'BL': (+1, 0),   # going right along bottom
}
```

Canvas: 4 px/mm with 40 mm padding, dark background, 10 mm grid.

---

### Phase 1 — `visualize_start_placements()`

For every piece with a corner variant, generates two images:

| File | Description |
|------|-------------|
| `dbg_place_P{idx}_TL.png` | Piece placed at top-left corner |
| `dbg_place_P{idx}_BL.png` | Piece placed at bottom-left corner |

**Key fix for BL**: edges must be swapped relative to TL:

```python
if position == 'TL':
    seg_h, seg_v = v.edges[0], v.edges[1]
else:  # BL
    seg_h, seg_v = v.edges[1], v.edges[0]
```

This gives the 90° rotation so that at BL the horiz segment lies along the bottom (going right) and the vert segment goes up along the left side.

The same swap is applied in `tree_search.py` for the BL start config.

**Placement function** `_place_contour(contour, seg_h, seg_v, position, px_per_mm)`:
1. Finds the shared endpoint of `seg_h` and `seg_v` → this is the piece's frame corner
2. Finds the far end of `seg_h` → computes `actual_horiz` direction
3. Rotates so `actual_horiz` aligns with `_EXPECTED_HORIZ[position]`
4. Translates so the piece's frame corner maps to `_PUZZLE_CORNERS_MM[position]`

---

### Phase 2 — `visualize_second_placements()`

For every first-piece scenario (corner piece at TL or BL), generates **one image per candidate second piece** — including invalid ones.

**File naming:**
```
dbg_step2_P{p1}_{TL|BL}_{n:03d}_P{p2}_{edge|corner}_{VALID|INVALID}.png
```

**Validity rules:**

| Candidate type | Valid condition |
|---------------|----------------|
| Edge piece on the same side | `current_length + seg_len < target − 15 mm` (leaves room for a closing corner) |
| Corner piece closing the side | `\|current_length + fwd_len − target\| ≤ 15 mm` |

**Corner piece orientation**: both `(edges[0], edges[1])` and `(edges[1], edges[0])` are tried as `(fwd, turn)` — each gets its own image (`ori0` / `ori1`).

**Visual elements per image:**
- First piece drawn (semi-transparent fill + outline + label)
- Second piece drawn on top in a different colour
- Cyan/orange segment markers (H = horizontal, V = vertical) for corner pieces
- Two-colour **progress bar** outside the frame showing cumulative coverage vs. target
- Large **`VALID`** (green) or **`INVALID`** (red) stamp top-right with the reason string

**New geometry function** `_place_contour_on_side(contour, seg, side, offset_mm, px_per_mm)`:
1. Rotates the contour so the segment aligns with the side direction `(+1,0)` or `(0,+1)`
2. Translates so the "start" endpoint (smallest coordinate along the side) lands at `offset_mm`
3. Reflects the piece body inward if it ended up on the wrong side of the frame edge

---

## `run.py` Integration

Steps run in order:

| Step label | Description | Key output files |
|-----------|-------------|-----------------|
| A1 | Detect and rectify A4 area | `a4_rectified.png` |
| A2 | Detect puzzle pieces | `pieces_detected.png` |
| A3 | Detect corners per piece | `corners_detected.png` |
| A4 | Classify pieces | `classified.png`, `debug.png`, `outside_segments.png` |
| 1  | Define puzzle frame | — |
| 2  | Extract border info | — |
| 3  | Generate variants | — |
| 4  | Pairwise similarity | — |
| 5  | Constraint config | — |
| 6-debug | Visual placement (step 1 + step 2) | `dbg_place_P*`, `dbg_step2_P*` |

Output is written to `output/{YYYYMMDD_HHMMSS}_{uuid4[:4]}/` for sortable run history.
All console output is mirrored to `run.log` in the same folder via a `_Tee` wrapper.

---

## Pending

- [ ] Verify `dbg_step2_P*` images — confirm valid candidates are geometrically correct
- [ ] Re-enable `tree_search.py` in `run.py` once geometry is confirmed
- [ ] Implement Step 7: geometric placement with overlap detection + visualization
- [ ] Implement Step 8: final output image
