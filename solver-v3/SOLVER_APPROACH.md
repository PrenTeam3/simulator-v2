# Puzzle Solver — Algorithmic Approach

## Overview

The solver takes the output of the puzzle analyzer (outside segments, piece classifications, contours) and determines how the 4 puzzle pieces fit together into a 190 × 128 mm rectangle.

---

## Pipeline

### Step 1 — Extract Border Info (`border_solver.extract_border_info`)

For each piece, collect all segments marked `is_outside=True` from the analyzer. Convert their pixel lengths to mm using:

```
px_per_mm = a4_image_width_px / 297.0
length_mm = segment['length'] / px_per_mm
```

Each piece dict contains:
- `piece_idx` — index of the piece
- `type` — `'corner'` / `'edge'` / `'inner'` (from classifier, not fully trusted)
- `centroid_px` — pixel centroid of the piece contour
- `px_per_mm` — scale factor
- `outside_segments` — list of `{seg_id, piece_idx, length_mm, p1, p2}`

---

### Step 2 — Find Dimension-Valid Combinations (`border_solver.find_corner_combinations`)

#### Strategy: Corner-first with fallback

**Priority 1:** Use all pieces classified as `corner` (most reliable case).
**Fallback:** If no solution found, try all pieces regardless of classification (classification may be wrong).

#### For each permutation of 4 corner pieces → 4 puzzle positions (TL, TR, BR, BL):

Try all **4! = 24 permutations** × **2 orientations per piece** × **all segment pairs** = up to **384+ combinations**.

Each piece contributes two outside segments to its puzzle corner:
- `seg_horiz` → contributes length to the **top or bottom** edge (190 mm sides)
- `seg_vert`  → contributes length to the **left or right** edge (128 mm sides)

Both orientations (which segment is horiz vs vert) are tried because the puzzle is a **rectangle, not a square** — the two segments have different lengths.

#### Validity checks in `_check_corner_assignment`:

**1. Dimension check** — side lengths must sum correctly (within ±15 mm tolerance):
```
top    = TL.seg_horiz + TR.seg_horiz  ≈ 190 mm
bottom = BL.seg_horiz + BR.seg_horiz  ≈ 190 mm
left   = TL.seg_vert  + BL.seg_vert   ≈ 128 mm
right  = TR.seg_vert  + BR.seg_vert   ≈ 128 mm
```

**2. Centroid check** — the piece body must land *inside* the puzzle canvas after placement.
This filters out geometrically impossible orientations where the piece would face the wrong way. Uses the same transformation logic as the actual placement step.

---

### Step 3 — Geometric Placement (`piece_placer.place_piece`)

For each valid combination, place every piece geometrically onto the 190 × 128 mm canvas.

**Per piece:**

1. **Find the frame corner** — the shared endpoint of `seg_horiz` and `seg_vert` (midpoint of closest endpoint pair between the two segments).

2. **Compute rotation** — align `seg_horiz` direction to the expected direction for that position:
   - TL: horiz → (+1, 0) right,  vert → (0, +1) down
   - TR: horiz → (−1, 0) left,   vert → (0, +1) down
   - BR: horiz → (−1, 0) left,   vert → (0, −1) up
   - BL: horiz → (+1, 0) right,  vert → (0, −1) up

3. **Transform contour:**
   ```
   contour_mm = contour_px / px_per_mm
   centered   = contour_mm − frame_corner_mm
   rotated    = R × centered
   placed     = rotated + puzzle_corner_mm
   ```

---

### Step 4 — Overlap Check (`piece_placer.check_and_draw`)

For each combination with all 4 pieces successfully placed:

- Draw all 4 pieces onto a pixel canvas (5 px/mm resolution → 950 × 640 px)
- Count pixels covered by more than one piece
- Convert to mm² and compare against a tolerance (default: 50 mm²)
- Combinations within tolerance are marked **FITS**

Combinations where fewer than 4 pieces are placed are automatically rejected.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Corner-first search | Massively reduces search space (24 × 16 vs. thousands of combinations) |
| Try both orientations per piece | Puzzle is rectangular (190 ≠ 128), so segment roles can't be assumed |
| Piece type not fully trusted | Classifier can misclassify; fallback ignores type |
| Centroid check during matching | Filters geometrically wrong orientations early, before expensive placement |
| 15 mm length tolerance | Accounts for contour detection noise and segment measurement error |
| 50 mm² overlap tolerance | Small overlaps due to contour noise are acceptable |

---

## File Structure

```
puzzle_solver/
├── border_solver.py    — combination finding (Steps 1 & 2)
└── piece_placer.py     — geometric placement & overlap check (Steps 3 & 4)
```

Called from `run.py` after the full analyzer pipeline completes.
