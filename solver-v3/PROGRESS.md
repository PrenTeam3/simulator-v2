# Puzzle Solver v3 — Implementation Progress

## Overview

The solver determines which puzzle pieces go where in a fixed frame (**190 × 128 mm**, landscape A4).
The approach is a progressive tree search: start from one known corner, fill a side piece by piece,
and at each step visualise which candidates are valid and why.

---

## Architecture

The solver is split into numbered steps (modules), each building on the previous.

```
puzzle_solverv2/
├── frame.py              # Step 1 — frame definition & px/mm scale
├── border_info.py        # Step 2 — per-piece outside segments & corner pairs
├── variants.py           # Step 3 — corner / edge variants per piece
├── similarity.py         # Step 4 — pairwise duplicate detection
├── constraints.py        # Step 5 — 5 validity constraints (C1–C5)
├── _placement_types.py   # Step 6 support — constants, Candidate, occupancy helpers
├── _placement_geometry.py# Step 6 support — contour transforms & candidate generation
├── _placement_canvas.py  # Step 6 support — canvas creation & primitive drawing
├── _placement_drawing.py # Step 6 support — overlays: verdict, progress bar, legend
└── tree_search.py        # Step 6 — placement visualisation entry points
run.py                    # top-level runner
```

---

## Step 1 — Frame (`frame.py`)

- **Frame size**: 190 mm wide × 128 mm tall (landscape).
- **Scale**: `px_per_mm = a4_image_width_px / 297.0` (A4 landscape = 297 mm wide).
- `PuzzleFrame` dataclass: `width_mm`, `height_mm`, `px_per_mm`, with helpers `px_to_mm` / `mm_to_px`.

---

## Step 2 — Border Info (`border_info.py`)

For each detected piece:

- Collects all **outside segments** (`is_outside=True`) and converts lengths to mm.
- Segment IDs follow format `P{piece_idx}S{local_seg_idx}`.
- Detects **corner pairs**: two adjacent outside segments that meet at a ~90° frame corner
  (stored as `frame_corners` in the classification output).
- Stores piece **centroid in pixels** for later constraint checks.

Output: `PieceBorderInfo` per piece with `outside_segments`, `corner_pairs`, `straight_edges`, `centroid_px`.

---

## Step 3 — Variants (`variants.py`)

From the border info, generates all ways a piece can sit against the frame:

| Variant type | Edges | Description |
|---|---|---|
| `corner` | 2 segments | Two adjacent outside segments that together fill a frame corner |
| `edge` | 1 segment | A single outside segment placed along one frame side |

Output: `PieceVariants` per piece, each containing a `list[Variant]`.

---

## Step 4 — Similarity (`similarity.py`)

Pairwise comparison of all pieces to flag likely duplicates:

1. **Area check** (cheap): skip shape comparison if area differs by > 5%.
2. **Hu Moments** (`cv2.matchShapes`): rotation-invariant shape score; `< 0.10` → likely duplicate.

Output: `SimilarityResult` per unique pair. Used by Step 6 to avoid redundant symmetric searches.

---

## Step 5 — Constraints (`constraints.py`)

Defines `PlacedPiece` and five constraints checked in order (cheapest first):

| ID | Constraint |
|---|---|
| C4 | Every piece has at least one edge assigned to a frame side |
| C3 | Every piece used exactly once |
| C1 | Exactly 4 frame corners filled — one per position (TL / TR / BR / BL) |
| C2 | Every frame side covered within ±15 mm of target |
| C5 | Every corner piece centroid lands inside the frame after rotation |

`check_all()` applies them in sequence and returns `(passed, reason)`.

---

## Step 6 — Tree Search & Visualisation

### Core idea

Start from a known frame corner (TL or BL), place pieces along the top or bottom side one by one.
At each step, generate all candidates, evaluate them, and save a debug image for every attempt.

### Key constants (`_placement_types.py`)

| Constant | Value | Meaning |
|---|---|---|
| `PUZZLE_WIDTH_MM` | 190.0 | Target length for top/bottom sides |
| `PUZZLE_HEIGHT_MM` | 128.0 | Target length for left/right sides |
| `_SIDE_TOLERANCE` | 30.0 mm | Allowed error in length matching |
| `_SECOND_CONFIGS` | `{'TL': ('top', 190, 'TR', True), 'BL': ('bottom', 190, 'BR', True)}` | Config for the first side being filled |

### `Candidate` dataclass

Returned by `_build_candidates`. Fields:

| Field | Type | Description |
|---|---|---|
| `pv` | `PieceVariants` | Source piece |
| `variant` | `Variant` | Which variant was used |
| `placed_mm` | `ndarray (N,2)` | Contour in puzzle-mm coords |
| `seg_len` | `float` | Forward segment length (mm) |
| `valid` | `bool` | Passes all placement checks |
| `reason` | `str` | Human-readable explanation |
| `label` | `str` | Short label for canvas text |
| `seg_h / seg_v` | `dict \| None` | H/V segments (corner pieces only) |
| `is_corner` | `bool` | Whether this is a corner variant |

### Occupancy helpers

`empty_occupancy()` → `{'top': [], 'right': [], 'bottom': [], 'left': []}` where each entry is
`{'piece_idx', 'seg_id', 'length_mm'}`. Updated via `occ_add()` and `occ_add_candidate()`.

### Geometry (`_placement_geometry.py`)

**`_place_contour(contour, seg_h, seg_v, position, px_per_mm)`**
- Snaps a corner piece to a frame corner by rotation.
- Finds the shared endpoint of the two segments, computes the rotation angle, applies it.

**`_place_contour_on_side(contour, seg, side, offset_mm, px_per_mm, start_from_end=False)`**
- Places an edge piece so its segment lies along a frame side at `offset_mm`.
- Tries **0° then 180° rotation** (pure rotation, no mirror/flip), picks whichever puts the
  piece centroid inward (inside the frame).
- `start_from_end=False` (default): `offset_mm` is measured from the near end of the side
  (TL/TR for top, TR/BR for right); the segment's **minimum** coordinate is placed at `offset_mm`.
- `start_from_end=True`: `offset_mm` is measured from the **far end** of the side (BR for right);
  the segment's **maximum** coordinate is placed at `side_length - offset_mm`.
  Used when traversing the right side bottom-up (BL→BR→TR path).

**`_build_candidates(..., start_from_end=False)`**
- Iterates all pieces and variants not already used.
- **Edge variant**: valid if `offset_mm + seg_len < target - tolerance` and centroid inside frame.
  If it nearly fills the side (within tolerance), it is marked invalid — must be a corner piece instead.
- **Corner variant**: valid if `|offset_mm + fwd_len - target| ≤ tolerance` and centroid inside frame.
  Both orientations of the two edges are tried.
- `start_from_end` is forwarded to `_place_contour_on_side` for edge variants.

### Canvas & drawing (`_placement_canvas.py`, `_placement_drawing.py`)

- Canvas: `CANVAS_PX_PER_MM = 4`, `PADDING_MM = 40`, dark grey background with frame rectangle and 10 mm grid.
- `_draw_piece`: filled semi-transparent polygon + outline + centroid label.
- `_draw_segments`: draws the H (cyan) and V (orange) segments of a corner piece in puzzle-mm space.
- `_mark_corner`: green cross marker at a frame corner.
- `_draw_verdict`: VALID (green) / INVALID (red) banner top-right.
- `_draw_progress_bar`: two-colour bar above/below the frame showing cumulative side coverage.
- `_draw_legend`: bottom area text showing side occupancy (with `[FULL]` when side is covered)
  and a list of remaining candidate piece indices.

### Visualisation functions (`tree_search.py`)

#### `visualize_start_placements`

For each piece that has a corner variant, saves `start.png` showing it placed at both TL and BL.
Shows the piece, its H/V segments, the corner marker, and which pieces could follow.

#### `visualize_second_placements`

For every (first-piece, corner-variant, start-position) combination, generates images for every
possible second piece:

- **Piece 2 canvas**: piece 1 + candidate piece 2 + validity verdict + occupancy legend.
- Output: `placements/P{1}/{TL|BL}/P{2}/{n:03d}_{type}_{VALID|INVALID}.png`

**Piece 3 — Scenario A** (piece 2 was a valid edge piece):
- Piece 3 continues along the **same side** (top or bottom).
- Valid piece 3 must be a corner that closes the side within tolerance.
- `base_2` canvas shows piece 1 + piece 2; piece 3 is drawn on top.
- Output: `placements/P{1}/{TL|BL}/P{2}/P{3}/{m:03d}_{type}_{VALID|INVALID}.png`

**Piece 3 — Scenario B** (piece 2 was a valid corner piece):
- Piece 2 is at TR or BR, with a vertical segment (`seg_v`) already covering part of the right side.
- Piece 3 continues along the **right side**, placed adjacent to piece 2.
- Parameters depend on which corner piece 2 occupies:

| Start | Piece 2 at | offset_b | target_b | end_b | start_from_end |
|---|---|---|---|---|---|
| TL | TR | `seg_v` (from TR, top-down) | 128 mm | BR | False |
| BL | BR | `seg_v` (from BR, bottom-up) | 128 mm | TR | **True** |

- `start_from_end=True` for the BL case ensures piece 3 is placed **adjacent to piece 2** (just
  above it on the right side) rather than at the opposite TR corner.
- Output: `placements/P{1}/{TL|BL}/P{2}/B_P{3}/{m:03d}_{type}_{VALID|INVALID}.png`
  (`B_` prefix distinguishes Scenario B from Scenario A subfolders).

---

## Output folder structure

```
output/
└── placements/
    └── P{piece1}/
        └── {TL|BL}/
            ├── start.png
            └── P{piece2}/
                ├── 001_corner_VALID.png
                ├── 002_edge_INVALID.png
                ├── P{piece3}/           ← Scenario A (piece 2 was edge, piece 3 on same side)
                │   ├── 001_corner_VALID.png
                │   └── ...
                └── B_P{piece3}/         ← Scenario B (piece 2 was corner, piece 3 on right side)
                    ├── 001_edge_VALID.png
                    └── ...
```

---

## Key design decisions

| Decision | Rationale |
|---|---|
| Tolerance = 30 mm | Generous to handle real-world measurement noise |
| Rotation only, no flipping | Physical pieces cannot be flipped; 180° rotation is the correct alternative orientation |
| Validity logged but not filtered | All candidates saved to disk so invalid ones can be visually inspected |
| Legend shows piece indices only | Simpler to read than full segment combinations |
| `[FULL]` marker in legend | Immediately visible when a side is satisfied |
| Files split ≤ 250 lines each | Easier to navigate; each file has one clear responsibility |
| `start_from_end` for BL right-side | BL→BR traverses the right side bottom-up; placing at offset=0 (TR) would push pieces outside the frame and look disconnected from piece 2 |

---

## Pending

- [ ] **Piece 4+**: generalise the search loop to continue around the remaining sides (left side after Scenario B closes at TR/BR).
- [ ] **Full tree search**: connect visualisation to actual constraint evaluation (`check_all`).
