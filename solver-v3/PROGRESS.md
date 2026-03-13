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

Start from a known frame corner (TL or BL), place pieces around the frame side by side.
At each step, generate all candidates, evaluate them, and optionally save a debug image per attempt.
The search is fully recursive and generic — the same `_search_step` function handles every depth.

### Key constants (`_placement_types.py`)

| Constant | Value | Meaning |
|---|---|---|
| `PUZZLE_WIDTH_MM` | 190.0 | Target length for top/bottom sides |
| `PUZZLE_HEIGHT_MM` | 128.0 | Target length for left/right sides |
| `_SIDE_TOLERANCE` | 30.0 mm | Allowed error in length matching |
| `_SECOND_CONFIGS` | `{'TL': ('top', 190, 'TR', True), 'BL': ('bottom', 190, 'BR', True)}` | Config for the first side being filled |
| `_CORNER_TURN` | 8-entry dict | Maps `(current_side, end_pos)` → `(turn_side, next_side, next_end_pos, next_fwd_is_horiz, next_start_from_end)` for all clockwise and counter-clockwise transitions |
| `_SIDE_TARGET` | dict | Target length in mm per side name |

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
| `seg_h / seg_v` | `dict \| None` | H/V segments (corner pieces only) — H is always horizontal, V always vertical |
| `is_corner` | `bool` | Whether this is a corner variant |

### Occupancy helpers

`empty_occupancy()` → `{'top': [], 'right': [], 'bottom': [], 'left': []}` where each entry is
`{'piece_idx', 'seg_id', 'length_mm'}`. Updated via `occ_add()` and `occ_add_candidate()`.

`occ_add_candidate()` takes `fwd_is_horiz` to correctly assign which segment goes along the
forward side vs the turn side:
- `fwd_is_horiz=True` (top/bottom): `seg_h` is the forward segment, `seg_v` is the turn segment
- `fwd_is_horiz=False` (right/left): `seg_v` is the forward segment, `seg_h` is the turn segment

### Geometry (`_placement_geometry.py`)

**`_place_contour(contour, seg_h, seg_v, position, px_per_mm)`**
- Snaps a corner piece to a frame corner by rotation.
- Finds the shared endpoint of the two segments, computes the rotation angle, applies it.

**`_place_contour_on_side(contour, seg, side, offset_mm, px_per_mm, start_from_end=False)`**
- Places an edge piece so its segment lies along a frame side at `offset_mm`.
- Tries **0° then 180° rotation** (pure rotation, no mirror/flip), picks whichever puts the
  piece centroid inward (inside the frame).
- `start_from_end=False` (default): `offset_mm` is measured from the near end of the side;
  the segment's **minimum** coordinate is placed at `offset_mm`.
- `start_from_end=True`: `offset_mm` is measured from the **far end** of the side;
  the segment's **maximum** coordinate is placed at `side_length - offset_mm`.
  Used when traversing bottom-up or right-to-left.

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

### Tree search (`tree_search.py`)

#### `_SearchState` dataclass

Carries all traversal context between recursive calls:
`used`, `side`, `offset_mm`, `target_mm`, `end_pos`, `fwd_is_horiz`, `start_from_end`,
`turn_side`, `occ`, `base_canvas`, `ox`, `oy`, `folder`, `prev_color`, `folder_prefix`.

#### `_search_step`

Recursive function. Places one piece at the current depth, saves its image (depending on mode),
then recurses for each valid candidate. Corner pieces trigger a side-transition via `_CORNER_TURN`;
edge pieces continue on the same side. Stops at `depth > max_depth`.

#### `visualize_start_placements`

For each piece that has a corner variant, saves `start.png` showing it placed at both TL and BL.

#### `visualize_second_placements`

Entry point for the recursive search. Sets up piece 1 state and calls `_search_step` starting
at `depth=2`. Returns `list[str]` of all valid complete branches (always, regardless of mode).

**Output mode** (controlled by `mode` parameter in `run.py`):

| Mode | Images saved | Use case |
|---|---|---|
| `console_only` | None | Fast iteration, branch summary only |
| `valid_only` | Only fully-valid branches (all steps valid to `max_depth`) | Focused review |
| `all` | Every candidate, valid and invalid | Full debug |

**Valid branch summary** is always printed to console at the end of step 6, e.g.:
```
P2@TL → P0 edge(45mm) → P3@TR corner(88mm)
```

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
                ├── P{piece3}/       ← piece 3 on same side as piece 2 (piece 2 was edge)
                │   └── ...
                └── B_P{piece3}/     ← piece 3 on new side (piece 2 was corner, B_ = side turn)
                    ├── P{piece4}/   ← piece 4 on same side as piece 3
                    └── B_P{piece4}/ ← piece 4 on new side (piece 3 was corner)
```

The `B_` prefix on a subfolder means the piece inside it is the **first piece on a new side**
(i.e. the previous piece was a corner that turned the traversal direction).

---

## Key design decisions

| Decision | Rationale |
|---|---|
| Tolerance = 30 mm | Generous to handle real-world measurement noise |
| Rotation only, no flipping | Physical pieces cannot be flipped; 180° rotation is the correct alternative orientation |
| All candidates traversed | Tree traversal always runs fully; mode only controls what is written to disk |
| `valid_branches` returned | `visualize_second_placements` returns the branch list for downstream steps (step 7+) |
| `_CORNER_TURN` lookup table | 8 entries cover all clockwise and counter-clockwise transitions; raising `max_depth` automatically navigates more sides |
| `fwd_is_horiz` in `occ_add_candidate` | `seg_h`/`seg_v` are geometric (H=horizontal, V=vertical); which one is the *forward* segment depends on the current side's orientation |
| `[FULL]` marker in legend | Immediately visible when a side is satisfied |
| Files split ≤ 250 lines each | Easier to navigate; each file has one clear responsibility |
| `start_from_end` for counter-clockwise traversal | BL→BR→TR traverses the right side bottom-up; offset measured from the far end keeps pieces adjacent to the previous corner |

---

## Fixes (2026-03-12)

### Fix 1 — Wrong offset after side turns (`tree_search.py:219`)

**Root cause**: `seg_h`/`seg_v` are geometric labels (H = horizontal, V = vertical). When a corner
piece closes a side and the traversal turns, the *turn* segment (the one covering the beginning
of the next side) is `seg_v` when `fwd_is_horiz=True` but `seg_h` when `fwd_is_horiz=False`.
The code always used `cand.seg_v['length_mm']`, which was wrong for right/left sides.

**Fix**: `tree_search.py` line 219 changed from:
```python
offset_mm=cand.seg_v['length_mm'],
```
to:
```python
offset_mm=(cand.seg_v if state.fwd_is_horiz else cand.seg_h)['length_mm'],
```

This correctly selects the turn segment's length as the starting offset on the new side,
regardless of which axis the current side runs along.

---

### Fix 2 — Segment ruler lines drawn on frame edge, not outside it (`_placement_canvas.py`)

**Root cause**: `_draw_segments` drew the H/V segment lines at their exact puzzle-mm position,
which lies on the frame edge itself. The lines were visually swallowed by the piece polygon and
the frame border, making them unreadable.

**Fix**: After transforming each segment endpoint to puzzle-mm space, the midpoint is compared
against the four frame edges (within `_RULER_MARGIN_MM = 15 mm`). The line is then shifted
`_RULER_OFFSET_MM = 8 mm` outward into the padding area before drawing. The label moves with it.

---

## Known Issues

*(none currently known)*

---

## Step 7 — Constraint Validation (`run.py`, `tree_search.py`)

### What was built

Step 7 applies `check_all()` from `constraints.py` to every valid branch produced by Step 6,
without reconstructing anything. The `PlacedPiece` objects are built **during** the tree search
and carried forward, so Step 7 is a pure post-processing loop.

### Changes

**`tree_search.py`**

- Added `from puzzle_solverv2.constraints import PlacedPiece`.
- `_search_step` gains two new parameters:
  - `placed: list` — the `list[PlacedPiece]` accumulated so far along the current branch.
  - `centroid_by_idx: dict` — lookup `{piece_idx: centroid_px}` built once from `pieces_border`.
- At each valid candidate a `PlacedPiece` is constructed inline and appended → `next_placed`.
- `results` changed from `list[str]` to `list[tuple[str, list[PlacedPiece]]]`; at `depth == max_depth`
  both the branch string and the full placed list are stored.
- `visualize_second_placements` gains a `pieces_border` parameter (optional, defaults to `None`).
  It builds `centroid_by_idx`, creates the initial `PlacedPiece` for piece 1 (the starting corner),
  and now returns `list[tuple[str, list]]` instead of `list[str]`.
- The summary print loop updated to unpack `(branch_str, _)`.

**`run.py`**

- Imports `check_all` from `puzzle_solverv2.constraints`.
- Passes `pieces_border=pieces_border` to `visualize_second_placements`.
- New **Step 7** block:
  ```python
  for branch_str, placed in valid_branches:
      ok, reason = check_all(placed, frame, total_pieces)
      if ok:   log_ok(f"PASS: {branch_str}")
      else:    log(f"FAIL [{reason}]: {branch_str}")
  ```
  Prints `N / M branch(es) passed all constraints` at the end.

### Design note

`PlacedPiece` objects are built once during the search (no re-running of placements).
Each recursive call extends the immutable `placed + [new_piece]` list, so backtracking is
naturally correct — sibling branches get independent copies.

---

## Pending

- [ ] **Full frame search**: raise `max_depth` beyond 3 once placement geometry is verified correct
  through visual inspection of multi-side traversal outputs.
