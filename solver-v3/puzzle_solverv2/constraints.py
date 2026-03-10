"""Step 5 — Constraint definitions.

Defines the PlacedPiece data structure and all 5 constraints that a valid
combination must satisfy. Applied in Step 6 (tree search) in order:
cheapest first, most expensive (centroid) last.

Constraints:
  C1 — Exactly 4 frame corners filled (TL, TR, BR, BL each exactly once)
  C2 — Every frame side fully covered within tolerance (±15mm)
  C3 — Every piece used exactly once (no duplicates)
  C4 — Every piece touches the frame (has at least one assigned edge)
  C5 — Every piece centroid lands inside the frame after rotation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from puzzle_solverv2.frame import PuzzleFrame, PUZZLE_WIDTH_MM, PUZZLE_HEIGHT_MM
from puzzle_solverv2.variants import Variant


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

POSITIONS      = ('TL', 'TR', 'BR', 'BL')
SIDE_TOLERANCE = 15.0   # mm — ±15mm per side (from algorithm doc)

# Expected horizontal direction per position (in puzzle mm coords)
_EXPECTED_HORIZ = {
    'TL': np.array([ 1.0,  0.0]),
    'TR': np.array([-1.0,  0.0]),
    'BR': np.array([-1.0,  0.0]),
    'BL': np.array([ 1.0,  0.0]),
}

# Puzzle corner in mm per position
_PUZZLE_CORNER_MM = {
    'TL': np.array([0.0,              0.0             ]),
    'TR': np.array([PUZZLE_WIDTH_MM,  0.0             ]),
    'BR': np.array([PUZZLE_WIDTH_MM,  PUZZLE_HEIGHT_MM]),
    'BL': np.array([0.0,              PUZZLE_HEIGHT_MM]),
}


# ─────────────────────────────────────────────
#  PlacedPiece — the unit Step 6 builds
# ─────────────────────────────────────────────

@dataclass
class PlacedPiece:
    """One piece placed in the frame during the tree search."""
    piece_idx:   int
    variant:     Variant
    side:        str                  # 'top' | 'right' | 'bottom' | 'left'
    position:    Optional[str]        # 'TL'|'TR'|'BR'|'BL' for corner, None for edge
    horiz_seg:   Optional[dict]       # corner only: the segment along the horizontal side
    vert_seg:    Optional[dict]       # corner only: the segment along the vertical side
    centroid_px: Optional[tuple]      # (cx, cy) in pixels
    px_per_mm:   float


# ─────────────────────────────────────────────
#  Rotation helper (used by C5)
# ─────────────────────────────────────────────

def _find_frame_corner_px(seg_h: dict, seg_v: dict) -> np.ndarray:
    """Return the shared endpoint (closest pair) between seg_horiz and seg_vert."""
    pts_h = [np.array(seg_h['p1'], dtype=float), np.array(seg_h['p2'], dtype=float)]
    pts_v = [np.array(seg_v['p1'], dtype=float), np.array(seg_v['p2'], dtype=float)]
    best_dist, best = float('inf'), (pts_h[0], pts_v[0])
    for ph in pts_h:
        for pv in pts_v:
            d = np.linalg.norm(ph - pv)
            if d < best_dist:
                best_dist, best = d, (ph, pv)
    return (best[0] + best[1]) / 2.0


def _far_end(seg: dict, frame_corner: np.ndarray) -> np.ndarray:
    p1 = np.array(seg['p1'], dtype=float)
    p2 = np.array(seg['p2'], dtype=float)
    return p2 if np.linalg.norm(p1 - frame_corner) < np.linalg.norm(p2 - frame_corner) else p1


def _transform_centroid(centroid_px: tuple, seg_horiz: dict, seg_vert: dict,
                         position: str, px_per_mm: float) -> Optional[np.ndarray]:
    """
    Transform a centroid from pixel coords to puzzle-mm coords.
    Returns None if the transformation is degenerate.
    """
    puzzle_corner_mm = _PUZZLE_CORNER_MM[position]
    expected_horiz   = _EXPECTED_HORIZ[position]

    frame_corner_px = _find_frame_corner_px(seg_horiz, seg_vert)
    frame_corner_mm = frame_corner_px / px_per_mm

    far_h_px     = _far_end(seg_horiz, frame_corner_px)
    far_h_mm     = far_h_px / px_per_mm
    actual_horiz = far_h_mm - frame_corner_mm
    norm = np.linalg.norm(actual_horiz)
    if norm < 1e-6:
        return None
    actual_horiz /= norm

    angle = math.atan2(expected_horiz[1], expected_horiz[0]) - \
            math.atan2(actual_horiz[1],   actual_horiz[0])
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])

    pt_mm    = np.array(centroid_px, dtype=float) / px_per_mm
    centered = pt_mm - frame_corner_mm
    return R @ centered + puzzle_corner_mm


# ─────────────────────────────────────────────
#  Individual constraints
# ─────────────────────────────────────────────

def check_c1_corners(placed: list[PlacedPiece]) -> tuple[bool, str]:
    """C1 — Exactly 4 corners, one per position (TL/TR/BR/BL)."""
    corners = [p for p in placed if p.variant.type == 'corner' and p.position]
    if len(corners) != 4:
        return False, f"expected 4 corner pieces, got {len(corners)}"
    positions_used = [p.position for p in corners]
    for pos in POSITIONS:
        if positions_used.count(pos) != 1:
            return False, f"position {pos} not filled exactly once"
    return True, "ok"


def check_c2_side_coverage(placed: list[PlacedPiece], frame: PuzzleFrame,
                            tolerance: float = SIDE_TOLERANCE) -> tuple[bool, str]:
    """C2 — Each side length within frame target ± tolerance."""
    side_lengths: dict[str, float] = {'top': 0.0, 'right': 0.0, 'bottom': 0.0, 'left': 0.0}

    for p in placed:
        if p.variant.type == 'corner' and p.horiz_seg and p.vert_seg:
            # Horiz segment contributes to top or bottom depending on position
            horiz_side = 'top'    if p.position in ('TL', 'TR') else 'bottom'
            vert_side  = 'left'   if p.position in ('TL', 'BL') else 'right'
            side_lengths[horiz_side] += p.horiz_seg['length_mm']
            side_lengths[vert_side]  += p.vert_seg['length_mm']
        else:
            side_lengths[p.side] += p.variant.total_length_mm

    targets = {
        'top':    frame.width_mm,
        'bottom': frame.width_mm,
        'left':   frame.height_mm,
        'right':  frame.height_mm,
    }
    for side, target in targets.items():
        diff = abs(side_lengths[side] - target)
        if diff > tolerance:
            return False, (f"side '{side}': {side_lengths[side]:.1f}mm "
                           f"vs target {target:.1f}mm (diff={diff:.1f}mm > ±{tolerance}mm)")
    return True, "ok"


def check_c3_no_duplicates(placed: list[PlacedPiece], total_pieces: int) -> tuple[bool, str]:
    """C3 — Each piece used exactly once."""
    indices = [p.piece_idx for p in placed]
    if len(indices) != total_pieces:
        return False, f"expected {total_pieces} pieces, got {len(indices)}"
    if len(set(indices)) != total_pieces:
        dupes = [i for i in set(indices) if indices.count(i) > 1]
        return False, f"duplicate piece(s): {dupes}"
    return True, "ok"


def check_c4_touches_frame(placed: list[PlacedPiece]) -> tuple[bool, str]:
    """C4 — Every piece has at least one edge assigned to a frame side."""
    for p in placed:
        if not p.variant.edges:
            return False, f"piece {p.piece_idx} has no edges assigned"
    return True, "ok"


def check_c5_centroid_inside(placed: list[PlacedPiece], frame: PuzzleFrame,
                               margin: float = 10.0) -> tuple[bool, str]:
    """C5 — Every piece centroid lands inside the frame after rotation.

    Only checked for corner pieces (have horiz_seg + vert_seg for full transform).
    Edge pieces are skipped — their centroid check is deferred to Step 7.
    """
    for p in placed:
        if p.centroid_px is None:
            continue
        if p.variant.type != 'corner' or not p.horiz_seg or not p.vert_seg or not p.position:
            continue  # edge pieces deferred to Step 7

        pt = _transform_centroid(
            p.centroid_px, p.horiz_seg, p.vert_seg, p.position, p.px_per_mm
        )
        if pt is None:
            continue  # degenerate segment — don't reject

        in_x = -margin < pt[0] < frame.width_mm  + margin
        in_y = -margin < pt[1] < frame.height_mm + margin
        if not (in_x and in_y):
            return False, (f"piece {p.piece_idx} centroid ({pt[0]:.1f}, {pt[1]:.1f})mm "
                           f"outside frame ({frame.width_mm}x{frame.height_mm}mm)")
    return True, "ok"


# ─────────────────────────────────────────────
#  Combined check (all constraints in order)
# ─────────────────────────────────────────────

def check_all(placed: list[PlacedPiece], frame: PuzzleFrame,
              total_pieces: int) -> tuple[bool, str]:
    """
    Apply all 5 constraints in order (cheapest first).
    Returns (passed, reason) — reason is 'ok' or the failing constraint message.
    """
    for check, args in [
        (check_c4_touches_frame,  (placed,)),
        (check_c3_no_duplicates,  (placed, total_pieces)),
        (check_c1_corners,        (placed,)),
        (check_c2_side_coverage,  (placed, frame)),
        (check_c5_centroid_inside,(placed, frame)),
    ]:
        ok, reason = check(*args)
        if not ok:
            return False, reason
    return True, "ok"


# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────

def log_constraints(frame: PuzzleFrame) -> None:
    """Print the active constraint configuration."""
    print(f"  C1  Exactly 4 corners — positions: {POSITIONS}")
    print(f"  C2  Side coverage — top/bottom: {frame.width_mm}mm ±{SIDE_TOLERANCE}mm"
          f"  |  left/right: {frame.height_mm}mm ±{SIDE_TOLERANCE}mm")
    print(f"  C3  Each piece used exactly once")
    print(f"  C4  Each piece touches the frame")
    print(f"  C5  Centroid inside frame (corner pieces, margin ±10mm)")
