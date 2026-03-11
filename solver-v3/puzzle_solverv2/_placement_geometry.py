"""Geometry helpers for placing puzzle pieces in puzzle-mm coordinates."""

from __future__ import annotations

import math

import numpy as np

from puzzle_solverv2.frame import PUZZLE_WIDTH_MM, PUZZLE_HEIGHT_MM


# ─────────────────────────────────────────────
#  Shared constants
# ─────────────────────────────────────────────

# Puzzle corner positions in mm (within the padded canvas)
_PUZZLE_CORNERS_MM = {
    'TL': np.array([0.0,             0.0            ]),
    'TR': np.array([PUZZLE_WIDTH_MM, 0.0            ]),
    'BR': np.array([PUZZLE_WIDTH_MM, PUZZLE_HEIGHT_MM]),
    'BL': np.array([0.0,             PUZZLE_HEIGHT_MM]),
}

# Expected horiz direction per position (going away from the corner along the frame)
_EXPECTED_HORIZ = {
    'TL': np.array([ 1.0,  0.0]),   # going right along top
    'TR': np.array([-1.0,  0.0]),   # going left along top
    'BR': np.array([-1.0,  0.0]),   # going left along bottom
    'BL': np.array([ 1.0,  0.0]),   # going right along bottom
}

# Side tolerance (mm) — same as tree search
_SIDE_TOLERANCE = 15.0

# For each start position: (side_name, target_mm, closing_corner_pos, fwd_is_horiz)
_SECOND_CONFIGS = {
    'TL': ('top',    PUZZLE_WIDTH_MM,  'TR', True),
    'BL': ('bottom', PUZZLE_WIDTH_MM,  'BR', True),
}


# ─────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────

def _corner_transform(
    seg_h: dict,
    seg_v: dict,
    position: str,
    px_per_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Compute the rotation matrix and corner offsets for placing/drawing at `position`.

    Returns (R, frame_corner_mm, puzzle_corner_mm), or None if the segment is degenerate.
    Absorbs the shared logic of finding the shared frame corner, the far end of seg_h,
    and computing the alignment rotation.
    """
    pts_h = [np.array(seg_h['p1'], dtype=float), np.array(seg_h['p2'], dtype=float)]
    pts_v = [np.array(seg_v['p1'], dtype=float), np.array(seg_v['p2'], dtype=float)]
    best_dist, best = float('inf'), (pts_h[0], pts_v[0])
    for ph in pts_h:
        for pv in pts_v:
            d = np.linalg.norm(ph - pv)
            if d < best_dist:
                best_dist, best = d, (ph, pv)
    frame_corner_px = (best[0] + best[1]) / 2.0
    frame_corner_mm = frame_corner_px / px_per_mm

    # Far end of seg_h (the end farthest from the shared corner)
    p1h = np.array(seg_h['p1'], dtype=float)
    p2h = np.array(seg_h['p2'], dtype=float)
    far_h_px = p2h if np.linalg.norm(p1h - frame_corner_px) < np.linalg.norm(p2h - frame_corner_px) else p1h
    actual_horiz = far_h_px / px_per_mm - frame_corner_mm
    norm = np.linalg.norm(actual_horiz)
    if norm < 1e-6:
        return None

    actual_horiz /= norm
    expected_horiz = _EXPECTED_HORIZ[position]
    angle = math.atan2(expected_horiz[1], expected_horiz[0]) - \
            math.atan2(actual_horiz[1],   actual_horiz[0])
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return R, frame_corner_mm, _PUZZLE_CORNERS_MM[position]


def _place_contour(
    contour_flat: np.ndarray,   # (N, 2) float32, in pixels
    seg_h:        dict,
    seg_v:        dict,
    position:     str,          # 'TL', 'TR', 'BR', or 'BL'
    px_per_mm:    float,
) -> np.ndarray:
    """Transform contour from pixel coords into puzzle-mm coords for the given position."""
    result = _corner_transform(seg_h, seg_v, position, px_per_mm)
    if result is None:
        return contour_flat.astype(float) / px_per_mm
    R, frame_corner_mm, puzzle_corner_mm = result
    contour_mm = contour_flat.astype(float) / px_per_mm
    return (R @ (contour_mm - frame_corner_mm).T).T + puzzle_corner_mm


def _place_contour_on_side(
    contour_flat: np.ndarray,   # (N, 2) float32, in pixels
    seg:          dict,
    side:         str,          # 'top' | 'bottom' | 'left' | 'right'
    offset_mm:    float,        # start offset along the side from its origin corner
    px_per_mm:    float,
) -> np.ndarray:
    """
    Place an edge piece so that `seg` lies along the given frame side
    starting at offset_mm.  The piece body is reflected inward if needed.
    Returns (N, 2) float64 in puzzle-mm coords.
    """
    horiz = side in ('top', 'bottom')
    ax, perp = (0, 1) if horiz else (1, 0)
    side_axis = {
        'top': 0.0, 'bottom': PUZZLE_HEIGHT_MM,
        'left': 0.0, 'right': PUZZLE_WIDTH_MM,
    }[side]
    target_dir = np.array([1.0, 0.0]) if horiz else np.array([0.0, 1.0])

    p1 = np.array(seg['p1'], dtype=float) / px_per_mm
    p2 = np.array(seg['p2'], dtype=float) / px_per_mm
    seg_vec = p2 - p1
    norm = np.linalg.norm(seg_vec)
    if norm < 1e-6:
        return contour_flat.astype(float) / px_per_mm
    seg_vec /= norm

    angle = math.atan2(target_dir[1], target_dir[0]) - math.atan2(seg_vec[1], seg_vec[0])
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])

    contour_mm = contour_flat.astype(float) / px_per_mm
    rotated = (R @ contour_mm.T).T
    rp1, rp2 = R @ p1, R @ p2

    start_along = min(rp1[ax], rp2[ax])
    seg_perp    = (rp1[perp] + rp2[perp]) / 2.0
    translation = np.zeros(2)
    translation[ax]   = offset_mm - start_along
    translation[perp] = side_axis - seg_perp

    placed   = rotated + translation
    centroid = np.mean(placed, axis=0)

    # Reflect piece body inward if it ended up on the wrong side of the frame edge
    inward_wrong = (side == 'top'    and centroid[1] < side_axis) or \
                   (side == 'bottom' and centroid[1] > side_axis) or \
                   (side == 'left'   and centroid[0] < side_axis) or \
                   (side == 'right'  and centroid[0] > side_axis)
    if inward_wrong:
        placed[:, perp] = 2 * side_axis - placed[:, perp]

    return placed
