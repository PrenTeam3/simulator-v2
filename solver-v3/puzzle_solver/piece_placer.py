"""Geometric placement of puzzle pieces onto the solved puzzle canvas.

Given a corner assignment (from border_solver), places each piece by:
  1. Finding the frame corner (shared endpoint of seg_horiz and seg_vert)
  2. Computing the rotation to align the piece with the puzzle edges
  3. Transforming the contour into puzzle-mm coordinates
  4. Drawing and checking for overlaps
"""

from __future__ import annotations

import cv2
import numpy as np

PUZZLE_WIDTH_MM  = 190.0
PUZZLE_HEIGHT_MM = 128.0

# Resolution for the placement canvas (pixels per mm)
CANVAS_PX_PER_MM = 5

# Expected positions and directions in puzzle coordinate space
# horiz = direction the horizontal outside seg points away from the corner
# vert  = direction the vertical outside seg points away from the corner
POSITION_INFO = {
    'TL': {'corner_mm': np.array([0.0,            0.0           ]), 'horiz_dir': np.array([ 1.0,  0.0]), 'vert_dir': np.array([0.0,  1.0])},
    'TR': {'corner_mm': np.array([PUZZLE_WIDTH_MM, 0.0           ]), 'horiz_dir': np.array([-1.0,  0.0]), 'vert_dir': np.array([0.0,  1.0])},
    'BR': {'corner_mm': np.array([PUZZLE_WIDTH_MM, PUZZLE_HEIGHT_MM]), 'horiz_dir': np.array([-1.0,  0.0]), 'vert_dir': np.array([0.0, -1.0])},
    'BL': {'corner_mm': np.array([0.0,            PUZZLE_HEIGHT_MM]), 'horiz_dir': np.array([ 1.0,  0.0]), 'vert_dir': np.array([0.0, -1.0])},
}

PIECE_COLORS = {
    'TL': (255,  80,  80),   # red-ish
    'TR': ( 80, 255,  80),   # green-ish
    'BR': ( 80,  80, 255),   # blue-ish
    'BL': (255, 255,  80),   # yellow-ish
}


def _find_frame_corner_px(seg_h: dict, seg_v: dict) -> np.ndarray:
    """
    Find the shared endpoint (frame corner in pixels) between seg_horiz and seg_vert.
    Tries all 4 combinations of endpoints and returns the midpoint of the closest pair.
    """
    endpoints_h = [np.array(seg_h['p1'], dtype=float), np.array(seg_h['p2'], dtype=float)]
    endpoints_v = [np.array(seg_v['p1'], dtype=float), np.array(seg_v['p2'], dtype=float)]

    best_dist = float('inf')
    best_pair = (endpoints_h[0], endpoints_v[0])
    for ph in endpoints_h:
        for pv in endpoints_v:
            d = np.linalg.norm(ph - pv)
            if d < best_dist:
                best_dist = d
                best_pair = (ph, pv)

    return (best_pair[0] + best_pair[1]) / 2.0


def _far_end(seg: dict, frame_corner: np.ndarray) -> np.ndarray:
    """Return the endpoint of seg that is farthest from frame_corner (the 'far' end)."""
    p1 = np.array(seg['p1'], dtype=float)
    p2 = np.array(seg['p2'], dtype=float)
    return p2 if np.linalg.norm(p1 - frame_corner) < np.linalg.norm(p2 - frame_corner) else p1


def _rotation_matrix(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def transform_point(
    point_px: tuple,
    seg_horiz: dict,
    seg_vert: dict,
    position: str,
    px_per_mm: float,
) -> np.ndarray | None:
    """
    Transform a single point (in pixels) using the same logic as place_piece.
    Returns the point in puzzle-mm coordinates, or None if transformation fails.
    """
    pos_info         = POSITION_INFO[position]
    puzzle_corner_mm = pos_info['corner_mm']
    expected_horiz   = pos_info['horiz_dir']

    frame_corner_px = _find_frame_corner_px(seg_horiz, seg_vert)
    frame_corner_mm = frame_corner_px / px_per_mm

    far_h_px     = _far_end(seg_horiz, frame_corner_px)
    far_h_mm     = far_h_px / px_per_mm
    actual_horiz = far_h_mm - frame_corner_mm
    norm = np.linalg.norm(actual_horiz)
    if norm < 1e-6:
        return None
    actual_horiz /= norm

    angle = np.arctan2(expected_horiz[1], expected_horiz[0]) - \
            np.arctan2(actual_horiz[1],   actual_horiz[0])
    R = _rotation_matrix(angle)

    pt_mm    = np.array(point_px, dtype=float) / px_per_mm
    centered = pt_mm - frame_corner_mm
    return R @ centered + puzzle_corner_mm


def place_piece(
    contour_flat: np.ndarray,   # shape (N, 2) float32, in pixels
    seg_horiz: dict,
    seg_vert: dict,
    position: str,
    px_per_mm: float,
) -> np.ndarray:
    """
    Transform a piece contour into puzzle-mm coordinates.

    Returns the transformed contour as shape (N, 2) float64 in mm.
    Raises ValueError if the placement falls outside the puzzle canvas.
    """
    pos_info         = POSITION_INFO[position]
    puzzle_corner_mm = pos_info['corner_mm']
    expected_horiz   = pos_info['horiz_dir']

    # 1. Find frame corner in pixels, convert to mm
    frame_corner_px = _find_frame_corner_px(seg_horiz, seg_vert)
    frame_corner_mm = frame_corner_px / px_per_mm

    # 2. Compute actual horiz direction from frame corner
    far_h_px     = _far_end(seg_horiz, frame_corner_px)
    far_h_mm     = far_h_px / px_per_mm
    actual_horiz = far_h_mm - frame_corner_mm
    norm = np.linalg.norm(actual_horiz)
    if norm < 1e-6:
        raise ValueError(f"Degenerate horiz segment for position {position}")
    actual_horiz /= norm

    # 3. Compute rotation to align actual_horiz → expected_horiz
    angle = np.arctan2(expected_horiz[1], expected_horiz[0]) - \
            np.arctan2(actual_horiz[1],   actual_horiz[0])
    R = _rotation_matrix(angle)

    # 4. Transform contour: center on frame corner, rotate, translate to puzzle corner
    contour_mm = contour_flat.astype(float) / px_per_mm
    centered   = contour_mm - frame_corner_mm
    rotated    = (R @ centered.T).T
    placed     = rotated + puzzle_corner_mm

    return placed


def check_and_draw(
    assignment: dict,            # from find_corner_combinations
    corners_list: list[dict],    # from puzzle analyzer
    px_per_mm: float,
    overlap_tolerance_mm2: float = 50.0,
) -> tuple[np.ndarray, bool]:
    """
    Place all 4 pieces and draw them on a canvas.
    Returns (canvas_image, has_overlap).

    overlap_tolerance_mm2: total overlapping area in mm² below which placement
                           is still considered valid (accounts for contour noise).
    """
    canvas_h = int(PUZZLE_HEIGHT_MM * CANVAS_PX_PER_MM)
    canvas_w = int(PUZZLE_WIDTH_MM  * CANVAS_PX_PER_MM)

    canvas  = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 40  # dark background
    counter = np.zeros((canvas_h, canvas_w), dtype=np.int32)          # overlap counter

    # Draw puzzle border
    cv2.rectangle(canvas, (0, 0), (canvas_w - 1, canvas_h - 1), (200, 200, 200), 2)

    pieces_placed = 0
    for pos in ('TL', 'TR', 'BR', 'BL'):
        a            = assignment[pos]
        piece_idx    = a['piece_idx']
        seg_horiz    = a['seg_horiz']
        seg_vert     = a['seg_vert']
        contour_flat = corners_list[piece_idx]['contour_flat']

        try:
            placed_mm = place_piece(contour_flat, seg_horiz, seg_vert, pos, px_per_mm)
        except ValueError as e:
            print(f"  SKIP piece {piece_idx} at {pos}: {e}")
            continue

        # Convert mm → canvas pixels
        placed_px = (placed_mm * CANVAS_PX_PER_MM).astype(np.int32)
        pts = placed_px.reshape(-1, 1, 2)

        color = PIECE_COLORS[pos]

        # Draw filled piece (semi-transparent via overlay)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

        # Draw contour outline
        cv2.polylines(canvas, [pts], True, color, 2)

        # Update overlap counter
        mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        counter += mask.astype(np.int32)

        # Label the piece
        cx = int(np.mean(placed_px[:, 0]))
        cy = int(np.mean(placed_px[:, 1]))
        label = f"P{piece_idx} {pos}"
        cv2.putText(canvas, label, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(canvas, label, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        pieces_placed += 1

    if pieces_placed < 4:
        print(f"  WARNING: Only {pieces_placed}/4 pieces placed — combination rejected.")
        return canvas, True  # treat incomplete placement as invalid

    # Compute overlap
    overlap_mask  = counter > 1
    n_overlap_px  = int(np.sum(overlap_mask))
    n_overlap_mm2 = n_overlap_px / (CANVAS_PX_PER_MM ** 2)
    has_overlap   = n_overlap_mm2 > overlap_tolerance_mm2

    if n_overlap_px > 0:
        canvas[overlap_mask] = (0, 0, 255)
        print(f"  Overlap area: {n_overlap_mm2:.1f} mm²  "
              f"(tolerance: {overlap_tolerance_mm2:.1f} mm²)  "
              f"→ {'FAIL' if has_overlap else 'within tolerance'}")
    else:
        print("  No overlap detected.")

    return canvas, has_overlap
