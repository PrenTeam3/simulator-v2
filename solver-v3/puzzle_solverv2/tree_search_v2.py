"""Placement debug — visual check of TL and BL starting positions.

For every piece that has a corner variant, generates two images:
  - piece placed at TL (top-left of the puzzle frame)
  - piece placed at BL (bottom-left of the puzzle frame)

Each image shows the puzzle frame with 40mm padding on all sides so you can
clearly see whether the piece lands inside or outside the frame.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import math
import numpy as np

from puzzle_solverv2.frame import PuzzleFrame, PUZZLE_WIDTH_MM, PUZZLE_HEIGHT_MM
from puzzle_solverv2.border_info import PieceBorderInfo
from puzzle_solverv2.variants import PieceVariants


# ─────────────────────────────────────────────
#  Canvas settings
# ─────────────────────────────────────────────

CANVAS_PX_PER_MM = 4
PADDING_MM       = 40   # extra space around the frame on all sides

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

def _find_frame_corner_px(seg_h: dict, seg_v: dict) -> np.ndarray:
    """Return the shared endpoint (closest pair) between seg_h and seg_v."""
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


def _place_contour(
    contour_flat: np.ndarray,   # (N, 2) float32, in pixels
    seg_h:        dict,
    seg_v:        dict,
    position:     str,          # 'TL', 'TR', 'BR', or 'BL'
    px_per_mm:    float,
) -> np.ndarray:
    """
    Transform contour from pixel coords into puzzle-mm coords for the given position.
    Returns (N, 2) float64 in mm.
    """
    puzzle_corner_mm = _PUZZLE_CORNERS_MM[position]
    expected_horiz   = _EXPECTED_HORIZ[position]

    frame_corner_px = _find_frame_corner_px(seg_h, seg_v)
    frame_corner_mm = frame_corner_px / px_per_mm

    far_h_px     = _far_end(seg_h, frame_corner_px)
    far_h_mm     = far_h_px / px_per_mm
    actual_horiz = far_h_mm - frame_corner_mm
    norm         = np.linalg.norm(actual_horiz)
    if norm < 1e-6:
        return contour_flat.astype(float) / px_per_mm   # degenerate — return as-is

    actual_horiz /= norm
    angle = math.atan2(expected_horiz[1], expected_horiz[0]) - \
            math.atan2(actual_horiz[1],   actual_horiz[0])
    c, s = math.cos(angle), math.sin(angle)
    R    = np.array([[c, -s], [s, c]])

    contour_mm = contour_flat.astype(float) / px_per_mm
    centered   = contour_mm - frame_corner_mm
    rotated    = (R @ centered.T).T
    return rotated + puzzle_corner_mm


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
    target_dir = {
        'top':    np.array([1.0, 0.0]),
        'bottom': np.array([1.0, 0.0]),
        'left':   np.array([0.0, 1.0]),
        'right':  np.array([0.0, 1.0]),
    }[side]
    side_axis = {
        'top':    0.0,
        'bottom': PUZZLE_HEIGHT_MM,
        'left':   0.0,
        'right':  PUZZLE_WIDTH_MM,
    }[side]

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
    rotated    = (R @ contour_mm.T).T
    rp1 = R @ p1
    rp2 = R @ p2

    if side in ('top', 'bottom'):
        start_x = min(rp1[0], rp2[0])
        seg_y   = (rp1[1] + rp2[1]) / 2.0
        translation = np.array([offset_mm - start_x, side_axis - seg_y])
    else:
        start_y = min(rp1[1], rp2[1])
        seg_x   = (rp1[0] + rp2[0]) / 2.0
        translation = np.array([side_axis - seg_x, offset_mm - start_y])

    placed   = rotated + translation
    centroid = np.mean(placed, axis=0)

    # Reflect piece body inward if it ended up on the wrong side of the frame edge
    if side == 'top'    and centroid[1] < side_axis:
        placed[:, 1] = 2 * side_axis - placed[:, 1]
    elif side == 'bottom' and centroid[1] > side_axis:
        placed[:, 1] = 2 * side_axis - placed[:, 1]
    elif side == 'left'   and centroid[0] < side_axis:
        placed[:, 0] = 2 * side_axis - placed[:, 0]
    elif side == 'right'  and centroid[0] > side_axis:
        placed[:, 0] = 2 * side_axis - placed[:, 0]

    return placed


# ─────────────────────────────────────────────
#  Drawing
# ─────────────────────────────────────────────

def _make_canvas() -> tuple[np.ndarray, int, int]:
    """Create a blank canvas with padding. Returns (image, frame_origin_x_px, frame_origin_y_px)."""
    total_w = int((PUZZLE_WIDTH_MM  + 2 * PADDING_MM) * CANVAS_PX_PER_MM)
    total_h = int((PUZZLE_HEIGHT_MM + 2 * PADDING_MM) * CANVAS_PX_PER_MM)
    canvas  = np.full((total_h, total_w, 3), 40, dtype=np.uint8)   # dark background

    # Frame origin in canvas pixels
    ox = int(PADDING_MM * CANVAS_PX_PER_MM)
    oy = int(PADDING_MM * CANVAS_PX_PER_MM)

    # Draw frame rectangle
    fw = int(PUZZLE_WIDTH_MM  * CANVAS_PX_PER_MM)
    fh = int(PUZZLE_HEIGHT_MM * CANVAS_PX_PER_MM)
    cv2.rectangle(canvas, (ox, oy), (ox + fw, oy + fh), (200, 200, 200), 2)

    # Label corners
    for label, (mx, my) in [('TL', (0, 0)), ('TR', (PUZZLE_WIDTH_MM, 0)),
                              ('BR', (PUZZLE_WIDTH_MM, PUZZLE_HEIGHT_MM)),
                              ('BL', (0, PUZZLE_HEIGHT_MM))]:
        px = ox + int(mx * CANVAS_PX_PER_MM)
        py = oy + int(my * CANVAS_PX_PER_MM)
        cv2.circle(canvas, (px, py), 5, (200, 200, 200), -1)
        cv2.putText(canvas, label, (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Grid lines at 10mm intervals (subtle)
    for x_mm in range(0, int(PUZZLE_WIDTH_MM) + 1, 10):
        x_px = ox + int(x_mm * CANVAS_PX_PER_MM)
        cv2.line(canvas, (x_px, oy), (x_px, oy + fh), (60, 60, 60), 1)
    for y_mm in range(0, int(PUZZLE_HEIGHT_MM) + 1, 10):
        y_px = oy + int(y_mm * CANVAS_PX_PER_MM)
        cv2.line(canvas, (ox, y_px), (ox + fw, y_px), (60, 60, 60), 1)

    return canvas, ox, oy


def _draw_piece(
    canvas:    np.ndarray,
    placed_mm: np.ndarray,   # (N, 2) contour in puzzle-mm coords
    ox:        int,
    oy:        int,
    color:     tuple,
    label:     str,
) -> None:
    """Draw a placed piece on the canvas."""
    # Convert mm → canvas pixels (with padding offset)
    pts = (placed_mm * CANVAS_PX_PER_MM).astype(np.int32)
    pts[:, 0] += ox
    pts[:, 1] += oy
    pts_cv = pts.reshape(-1, 1, 2)

    # Fill semi-transparent
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts_cv], color)
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

    # Outline
    cv2.polylines(canvas, [pts_cv], True, color, 2)

    # Label at centroid
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    cv2.putText(canvas, label, (cx - 20, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
    cv2.putText(canvas, label, (cx - 20, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _draw_segments(
    canvas:    np.ndarray,
    seg_h:     dict,
    seg_v:     dict,
    position:  str,
    px_per_mm: float,
    ox:        int,
    oy:        int,
) -> None:
    """Draw the horiz and vert segments used for placement in contrasting colors."""
    puzzle_corner_mm = _PUZZLE_CORNERS_MM[position]
    expected_horiz   = _EXPECTED_HORIZ[position]

    frame_corner_px = _find_frame_corner_px(seg_h, seg_v)
    frame_corner_mm = frame_corner_px / px_per_mm

    far_h_px     = _far_end(seg_h, frame_corner_px)
    far_h_mm     = far_h_px / px_per_mm
    actual_horiz = far_h_mm - frame_corner_mm
    norm         = np.linalg.norm(actual_horiz)
    if norm < 1e-6:
        return

    actual_horiz /= norm
    angle = math.atan2(expected_horiz[1], expected_horiz[0]) - \
            math.atan2(actual_horiz[1],   actual_horiz[0])
    c, s = math.cos(angle), math.sin(angle)
    R    = np.array([[c, -s], [s, c]])

    def _transform(p_px):
        pt_mm    = np.array(p_px, dtype=float) / px_per_mm
        centered = pt_mm - frame_corner_mm
        return R @ centered + puzzle_corner_mm

    for seg, color, tag in [(seg_h, (0, 220, 255), 'H'), (seg_v, (255, 180, 0), 'V')]:
        p1 = _transform(seg['p1'])
        p2 = _transform(seg['p2'])
        px1 = (ox + int(p1[0] * CANVAS_PX_PER_MM), oy + int(p1[1] * CANVAS_PX_PER_MM))
        px2 = (ox + int(p2[0] * CANVAS_PX_PER_MM), oy + int(p2[1] * CANVAS_PX_PER_MM))
        cv2.line(canvas, px1, px2, color, 3)
        mid = ((px1[0] + px2[0]) // 2, (px1[1] + px2[1]) // 2)
        cv2.putText(canvas, f"{tag}:{seg['seg_id']} {seg['length_mm']:.0f}mm",
                    (mid[0] + 4, mid[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


# ─────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────

def visualize_start_placements(
    pieces_border:   list[PieceBorderInfo],
    pieces_variants: list[PieceVariants],
    corners_list:    list[dict],
    frame:           PuzzleFrame,
    output_dir:      Path,
) -> None:
    """
    For each piece with a corner variant, save two images:
      dbg_place_P{idx}_TL.png
      dbg_place_P{idx}_BL.png

    Each image shows the piece placed at TL or BL within the padded frame.
    Horiz segment = cyan, vert segment = orange.
    """
    border_by_idx = {p.piece_idx: p for p in pieces_border}
    COLORS = [
        (100, 200, 255),
        (100, 255, 150),
        (255, 150, 100),
        (200, 100, 255),
        (255, 255, 100),
        (100, 255, 255),
    ]

    for pv in pieces_variants:
        corner_variants = [v for v in pv.variants if v.type == 'corner']
        if not corner_variants:
            continue

        pb          = border_by_idx[pv.piece_idx]
        contour_raw = corners_list[pv.piece_idx]['contour_flat']
        color       = COLORS[pv.piece_idx % len(COLORS)]

        for v in corner_variants:
            for position in ('TL', 'BL'):
                # TL: edges[0] → horiz (along top, going right),  edges[1] → vert (↓ along left)
                # BL: edges[1] → horiz (along bottom, going right),edges[0] → vert (↑ along left)
                # Swapping gives the 90° rotation needed to go from TL to BL
                if position == 'TL':
                    seg_h, seg_v = v.edges[0], v.edges[1]
                else:
                    seg_h, seg_v = v.edges[1], v.edges[0]

                canvas, ox, oy = _make_canvas()

                # Place contour
                placed_mm = _place_contour(
                    contour_raw, seg_h, seg_v, position, frame.px_per_mm
                )
                label = f"P{pv.piece_idx}@{position}"
                _draw_piece(canvas, placed_mm, ox, oy, color, label)

                # Draw the two segments used for placement
                _draw_segments(canvas, seg_h, seg_v, position, frame.px_per_mm, ox, oy)

                # Mark the expected frame corner
                fx = ox + int(_PUZZLE_CORNERS_MM[position][0] * CANVAS_PX_PER_MM)
                fy = oy + int(_PUZZLE_CORNERS_MM[position][1] * CANVAS_PX_PER_MM)
                cv2.drawMarker(canvas, (fx, fy), (0, 255, 0),
                               cv2.MARKER_CROSS, 20, 2)
                cv2.putText(canvas, f"expected {position}",
                            (fx + 8, fy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                # Title
                centroid_info = f"centroid={pb.centroid_px}" if pb.centroid_px else ""
                title = (f"P{pv.piece_idx} at {position}  "
                         f"H:{seg_h['seg_id']}({seg_h['length_mm']:.1f}mm)  "
                         f"V:{seg_v['seg_id']}({seg_v['length_mm']:.1f}mm)  "
                         f"{centroid_info}")
                cv2.putText(canvas, title, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

                fname = output_dir / f"dbg_place_P{pv.piece_idx}_{position}.png"
                cv2.imwrite(str(fname), canvas)
                print(f"  [OUT] {fname.name}")


def _draw_verdict(canvas: np.ndarray, valid: bool, reason: str) -> None:
    """Stamp a large VALID / INVALID banner in the top-right corner."""
    h, w = canvas.shape[:2]
    text  = "VALID"   if valid else "INVALID"
    color = (0, 220, 80) if valid else (0, 60, 220)
    # large outline + fill
    cv2.putText(canvas, text, (w - 220, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 6)
    cv2.putText(canvas, text, (w - 220, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    cv2.putText(canvas, reason, (w - 350, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)


def _draw_progress_bar(
    canvas: np.ndarray,
    ox: int, oy: int,
    side_name: str,
    seg1_len: float,
    seg2_len: float,
    color1: tuple,
    color2: tuple,
    target: float,
) -> None:
    """Draw a two-colour progress bar outside the frame showing coverage."""
    if side_name == 'top':
        bar_y = oy - 8
    elif side_name == 'bottom':
        bar_y = oy + int(PUZZLE_HEIGHT_MM * CANVAS_PX_PER_MM) + 8
    else:
        return  # left/right not needed here

    x0 = ox
    x1 = ox + int(seg1_len * CANVAS_PX_PER_MM)
    x2 = ox + int((seg1_len + seg2_len) * CANVAS_PX_PER_MM)
    xt = ox + int(target * CANVAS_PX_PER_MM)

    cv2.line(canvas, (x0, bar_y), (x1, bar_y), color1, 4)
    cv2.line(canvas, (x1, bar_y), (x2, bar_y), color2, 4)
    # target marker
    cv2.line(canvas, (xt, bar_y - 6), (xt, bar_y + 6), (200, 200, 200), 2)
    cv2.putText(canvas, f"{target:.0f}mm", (xt - 20, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    # total label
    total = seg1_len + seg2_len
    cv2.putText(canvas, f"{total:.0f}mm", (x2 + 4, bar_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color2, 1)


def visualize_second_placements(
    pieces_border:   list[PieceBorderInfo],
    pieces_variants: list[PieceVariants],
    corners_list:    list[dict],
    frame:           PuzzleFrame,
    output_dir:      Path,
    tolerance:       float = _SIDE_TOLERANCE,
) -> None:
    """
    For every first-piece scenario (corner piece at TL or BL) and every possible
    second piece, save one image clearly labelled VALID or INVALID.

    Validity rules:
      - Edge piece:   new_total < target - tolerance  (room left for closing corner)
      - Corner piece: |new_total - target| <= tolerance  (closes the side within ±tol)

    Files: dbg_step2_P{p1}_{pos}_{n:03d}_P{p2}_{type}_{VALID|INVALID}.png
    """
    COLORS = [
        (100, 200, 255),
        (100, 255, 150),
        (255, 150, 100),
        (200, 100, 255),
        (255, 255, 100),
        (100, 255, 255),
    ]

    for pv in pieces_variants:
        corner_variants = [v for v in pv.variants if v.type == 'corner']
        if not corner_variants:
            continue

        contour_1 = corners_list[pv.piece_idx]['contour_flat']
        color_1   = COLORS[pv.piece_idx % len(COLORS)]

        for v in corner_variants:
            for position in ('TL', 'BL'):
                if position == 'TL':
                    seg_h, seg_v = v.edges[0], v.edges[1]
                else:
                    seg_h, seg_v = v.edges[1], v.edges[0]

                side_name, target, end_pos, fwd_is_horiz = _SECOND_CONFIGS[position]
                current_length = seg_h['length_mm']
                n = 0   # image counter per scenario

                # Pre-render first piece so we can copy() it per candidate
                base_canvas, ox, oy = _make_canvas()
                placed_1 = _place_contour(contour_1, seg_h, seg_v, position, frame.px_per_mm)
                _draw_piece(base_canvas, placed_1, ox, oy, color_1,
                            f"P{pv.piece_idx}@{position}")
                _draw_segments(base_canvas, seg_h, seg_v, position, frame.px_per_mm, ox, oy)
                fx = ox + int(_PUZZLE_CORNERS_MM[position][0] * CANVAS_PX_PER_MM)
                fy = oy + int(_PUZZLE_CORNERS_MM[position][1] * CANVAS_PX_PER_MM)
                cv2.drawMarker(base_canvas, (fx, fy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

                # ── Iterate all other pieces and all their variants ────────────
                for pv2 in pieces_variants:
                    if pv2.piece_idx == pv.piece_idx:
                        continue

                    contour_2 = corners_list[pv2.piece_idx]['contour_flat']
                    color_2   = COLORS[pv2.piece_idx % len(COLORS)]

                    for v2 in pv2.variants:

                        candidates_for_v2 = []   # list of (placed_2, seg2_len, valid, reason, label, seg_h2, seg_v2, is_corner)

                        if v2.type == 'edge':
                            seg2    = v2.edges[0]
                            new_len = current_length + seg2['length_mm']
                            placed_2 = _place_contour_on_side(
                                contour_2, seg2, side_name, current_length, frame.px_per_mm
                            )
                            if new_len < target - tolerance:
                                valid  = True
                                reason = f"{current_length:.0f}+{seg2['length_mm']:.0f}={new_len:.0f}mm  (room={target-new_len:.0f}mm)"
                            else:
                                valid  = False
                                reason = f"{current_length:.0f}+{seg2['length_mm']:.0f}={new_len:.0f}mm  overshoot (tol={tolerance:.0f}mm)"
                            candidates_for_v2.append(
                                (placed_2, seg2['length_mm'], valid, reason,
                                 f"P{pv2.piece_idx} edge {seg2['seg_id']}({seg2['length_mm']:.0f}mm)",
                                 None, None, False)
                            )

                        elif v2.type == 'corner' and len(v2.edges) >= 2:
                            for ori_n, (fwd, turn) in enumerate(
                                [(v2.edges[0], v2.edges[1]), (v2.edges[1], v2.edges[0])]
                            ):
                                new_len = current_length + fwd['length_mm']
                                diff    = abs(new_len - target)
                                seg_h2  = fwd  if fwd_is_horiz else turn
                                seg_v2  = turn if fwd_is_horiz else fwd
                                placed_2 = _place_contour(
                                    contour_2, seg_h2, seg_v2, end_pos, frame.px_per_mm
                                )
                                if diff <= tolerance:
                                    valid  = True
                                    reason = (f"{current_length:.0f}+{fwd['length_mm']:.0f}"
                                              f"={new_len:.0f}mm  diff={diff:.1f}mm ≤ {tolerance:.0f}mm")
                                else:
                                    valid  = False
                                    reason = (f"{current_length:.0f}+{fwd['length_mm']:.0f}"
                                              f"={new_len:.0f}mm  diff={diff:.1f}mm > {tolerance:.0f}mm")
                                candidates_for_v2.append(
                                    (placed_2, fwd['length_mm'], valid, reason,
                                     f"P{pv2.piece_idx}@{end_pos} ori{ori_n} fwd={fwd['seg_id']}({fwd['length_mm']:.0f}mm)",
                                     seg_h2, seg_v2, True)
                                )

                        for placed_2, seg2_len, valid, reason, label, sh2, sv2, is_corner in candidates_for_v2:
                            n += 1
                            canvas = base_canvas.copy()

                            # Second piece
                            _draw_piece(canvas, placed_2, ox, oy, color_2,
                                        f"P{pv2.piece_idx}")
                            if is_corner and sh2 and sv2:
                                _draw_segments(canvas, sh2, sv2, end_pos,
                                               frame.px_per_mm, ox, oy)
                                ex = ox + int(_PUZZLE_CORNERS_MM[end_pos][0] * CANVAS_PX_PER_MM)
                                ey = oy + int(_PUZZLE_CORNERS_MM[end_pos][1] * CANVAS_PX_PER_MM)
                                cv2.drawMarker(canvas, (ex, ey), (0, 255, 0),
                                               cv2.MARKER_CROSS, 20, 2)

                            # Progress bar
                            _draw_progress_bar(canvas, ox, oy, side_name,
                                               current_length, seg2_len,
                                               color_1, color_2, target)

                            # Title line
                            title = (f"P{pv.piece_idx}@{position}  H={current_length:.0f}mm"
                                     f"  →{side_name}({target:.0f}mm)  +  {label}")
                            cv2.putText(canvas, title, (10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

                            # VALID / INVALID verdict
                            _draw_verdict(canvas, valid, reason)

                            tag   = "VALID" if valid else "INVALID"
                            fname = (output_dir /
                                     f"dbg_step2_P{pv.piece_idx}_{position}"
                                     f"_{n:03d}_P{pv2.piece_idx}_{v2.type}_{tag}.png")
                            cv2.imwrite(str(fname), canvas)
                            print(f"  [OUT] {fname.name}")
