"""Drawing helpers for rendering placed puzzle pieces onto a canvas."""

from __future__ import annotations

import cv2
import numpy as np

from puzzle_solverv2.frame import PUZZLE_WIDTH_MM, PUZZLE_HEIGHT_MM
from puzzle_solverv2._placement_geometry import _PUZZLE_CORNERS_MM, _SIDE_TOLERANCE, _corner_transform, _place_contour


# ─────────────────────────────────────────────
#  Canvas settings
# ─────────────────────────────────────────────

CANVAS_PX_PER_MM = 4
PADDING_MM       = 40   # extra space around the frame on all sides

# Colors shared by both public functions
COLORS = [
    (100, 200, 255),
    (100, 255, 150),
    (255, 150, 100),
    (200, 100, 255),
    (255, 255, 100),
    (100, 255, 255),
]


# ─────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────

def _to_px(pt_mm: np.ndarray, ox: int, oy: int) -> tuple[int, int]:
    """Convert a puzzle-mm point to canvas pixel coordinates."""
    return (ox + int(pt_mm[0] * CANVAS_PX_PER_MM),
            oy + int(pt_mm[1] * CANVAS_PX_PER_MM))


def _make_canvas() -> tuple[np.ndarray, int, int]:
    """Create a blank canvas with padding. Returns (image, frame_origin_x_px, frame_origin_y_px)."""
    total_w = int((PUZZLE_WIDTH_MM  + 2 * PADDING_MM) * CANVAS_PX_PER_MM)
    total_h = int((PUZZLE_HEIGHT_MM + 2 * PADDING_MM) * CANVAS_PX_PER_MM)
    canvas  = np.full((total_h, total_w, 3), 40, dtype=np.uint8)

    ox = int(PADDING_MM * CANVAS_PX_PER_MM)
    oy = int(PADDING_MM * CANVAS_PX_PER_MM)
    fw = int(PUZZLE_WIDTH_MM  * CANVAS_PX_PER_MM)
    fh = int(PUZZLE_HEIGHT_MM * CANVAS_PX_PER_MM)
    cv2.rectangle(canvas, (ox, oy), (ox + fw, oy + fh), (200, 200, 200), 2)

    for label, (mx, my) in [('TL', (0, 0)), ('TR', (PUZZLE_WIDTH_MM, 0)),
                              ('BR', (PUZZLE_WIDTH_MM, PUZZLE_HEIGHT_MM)),
                              ('BL', (0, PUZZLE_HEIGHT_MM))]:
        px = ox + int(mx * CANVAS_PX_PER_MM)
        py = oy + int(my * CANVAS_PX_PER_MM)
        cv2.circle(canvas, (px, py), 5, (200, 200, 200), -1)
        cv2.putText(canvas, label, (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

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
    pts = (placed_mm * CANVAS_PX_PER_MM).astype(np.int32)
    pts[:, 0] += ox
    pts[:, 1] += oy
    pts_cv = pts.reshape(-1, 1, 2)

    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts_cv], color)
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
    cv2.polylines(canvas, [pts_cv], True, color, 2)

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
    result = _corner_transform(seg_h, seg_v, position, px_per_mm)
    if result is None:
        return
    R, frame_corner_mm, puzzle_corner_mm = result

    def _transform(p_px):
        pt_mm = np.array(p_px, dtype=float) / px_per_mm
        return R @ (pt_mm - frame_corner_mm) + puzzle_corner_mm

    for seg, color, tag in [(seg_h, (0, 220, 255), 'H'), (seg_v, (255, 180, 0), 'V')]:
        p1, p2 = _transform(seg['p1']), _transform(seg['p2'])
        px1, px2 = _to_px(p1, ox, oy), _to_px(p2, ox, oy)
        cv2.line(canvas, px1, px2, color, 3)
        mid = ((px1[0] + px2[0]) // 2, (px1[1] + px2[1]) // 2)
        cv2.putText(canvas, f"{tag}:{seg['seg_id']} {seg['length_mm']:.0f}mm",
                    (mid[0] + 4, mid[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def _draw_verdict(canvas: np.ndarray, valid: bool, reason: str) -> None:
    """Stamp a large VALID / INVALID banner in the top-right corner."""
    h, w = canvas.shape[:2]
    text  = "VALID"   if valid else "INVALID"
    color = (0, 220, 80) if valid else (0, 60, 220)
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
    cv2.line(canvas, (xt, bar_y - 6), (xt, bar_y + 6), (200, 200, 200), 2)
    cv2.putText(canvas, f"{target:.0f}mm", (xt - 20, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    total = seg1_len + seg2_len
    cv2.putText(canvas, f"{total:.0f}mm", (x2 + 4, bar_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color2, 1)


# ─────────────────────────────────────────────
#  Legend helpers
# ─────────────────────────────────────────────

_SIDE_TARGETS = {
    'top':    PUZZLE_WIDTH_MM,
    'bottom': PUZZLE_WIDTH_MM,
    'right':  PUZZLE_HEIGHT_MM,
    'left':   PUZZLE_HEIGHT_MM,
}


def _side_lines(side_occ: dict) -> list[str]:
    """Format side-occupancy state as text lines for the legend."""
    lines = ['FRAME SIDES:']
    for side in ('top', 'right', 'bottom', 'left'):
        segs   = side_occ.get(side, [])
        target = _SIDE_TARGETS[side]
        if segs:
            total = sum(s['length_mm'] for s in segs)
            parts = ' + '.join(f"P{s['piece_idx']}*{s['seg_id']}({s['length_mm']:.0f}mm)" for s in segs)
            full  = ' [FULL]' if total >= target - _SIDE_TOLERANCE else ''
            lines.append(f"  {side.upper():<8} {parts} = {total:.0f}/{target:.0f}mm{full}")
        else:
            lines.append(f"  {side.upper():<8} --  (target {target:.0f}mm)")
    return lines


def _candidate_lines(candidates: list) -> list[str]:
    """Format a candidates list as text lines for the legend."""
    if not candidates:
        return ['NEXT: (none)']
    lines = ['NEXT CANDIDATES:']
    for pv, v, placed, seg_len, valid, reason, label, sh, sv, is_corner in candidates:
        tag = 'OK ' if valid else 'NO '
        lines.append(f'  {tag}  {label}  —  {reason}')
    return lines


def _draw_legend(canvas: np.ndarray, ox: int, oy: int, lines: list[str]) -> None:
    """Draw legend text lines in the bottom padding area."""
    y = oy + int(PUZZLE_HEIGHT_MM * CANVAS_PX_PER_MM) + 20
    for line in lines:
        if y >= canvas.shape[0] - 4:
            break
        cv2.putText(canvas, line, (ox, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)
        y += 13
