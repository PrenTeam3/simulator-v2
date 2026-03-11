"""Canvas creation and primitive piece-drawing helpers."""

from __future__ import annotations

import cv2
import numpy as np

from puzzle_solverv2.frame import PUZZLE_WIDTH_MM, PUZZLE_HEIGHT_MM
from puzzle_solverv2._placement_types import _PUZZLE_CORNERS_MM
from puzzle_solverv2._placement_geometry import _corner_transform


CANVAS_PX_PER_MM: int   = 4
PADDING_MM:       float = 40.0

COLORS: list[tuple[int, int, int]] = [
    (100, 200, 255),
    (100, 255, 150),
    (255, 150, 100),
    (200, 100, 255),
    (255, 255, 100),
    (100, 255, 255),
]


def _to_px(pt_mm: np.ndarray, ox: int, oy: int) -> tuple[int, int]:
    """Convert a puzzle-mm point to canvas pixel coordinates."""
    return (ox + int(pt_mm[0] * CANVAS_PX_PER_MM),
            oy + int(pt_mm[1] * CANVAS_PX_PER_MM))


def _make_canvas() -> tuple[np.ndarray, int, int]:
    """Create a blank canvas with padding around the puzzle frame.

    Returns (image, frame_origin_x_px, frame_origin_y_px).
    """
    total_w = int((PUZZLE_WIDTH_MM  + 2 * PADDING_MM) * CANVAS_PX_PER_MM)
    total_h = int((PUZZLE_HEIGHT_MM + 2 * PADDING_MM) * CANVAS_PX_PER_MM)
    canvas  = np.full((total_h, total_w, 3), 40, dtype=np.uint8)

    ox = int(PADDING_MM * CANVAS_PX_PER_MM)
    oy = int(PADDING_MM * CANVAS_PX_PER_MM)
    fw = int(PUZZLE_WIDTH_MM  * CANVAS_PX_PER_MM)
    fh = int(PUZZLE_HEIGHT_MM * CANVAS_PX_PER_MM)
    cv2.rectangle(canvas, (ox, oy), (ox + fw, oy + fh), (200, 200, 200), 2)

    for lbl, (mx, my) in [('TL', (0, 0)), ('TR', (PUZZLE_WIDTH_MM, 0)),
                           ('BR', (PUZZLE_WIDTH_MM, PUZZLE_HEIGHT_MM)),
                           ('BL', (0, PUZZLE_HEIGHT_MM))]:
        px = ox + int(mx * CANVAS_PX_PER_MM)
        py = oy + int(my * CANVAS_PX_PER_MM)
        cv2.circle(canvas, (px, py), 5, (200, 200, 200), -1)
        cv2.putText(canvas, lbl, (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    for x_mm in range(0, int(PUZZLE_WIDTH_MM) + 1, 10):
        xp = ox + int(x_mm * CANVAS_PX_PER_MM)
        cv2.line(canvas, (xp, oy), (xp, oy + fh), (60, 60, 60), 1)
    for y_mm in range(0, int(PUZZLE_HEIGHT_MM) + 1, 10):
        yp = oy + int(y_mm * CANVAS_PX_PER_MM)
        cv2.line(canvas, (ox, yp), (ox + fw, yp), (60, 60, 60), 1)

    return canvas, ox, oy


def _draw_piece(
    canvas:    np.ndarray,
    placed_mm: np.ndarray,
    ox:        int,
    oy:        int,
    color:     tuple[int, int, int],
    label:     str,
) -> None:
    """Draw a placed piece (filled polygon + outline + centroid label)."""
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
    """Draw the H/V corner segments in contrasting colours."""
    result = _corner_transform(seg_h, seg_v, position, px_per_mm)
    if result is None:
        return
    R, frame_corner_mm, puzzle_corner_mm = result

    def _transform(p_px: list) -> np.ndarray:
        pt_mm = np.array(p_px, dtype=float) / px_per_mm
        return R @ (pt_mm - frame_corner_mm) + puzzle_corner_mm

    for seg, color, tag in [(seg_h, (0, 220, 255), 'H'), (seg_v, (255, 180, 0), 'V')]:
        p1, p2 = _transform(seg['p1']), _transform(seg['p2'])
        px1, px2 = _to_px(p1, ox, oy), _to_px(p2, ox, oy)
        cv2.line(canvas, px1, px2, color, 3)
        mid = ((px1[0] + px2[0]) // 2, (px1[1] + px2[1]) // 2)
        cv2.putText(canvas, f"{tag}:{seg['seg_id']} {seg['length_mm']:.0f}mm",
                    (mid[0] + 4, mid[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def _mark_corner(
    canvas: np.ndarray,
    pos:    str,
    ox:     int,
    oy:     int,
    label:  str | None = None,
) -> None:
    """Draw a green cross at the expected puzzle corner, with an optional text label."""
    cx, cy = _to_px(_PUZZLE_CORNERS_MM[pos], ox, oy)
    cv2.drawMarker(canvas, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    if label is not None:
        cv2.putText(canvas, label, (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
