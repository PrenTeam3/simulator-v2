"""Higher-level drawing overlays: verdict banner, progress bar, and legend."""

from __future__ import annotations

import cv2
import numpy as np

from puzzle_solverv2.frame import PUZZLE_WIDTH_MM, PUZZLE_HEIGHT_MM
from puzzle_solverv2._placement_types import Candidate, _SIDE_TOLERANCE
from puzzle_solverv2._placement_canvas import CANVAS_PX_PER_MM


# ── Overlays ──────────────────────────────────────────────────────────────────

def _draw_verdict(canvas: np.ndarray, valid: bool, reason: str) -> None:
    """Stamp a VALID / INVALID banner in the top-right corner.

    The reason string is split on double-spaces into separate lines so each
    metric (length, diff, overlap) is readable on its own row.
    """
    h, w = canvas.shape[:2]
    text  = "VALID"   if valid else "INVALID"
    color = (0, 220, 80) if valid else (0, 60, 220)
    cv2.putText(canvas, text, (w - 220, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 6)
    cv2.putText(canvas, text, (w - 220, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    parts = [p.strip() for p in reason.split('  ') if p.strip()]
    x, y  = w - 350, 75
    for part in parts:
        cv2.putText(canvas, part, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        y += 16


def _draw_progress_bar(
    canvas:    np.ndarray,
    ox:        int,
    oy:        int,
    side_name: str,
    seg1_len:  float,
    seg2_len:  float,
    color1:    tuple[int, int, int],
    color2:    tuple[int, int, int],
    target:    float,
) -> None:
    """Draw a two-colour progress bar outside the frame showing cumulative side coverage."""
    if side_name == 'top':
        bar_y = oy - 8
    elif side_name == 'bottom':
        bar_y = oy + int(PUZZLE_HEIGHT_MM * CANVAS_PX_PER_MM) + 8
    else:
        return  # left/right sides not visualised here

    x0 = ox
    x1 = ox + int(seg1_len * CANVAS_PX_PER_MM)
    x2 = ox + int((seg1_len + seg2_len) * CANVAS_PX_PER_MM)
    xt = ox + int(target * CANVAS_PX_PER_MM)

    cv2.line(canvas, (x0, bar_y), (x1, bar_y), color1, 4)
    cv2.line(canvas, (x1, bar_y), (x2, bar_y), color2, 4)
    cv2.line(canvas, (xt, bar_y - 6), (xt, bar_y + 6), (200, 200, 200), 2)
    cv2.putText(canvas, f"{target:.0f}mm", (xt - 20, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(canvas, f"{seg1_len + seg2_len:.0f}mm", (x2 + 4, bar_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color2, 1)


# ── Legend ────────────────────────────────────────────────────────────────────

_SIDE_TARGETS: dict[str, float] = {
    'top':    PUZZLE_WIDTH_MM,
    'bottom': PUZZLE_WIDTH_MM,
    'right':  PUZZLE_HEIGHT_MM,
    'left':   PUZZLE_HEIGHT_MM,
}


def _side_lines(side_occ: dict[str, list]) -> list[str]:
    """Format side-occupancy as text lines for the legend."""
    lines = ['FRAME SIDES:']
    for side in ('top', 'right', 'bottom', 'left'):
        segs   = side_occ.get(side, [])
        target = _SIDE_TARGETS[side]
        if segs:
            total = sum(s['length_mm'] for s in segs)
            parts = ' + '.join(
                f"P{s['piece_idx']}*{s['seg_id']}({s['length_mm']:.0f}mm)" for s in segs
            )
            full = ' [FULL]' if total >= target - _SIDE_TOLERANCE else ''
            lines.append(f"  {side.upper():<8} {parts} = {total:.0f}/{target:.0f}mm{full}")
        else:
            lines.append(f"  {side.upper():<8} --  (target {target:.0f}mm)")
    return lines


def _candidate_lines(candidates: list[Candidate]) -> list[str]:
    """Format next-piece candidates as a compact piece-index list."""
    if not candidates:
        return ['NEXT PIECES: (none)']
    seen: list[int] = []
    for c in candidates:
        if c.pv.piece_idx not in seen:
            seen.append(c.pv.piece_idx)
    return [f"NEXT PIECES: {', '.join(f'P{i}' for i in seen)}"]


def _draw_legend(canvas: np.ndarray, ox: int, oy: int, lines: list[str]) -> None:
    """Draw legend text lines in the bottom padding area."""
    y = oy + int(PUZZLE_HEIGHT_MM * CANVAS_PX_PER_MM) + 20
    for line in lines:
        if y >= canvas.shape[0] - 4:
            break
        cv2.putText(canvas, line, (ox, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)
        y += 13
