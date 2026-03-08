"""Detect outside (frame-border) straight segments using forbidden zone checks.

Uses the exact same zone construction as criterion 4 in frame_corner_detector:
  - Zone at p1: corner=p2, far_endpoint=p1, bisector=inward
  - Zone at p2: corner=p1, far_endpoint=p2, bisector=inward
where 'inward' is the perpendicular direction toward the piece centroid.
"""

from __future__ import annotations

import cv2
import numpy as np

from puzzle_analyzer.frame_corner_detector import (
    _build_forbidden_zone,
    _zone_violations,
)


# ─────────────────────────────────────────────
#  Outside segment detection
# ─────────────────────────────────────────────

def _check_segment_outside(seg: dict, contour_flat: np.ndarray, centroid) -> dict:
    p1 = np.array(seg['p1'], dtype=float)
    p2 = np.array(seg['p2'], dtype=float)

    # Inward = perpendicular to segment pointing toward centroid (acts as bisector)
    seg_vec = p2 - p1
    perp = np.array([-seg_vec[1], seg_vec[0]], dtype=float)
    pnorm = np.linalg.norm(perp)
    inward = None
    if pnorm > 1e-6 and centroid is not None:
        perp /= pnorm
        to_center = np.array(centroid, dtype=float) - (p1 + p2) / 2
        inward = perp if np.dot(perp, to_center) > 0 else -perp

    # Same call pattern as criterion 4: _build_forbidden_zone(corner, far_endpoint, bisector)
    zone1 = _build_forbidden_zone(p2, p1, inward)  # zone at p1
    zone2 = _build_forbidden_zone(p1, p2, inward)  # zone at p2

    v1 = _zone_violations(zone1, contour_flat, p1) if zone1 is not None else []
    v2 = _zone_violations(zone2, contour_flat, p2) if zone2 is not None else []

    return {
        'is_outside': len(v1) == 0 and len(v2) == 0,
        'zone1': zone1, 'zone2': zone2,
        'violations1': v1, 'violations2': v2,
    }


def detect_outside_segments(corners_info: dict) -> None:
    """
    Enrich each segment dict in `corners_info['all_segments']` in place with:
        is_outside, zone1, zone2, violations1, violations2
    Non-straight segments get is_outside=False and no zones.
    """
    contour_flat = corners_info['contour_flat']
    centroid = corners_info['centroid']
    for seg in corners_info['all_segments']:
        if not seg['is_straight']:
            seg['is_outside'] = False
            seg['zone1'] = seg['zone2'] = None
            seg['violations1'] = seg['violations2'] = []
        else:
            seg.update(_check_segment_outside(seg, contour_flat, centroid))


# ─────────────────────────────────────────────
#  Debug drawing
# ─────────────────────────────────────────────

def draw_outside_segments_debug(
    image: np.ndarray,
    contours: list,
    corners_list: list,
    classifications: list,
) -> np.ndarray:
    """
    Debug image for outside segment detection.

    Green line      — outside straight segment (both zones clear)
    Cyan line       — straight but not outside (zone violated)
    Grey line       — not straight
    Yellow zone     — clear forbidden zone
    Red zone        — violated forbidden zone
    Red dots        — contour points inside violated zone
    """
    out = image.copy()

    for contour, info, cls in zip(contours, corners_list, classifications):
        for sr in info['all_segments']:
            p1, p2 = sr['p1'], sr['p2']

            if not sr['is_straight']:
                cv2.line(out, p1, p2, (80, 80, 80), 1)
                continue

            # Draw zones first (underneath the segment line)
            for zone, violations in ((sr['zone1'], sr['violations1']),
                                     (sr['zone2'], sr['violations2'])):
                if zone is None:
                    continue
                pts = zone.astype(np.int32).reshape(-1, 1, 2)
                clear = len(violations) == 0
                fill_color   = (0, 230, 230) if clear else (0, 0, 220)
                border_color = (0, 180, 180) if clear else (0, 0, 180)

                overlay = out.copy()
                cv2.fillPoly(overlay, [pts], fill_color)
                cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
                cv2.polylines(out, [pts], True, border_color, 1)

                for vpt in violations:
                    cv2.circle(out, (int(vpt[0]), int(vpt[1])), 4, (0, 0, 255), -1)

            # Segment line
            line_color = (0, 200, 0) if sr['is_outside'] else (0, 255, 255)
            cv2.line(out, p1, p2, line_color, 3)

    return out
