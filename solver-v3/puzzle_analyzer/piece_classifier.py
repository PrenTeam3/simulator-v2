"""Classify puzzle pieces as corner, edge, or inner based on straight edges."""

from __future__ import annotations

import cv2
import numpy as np

from puzzle_analyzer.frame_corner_detector import detect_frame_corners
from puzzle_analyzer.outside_line_detector import detect_outside_segments, draw_outside_segments_debug

CORNER = 'corner'
EDGE   = 'edge'
INNER  = 'inner'

TYPE_COLORS = {
    CORNER: (0, 0, 255),
    EDGE:   (0, 165, 255),
    INNER:  (255, 255, 0),
}


# ─────────────────────────────────────────────
#  Piece classification
# ─────────────────────────────────────────────

def classify_piece(corners_info, debug=False, piece_idx=None):
    detect_outside_segments(corners_info)  # enriches all_segments in place
    frame_corners, candidates = detect_frame_corners(
        corners_info, debug=debug, piece_idx=piece_idx
    )
    num_straight = sum(1 for s in corners_info['all_segments'] if s['is_straight'])

    if frame_corners:
        piece_type = CORNER
    elif num_straight > 0:
        piece_type = EDGE
    else:
        piece_type = INNER

    return {
        'type': piece_type,
        'frame_corners': frame_corners,
        'candidates': candidates,
        'num_straight': num_straight,
    }


# ─────────────────────────────────────────────
#  Drawing — classification result
# ─────────────────────────────────────────────

def draw_classification(image, contours, corners_list, classifications):
    out = image.copy()
    for contour, info, cls in zip(contours, corners_list, classifications):
        color = TYPE_COLORS[cls['type']]
        if info['convex_hull'] is not None:
            cv2.polylines(out, [info['convex_hull']], True, (255, 255, 255), 1)
        cv2.drawContours(out, [contour], 0, color, 3)
        # Collect segments that belong to a frame corner for special highlighting
        frame_segs = set()
        for fc in cls['frame_corners']:
            frame_segs.add((fc['prev_seg']['p1'], fc['prev_seg']['p2']))
            frame_segs.add((fc['next_seg']['p1'], fc['next_seg']['p2']))
        for seg in info['all_segments']:
            if seg['is_straight']:
                seg_key = (seg['p1'], seg['p2'])
                if seg_key in frame_segs:
                    line_color = (0, 255, 0)    # green  — frame corner segment
                elif seg.get('is_outside', False):
                    line_color = (0, 255, 255)  # yellow — outside segment
                else:
                    line_color = (255, 0, 255)  # magenta — other straight
                cv2.line(out, seg['p1'], seg['p2'], line_color, 3)
        for fc in cls['frame_corners']:
            cv2.circle(out, fc['corner'], 12, (0, 255, 0), 3)
            cv2.circle(out, fc['corner'], 5, (0, 255, 0), -1)
        if info['centroid']:
            cx, cy = info['centroid']
            label = f"{cls['type']}  ({cls['num_straight']} straight)"
            cv2.putText(out, label, (cx - 70, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
            cv2.putText(out, label, (cx - 70, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


# ─────────────────────────────────────────────
#  Drawing — debug (mirrors v2 SVG helper elements)
# ─────────────────────────────────────────────

def draw_debug(image, contours, corners_list, classifications):
    """
    Draw all debug elements matching the v2 SVG helper visualization:

      Magenta lines     — straight segments (labelled S0, S1 …)
      Grey lines        — curved segments (labelled S0, S1 …)
      Orange dots       — outer corners (labelled O0, O1 …)
      Green dots        — inner corners
      Green circle+C    — piece centroid
      Blue arrow        — bisector (inward arrow at candidate corners)
      Yellow zone       — forbidden zone that is CLEAR
      Red zone          — forbidden zone that is VIOLATED
      Red dots          — contour points found inside a violated zone
      Blue crosshair+F  — confirmed frame corner
    """
    out = image.copy()
    h, w = out.shape[:2]
    mm_per_px_x = 297.0 / w
    mm_per_px_y = 210.0 / h

    for contour, info, cls in zip(contours, corners_list, classifications):
        contour_flat = info['contour_flat']

        # ── All segments (magenta=straight, grey=curved) ─────────────
        for s_idx, seg in enumerate(info['all_segments']):
            p1, p2 = seg['p1'], seg['p2']
            color = (255, 0, 255) if seg['is_straight'] else (120, 120, 120)
            cv2.line(out, p1, p2, color, 2)
            dx_mm = (p2[0] - p1[0]) * mm_per_px_x
            dy_mm = (p2[1] - p1[1]) * mm_per_px_y
            length_mm = np.sqrt(dx_mm ** 2 + dy_mm ** 2)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            label = f"S{s_idx} {length_mm:.0f}mm"
            cv2.putText(out, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3)
            cv2.putText(out, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # ── Inner corners (green) ────────────────────────────────────
        for pt in info['inner_corners']:
            cv2.circle(out, pt, 5, (0, 200, 0), -1)

        # ── Outer corners (orange) ───────────────────────────────────
        for o_idx, pt in enumerate(info['outer_corners']):
            cv2.circle(out, pt, 6, (0, 140, 255), -1)
            cv2.putText(out, f"O{o_idx}", (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

        # ── Centroid (green circle + C) ──────────────────────────────
        if info['centroid']:
            cx, cy = info['centroid']
            cv2.circle(out, (cx, cy), 8, (0, 200, 0), 2)
            cv2.putText(out, "C", (cx + 10, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

        # ── Candidates: zones, bisector, frame corners ───────────────
        for cand in cls['candidates']:
            if not cand['passed_c1'] or not cand['passed_c2'] or not cand['passed_c3']:
                continue  # only draw from C4 stage onward

            corner_pt = cand['corner']
            bisector  = cand['bisector']

            # Bisector arrow (blue) — scaled to 60px length
            if bisector is not None:
                tip = (int(corner_pt[0] + bisector[0] * 60),
                       int(corner_pt[1] + bisector[1] * 60))
                cv2.arrowedLine(out, corner_pt, tip, (255, 100, 0), 2, tipLength=0.3)

            # Forbidden zones
            for zone, violations in ((cand['zone1'], cand['violations1']),
                                     (cand['zone2'], cand['violations2'])):
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

            # Confirmed frame corner: blue crosshair circle + F label
            if cand['confirmed']:
                cv2.circle(out, corner_pt, 10, (255, 80, 0), 2)
                cv2.line(out, (corner_pt[0] - 6, corner_pt[1]),
                              (corner_pt[0] + 6, corner_pt[1]), (255, 80, 0), 2)
                cv2.line(out, (corner_pt[0], corner_pt[1] - 6),
                              (corner_pt[0], corner_pt[1] + 6), (255, 80, 0), 2)
                cv2.putText(out, "F", (corner_pt[0] + 12, corner_pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 0), 2)

    return out
