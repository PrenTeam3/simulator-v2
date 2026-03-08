"""Detect frame corners in puzzle pieces using 4 geometric criteria."""

from __future__ import annotations

import cv2
import numpy as np


# ─────────────────────────────────────────────
#  Criterion 1 — Connection
# ─────────────────────────────────────────────

def _criterion_1_connection(corner, approx_flat, all_segments):
    corner_arr = np.array(corner, dtype=float)
    corner_idx = None
    for i, pt in enumerate(approx_flat):
        if np.allclose(pt, corner_arr, atol=1.5):
            corner_idx = i
            break
    if corner_idx is None:
        return False, None, None
    n = len(all_segments)
    prev_seg = all_segments[(corner_idx - 1) % n]
    next_seg = all_segments[corner_idx % n]
    if prev_seg['is_straight'] and next_seg['is_straight']:
        return True, prev_seg, next_seg
    return False, prev_seg, next_seg


# ─────────────────────────────────────────────
#  Criterion 2 — Angle ~90°
# ─────────────────────────────────────────────

def _criterion_2_angle(corner, prev_seg, next_seg, angle_tolerance=15.0):
    p_curr = np.array(corner, dtype=float)
    p_prev = np.array(prev_seg['p1'], dtype=float)
    p_next = np.array(next_seg['p2'], dtype=float)
    v_prev = p_prev - p_curr
    v_next = p_next - p_curr
    n1, n2 = np.linalg.norm(v_prev), np.linalg.norm(v_next)
    if n1 < 1e-6 or n2 < 1e-6:
        return False, 0.0
    cos_a = np.clip(np.dot(v_prev, v_next) / (n1 * n2), -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cos_a)))
    return abs(angle - 90.0) <= angle_tolerance, angle


# ─────────────────────────────────────────────
#  Criterion 3 — Inward arrow
# ─────────────────────────────────────────────

def _criterion_3_inward_arrow(corner, prev_seg, next_seg, centroid):
    if centroid is None:
        return False
    p_curr = np.array(corner, dtype=float)
    p_prev = np.array(prev_seg['p1'], dtype=float)
    p_next = np.array(next_seg['p2'], dtype=float)
    v_prev = p_prev - p_curr
    v_next = p_next - p_curr
    to_center = np.array(centroid, dtype=float) - p_curr
    norm = np.linalg.norm(to_center)
    if norm < 1e-6:
        return False
    to_center /= norm
    v_prev_norm = v_prev / np.linalg.norm(v_prev)
    v_next_norm = v_next / np.linalg.norm(v_next)
    return float(np.dot(to_center, v_prev_norm)) > 0 and float(np.dot(to_center, v_next_norm)) > 0


# ─────────────────────────────────────────────
#  Criterion 4 — Forbidden zones
# ─────────────────────────────────────────────

def _rotate_vec(v: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _build_forbidden_zone(corner, far_endpoint, bisector,
                          zone_forward=200.0, zone_backward=50.0,
                          zone_width=50.0, zone_offset=20.0,
                          tilt_deg=10.0):
    direction = far_endpoint - corner
    dist = np.linalg.norm(direction)
    if dist < 1:
        return None
    dir_norm = direction / dist

    # Rotate the extension direction inward toward the bisector by tilt_deg
    if bisector is not None:
        cross = dir_norm[0] * bisector[1] - dir_norm[1] * bisector[0]
        tilt = np.radians(tilt_deg) if cross > 0 else -np.radians(tilt_deg)
        dir_tilted = _rotate_vec(dir_norm, tilt)
    else:
        dir_tilted = dir_norm

    # Perpendicular offset: away from bisector (outside the angle)
    perp = np.array([-dir_norm[1], dir_norm[0]], dtype=float)
    if bisector is not None and np.dot(bisector, perp) > 0:
        perp = -perp

    origin     = far_endpoint + perp * zone_offset
    zone_start = origin - dir_tilted * zone_backward
    zone_end   = origin + dir_tilted * zone_forward
    return np.array([
        zone_start,
        zone_end,
        zone_end   + perp * zone_width,
        zone_start + perp * zone_width,
    ], dtype=np.float32)


def _zone_violations(zone, contour_flat, corner, exclusion_radius=20.0):
    """Return list of contour points found inside the zone (excluding near corner)."""
    zone_cv = zone.reshape(-1, 1, 2)
    hits = []
    for pt in contour_flat:
        if np.linalg.norm(pt - corner) < exclusion_radius:
            continue
        if cv2.pointPolygonTest(zone_cv, (float(pt[0]), float(pt[1])), False) >= 0:
            hits.append(pt)
    return hits


def _criterion_4_forbidden_zones(corner, prev_seg, next_seg):
    """
    Returns (passed, zone1, zone2, violations1, violations2, bisector).

    Reads pre-computed zone results from the segment dicts (populated by
    detect_outside_segments). Uses the far-endpoint zone of each segment:
      prev_seg → zone1 / violations1  (zone at prev_seg['p1'], the far end)
      next_seg → zone2 / violations2  (zone at next_seg['p2'], the far end)

    Also computes the corner bisector for debug visualisation.
    """
    p_curr = np.array(corner, dtype=float)
    p_prev = np.array(prev_seg['p1'], dtype=float)
    p_next = np.array(next_seg['p2'], dtype=float)
    v_prev = p_prev - p_curr
    v_next = p_next - p_curr
    n1, n2 = np.linalg.norm(v_prev), np.linalg.norm(v_next)
    if n1 < 1e-6 or n2 < 1e-6:
        return True, None, None, [], [], None

    bisector = v_prev / n1 + v_next / n2
    b_norm = np.linalg.norm(bisector)
    bisector = bisector / b_norm if b_norm > 0.1 else None

    # Read pre-computed zone data from the segments
    zone1 = prev_seg.get('zone1')        # zone at far endpoint of prev_seg
    v1    = prev_seg.get('violations1', [])
    zone2 = next_seg.get('zone2')        # zone at far endpoint of next_seg
    v2    = next_seg.get('violations2', [])

    passed = (len(v1) == 0) and (len(v2) == 0)
    return passed, zone1, zone2, v1, v2, bisector


# ─────────────────────────────────────────────
#  Frame corner detection
# ─────────────────────────────────────────────

def detect_frame_corners(corners_info, angle_tolerance=15.0, debug=False, piece_idx=None):
    """
    Apply all 4 criteria to each outer corner.

    Returns (frame_corners, candidates):
        frame_corners : confirmed corners
        candidates    : all corners that passed C1, with full debug info
    """
    outer_corners = corners_info['outer_corners']
    approx_flat   = corners_info['approx_flat']
    all_segments  = corners_info['all_segments']
    centroid      = corners_info['centroid']

    label = f"Piece {piece_idx}" if piece_idx is not None else "Piece ?"

    if debug:
        n_straight = sum(1 for s in all_segments if s['is_straight'])
        print(f"\n{'='*60}")
        print(f"[{label}] Frame corner detection")
        print(f"  Outer corners : {len(outer_corners)}")
        print(f"  All segments  : {len(all_segments)} ({n_straight} straight)")
        print(f"  Centroid      : {centroid}")
        print(f"  Angle tol     : ±{angle_tolerance}°")

    frame_corners = []
    candidates = []

    for c_num, corner in enumerate(outer_corners):
        cand = {
            'corner': corner, 'c_num': c_num,
            'passed_c1': False, 'passed_c2': False,
            'passed_c3': False, 'passed_c4': False,
            'angle': None, 'prev_seg': None, 'next_seg': None,
            'zone1': None, 'zone2': None,
            'violations1': [], 'violations2': [],
            'bisector': None, 'confirmed': False,
        }

        if debug:
            print(f"\n  [{c_num}] Corner {corner}")

        ok1, prev_seg, next_seg = _criterion_1_connection(corner, approx_flat, all_segments)
        cand['passed_c1'] = ok1
        cand['prev_seg'] = prev_seg
        cand['next_seg'] = next_seg
        if debug:
            prev_str = prev_seg['is_straight'] if prev_seg else 'N/A'
            next_str = next_seg['is_straight'] if next_seg else 'N/A'
            print(f"    C1 Connection   : {'PASS' if ok1 else 'FAIL'}  (prev={prev_str}, next={next_str})")
        if not ok1:
            candidates.append(cand)
            continue

        ok2, angle = _criterion_2_angle(corner, prev_seg, next_seg, angle_tolerance)
        cand['passed_c2'] = ok2
        cand['angle'] = angle
        if debug:
            print(f"    C2 Angle        : {'PASS' if ok2 else 'FAIL'}  ({angle:.1f}°, need {90-angle_tolerance:.0f}°–{90+angle_tolerance:.0f}°)")
        if not ok2:
            candidates.append(cand)
            continue

        ok3 = _criterion_3_inward_arrow(corner, prev_seg, next_seg, centroid)
        cand['passed_c3'] = ok3
        if debug:
            print(f"    C3 Inward arrow : {'PASS' if ok3 else 'FAIL'}")
        if not ok3:
            candidates.append(cand)
            continue

        ok4, zone1, zone2, v1, v2, bisector = _criterion_4_forbidden_zones(
            corner, prev_seg, next_seg
        )
        cand['passed_c4'] = ok4
        cand['zone1'] = zone1
        cand['zone2'] = zone2
        cand['violations1'] = v1
        cand['violations2'] = v2
        cand['bisector'] = bisector
        if debug:
            print(f"    C4 Forbidden zones: {'PASS' if ok4 else 'FAIL'}"
                  f"  (zone1 hits={len(v1)}, zone2 hits={len(v2)})")

        candidates.append(cand)

        if ok4:
            cand['confirmed'] = True
            if debug:
                print(f"    >>> FRAME CORNER CONFIRMED")
            frame_corners.append({
                'corner': corner, 'angle': angle,
                'prev_seg': prev_seg, 'next_seg': next_seg,
                'zone1': zone1, 'zone2': zone2,
                'bisector': bisector,
            })

    if debug:
        print(f"\n  Result: {len(frame_corners)} frame corner(s) confirmed")
        print(f"{'='*60}")

    return frame_corners, candidates
