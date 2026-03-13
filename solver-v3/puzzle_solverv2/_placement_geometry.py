"""Geometry helpers: contour transforms and candidate generation."""

from __future__ import annotations

import math

import numpy as np

import cv2

from puzzle_solverv2.frame import PUZZLE_WIDTH_MM, PUZZLE_HEIGHT_MM
from puzzle_solverv2.config import MAX_OVERLAP_MM2
from puzzle_solverv2._placement_types import (
    Candidate,
    _EXPECTED_HORIZ,
    _PUZZLE_CORNERS_MM,
    _SIDE_TOLERANCE,
)


def _corner_transform(
    seg_h:     dict,
    seg_v:     dict,
    position:  str,
    px_per_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute rotation matrix and corner offsets for placing a piece at `position`.

    Returns (R, frame_corner_mm, puzzle_corner_mm), or None if the segment is degenerate.
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

    p1h = np.array(seg_h['p1'], dtype=float)
    p2h = np.array(seg_h['p2'], dtype=float)
    far_h_px = (p2h
                if np.linalg.norm(p1h - frame_corner_px) < np.linalg.norm(p2h - frame_corner_px)
                else p1h)
    actual_horiz = far_h_px / px_per_mm - frame_corner_mm
    norm = np.linalg.norm(actual_horiz)
    if norm < 1e-6:
        return None

    actual_horiz /= norm
    expected = _EXPECTED_HORIZ[position]
    angle = math.atan2(expected[1], expected[0]) - math.atan2(actual_horiz[1], actual_horiz[0])
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]]), frame_corner_mm, _PUZZLE_CORNERS_MM[position]


def _place_contour(
    contour_flat: np.ndarray,
    seg_h:        dict,
    seg_v:        dict,
    position:     str,
    px_per_mm:    float,
) -> np.ndarray:
    """Transform a contour (pixels) into puzzle-mm coords snapped to a frame corner."""
    result = _corner_transform(seg_h, seg_v, position, px_per_mm)
    if result is None:
        return contour_flat.astype(float) / px_per_mm
    R, frame_corner_mm, puzzle_corner_mm = result
    contour_mm = contour_flat.astype(float) / px_per_mm
    return (R @ (contour_mm - frame_corner_mm).T).T + puzzle_corner_mm


def _place_contour_on_side(
    contour_flat:   np.ndarray,
    seg:            dict,
    side:           str,
    offset_mm:      float,
    px_per_mm:      float,
    start_from_end: bool = False,
) -> np.ndarray:
    """Place an edge piece so that `seg` lies along the frame side at offset_mm.

    Tries 0° then 180° rotation to ensure the piece body faces inward.
    Returns (N, 2) float64 in puzzle-mm coords.

    start_from_end=True: offset_mm is measured from the far end of the side
    (right→BR, left→BL) so the segment's far endpoint lands at
    (side_length - offset_mm).  Use this when traversing bottom-up / right-to-left.
    """
    horiz = side in ('top', 'bottom')
    ax, perp = (0, 1) if horiz else (1, 0)
    side_axis: float = {
        'top': 0.0, 'bottom': PUZZLE_HEIGHT_MM,
        'left': 0.0, 'right': PUZZLE_WIDTH_MM,
    }[side]
    side_length: float = {
        'top': PUZZLE_WIDTH_MM,  'bottom': PUZZLE_WIDTH_MM,
        'left': PUZZLE_HEIGHT_MM, 'right': PUZZLE_HEIGHT_MM,
    }[side]
    target_dir = np.array([1.0, 0.0]) if horiz else np.array([0.0, 1.0])
    inward_pos = side in ('top', 'left')

    p1 = np.array(seg['p1'], dtype=float) / px_per_mm
    p2 = np.array(seg['p2'], dtype=float) / px_per_mm
    seg_vec = p2 - p1
    norm = np.linalg.norm(seg_vec)
    if norm < 1e-6:
        return contour_flat.astype(float) / px_per_mm
    seg_vec /= norm

    base_angle = math.atan2(target_dir[1], target_dir[0]) - math.atan2(seg_vec[1], seg_vec[0])
    contour_mm = contour_flat.astype(float) / px_per_mm
    placed = contour_mm  # fallback

    for extra in (0, math.pi):
        a = base_angle + extra
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, -s], [s, c]])
        rotated = (R @ contour_mm.T).T
        rp1, rp2 = R @ p1, R @ p2
        seg_perp      = (rp1[perp] + rp2[perp]) / 2.0
        centroid_perp = float(np.mean(rotated[:, perp]))
        rel = centroid_perp - seg_perp
        if (inward_pos and rel > 0) or (not inward_pos and rel < 0):
            translation = np.zeros(2)
            if start_from_end:
                end_along = max(rp1[ax], rp2[ax])
                translation[ax] = (side_length - offset_mm) - end_along
            else:
                start_along = min(rp1[ax], rp2[ax])
                translation[ax] = offset_mm - start_along
            translation[perp] = side_axis - seg_perp
            placed = rotated + translation
            break

    return placed


_OVERLAP_SCALE = 2  # px per mm for rasterised overlap check (coarse but fast)


def _polygon_overlap_mm2(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Return the approximate overlap area (mm²) between two polygons.

    Uses rasterisation at _OVERLAP_SCALE px/mm so no external dependency is needed.
    """
    all_pts = np.vstack([poly_a, poly_b])
    min_xy  = all_pts.min(axis=0) - 1.0

    def to_px(poly: np.ndarray) -> np.ndarray:
        return ((poly - min_xy) * _OVERLAP_SCALE).astype(np.int32)

    pts_a = to_px(poly_a)
    pts_b = to_px(poly_b)
    size  = to_px(all_pts).max(axis=0) + 2
    h, w  = int(size[1]), int(size[0])

    mask_a = np.zeros((h, w), dtype=np.uint8)
    mask_b = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_a, [pts_a], 1)
    cv2.fillPoly(mask_b, [pts_b], 1)

    overlap_px = int(np.count_nonzero(mask_a & mask_b))
    return overlap_px / (_OVERLAP_SCALE * _OVERLAP_SCALE)


def _build_candidates(
    pieces_variants:  list,
    used_idxs:        set[int],
    corners_list:     list[dict],
    px_per_mm:        float,
    side_name:        str,
    offset_mm:        float,
    target:           float,
    end_pos:          str,
    fwd_is_horiz:     bool,
    tolerance:        float,
    start_from_end:   bool = False,
    placed_contours:  list | None = None,
) -> list[Candidate]:
    """Return all next-piece candidates for a given side/offset.

    Excludes pieces in used_idxs and applies length, centroid, and overlap checks.
    placed_contours: list of already-placed contours (puzzle-mm) for overlap detection.
    """
    results: list[Candidate] = []

    for pv in pieces_variants:
        if pv.piece_idx in used_idxs:
            continue
        contour = corners_list[pv.piece_idx]['contour_flat']

        for v in pv.variants:

            if v.type == 'edge':
                seg     = v.edges[0]
                new_len = offset_mm + seg['length_mm']
                placed  = _place_contour_on_side(contour, seg, side_name, offset_mm, px_per_mm,
                                                  start_from_end)
                if new_len < target - tolerance:
                    valid  = True
                    reason = (f"{offset_mm:.0f}+{seg['length_mm']:.0f}={new_len:.0f}mm"
                              f"  (room={target - new_len:.0f}mm)")
                else:
                    valid  = False
                    reason = (f"{offset_mm:.0f}+{seg['length_mm']:.0f}={new_len:.0f}mm"
                              f"  nearly fills side -- must be corner")
                if valid:
                    cen = np.mean(placed, axis=0)
                    if not (0 <= cen[0] <= PUZZLE_WIDTH_MM and 0 <= cen[1] <= PUZZLE_HEIGHT_MM):
                        valid, reason = False, f"centroid ({cen[0]:.0f},{cen[1]:.0f})mm outside frame"
                if valid and placed_contours:
                    for prev in placed_contours:
                        overlap = _polygon_overlap_mm2(placed, prev)
                        if overlap > MAX_OVERLAP_MM2:
                            valid  = False
                            reason = f"overlap {overlap:.0f}mm² > {MAX_OVERLAP_MM2:.0f}mm² limit"
                            break
                results.append(Candidate(
                    pv=pv, variant=v, placed_mm=placed, seg_len=seg['length_mm'],
                    valid=valid, reason=reason,
                    label=f"P{pv.piece_idx} edge {seg['seg_id']}({seg['length_mm']:.0f}mm)",
                    seg_h=None, seg_v=None,
                ))

            elif v.type == 'corner' and len(v.edges) >= 2:
                for ori_n, (fwd, turn) in enumerate(
                    [(v.edges[0], v.edges[1]), (v.edges[1], v.edges[0])]
                ):
                    new_len = offset_mm + fwd['length_mm']
                    diff    = abs(new_len - target)
                    sh      = fwd  if fwd_is_horiz else turn
                    sv      = turn if fwd_is_horiz else fwd
                    placed  = _place_contour(contour, sh, sv, end_pos, px_per_mm)
                    if diff <= tolerance:
                        valid  = True
                        reason = (f"{offset_mm:.0f}+{fwd['length_mm']:.0f}={new_len:.0f}mm"
                                  f"  diff={diff:.1f}mm <= {tolerance:.0f}mm")
                    else:
                        valid  = False
                        reason = (f"{offset_mm:.0f}+{fwd['length_mm']:.0f}={new_len:.0f}mm"
                                  f"  diff={diff:.1f}mm > {tolerance:.0f}mm")
                    if valid:
                        cen = np.mean(placed, axis=0)
                        if not (0 <= cen[0] <= PUZZLE_WIDTH_MM and 0 <= cen[1] <= PUZZLE_HEIGHT_MM):
                            valid, reason = False, f"centroid ({cen[0]:.0f},{cen[1]:.0f})mm outside frame"
                    if valid and placed_contours:
                        for prev in placed_contours:
                            overlap = _polygon_overlap_mm2(placed, prev)
                            if overlap > MAX_OVERLAP_MM2:
                                valid  = False
                                reason = f"overlap {overlap:.0f}mm² > {MAX_OVERLAP_MM2:.0f}mm² limit"
                                break
                    results.append(Candidate(
                        pv=pv, variant=v, placed_mm=placed, seg_len=fwd['length_mm'],
                        valid=valid, reason=reason,
                        label=(f"P{pv.piece_idx}@{end_pos} ori{ori_n}"
                               f" fwd={fwd['seg_id']}({fwd['length_mm']:.0f}mm)"),
                        seg_h=sh, seg_v=sv,
                    ))

    return results
