"""Detect and classify corners in puzzle piece contours."""

from __future__ import annotations

import cv2
import numpy as np


def _find_nearest_contour_index(point: np.ndarray, contour_flat: np.ndarray) -> int:
    distances = np.linalg.norm(contour_flat - point, axis=1)
    return int(np.argmin(distances))


def _get_contour_slice(p1: np.ndarray, p2: np.ndarray, contour_flat: np.ndarray) -> np.ndarray:
    """Return the contour points that lie between p1 and p2."""
    idx1 = _find_nearest_contour_index(p1, contour_flat)
    idx2 = _find_nearest_contour_index(p2, contour_flat)
    if idx1 == idx2:
        return contour_flat[idx1:idx1 + 1]
    if idx1 < idx2:
        return contour_flat[idx1:idx2 + 1]
    return np.concatenate((contour_flat[idx1:], contour_flat[:idx2 + 1]))


def _is_straight_pca(
    p1: np.ndarray,
    p2: np.ndarray,
    contour_flat: np.ndarray,
    base_px: float = 6.0,
    length_factor: float = 0.05,
) -> bool:
    """
    Check straightness using PCA best-fit line.
    Threshold scales with segment length so longer edges get proportionally more tolerance.
    """
    pts = _get_contour_slice(p1, p2, contour_flat)
    if len(pts) < 3:
        return True

    length = float(np.linalg.norm(p2 - p1))
    threshold = max(base_px, length * length_factor)

    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    line_dir = vt[0]

    projections = np.dot(centered, line_dir)
    reconstructed = np.outer(projections, line_dir)
    residuals = np.linalg.norm(centered - reconstructed, axis=1)
    return bool(np.max(residuals) <= threshold)


def _build_all_segments(
    approx_flat: np.ndarray,
    contour_flat: np.ndarray,
    min_edge_length: int = 5,
) -> list[dict]:
    """
    Build a segment dict for every edge in the approx polygon.
    Each segment knows its approx_index and whether it is straight.
    This full list is needed to check what's on either side of a corner.
    """
    n = len(approx_flat)
    segments = []
    for i in range(n):
        p1 = approx_flat[i]
        p2 = approx_flat[(i + 1) % n]
        length = float(np.linalg.norm(p2 - p1))
        is_straight = (
            length >= min_edge_length and
            _is_straight_pca(p1, p2, contour_flat)
        )
        segments.append({
            'approx_index': i,
            'p1': tuple(p1.astype(int)),
            'p2': tuple(p2.astype(int)),
            'length': length,
            'is_straight': is_straight,
        })
    return segments


def _classify_corners(
    approx_poly: np.ndarray,
    approx_flat: np.ndarray,
    convex_hull: np.ndarray,
    hull_indices: np.ndarray,
) -> tuple[list, list]:
    inner_pts: set = set()
    try:
        defects = cv2.convexityDefects(approx_poly, hull_indices)
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                if d > 550:
                    inner_pts.add(tuple(approx_flat[f].astype(int)))
    except Exception:
        pass

    outer_corners, inner_corners = [], []
    n = len(approx_flat)

    for i in range(n):
        p_curr = approx_flat[i]
        p_tuple = tuple(p_curr.astype(int))

        if p_tuple in inner_pts:
            inner_corners.append(p_tuple)
            continue

        p_prev = approx_flat[(i - 1) % n]
        p_next = approx_flat[(i + 1) % n]
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue

        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        dist = cv2.pointPolygonTest(convex_hull, (float(p_curr[0]), float(p_curr[1])), True)

        if dist < -2.8:
            inner_corners.append(p_tuple)
        elif angle < 132:
            (inner_corners if dist < 0 else outer_corners).append(p_tuple)
        elif angle < 166:
            outer_corners.append(p_tuple)

    return outer_corners, inner_corners


def detect_corners(contour: np.ndarray, min_edge_length: int = 5) -> dict:
    """
    Detect and classify corners in a single puzzle piece contour.

    Returns:
        dict with keys:
            'outer_corners'   : list of (x, y) convex corner points
            'inner_corners'   : list of (x, y) concave corner points
            'all_segments'    : every approx polygon edge with 'is_straight' flag
            'approx_flat'     : approx polygon points as float32 array
            'contour_flat'    : original contour points as float32 array
            'centroid'        : (cx, cy) of the piece
            'convex_hull'     : convex hull array for drawing
    """
    if len(contour) < 3:
        return {
            'outer_corners': [], 'inner_corners': [],
            'all_segments': [], 'approx_flat': np.array([]),
            'contour_flat': np.array([]), 'centroid': None,
            'convex_hull': None,
        }

    contour_flat = contour.reshape(-1, 2).astype(np.float32)
    perimeter = cv2.arcLength(contour, True)
    approx_poly = cv2.approxPolyDP(contour, 0.012 * perimeter, True)
    approx_flat = approx_poly.reshape(-1, 2).astype(np.float32)

    convex_hull_pts = cv2.convexHull(approx_poly)
    hull_indices = cv2.convexHull(approx_poly, returnPoints=False).flatten()

    outer_corners, inner_corners = _classify_corners(
        approx_poly, approx_flat, convex_hull_pts, hull_indices
    )
    all_segments = _build_all_segments(approx_flat, contour_flat, min_edge_length)

    moments = cv2.moments(contour)
    if moments['m00'] != 0:
        centroid = (int(moments['m10'] / moments['m00']),
                    int(moments['m01'] / moments['m00']))
    else:
        centroid = None

    return {
        'outer_corners': outer_corners,
        'inner_corners': inner_corners,
        'all_segments': all_segments,
        'approx_flat': approx_flat,
        'contour_flat': contour_flat,
        'centroid': centroid,
        'convex_hull': convex_hull_pts,
    }


def draw_corners(image: np.ndarray, contours: list, corners_list: list) -> np.ndarray:
    """
    Draw corner detection results on a copy of the image.

    - Yellow lines:  straight edges
    - Orange dots:   outer (convex) corners
    - Green dots:    inner (concave) corners
    - White outline: convex hull of each piece
    """
    out = image.copy()
    for info in corners_list:
        for seg in info['all_segments']:
            if seg['is_straight']:
                cv2.line(out, seg['p1'], seg['p2'], (0, 255, 255), 2)
        for pt in info['outer_corners']:
            cv2.circle(out, pt, 7, (0, 140, 255), -1)
        for pt in info['inner_corners']:
            cv2.circle(out, pt, 7, (0, 255, 0), -1)
        if info['convex_hull'] is not None:
            cv2.polylines(out, [info['convex_hull']], True, (255, 255, 255), 1)
    return out
