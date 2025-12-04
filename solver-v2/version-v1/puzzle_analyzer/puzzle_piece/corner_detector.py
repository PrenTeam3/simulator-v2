import cv2
import numpy as np
from typing import List, Tuple, Dict
from puzzle_analyzer.puzzle_piece.curve_fitter import CurveFitter


class CornerDetector:
    """Handles corner classification for puzzle pieces using curve fitting."""

    @staticmethod
    def classify_corners(approx_poly, approx_poly_flat, convex_hull, contour_flat: np.ndarray = None):
        """
        Classify all polygon corners as inner (concave) or outer (convex).
        Uses convexity defects for better inner corner detection.

        Returns:
            Tuple of (outer_corners, inner_corners, curved_points, corner_ids)
            where corner_ids is a dict mapping corner coordinates to their ID number
        """
        outer_corners = []
        inner_corners = []
        curved_points = []
        corner_ids: Dict[Tuple[int, int], int] = {}
        corner_counter = 0

        approx = approx_poly_flat
        n = len(approx)

        if n < 3:
            return outer_corners, inner_corners, curved_points

        # Get convexity defects to identify inner corners
        hull_indices = cv2.convexHull(approx_poly, returnPoints=False)

        inner_pts = set()
        if len(hull_indices) > 3:
            try:
                defects = cv2.convexityDefects(approx_poly, hull_indices)

                # Mark defect points as inner corners
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        far = tuple(approx_poly[f][0])
                        # Only consider significant defects
                        if d > 700:
                            inner_pts.add(far)
            except:
                pass

        # Classify each corner
        for i in range(n):
            p_curr = approx[i]
            p_tuple = tuple(p_curr)

            # Check if this is a defect point (inner corner)
            if p_tuple in inner_pts:
                corner_ids[p_tuple] = corner_counter
                inner_corners.append(p_tuple)
                corner_counter += 1
                continue

            # Calculate angle at this corner
            p_prev = approx[(i - 1) % n]
            p_next = approx[(i + 1) % n]

            v1 = p_prev - p_curr
            v2 = p_next - p_curr

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                continue

            # Calculate angle
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            # Check if point is inside convex hull
            dist = cv2.pointPolygonTest(
                convex_hull,
                (float(p_curr[0]), float(p_curr[1])),
                True
            )

            # Classification logic
            if dist < -3:
                corner_ids[p_tuple] = corner_counter
                inner_corners.append(p_tuple)
                corner_counter += 1
            elif angle < 165:
                if angle < 130 and dist < 0:
                    corner_ids[p_tuple] = corner_counter
                    inner_corners.append(p_tuple)
                    corner_counter += 1
                else:
                    corner_ids[p_tuple] = corner_counter
                    outer_corners.append(p_tuple)
                    corner_counter += 1
            else:
                corner_ids[p_tuple] = corner_counter
                curved_points.append(p_tuple)
                corner_counter += 1

        # Refine classification using curve fitting if contour is available
        # This detects local extrema in curves as corners
        if contour_flat is not None and len(contour_flat) > 20:
            outer_corners, inner_corners, curved_points = CornerDetector._refine_corners_via_curve_fitting(
                contour_flat, approx_poly_flat, outer_corners, inner_corners, curved_points
            )

            # Detect additional corners from curve local extrema
            additional_corners = CornerDetector._detect_curve_extrema_corners(
                contour_flat, outer_corners, inner_corners, curved_points
            )

            if additional_corners:
                print(f"    [CURVE EXTREMA] Detected {len(additional_corners)} additional curved junction corners")

            # Merge additional corners with existing ones, avoiding duplicates
            for corner in additional_corners:
                is_duplicate = any(
                    np.linalg.norm(np.array(corner) - np.array(existing)) < 15
                    for existing_list in [outer_corners, inner_corners, curved_points]
                    for existing in existing_list
                )
                if not is_duplicate:
                    curved_points.append(corner)
                    # Add corner to corner_ids dictionary
                    if corner not in corner_ids:
                        corner_ids[corner] = corner_counter
                        corner_counter += 1

        # Print corner summary
        print(f"\n  Corner Classification Summary (with curve fitting refinement):")
        print(f"    Total corners detected: {corner_counter}")
        if outer_corners:
            print(f"    Outer corners: {', '.join([f'C{corner_ids.get(c, -1)}' for c in outer_corners if c in corner_ids])}")
        if inner_corners:
            print(f"    Inner corners: {', '.join([f'C{corner_ids.get(c, -1)}' for c in inner_corners if c in corner_ids])}")
        if curved_points:
            print(f"    Curved points: {', '.join([f'C{corner_ids.get(c, -1)}' for c in curved_points if c in corner_ids])}")

        return outer_corners, inner_corners, curved_points, corner_ids

    @staticmethod
    def _refine_corners_via_curve_fitting(contour_flat: np.ndarray, approx_poly_flat: np.ndarray,
                                         outer_corners: List, inner_corners: List, curved_points: List) \
            -> Tuple[List, List, List]:
        """
        Refine corner classification using B-spline curve fitting and curvature analysis.
        More robust to camera angle and lighting variations.

        Returns:
            Refined (outer_corners, inner_corners, curved_points)
        """
        try:
            # Fit B-spline to the contour
            smoothed_contour = CurveFitter.smooth_contour_points(contour_flat, kernel_size=7)
            bspline = CurveFitter.fit_contour_with_bspline(smoothed_contour, smoothing=0.05)

            if bspline is None:
                return outer_corners, inner_corners, curved_points

            # Detect corners from curvature
            curve_corners = CurveFitter.detect_corners_from_curve(bspline, num_samples=500, curvature_threshold=0.03)

            if not curve_corners:
                return outer_corners, inner_corners, curved_points

            # Use curve-detected corners as reference
            # Reclassify existing corners based on proximity to high-curvature points
            refined_inner = []
            refined_outer = []
            refined_curved = []

            for corner in outer_corners:
                is_near_curve_corner = any(
                    np.linalg.norm(np.array(corner) - np.array(cc)) < 20
                    for cc in curve_corners
                )
                if is_near_curve_corner:
                    # High curvature at outer corner suggests it's more complex
                    refined_curved.append(corner)
                else:
                    refined_outer.append(corner)

            for corner in inner_corners:
                refined_inner.append(corner)

            for corner in curved_points:
                is_near_curve_corner = any(
                    np.linalg.norm(np.array(corner) - np.array(cc)) < 20
                    for cc in curve_corners
                )
                # Already marked as curved, keep it
                refined_curved.append(corner)

            return refined_outer, refined_inner, refined_curved

        except Exception as e:
            print(f"    [WARNING] Curve fitting refinement failed: {e}")
            return outer_corners, inner_corners, curved_points

    @staticmethod
    def _detect_curve_extrema_corners(contour_flat: np.ndarray,
                                     outer_corners: List, inner_corners: List, curved_points: List) \
            -> List[Tuple[int, int]]:
        """
        Detect additional corners from local extrema in the contour curve.
        These are points where the curve direction changes significantly (peaks/valleys).

        Uses direct contour analysis instead of B-spline fitting for robustness.

        Returns:
            List of newly detected corner coordinates
        """
        try:
            # Use lighter smoothing to preserve wavy details
            smoothed_contour = CurveFitter.smooth_contour_points(contour_flat, kernel_size=5)

            # Calculate tangent direction at each point
            tangent_vectors = np.diff(smoothed_contour, axis=0)

            # Pad to same length as contour
            tangent_vectors = np.vstack([tangent_vectors, tangent_vectors[-1:]])

            # Calculate tangent angles
            tangent_angles = np.arctan2(tangent_vectors[:, 1], tangent_vectors[:, 0])

            # Smooth angles to reduce noise (handle angle wrapping)
            angle_diffs = np.diff(tangent_angles)
            # Handle wrapping at ±π
            angle_diffs = np.arctan2(np.sin(angle_diffs), np.cos(angle_diffs))
            angle_diffs = np.hstack([angle_diffs, [angle_diffs[-1]]])

            # Absolute angle change rate
            angle_change = np.abs(angle_diffs)

            # Light smoothing of angle change (preserve small peaks)
            kernel_size = 5
            sigma = kernel_size / 4.0
            x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
            kernel = np.exp(-(x**2) / (2 * sigma**2))
            kernel = kernel / np.sum(kernel)

            # Pad for circular convolution
            half_k = kernel_size // 2
            padded = np.concatenate([angle_change[-half_k:], angle_change, angle_change[:half_k]])
            smoothed_angle_change = np.convolve(padded, kernel, mode='same')[half_k:-half_k]

            # Normalize
            max_change = np.max(smoothed_angle_change)
            if max_change > 1e-6:
                normalized_change = smoothed_angle_change / max_change
            else:
                normalized_change = np.zeros_like(smoothed_angle_change)

            # Find local maxima in direction change
            corners = []
            # Very small window to catch even small local extrema
            window_size = max(2, len(smoothed_angle_change) // 60)

            for i in range(window_size, len(normalized_change) - window_size):
                is_local_max = (
                    normalized_change[i] > np.max(normalized_change[i-window_size:i]) and
                    normalized_change[i] > np.max(normalized_change[i+1:i+window_size+1])
                )

                # Moderate threshold to catch meaningful direction changes
                # 10% of max angle change - balanced between too few and too many
                if is_local_max and normalized_change[i] > 0.25:  # 10% of max angle change
                    corner = tuple(smoothed_contour[i].astype(int))
                    corners.append(corner)

            return corners

        except Exception as e:
            print(f"    [WARNING] Curve extrema detection failed: {e}")
            return []

    @staticmethod
    def is_point_an_inner_corner(point: np.ndarray, inner_corners: List[Tuple[int, int]],
                                  tolerance: float = 15.0) -> bool:
        """
        Check if a given point is at or very near an inner (concave) corner.
        """
        for inner_corner in inner_corners:
            dist = np.linalg.norm(point - np.array(inner_corner))
            if dist < tolerance:
                return True
        return False
