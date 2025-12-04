import numpy as np
from typing import List, Tuple
from puzzle_analyzer.geometry import LineSegment
from puzzle_analyzer.puzzle_piece.curve_fitter import CurveFitter


class SegmentDetector:
    """Handles straight segment detection for puzzle pieces using curve fitting."""

    @staticmethod
    def detect_straight_segments(approx_poly_flat, approx_indices, contour_flat,
                                  min_edge_length: int, straightness_tol: float = 5.0) -> List[LineSegment]:
        """
        Identify all straight line segments along the contour using robust curve fitting.
        Reduces false positives from camera angle and shadow effects.
        """
        straight_segments = []
        approx = approx_poly_flat
        n = len(approx)

        if n < 2:
            return straight_segments

        print(f"  Checking {n} polygon segments with curve fitting...")

        for i in range(n):
            p1 = approx[i]
            p2 = approx[(i + 1) % n]

            # Get the indices in the *original* contour
            idx1 = approx_indices[i]
            idx2 = approx_indices[(i + 1) % n]

            length = np.linalg.norm(p2 - p1)

            # Check minimum length
            if length < min_edge_length:
                continue

            # Get segment points from contour
            if idx1 <= idx2:
                segment_pts = contour_flat[idx1:idx2 + 1]
            else:  # Wrap-around case
                segment_pts = np.concatenate((contour_flat[idx1:], contour_flat[0:idx2 + 1]))

            # Check straightness using curve fitting
            is_straight, residual_std = SegmentDetector._is_segment_straight_via_curve(
                segment_pts, straightness_tol
            )

            if is_straight:
                segment = LineSegment(
                    p1=tuple(p1),
                    p2=tuple(p2),
                    length=length
                )
                straight_segments.append(segment)
                print(f"    [OK] Straight segment: length={length:.1f}, residual_std={residual_std:.2f}")
            else:
                print(f"    [NO] Curved segment: length={length:.1f}, residual_std={residual_std:.2f}")

        return straight_segments

    @staticmethod
    def _is_segment_straight_via_curve(segment_pts: np.ndarray, tolerance: float) -> Tuple[bool, float]:
        """
        Check if a segment is straight using polynomial curve fitting.
        More robust to camera angle and shadow effects than pixel-by-pixel deviation.

        Returns (is_straight, residual_std_dev)
        """
        if len(segment_pts) < 5:
            return True, 0.0

        # Use curve fitting approach
        is_straight, residual_std = CurveFitter.is_segment_straight_via_curve(
            segment_pts, tolerance
        )

        return is_straight, residual_std
