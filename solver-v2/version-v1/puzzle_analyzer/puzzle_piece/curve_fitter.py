"""Curve fitting module for robust contour analysis using vectorization instead of pixel-by-pixel detection."""

import numpy as np
from scipy.interpolate import splprep, BSpline
from typing import List, Tuple, Optional
import cv2


class CurveFitter:
    """Handles B-spline fitting of contours for robust edge detection."""

    @staticmethod
    def fit_contour_with_bspline(contour_pts: np.ndarray, smoothing: float = 0.1) -> Optional[BSpline]:
        """
        Fit a B-spline to contour points.

        Args:
            contour_pts: Array of contour points (Nx2)
            smoothing: Smoothing factor (0-1). Higher = smoother curve.
                      0 = exact fit, 1 = very smooth

        Returns:
            BSpline representation of the contour, or None if fitting fails
        """
        if len(contour_pts) < 4:
            return None

        try:
            # Ensure we have enough unique points
            unique_pts = np.unique(contour_pts, axis=0)
            if len(unique_pts) < 4:
                return None

            # Prepare contour data
            contour_data = contour_pts.astype(np.float64)

            # Ensure minimum point count for spline fitting
            if len(contour_data) < 10:
                # If too few points, just return None - can't fit well
                return None

            # Try different spline degrees and smoothing values
            spline_degrees = [min(3, len(contour_data) - 1),
                             min(2, len(contour_data) - 1),
                             1]

            for k in spline_degrees:
                if k < 1 or k >= len(contour_data):
                    continue

                try:
                    # Adjust smoothing based on contour length
                    # The smoothing factor affects how closely the spline fits the data
                    s_value = max(0.1, smoothing * len(contour_data) * 0.1)

                    tck, u = splprep(
                        contour_data.T,
                        s=s_value,
                        k=k,
                        per=False
                    )

                    bspline = BSpline(*tck)
                    return bspline
                except Exception:
                    # Try next degree
                    continue

            # If all degrees failed, return None
            return None

        except Exception as e:
            # Silently return None to avoid clutter
            return None

    @staticmethod
    def get_curve_derivatives(bspline: BSpline, u_vals: np.ndarray, derivative_order: int = 1) -> np.ndarray:
        """
        Calculate derivatives of the fitted curve.

        Args:
            bspline: BSpline object
            u_vals: Parameter values (0 to 1) where to evaluate derivatives
            derivative_order: Order of derivative (1=tangent, 2=curvature)

        Returns:
            Array of derivative vectors
        """
        try:
            derivatives = bspline(u_vals, derivative_order)
            return derivatives
        except Exception:
            return None

    @staticmethod
    def sample_curve(bspline: BSpline, num_samples: int = 200) -> np.ndarray:
        """
        Sample points from the fitted B-spline curve.

        Args:
            bspline: BSpline object
            num_samples: Number of points to sample

        Returns:
            Array of sampled points (num_samples x 2)
        """
        u_vals = np.linspace(0, 1, num_samples)
        sampled = bspline(u_vals).T
        return sampled

    @staticmethod
    def detect_corners_from_curve(bspline: BSpline, num_samples: int = 500,
                                  curvature_threshold: float = 0.05,
                                  direction_change_threshold: float = 0.15) -> List[Tuple[int, int]]:
        """
        Detect corners by finding:
        1. Peaks in curvature (sharp turns)
        2. Local extrema in direction (directional changes along curve)

        Args:
            bspline: BSpline object
            num_samples: Number of points to sample for analysis
            curvature_threshold: Minimum curvature to consider as corner
            direction_change_threshold: Minimum normalized direction change to detect local extrema

        Returns:
            List of corner coordinates
        """
        if bspline is None:
            return []

        try:
            u_vals = np.linspace(0, 1, num_samples)

            # Get curve points and derivatives
            curve_pts = bspline(u_vals).T  # (num_samples, 2)
            first_deriv = bspline(u_vals, 1).T  # (num_samples, 2)
            second_deriv = bspline(u_vals, 2).T  # (num_samples, 2)

            # --- Method 1: Curvature-based corner detection ---
            dx = first_deriv[:, 0]
            dy = first_deriv[:, 1]
            d2x = second_deriv[:, 0]
            d2y = second_deriv[:, 1]

            denominator = (dx**2 + dy**2)**1.5
            denominator = np.where(denominator < 1e-6, 1e-6, denominator)

            curvature = np.abs((dx * d2y - dy * d2x) / denominator)

            # Normalize curvature
            max_curv = np.max(curvature)
            if max_curv > 0:
                curvature = curvature / max_curv

            corners = []
            window_size = max(5, num_samples // 40)

            for i in range(window_size, len(curvature) - window_size):
                local_max = curvature[i] > np.max(curvature[i-window_size:i])
                local_max = local_max and curvature[i] > np.max(curvature[i+1:i+window_size+1])

                if local_max and curvature[i] > curvature_threshold:
                    corner_pt = curve_pts[i]
                    corners.append((int(corner_pt[0]), int(corner_pt[1])))

            # --- Method 2: Local extrema in direction (tangent direction changes) ---
            # Calculate tangent angle at each point
            tangent_angles = np.arctan2(dy, dx)

            # Smooth the angles to avoid noise
            angles_smooth = CurveFitter._smooth_angles(tangent_angles)

            # Calculate second derivative of angle (rate of change of direction)
            angle_second_deriv = np.gradient(np.gradient(angles_smooth))

            # Normalize
            max_angle_change = np.max(np.abs(angle_second_deriv))
            if max_angle_change > 1e-6:
                angle_change_normalized = np.abs(angle_second_deriv) / max_angle_change
            else:
                angle_change_normalized = np.zeros_like(angle_second_deriv)

            # Find local extrema in direction (peaks and valleys in the tangent angle curve)
            direction_window = max(5, num_samples // 30)

            for i in range(direction_window, len(angle_change_normalized) - direction_window):
                is_local_extremum = (
                    angle_change_normalized[i] > np.max(angle_change_normalized[i-direction_window:i]) and
                    angle_change_normalized[i] > np.max(angle_change_normalized[i+1:i+direction_window+1])
                )

                if is_local_extremum and angle_change_normalized[i] > direction_change_threshold:
                    corner_pt = curve_pts[i]
                    new_corner = (int(corner_pt[0]), int(corner_pt[1]))

                    # Avoid duplicates (if already in corners from curvature method)
                    is_duplicate = any(
                        np.linalg.norm(np.array(new_corner) - np.array(existing)) < 15
                        for existing in corners
                    )

                    if not is_duplicate:
                        corners.append(new_corner)

            return corners

        except Exception as e:
            print(f"    [WARNING] Corner detection from curve failed: {e}")
            return []

    @staticmethod
    def _smooth_angles(angles: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Smooth angle values while handling angle wrapping (-π to π).

        Args:
            angles: Array of angle values in radians
            kernel_size: Size of smoothing window

        Returns:
            Smoothed angle array
        """
        # Convert to complex representation to handle angle wrapping
        complex_vals = np.exp(1j * angles)

        # Apply Gaussian smoothing to real and imaginary parts
        sigma = kernel_size / 4.0
        x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        kernel = np.exp(-(x**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)

        # Pad for circular convolution
        half_k = kernel_size // 2
        real_padded = np.concatenate([complex_vals.real[-half_k:], complex_vals.real, complex_vals.real[:half_k]])
        imag_padded = np.concatenate([complex_vals.imag[-half_k:], complex_vals.imag, complex_vals.imag[:half_k]])

        real_smooth = np.convolve(real_padded, kernel, mode='same')[half_k:-half_k]
        imag_smooth = np.convolve(imag_padded, kernel, mode='same')[half_k:-half_k]

        # Convert back to angles
        smoothed_angles = np.arctan2(imag_smooth, real_smooth)

        return smoothed_angles

    @staticmethod
    def fit_segment_to_line(segment_pts: np.ndarray, fit_degree: int = 1) -> Tuple[np.ndarray, float]:
        """
        Fit a line (or polynomial) to segment points using least squares.

        Args:
            segment_pts: Array of points (Nx2)
            fit_degree: Polynomial degree (1=line, 2=parabola, etc.)

        Returns:
            Tuple of (coefficients, residual_std_dev)
        """
        if len(segment_pts) < 3:
            return None, float('inf')

        try:
            # Fit polynomial
            x = segment_pts[:, 0]
            y = segment_pts[:, 1]

            coeffs = np.polyfit(x, y, fit_degree)
            poly = np.poly1d(coeffs)

            y_fitted = poly(x)
            residuals = y - y_fitted
            residual_std = np.std(residuals)

            return coeffs, residual_std
        except Exception:
            return None, float('inf')

    @staticmethod
    def is_segment_straight_via_curve(segment_pts: np.ndarray, tolerance: float = 5.0) -> Tuple[bool, float]:
        """
        Determine if a segment is straight using curve fitting.
        Uses low-degree polynomial fit and checks residuals.

        Args:
            segment_pts: Array of contour points
            tolerance: Maximum standard deviation of residuals to consider straight

        Returns:
            Tuple of (is_straight, residual_std_dev)
        """
        if len(segment_pts) < 5:
            return True, 0.0

        # Fit a line (degree=1)
        coeffs, residual_std = CurveFitter.fit_segment_to_line(segment_pts, fit_degree=1)

        if coeffs is None:
            return False, float('inf')

        # Check if residuals are small
        is_straight = residual_std <= tolerance

        return is_straight, residual_std

    @staticmethod
    def extract_straight_edges_from_curve(bspline: BSpline, num_samples: int = 500,
                                         straightness_threshold: float = 0.02) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Extract straight edge segments from a fitted B-spline curve.
        Uses curvature analysis to find straight sections.

        Args:
            bspline: BSpline object
            num_samples: Number of points to sample from curve
            straightness_threshold: Maximum curvature to be considered "straight"

        Returns:
            List of (start_point, end_point) tuples for straight segments
        """
        if bspline is None:
            return []

        try:
            u_vals = np.linspace(0, 1, num_samples)
            curve_pts = bspline(u_vals).T

            # Calculate curvature
            first_deriv = bspline(u_vals, 1).T
            second_deriv = bspline(u_vals, 2).T

            dx = first_deriv[:, 0]
            dy = first_deriv[:, 1]
            d2x = second_deriv[:, 0]
            d2y = second_deriv[:, 1]

            denominator = (dx**2 + dy**2)**1.5
            denominator = np.where(denominator < 1e-6, 1e-6, denominator)

            curvature = np.abs((dx * d2y - dy * d2x) / denominator)

            # Normalize
            max_curv = np.max(curvature)
            if max_curv > 0:
                curvature = curvature / max_curv

            # Find segments with low curvature
            is_straight = curvature < straightness_threshold

            # Group consecutive straight segments
            segments = []
            in_segment = False
            segment_start = 0

            for i in range(len(is_straight)):
                if is_straight[i] and not in_segment:
                    # Start of a straight segment
                    in_segment = True
                    segment_start = i
                elif not is_straight[i] and in_segment:
                    # End of a straight segment
                    in_segment = False
                    if i - segment_start > 5:  # Minimum segment length
                        pt_start = tuple(curve_pts[segment_start].astype(int))
                        pt_end = tuple(curve_pts[i - 1].astype(int))
                        segments.append((pt_start, pt_end))

            # Handle wrap-around for closed curves
            if in_segment and segment_start > 0:
                pt_start = tuple(curve_pts[segment_start].astype(int))
                pt_end = tuple(curve_pts[-1].astype(int))
                segments.append((pt_start, pt_end))

            return segments

        except Exception as e:
            print(f"    [WARNING] Straight edge extraction failed: {e}")
            return []

    @staticmethod
    def smooth_contour_points(contour_pts: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian smoothing to contour points to reduce noise from camera angle/shadows.

        Args:
            contour_pts: Original contour points (Nx2)
            kernel_size: Size of smoothing kernel (must be odd)

        Returns:
            Smoothed contour points
        """
        if len(contour_pts) < kernel_size:
            return contour_pts

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create 1D Gaussian kernel
        sigma = kernel_size / 4.0
        x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        kernel = np.exp(-(x**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)

        # Apply kernel to handle circular/closed contours
        half_k = kernel_size // 2
        x_padded = np.concatenate([contour_pts[-half_k:, 0], contour_pts[:, 0], contour_pts[:half_k, 0]])
        y_padded = np.concatenate([contour_pts[-half_k:, 1], contour_pts[:, 1], contour_pts[:half_k, 1]])

        x_smooth = np.convolve(x_padded, kernel, mode='same')[half_k:-half_k]
        y_smooth = np.convolve(y_padded, kernel, mode='same')[half_k:-half_k]

        return np.column_stack([x_smooth, y_smooth])
