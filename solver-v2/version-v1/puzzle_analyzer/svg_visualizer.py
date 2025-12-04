"""SVG vector graphics visualizer for puzzle piece analysis."""

import numpy as np
from typing import List, Tuple
from puzzle_analyzer.geometry import LineSegment


class SVGVisualizer:
    """Create scalable vector graphics (SVG) visualizations of puzzle pieces."""

    @staticmethod
    def create_svg_from_contours(contours: List[np.ndarray], image_shape: Tuple[int, int, int],
                                puzzle_pieces=None, filename: str = "contours.svg") -> str:
        """
        Create an SVG from detected contours (simple linear contours, no smoothing).

        Args:
            contours: List of OpenCV contours
            image_shape: Shape of original image (height, width, channels)
            puzzle_pieces: List of PuzzlePiece objects with analysis data
            filename: Output SVG filename

        Returns:
            SVG content as string
        """
        height, width = image_shape[0], image_shape[1]

        # Create SVG document
        svg_parts = []
        svg_parts.append(f'<?xml version="1.0" encoding="UTF-8"?>')
        svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
        svg_parts.append(f'<defs>')
        svg_parts.append(f'  <style>')
        svg_parts.append(f'    .contour {{ fill: none; stroke: #0088FF; stroke-width: 2; }}')
        svg_parts.append(f'    .contour-filled {{ fill: #E8F4FF; opacity: 0.3; stroke: #0088FF; stroke-width: 2; }}')
        svg_parts.append(f'    text {{ font-family: Arial, sans-serif; }}')
        svg_parts.append(f'  </style>')
        svg_parts.append(f'</defs>')

        # Add white background
        svg_parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')

        # Draw contours
        for idx, contour in enumerate(contours):
            contour_flat = contour[:, 0, :]
            if len(contour_flat) < 3:
                continue

            # Create simple linear path (no smoothing)
            path_data = SVGVisualizer._contour_to_linear_path(contour_flat)
            svg_parts.append(f'<path class="contour-filled" d="{path_data}"/>')
            svg_parts.append(f'<path class="contour" d="{path_data}"/>')

            # Add contour ID label
            centroid = np.mean(contour_flat, axis=0)
            svg_parts.append(f'<text x="{centroid[0]:.1f}" y="{centroid[1]:.1f}" '
                           f'font-size="12" fill="black" text-anchor="middle" dominant-baseline="middle">'
                           f'Contour {idx}</text>')

        # Draw puzzle piece analysis if available
        if puzzle_pieces:
            for piece_idx, piece in enumerate(puzzle_pieces):
                SVGVisualizer._add_piece_analysis_to_svg(svg_parts, piece, piece_idx)

        svg_parts.append(f'</svg>')

        svg_content = '\n'.join(svg_parts)

        # Save to file
        with open(filename, 'w') as f:
            f.write(svg_content)

        print(f"SVG saved to: {filename}")
        return svg_content

    @staticmethod
    def _contour_to_linear_path(contour_pts: np.ndarray) -> str:
        """
        Convert contour points to a simple linear SVG path (no smoothing).

        Args:
            contour_pts: Array of contour points

        Returns:
            SVG path data string
        """
        if len(contour_pts) < 2:
            return ""

        path_parts = []
        path_parts.append(f"M {contour_pts[0][0]:.1f} {contour_pts[0][1]:.1f}")

        for pt in contour_pts[1:]:
            path_parts.append(f"L {pt[0]:.1f} {pt[1]:.1f}")

        path_parts.append("Z")
        return " ".join(path_parts)

    @staticmethod
    def _contour_to_path(contour_pts: np.ndarray, smooth: bool = True,
                        max_points: int = None, tension: float = None) -> str:
        """
        Convert contour points to SVG path data with intelligent smoothing.
        Detects straight segments and curves separately for optimal SVG generation.

        Args:
            contour_pts: Array of contour points
            smooth: If True, use cubic Bézier curves for smooth paths
            max_points: Maximum contour points after downsampling (uses class default if None)
            tension: Catmull-Rom tension parameter (uses class default if None)

        Returns:
            SVG path data string
        """
        if len(contour_pts) < 2:
            return ""

        # Use class defaults if not specified
        if max_points is None:
            max_points = SVGVisualizer.MAX_CONTOUR_POINTS
        if tension is None:
            tension = SVGVisualizer.CATMULL_ROM_TENSION

        if not smooth or len(contour_pts) < 4:
            # Fallback to linear path if not enough points
            path_parts = []
            path_parts.append(f"M {contour_pts[0][0]:.1f} {contour_pts[0][1]:.1f}")
            for pt in contour_pts[1:]:
                path_parts.append(f"L {pt[0]:.1f} {pt[1]:.1f}")
            path_parts.append("Z")
            return " ".join(path_parts)

        # Use intelligent downsampling that preserves straight edges
        downsampled = SVGVisualizer._downsample_contour_smart(contour_pts, max_points=max_points)

        path_parts = []
        path_parts.append(f"M {downsampled[0][0]:.1f} {downsampled[0][1]:.1f}")

        # Generate path using either straight lines or curves depending on local geometry
        for i in range(len(downsampled) - 1):
            p0 = downsampled[i]
            p1 = downsampled[(i + 1) % len(downsampled)]

            # Get neighboring points for curvature analysis
            p_prev = downsampled[(i - 1) % len(downsampled)]
            p_next = downsampled[(i + 2) % len(downsampled)]

            # Check if this segment is relatively straight
            is_straight = SVGVisualizer._is_segment_straight(p_prev, p0, p1, p_next, threshold=2.0)

            if is_straight:
                # Use straight line for straight segments
                path_parts.append(f"L {p1[0]:.1f} {p1[1]:.1f}")
            else:
                # Use Catmull-Rom curve for curved segments
                cp1 = p0 + tension * (p1 - p_prev)
                cp2 = p1 - tension * (p_next - p0)
                path_parts.append(f"C {cp1[0]:.1f} {cp1[1]:.1f} {cp2[0]:.1f} {cp2[1]:.1f} {p1[0]:.1f} {p1[1]:.1f}")

        path_parts.append("Z")
        return " ".join(path_parts)

    @staticmethod
    def _is_segment_straight(p_prev: np.ndarray, p0: np.ndarray, p1: np.ndarray,
                            p_next: np.ndarray, threshold: float = 2.0) -> bool:
        """
        Determine if the segment from p0 to p1 is straight.
        Uses collinearity test with neighboring points.

        Args:
            p_prev: Previous point
            p0: Current point
            p1: Next point
            p_next: Point after next
            threshold: Maximum deviation angle (in degrees) to consider straight

        Returns:
            True if segment is straight, False otherwise
        """
        # Calculate direction vectors
        v0 = p0 - p_prev
        v1 = p1 - p0
        v2 = p_next - p1

        # Calculate magnitudes to avoid division by zero
        len_v0 = np.linalg.norm(v0)
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)

        if len_v0 < 1e-6 or len_v1 < 1e-6 or len_v2 < 1e-6:
            return True  # Degenerate segment, treat as straight

        # Normalize vectors
        v0_norm = v0 / len_v0
        v1_norm = v1 / len_v1
        v2_norm = v2 / len_v2

        # Calculate angle changes
        dot1 = np.dot(v0_norm, v1_norm)
        dot2 = np.dot(v1_norm, v2_norm)

        # Clamp to valid range for arccos
        dot1 = np.clip(dot1, -1.0, 1.0)
        dot2 = np.clip(dot2, -1.0, 1.0)

        # Calculate angles in degrees
        angle1 = np.degrees(np.arccos(dot1))
        angle2 = np.degrees(np.arccos(dot2))

        # Consider straight if both angles are close to 180 (nearly collinear)
        # Straight = both angles close to 180°, meaning little direction change
        straightness1 = 180.0 - abs(angle1 - 180.0)
        straightness2 = 180.0 - abs(angle2 - 180.0)

        avg_straightness = (straightness1 + straightness2) / 2.0

        return avg_straightness > (180.0 - threshold)

    @staticmethod
    def _downsample_contour_smart(contour_pts: np.ndarray, max_points: int = 500) -> np.ndarray:
        """
        Intelligently downsample contour points while preserving straight edges and corners.
        Uses Ramer-Douglas-Peucker algorithm variant that preserves important features.

        Args:
            contour_pts: Original contour points
            max_points: Maximum number of points to keep

        Returns:
            Downsampled contour points preserving straight edges and corners
        """
        if len(contour_pts) <= max_points:
            return contour_pts

        # Use Ramer-Douglas-Peucker algorithm
        epsilon = SVGVisualizer._calculate_epsilon(contour_pts, max_points)
        simplified = SVGVisualizer._rdp_simplify(contour_pts, epsilon)

        # If still too many points, use uniform sampling
        if len(simplified) > max_points:
            simplified = SVGVisualizer._downsample_contour(simplified, max_points)

        return simplified

    @staticmethod
    def _rdp_simplify(contour_pts: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Ramer-Douglas-Peucker algorithm for contour simplification.
        Removes points that are closer than epsilon to the line between endpoints.

        Args:
            contour_pts: Original contour points
            epsilon: Maximum distance threshold for point removal

        Returns:
            Simplified contour points
        """
        if len(contour_pts) < 3:
            return contour_pts

        # Find the point with maximum distance from line between start and end
        dmax = 0.0
        index = 0

        end_point = contour_pts[-1]
        start_point = contour_pts[0]

        for i in range(1, len(contour_pts) - 1):
            d = SVGVisualizer._point_to_line_distance(contour_pts[i], start_point, end_point)
            if d > dmax:
                dmax = d
                index = i

        # If max distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            # Recursive call for segments before and after the point of max distance
            rec1 = SVGVisualizer._rdp_simplify(contour_pts[:index + 1], epsilon)
            rec2 = SVGVisualizer._rdp_simplify(contour_pts[index:], epsilon)

            # Build the result list
            result = np.concatenate([rec1[:-1], rec2])
            return result
        else:
            # Return just start and end points
            return np.array([start_point, end_point])

    @staticmethod
    def _point_to_line_distance(point: np.ndarray, line_start: np.ndarray,
                               line_end: np.ndarray) -> float:
        """
        Calculate perpendicular distance from point to line segment.

        Args:
            point: Point to measure
            line_start: Start of line segment
            line_end: End of line segment

        Returns:
            Perpendicular distance
        """
        # Vector from line_start to line_end
        line_vec = line_end - line_start
        # Vector from line_start to point
        point_vec = point - line_start

        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq == 0:
            return np.linalg.norm(point_vec)

        # Project point onto line
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        projection = line_start + t * line_vec

        return np.linalg.norm(point - projection)

    @staticmethod
    def _calculate_epsilon(contour_pts: np.ndarray, max_points: int) -> float:
        """
        Calculate appropriate epsilon for RDP algorithm based on contour size and target point count.

        Args:
            contour_pts: Original contour points
            max_points: Target maximum number of points

        Returns:
            Epsilon threshold value
        """
        # Start with a reasonable epsilon based on contour size
        x_min, x_max = contour_pts[:, 0].min(), contour_pts[:, 0].max()
        y_min, y_max = contour_pts[:, 1].min(), contour_pts[:, 1].max()

        max_dimension = max(x_max - x_min, y_max - y_min)

        # Base epsilon as percentage of max dimension
        base_epsilon = max_dimension * 0.002

        # Adjust based on desired point count
        reduction_ratio = len(contour_pts) / max(max_points, 10)
        epsilon = base_epsilon * (reduction_ratio ** 0.5)

        return epsilon

    @staticmethod
    def _downsample_contour(contour_pts: np.ndarray, max_points: int = 500) -> np.ndarray:
        """
        Uniform downsampling of contour points.
        Used as fallback when RDP simplification doesn't reduce enough.

        Args:
            contour_pts: Original contour points
            max_points: Maximum number of points to keep

        Returns:
            Downsampled contour points
        """
        if len(contour_pts) <= max_points:
            return contour_pts

        # Calculate distance along contour for each point
        distances = np.zeros(len(contour_pts))
        for i in range(1, len(contour_pts)):
            distances[i] = distances[i - 1] + np.linalg.norm(contour_pts[i] - contour_pts[i - 1])

        total_distance = distances[-1]
        target_distance = total_distance / max_points

        # Sample points at regular intervals along the contour
        downsampled = []
        current_target = 0
        j = 0

        for i in range(max_points):
            current_target = target_distance * (i + 1)
            # Find the point closest to current_target distance
            while j < len(distances) - 1 and distances[j] < current_target:
                j += 1
            downsampled.append(contour_pts[j])

        return np.array(downsampled)

    @staticmethod
    def _add_piece_analysis_to_svg(svg_parts: List[str], piece, piece_idx: int):
        """Add only the piece name label to SVG."""
        # Add piece label only (at centroid position)
        svg_parts.append(f'<text x="{piece.centroid[0]:.1f}" y="{piece.centroid[1]:.1f}" '
                       f'font-size="14" font-weight="bold" fill="black" text-anchor="middle" '
                       f'dominant-baseline="middle">Piece {piece_idx}</text>')

    @staticmethod
    def create_interactive_svg(contours: List[np.ndarray], image_shape: Tuple[int, int, int],
                              puzzle_pieces=None, filename: str = "contours_interactive.svg") -> str:
        """
        Create an interactive SVG with simple linear contours (no smoothing).

        Args:
            contours: List of OpenCV contours
            image_shape: Shape of original image
            puzzle_pieces: List of PuzzlePiece objects
            filename: Output SVG filename

        Returns:
            SVG content as string
        """
        height, width = image_shape[0], image_shape[1]

        svg_parts = []
        svg_parts.append(f'<?xml version="1.0" encoding="UTF-8"?>')
        svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                        f'width="{width}" height="{height}" '
                        f'viewBox="0 0 {width} {height}" '
                        f'preserveAspectRatio="xMidYMid meet">')

        # Add CSS styles
        svg_parts.append(f'<defs>')
        svg_parts.append(f'  <style>')
        svg_parts.append(f'    svg {{ background: white; }}')
        svg_parts.append(f'    .contour {{ fill: none; stroke: #0088FF; stroke-width: 2; }}')
        svg_parts.append(f'    .contour-filled {{ fill: #E8F4FF; opacity: 0.3; stroke: #0088FF; stroke-width: 2; }}')
        svg_parts.append(f'    text {{ font-family: Arial, sans-serif; }}')
        svg_parts.append(f'  </style>')
        svg_parts.append(f'</defs>')

        # Add white background
        svg_parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')

        # Draw all contours with linear paths
        for idx, contour in enumerate(contours):
            contour_flat = contour[:, 0, :]
            if len(contour_flat) < 3:
                continue

            path_data = SVGVisualizer._contour_to_linear_path(contour_flat)
            svg_parts.append(f'<path class="contour-filled" d="{path_data}"/>')
            svg_parts.append(f'<path class="contour" d="{path_data}"/>')

        # Draw piece analysis
        if puzzle_pieces:
            for piece_idx, piece in enumerate(puzzle_pieces):
                SVGVisualizer._add_piece_analysis_to_svg(svg_parts, piece, piece_idx)

        svg_parts.append(f'</svg>')

        svg_content = '\n'.join(svg_parts)

        # Save to file
        with open(filename, 'w') as f:
            f.write(svg_content)

        print(f"Interactive SVG saved to: {filename}")
        return svg_content
