"""SVG path smoothing and optimization using B-splines."""

import xml.etree.ElementTree as ET
import numpy as np
from typing import List
from scipy.interpolate import splev, splprep


class SVGSmoother:
    """Smooth SVG paths using B-spline interpolation."""

    # Configuration
    SIMPLIFICATION_TOLERANCE = 2.0
    SMOOTHING_POINTS = 100
    CORNER_ANGLE_THRESHOLD = 180.0
    CORNER_MIN_DISTANCE = 5.0

    @staticmethod
    def rdp_fast(points, epsilon):
        """Fast vectorized RDP implementation."""
        def perpendicular_distance(pt, p1, p2):
            """Vectorized distance calculation."""
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            denom = dx*dx + dy*dy
            if denom < 1e-10:
                return np.sqrt((pt[0] - p1[0])**2 + (pt[1] - p1[1])**2)
            t = np.clip(((pt[0] - p1[0])*dx + (pt[1] - p1[1])*dy) / denom, 0, 1)
            closest_x = p1[0] + t * dx
            closest_y = p1[1] + t * dy
            return np.sqrt((pt[0] - closest_x)**2 + (pt[1] - closest_y)**2)

        def rdp_iter(pts, eps):
            if len(pts) < 3:
                return pts
            dmax = 0
            index = 0
            for i in range(1, len(pts) - 1):
                d = perpendicular_distance(pts[i], pts[0], pts[-1])
                if d > dmax:
                    dmax, index = d, i

            if dmax > eps:
                left = rdp_iter(pts[:index + 1], eps)
                right = rdp_iter(pts[index:], eps)
                return np.vstack([left[:-1], right])
            return np.array([pts[0], pts[-1]])

        return rdp_iter(points, epsilon)

    @staticmethod
    def is_straight_line(points, tolerance=0.5):
        """Check if points form a straight line."""
        if len(points) < 3:
            return True

        p1, p2 = points[0], points[-1]
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-6:
            return True

        line_unitvec = line_vec / line_len
        max_dist = 0

        for p in points[1:-1]:
            vec = p - p1
            proj = np.dot(vec, line_unitvec)
            closest = p1 + proj * line_unitvec
            dist = np.linalg.norm(p - closest)
            max_dist = max(max_dist, dist)

        return max_dist < tolerance

    @staticmethod
    def smooth_points(points, num_points, is_closed=False):
        """Smooth points using B-spline with aggressive smoothing."""
        if len(points) < 4:
            return points

        # If points are roughly linear, return simplified linear version
        if SVGSmoother.is_straight_line(points, tolerance=1.5):
            step = max(1, len(points) // max(2, num_points // 5))
            return points[::step]

        try:
            # Aggressive smoothing with optimized parameters
            s = len(points) * 1.5
            k = min(3, len(points) - 1)
            tck, u = splprep([points[:, 0], points[:, 1]], s=s, k=k, per=is_closed)
            u_new = np.linspace(0, 1, num_points)
            new_x, new_y = splev(u_new, tck)
            return np.column_stack([new_x, new_y])
        except:
            return points

    @staticmethod
    def detect_corners(points, angle_threshold=CORNER_ANGLE_THRESHOLD, min_distance=CORNER_MIN_DISTANCE):
        """Detect sharp corners, return their indices."""
        if len(points) < 3:
            return [0, len(points) - 1]

        corners = [0]
        for i in range(1, len(points) - 1):
            # Quick distance check
            dx = points[i, 0] - points[corners[-1], 0]
            dy = points[i, 1] - points[corners[-1], 1]
            if dx*dx + dy*dy < min_distance * min_distance:
                continue

            # Calculate angle using dot product (faster than np.linalg)
            v1 = points[i - 1] - points[i]
            v2 = points[i + 1] - points[i]
            v1_norm2 = v1[0]*v1[0] + v1[1]*v1[1]
            v2_norm2 = v2[0]*v2[0] + v2[1]*v2[1]

            if v1_norm2 > 1e-12 and v2_norm2 > 1e-12:
                cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (np.sqrt(v1_norm2) * np.sqrt(v2_norm2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                if cos_angle < 0.99:
                    corners.append(i)

        corners.append(len(points) - 1)
        return corners

    @staticmethod
    def create_smooth_path(points, is_closed=False):
        """Simplify, detect corners, and smooth path while preserving corners."""
        if len(points) < 4:
            return ""

        # First pass: aggressive RDP simplification
        simplified = SVGSmoother.rdp_fast(points, epsilon=SVGSmoother.SIMPLIFICATION_TOLERANCE)
        if len(simplified) < 4:
            return ""

        corner_idx = SVGSmoother.detect_corners(simplified)

        # If few corners, smooth entire path
        if len(corner_idx) <= 2:
            smoothed = SVGSmoother.smooth_points(simplified, SVGSmoother.SMOOTHING_POINTS, is_closed)
        else:
            # Smooth segments between corners
            smoothed = []
            total_len = np.sum([np.linalg.norm(simplified[j+1] - simplified[j]) for j in range(len(simplified)-1)])

            for i in range(len(corner_idx) - 1):
                seg = simplified[corner_idx[i]:corner_idx[i + 1] + 1]
                if len(seg) > 3:
                    # Allocate points based on segment length ratio
                    seg_len = np.sum([np.linalg.norm(seg[j+1] - seg[j]) for j in range(len(seg)-1)])
                    seg_points = max(3, int((seg_len / total_len) * SVGSmoother.SMOOTHING_POINTS * 0.8))
                    seg_smooth = SVGSmoother.smooth_points(seg, seg_points, is_closed=False)
                else:
                    seg_smooth = seg
                smoothed.extend(seg_smooth if i == 0 else seg_smooth[1:])
            smoothed = np.array(smoothed)

        if len(smoothed) < 2:
            return ""

        d = f"M {smoothed[0][0]:.2f},{smoothed[0][1]:.2f} "
        d += "L " + " ".join([f"{x:.2f},{y:.2f}" for x, y in smoothed[1:]])
        if is_closed:
            d += " Z"
        return d

    @staticmethod
    def smooth_svg_file(input_path: str, output_path: str):
        """
        Read SVG file, smooth all paths, and save to new file.

        Args:
            input_path: Path to input SVG file
            output_path: Path to output smoothed SVG file
        """
        print(f"Smoothing SVG: {input_path}")

        ET.register_namespace('', "http://www.w3.org/2000/svg")

        try:
            tree = ET.parse(input_path)
        except FileNotFoundError:
            print(f"Error: File not found '{input_path}'")
            return
        except ET.ParseError:
            print(f"Error: Invalid SVG file '{input_path}'")
            return

        paths = tree.getroot().findall('.//svg:path', {'svg': 'http://www.w3.org/2000/svg'})
        if not paths:
            print("No paths found in SVG")
            return

        print(f"Found {len(paths)} paths. Smoothing...")
        count = 0

        for i, path_elem in enumerate(paths):
            d = path_elem.get('d')
            if not d:
                continue

            # Parse path to points
            points = SVGSmoother._parse_svg_path_to_points(d)
            if len(points) == 0:
                continue

            # Create smoothed path
            d_smooth = SVGSmoother.create_smooth_path(
                points,
                d.strip().upper().endswith('Z')
            )

            if d_smooth:
                path_elem.set('d', d_smooth)
                count += 1

        try:
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            print(f"Smoothed {count} paths. Saved to '{output_path}'")
        except Exception as e:
            print(f"Error writing file: {e}")

    @staticmethod
    def _parse_svg_path_to_points(path_d: str) -> np.ndarray:
        """
        Parse simple SVG path (M, L, Z commands) to points.

        Args:
            path_d: SVG path d attribute

        Returns:
            Array of points
        """
        points = []
        path_d = path_d.strip()

        # Remove commands and split by coordinates
        import re
        # Extract all numeric values (coordinates)
        coords = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', path_d)

        # Group into (x, y) pairs
        for i in range(0, len(coords) - 1, 2):
            try:
                x = float(coords[i])
                y = float(coords[i + 1])
                points.append([x, y])
            except (ValueError, IndexError):
                continue

        if not points:
            return np.array([])

        return np.array(points)

    @staticmethod
    def extract_paths_from_svg(svg_path: str) -> List[dict]:
        """
        Extract all paths from an SVG file with their point data.
        Only extracts stroked contours (class="contour"), not filled ones.

        Args:
            svg_path: Path to SVG file

        Returns:
            List of dicts with 'points' and 'd' attributes
        """
        import xml.etree.ElementTree as ET

        try:
            tree = ET.parse(svg_path)
        except:
            return []

        paths = tree.getroot().findall('.//svg:path', {'svg': 'http://www.w3.org/2000/svg'})
        result = []

        for path_elem in paths:
            # Only process stroked contours, skip filled ones
            path_class = path_elem.get('class', '')
            if 'contour-filled' in path_class:
                continue

            d = path_elem.get('d', '')
            if d:
                points = SVGSmoother._parse_svg_path_to_points(d)
                if len(points) > 0:
                    result.append({'points': points, 'd': d})

        return result
