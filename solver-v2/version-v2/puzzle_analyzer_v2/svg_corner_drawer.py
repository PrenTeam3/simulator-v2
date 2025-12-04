"""Draw corners on smoothed SVG files."""

import xml.etree.ElementTree as ET
import numpy as np
from typing import List, Dict, Optional, Tuple
from puzzle_analyzer_v2.svg_smoother import SVGSmoother
from puzzle_analyzer_v2.corner_detector import CornerDetector


class SVGCornerDrawer:
    """Detects corners from smoothed SVG paths and draws them."""

    @staticmethod
    def _check_polygon_intersects_segments(zone_polygon: np.ndarray, all_segments: List[np.ndarray],
                                          tolerance: float = 2.0, corner_exclusion_pt: np.ndarray = None,
                                          corner_exclusion_radius: float = 10.0) -> Tuple[bool, List[np.ndarray]]:
        """
        Check if a polygon (forbidden zone) intersects with any line segments.

        Args:
            zone_polygon: 4 corner points of the forbidden zone rectangle
            all_segments: List of line segments (each segment is [p1, p2])
            tolerance: Distance tolerance for intersection detection
            corner_exclusion_pt: Corner point to exclude (10px radius around it won't count)
            corner_exclusion_radius: Radius around corner to exclude from checks

        Returns:
            Tuple of (has_intersection, list_of_intersection_points)
        """

        intersection_points = []

        # Check each edge of the forbidden zone polygon
        num_zone_edges = len(zone_polygon)
        for i in range(num_zone_edges):
            zone_p1 = zone_polygon[i]
            zone_p2 = zone_polygon[(i + 1) % num_zone_edges]

            # Check against all puzzle piece segments
            for segment in all_segments:
                seg_p1, seg_p2 = segment[0], segment[1]

                # Skip segments that are too close to the corner (within exclusion radius)
                if corner_exclusion_pt is not None:
                    dist_p1 = np.linalg.norm(seg_p1 - corner_exclusion_pt)
                    dist_p2 = np.linalg.norm(seg_p2 - corner_exclusion_pt)
                    # Skip if both points are within exclusion radius
                    if dist_p1 < corner_exclusion_radius and dist_p2 < corner_exclusion_radius:
                        continue

                # Check if line segments intersect or are very close
                intersect_pt = SVGCornerDrawer._get_segment_intersection_point(zone_p1, zone_p2, seg_p1, seg_p2, tolerance)
                if intersect_pt is not None:
                    intersection_points.append(intersect_pt)

        has_intersection = len(intersection_points) > 0
        return has_intersection, intersection_points

    @staticmethod
    def _get_segment_intersection_point(a1: np.ndarray, a2: np.ndarray,
                                        b1: np.ndarray, b2: np.ndarray,
                                        tolerance: float = 2.0) -> Optional[np.ndarray]:
        """
        Get the intersection point between two line segments if they intersect or are close.

        Returns:
            Intersection point as np.ndarray or None if no intersection
        """
        # Check for actual intersection first
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        # Standard line segment intersection test
        if ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2):
            # Calculate actual intersection point
            x1, y1 = a1[0], a1[1]
            x2, y2 = a2[0], a2[1]
            x3, y3 = b1[0], b1[1]
            x4, y4 = b2[0], b2[1]

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) > 1e-10:
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                intersect_x = x1 + t * (x2 - x1)
                intersect_y = y1 + t * (y2 - y1)
                return np.array([intersect_x, intersect_y])

        # Check distance from segment endpoints to the other segment
        def point_to_segment_distance_and_closest(point, seg_start, seg_end):
            """Distance from point to line segment and closest point on segment."""
            seg_vec = seg_end - seg_start
            point_vec = point - seg_start
            seg_len_sq = np.dot(seg_vec, seg_vec)

            if seg_len_sq < 1e-6:
                return np.linalg.norm(point - seg_start), seg_start

            t = max(0, min(1, np.dot(point_vec, seg_vec) / seg_len_sq))
            projection = seg_start + t * seg_vec
            return np.linalg.norm(point - projection), projection

        # Check all endpoint-to-segment distances
        dist_a1_b, closest_a1_b = point_to_segment_distance_and_closest(a1, b1, b2)
        if dist_a1_b < tolerance:
            return closest_a1_b

        dist_a2_b, closest_a2_b = point_to_segment_distance_and_closest(a2, b1, b2)
        if dist_a2_b < tolerance:
            return closest_a2_b

        dist_b1_a, closest_b1_a = point_to_segment_distance_and_closest(b1, a1, a2)
        if dist_b1_a < tolerance:
            return closest_b1_a

        dist_b2_a, closest_b2_a = point_to_segment_distance_and_closest(b2, a1, a2)
        if dist_b2_a < tolerance:
            return closest_b2_a

        return None

    @staticmethod
    def _segments_intersect_or_close(a1: np.ndarray, a2: np.ndarray,
                                      b1: np.ndarray, b2: np.ndarray,
                                      tolerance: float = 2.0) -> bool:
        """
        Check if two line segments intersect or are within tolerance distance.

        Args:
            a1, a2: Endpoints of first segment
            b1, b2: Endpoints of second segment
            tolerance: Distance tolerance

        Returns:
            True if segments intersect or are close
        """
        # Check for actual intersection
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        # Standard line segment intersection test
        if ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2):
            return True

        # Check distance from segment endpoints to the other segment
        def point_to_segment_distance(point, seg_start, seg_end):
            """Distance from point to line segment."""
            seg_vec = seg_end - seg_start
            point_vec = point - seg_start
            seg_len_sq = np.dot(seg_vec, seg_vec)

            if seg_len_sq < 1e-6:
                return np.linalg.norm(point - seg_start)

            t = max(0, min(1, np.dot(point_vec, seg_vec) / seg_len_sq))
            projection = seg_start + t * seg_vec
            return np.linalg.norm(point - projection)

        # Check if any endpoint is close to the other segment
        if point_to_segment_distance(a1, b1, b2) < tolerance:
            return True
        if point_to_segment_distance(a2, b1, b2) < tolerance:
            return True
        if point_to_segment_distance(b1, a1, a2) < tolerance:
            return True
        if point_to_segment_distance(b2, a1, a2) < tolerance:
            return True

        return False

    @staticmethod
    def detect_corners_from_smoothed_svg(svg_path: str, strictness: str = 'strict', debug: bool = False) -> List[Dict]:
        """
        Extract paths from smoothed SVG, detect corners, and return results.

        Args:
            svg_path: Path to smoothed SVG file
            strictness: Strictness level for straight edge detection
            debug: Enable debug logging for corner detection

        Returns:
            List of dicts with corner information for each path
        """
        paths_data = SVGSmoother.extract_paths_from_svg(svg_path)
        corners_list = []

        for piece_idx, path_data in enumerate(paths_data):
            points = path_data['points']
            if len(points) < 3:
                corners_list.append(None)
                continue

            # Convert points to OpenCV-like format for corner detection
            contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

            # Detect corners using the corner detector
            corner_info = CornerDetector.detect_corners(contour, strictness=strictness, debug=debug, piece_idx=piece_idx)
            corners_list.append(corner_info)

        return corners_list

    @staticmethod
    def add_corners_to_smoothed_svg(input_svg: str, output_svg: str,
                                   corners_list: List[Dict] = None, strictness: str = 'strict', debug: bool = False):
        """
        Add corner markings to a smoothed SVG. Creates two versions:
        - One with all helper visualizations (forbidden zones, bisectors, etc.)
        - One with only real results (corners, frame corners, segments)

        Args:
            input_svg: Path to smoothed SVG file
            output_svg: Path to output SVG with corners (will create both _with_helpers and _without_helpers versions)
            corners_list: Pre-detected corners (if None, will detect)
            strictness: Strictness level for straight edge detection
            debug: Enable debug logging for corner detection

        Returns:
            List of corner detection results for each piece
        """
        # Generate both output paths
        base_path = output_svg.rsplit('.', 1)[0]
        output_with_helpers = f"{base_path}_with_helpers.svg"
        output_without_helpers = f"{base_path}_without_helpers.svg"

        # Create version with all helpers
        corners_result = SVGCornerDrawer._add_corners_internal(input_svg, output_with_helpers, corners_list, strictness, debug, include_helpers=True)

        # Create version without helpers
        SVGCornerDrawer._add_corners_internal(input_svg, output_without_helpers, corners_list, strictness, debug, include_helpers=False)

        print(f"Created SVG with helpers: {output_with_helpers}")
        print(f"Created SVG without helpers: {output_without_helpers}")

        return corners_result

    @staticmethod
    def _add_corners_internal(input_svg: str, output_svg: str,
                             corners_list: List[Dict] = None, strictness: str = 'strict',
                             debug: bool = False, include_helpers: bool = True):
        """
        Internal method to add corner markings to a smoothed SVG.

        Args:
            input_svg: Path to smoothed SVG file
            output_svg: Path to output SVG with corners
            corners_list: Pre-detected corners (if None, will detect)
            strictness: Strictness level for straight edge detection
            debug: Enable debug logging for corner detection
            include_helpers: If True, include forbidden zones, bisectors, etc. If False, only real results.
        """
        # Detect corners from smoothed SVG if not provided
        if corners_list is None:
            corners_list = SVGCornerDrawer.detect_corners_from_smoothed_svg(
                input_svg, strictness=strictness, debug=debug
            )

        # Parse the SVG
        ET.register_namespace('', "http://www.w3.org/2000/svg")

        try:
            tree = ET.parse(input_svg)
            root = tree.getroot()
        except:
            print(f"Error: Could not parse SVG '{input_svg}'")
            return

        # Find all paths
        paths = root.findall('.//svg:path', {'svg': 'http://www.w3.org/2000/svg'})

        # If not including helpers, remove all contour-filled paths (keep only contour)
        if not include_helpers:
            paths_to_remove = []
            for path in paths:
                if path.get('class') == 'contour-filled':
                    paths_to_remove.append(path)

            # Remove the filled paths from the SVG
            for path in paths_to_remove:
                root.remove(path)

            # Refresh the paths list after removal
            paths = root.findall('.//svg:path', {'svg': 'http://www.w3.org/2000/svg'})

        # Extract all line segments from all puzzle pieces for collision detection
        # Store segments with their piece index
        all_segments_by_piece = []
        paths_data = SVGSmoother.extract_paths_from_svg(input_svg)
        for piece_idx, path_data in enumerate(paths_data):
            points = path_data['points']
            piece_segments = []
            # Create segments from consecutive points
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                piece_segments.append(np.array([p1, p2]))
            all_segments_by_piece.append(piece_segments)

        # Create a group for corners
        xmlns = "http://www.w3.org/2000/svg"
        corners_group = ET.Element('g', {
            'id': 'corners',
            'stroke-width': '2'
        })

        # Add corner circles and labels (only with helpers for visualization)
        # Without helpers: corners are stored as data attributes on the path elements
        # After removing contour-filled paths, all remaining paths are contour paths
        for piece_idx, (path_elem, corner_info) in enumerate(zip(paths, corners_list)):
            if corner_info is None:
                continue

            outer_corners = corner_info.get('outer_corners', [])
            inner_corners = corner_info.get('inner_corners', [])

            if not include_helpers:
                # For solver: add corner data as attributes on the path element
                path_elem.set('data-piece-id', str(piece_idx))

                # Add outer corners as data attribute
                outer_corners_str = ';'.join([f"{c[0]},{c[1]}" for c in outer_corners])
                if outer_corners_str:
                    path_elem.set('data-outer-corners', outer_corners_str)

                # Add inner corners as data attribute
                inner_corners_str = ';'.join([f"{c[0]},{c[1]}" for c in inner_corners])
                if inner_corners_str:
                    path_elem.set('data-inner-corners', inner_corners_str)
            else:
                # Draw outer corners (orange) - only with helpers
                for corner_idx, corner in enumerate(outer_corners):
                    circle = ET.Element('circle', {
                        'cx': str(corner[0]),
                        'cy': str(corner[1]),
                        'r': '6',
                        'fill': '#FF6600',
                        'stroke': '#FF3300'
                    })
                    corners_group.append(circle)

                    # Add label
                    text = ET.Element('text', {
                        'x': str(corner[0] + 10),
                        'y': str(corner[1] - 10),
                        'fill': '#FF6600',
                        'font-size': '10',
                        'font-family': 'Arial'
                    })
                    text.text = f'O{corner_idx}'
                    corners_group.append(text)

                # Draw inner corners (green) - only with helpers
                for corner_idx, corner in enumerate(inner_corners):
                    circle = ET.Element('circle', {
                        'cx': str(corner[0]),
                        'cy': str(corner[1]),
                        'r': '6',
                        'fill': '#00CC00',
                        'stroke': '#00AA00'
                    })
                    corners_group.append(circle)

                    # Add label
                    text = ET.Element('text', {
                        'x': str(corner[0] + 10),
                        'y': str(corner[1] - 10),
                        'fill': '#00CC00',
                        'font-size': '10',
                        'font-family': 'Arial'
                    })
                    text.text = f'I{corner_idx}'
                    corners_group.append(text)

        # Add forbidden zones visualization (only if include_helpers is True)
        # But we still need to check them to determine valid frame corners
        valid_frame_corners_by_piece = {}  # Store corners that passed forbidden zone test per piece

        for path_idx, (path_elem, corner_info) in enumerate(zip(paths, corners_list)):
            if corner_info is None:
                continue

            # Get segments from CURRENT piece only (to check if edge continues)
            segments_from_current_piece = []
            if path_idx < len(all_segments_by_piece):
                segments_from_current_piece = all_segments_by_piece[path_idx]

            forbidden_zones = corner_info.get('forbidden_zones', [])

            # Draw forbidden zones as light red rectangles
            for zone in forbidden_zones:
                corner = zone.get('corner')
                p1 = zone.get('p1')
                p2 = zone.get('p2')

                if corner and p1 and p2:
                    # First calculate bisector to determine which side is forbidden
                    corner_arr = np.array(corner)
                    p1_arr = np.array(p1)
                    p2_arr = np.array(p2)

                    # Vectors pointing FROM corner TOWARD the edge endpoints
                    v1 = p1_arr - corner_arr
                    v2 = p2_arr - corner_arr

                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)

                    bisector = None
                    angle_tolerance_rad = np.radians(20.0)  # ±20 degrees tolerance

                    if norm1 > 1 and norm2 > 1:
                        v1_norm = v1 / norm1
                        v2_norm = v2 / norm2

                        # Bisector: average of the two edge vectors
                        bisector = v1_norm + v2_norm
                        bisector_norm = np.linalg.norm(bisector)

                        if bisector_norm > 0.1:
                            bisector = bisector / bisector_norm

                            # Rotate bisector by +20 degrees to account for angle tolerance
                            # The forbidden zones should extend in the perpendicular directions
                            # with consideration for the ±20° tolerance
                            cos_angle = np.cos(angle_tolerance_rad)
                            sin_angle = np.sin(angle_tolerance_rad)
                            rotation_matrix = np.array([
                                [cos_angle, -sin_angle],
                                [sin_angle, cos_angle]
                            ])
                            # Apply rotation to bisector
                            bisector = rotation_matrix @ bisector

                    # Draw forbidden zone rectangles extending from each edge
                    # Zone 1: from p1, extending away from corner (and backwards)
                    direction1 = p1_arr - corner_arr
                    dist1 = np.linalg.norm(direction1)
                    if dist1 > 0:
                        dir1_norm = direction1 / dist1

                        # Apply 20° rotation TOWARDS the bisector (inward towards 90° angle center)
                        # Determine rotation direction based on bisector
                        tilt_angle = np.radians(20.0)

                        # Check if we should rotate clockwise or counter-clockwise to move towards bisector
                        if bisector is not None:
                            # Cross product tells us the rotation direction
                            cross = dir1_norm[0] * bisector[1] - dir1_norm[1] * bisector[0]
                            if cross > 0:
                                # Rotate counter-clockwise (positive angle)
                                tilt_angle = tilt_angle
                            else:
                                # Rotate clockwise (negative angle)
                                tilt_angle = -tilt_angle

                        cos_tilt = np.cos(tilt_angle)
                        sin_tilt = np.sin(tilt_angle)
                        tilt_matrix = np.array([
                            [cos_tilt, -sin_tilt],
                            [sin_tilt, cos_tilt]
                        ])
                        dir1_rotated = tilt_matrix @ dir1_norm

                        # Calculate perpendicular: should be OPPOSITE from bisector
                        zone_perp_dir = np.array([-dir1_norm[1], dir1_norm[0]])

                        # Check which side to place the zone (opposite from bisector)
                        if bisector is not None:
                            # If bisector points in the direction of this perpendicular, flip it
                            if np.dot(bisector, zone_perp_dir) > 0:
                                zone_perp_dir = -zone_perp_dir

                        # Shift the entire zone 15px perpendicular to the edge (away from corner)
                        zone_offset1 = p1_arr + zone_perp_dir * 15
                        zone_start1 = zone_offset1 - dir1_rotated * 30  # Extend 30px backwards from offset
                        zone_end1 = zone_offset1 + dir1_rotated * 150  # 150px forwards from offset

                        zone_perp = zone_perp_dir * 30  # 2x wider

                        # Create rectangle path for zone 1 (now includes backwards extension)
                        rect_points = [
                            tuple(zone_start1),
                            tuple(zone_end1),
                            tuple(zone_end1 + zone_perp),
                            tuple(zone_start1 + zone_perp)
                        ]

                        # Check if this forbidden zone intersects any puzzle segments from current piece
                        zone_polygon1 = np.array([
                            zone_start1,
                            zone_end1,
                            zone_end1 + zone_perp,
                            zone_start1 + zone_perp
                        ])
                        intersects1, intersection_pts1 = SVGCornerDrawer._check_polygon_intersects_segments(
                            zone_polygon1, segments_from_current_piece, tolerance=2.0,
                            corner_exclusion_pt=corner_arr, corner_exclusion_radius=10.0
                        )

                        # Store zone 1 info for later drawing
                        zone1_data = {
                            'rect_points': rect_points,
                            'intersects': intersects1,
                            'intersection_pts': intersection_pts1,
                            'zone_polygon': zone_polygon1
                        }

                    # Zone 2: from p2, extending away from corner (and backwards)
                    direction2 = p2_arr - corner_arr
                    dist2 = np.linalg.norm(direction2)
                    if dist2 > 0:
                        dir2_norm = direction2 / dist2

                        # Apply 20° rotation TOWARDS the bisector (inward towards 90° angle center)
                        # Determine rotation direction based on bisector
                        tilt_angle = np.radians(20.0)

                        # Check if we should rotate clockwise or counter-clockwise to move towards bisector
                        if bisector is not None:
                            # Cross product tells us the rotation direction
                            cross = dir2_norm[0] * bisector[1] - dir2_norm[1] * bisector[0]
                            if cross > 0:
                                # Rotate counter-clockwise (positive angle)
                                tilt_angle = tilt_angle
                            else:
                                # Rotate clockwise (negative angle)
                                tilt_angle = -tilt_angle

                        cos_tilt = np.cos(tilt_angle)
                        sin_tilt = np.sin(tilt_angle)
                        tilt_matrix = np.array([
                            [cos_tilt, -sin_tilt],
                            [sin_tilt, cos_tilt]
                        ])
                        dir2_rotated = tilt_matrix @ dir2_norm

                        # Calculate perpendicular: should be OPPOSITE from bisector
                        zone_perp_dir2 = np.array([-dir2_norm[1], dir2_norm[0]])

                        # Check which side to place the zone (opposite from bisector)
                        if bisector is not None:
                            # If bisector points in the direction of this perpendicular, flip it
                            if np.dot(bisector, zone_perp_dir2) > 0:
                                zone_perp_dir2 = -zone_perp_dir2

                        # Shift the entire zone 15px perpendicular to the edge (away from corner)
                        zone_offset2 = p2_arr + zone_perp_dir2 * 15
                        zone_start2 = zone_offset2 - dir2_rotated * 30  # Extend 30px backwards from offset
                        zone_end2 = zone_offset2 + dir2_rotated * 150  # 150px forwards from offset

                        zone_perp2 = zone_perp_dir2 * 30  # 2x wider

                        # Create rectangle path for zone 2 (now includes backwards extension)
                        rect_points = [
                            tuple(zone_start2),
                            tuple(zone_end2),
                            tuple(zone_end2 + zone_perp2),
                            tuple(zone_start2 + zone_perp2)
                        ]

                        # Check if this forbidden zone intersects any puzzle segments from current piece
                        zone_polygon2 = np.array([
                            zone_start2,
                            zone_end2,
                            zone_end2 + zone_perp2,
                            zone_start2 + zone_perp2
                        ])
                        intersects2, intersection_pts2 = SVGCornerDrawer._check_polygon_intersects_segments(
                            zone_polygon2, segments_from_current_piece, tolerance=2.0,
                            corner_exclusion_pt=corner_arr, corner_exclusion_radius=10.0
                        )

                        # Store zone 2 info
                        zone2_data = {
                            'rect_points': rect_points,
                            'intersects': intersects2,
                            'intersection_pts': intersection_pts2,
                            'zone_polygon': zone_polygon2
                        }

                    # Now draw both zones and determine if this is a valid frame corner
                    if 'zone1_data' in locals() and 'zone2_data' in locals():
                        is_valid_frame_corner = (not zone1_data['intersects']) and (not zone2_data['intersects'])

                        # Store valid frame corner for later
                        if is_valid_frame_corner:
                            if path_idx not in valid_frame_corners_by_piece:
                                valid_frame_corners_by_piece[path_idx] = []
                            valid_frame_corners_by_piece[path_idx].append(corner)

                        # Mark this corner in the corners_list with forbidden zone validation status
                        # Find this corner in the frame_corners list and mark it
                        if path_idx < len(corners_list) and corners_list[path_idx]:
                            frame_corners = corners_list[path_idx].get('frame_corners', [])
                            for fc in frame_corners:
                                if isinstance(fc, dict) and fc.get('corner') == corner:
                                    fc['forbidden_zone_clear'] = is_valid_frame_corner
                                    break

                        # Only draw helper visualizations if include_helpers is True
                        if include_helpers:
                            # Draw Zone 1
                            fill_color1 = '#FFFF66' if not zone1_data['intersects'] else '#FF6666'
                            stroke_color1 = '#CCCC00' if not zone1_data['intersects'] else '#FF3333'
                            path_str1 = f"M {zone1_data['rect_points'][0][0]:.0f} {zone1_data['rect_points'][0][1]:.0f} L {zone1_data['rect_points'][1][0]:.0f} {zone1_data['rect_points'][1][1]:.0f} L {zone1_data['rect_points'][2][0]:.0f} {zone1_data['rect_points'][2][1]:.0f} L {zone1_data['rect_points'][3][0]:.0f} {zone1_data['rect_points'][3][1]:.0f} Z"
                            zone_rect1 = ET.Element('path', {
                                'd': path_str1,
                                'fill': fill_color1,
                                'opacity': '0.3',
                                'stroke': stroke_color1,
                                'stroke-width': '2'
                            })
                            corners_group.append(zone_rect1)

                            # Mark intersection points for Zone 1 if red
                            if zone1_data['intersects']:
                                for int_pt in zone1_data['intersection_pts']:
                                    int_circle = ET.Element('circle', {
                                        'cx': str(int(int_pt[0])),
                                        'cy': str(int(int_pt[1])),
                                        'r': '4',
                                        'fill': '#FF0000',
                                        'stroke': '#AA0000',
                                        'stroke-width': '1'
                                    })
                                    corners_group.append(int_circle)

                            # Draw Zone 2
                            fill_color2 = '#FFFF66' if not zone2_data['intersects'] else '#FF6666'
                            stroke_color2 = '#CCCC00' if not zone2_data['intersects'] else '#FF3333'
                            path_str2 = f"M {zone2_data['rect_points'][0][0]:.0f} {zone2_data['rect_points'][0][1]:.0f} L {zone2_data['rect_points'][1][0]:.0f} {zone2_data['rect_points'][1][1]:.0f} L {zone2_data['rect_points'][2][0]:.0f} {zone2_data['rect_points'][2][1]:.0f} L {zone2_data['rect_points'][3][0]:.0f} {zone2_data['rect_points'][3][1]:.0f} Z"
                            zone_rect2 = ET.Element('path', {
                                'd': path_str2,
                                'fill': fill_color2,
                                'opacity': '0.3',
                                'stroke': stroke_color2,
                                'stroke-width': '2'
                            })
                            corners_group.append(zone_rect2)

                            # Mark intersection points for Zone 2 if red
                            if zone2_data['intersects']:
                                for int_pt in zone2_data['intersection_pts']:
                                    int_circle = ET.Element('circle', {
                                        'cx': str(int(int_pt[0])),
                                        'cy': str(int(int_pt[1])),
                                        'r': '4',
                                        'fill': '#FF0000',
                                        'stroke': '#AA0000',
                                        'stroke-width': '1'
                                    })
                                    corners_group.append(int_circle)

                        # Mark frame corner (both versions get this)
                        if is_valid_frame_corner:
                            # Draw a larger circle for valid frame corner
                            # Helpers: Green, Without helpers: Blue (more professional)
                            marker_color = '#00FF00' if include_helpers else '#0066FF'

                            frame_circle = ET.Element('circle', {
                                'cx': str(corner[0]),
                                'cy': str(corner[1]),
                                'r': '8',
                                'fill': 'none',
                                'stroke': marker_color,
                                'stroke-width': '3'
                            })
                            corners_group.append(frame_circle)

                            # Add a cross mark inside
                            cross_line1 = ET.Element('line', {
                                'x1': str(corner[0] - 6),
                                'y1': str(corner[1]),
                                'x2': str(corner[0] + 6),
                                'y2': str(corner[1]),
                                'stroke': marker_color,
                                'stroke-width': '2'
                            })
                            corners_group.append(cross_line1)

                            cross_line2 = ET.Element('line', {
                                'x1': str(corner[0]),
                                'y1': str(corner[1] - 6),
                                'x2': str(corner[0]),
                                'y2': str(corner[1] + 6),
                                'stroke': marker_color,
                                'stroke-width': '2'
                            })
                            corners_group.append(cross_line2)

                            # Add label for frame corner
                            if not include_helpers:
                                frame_label = ET.Element('text', {
                                    'x': str(corner[0] + 12),
                                    'y': str(corner[1] - 12),
                                    'fill': marker_color,
                                    'font-size': '11',
                                    'font-family': 'Arial',
                                    'font-weight': 'bold'
                                })
                                frame_label.text = 'FC'
                                corners_group.append(frame_label)

                    # Add bisector and direction arrows (only with helpers)
                    # Use the bisector we already calculated above
                    if include_helpers and bisector is not None:
                        # Draw bisector as a blue arrow pointing inward (toward center)
                        bisector_end = corner_arr + bisector * 60
                        arrow_line = ET.Element('line', {
                            'x1': str(int(corner[0])),
                            'y1': str(int(corner[1])),
                            'x2': str(int(bisector_end[0])),
                            'y2': str(int(bisector_end[1])),
                            'stroke': '#0088FF',
                            'stroke-width': '2',
                            'marker-end': 'url(#arrowhead-blue)'
                        })
                        corners_group.append(arrow_line)

        # Add SVG arrow markers (once, at the beginning of the group)
        # Blue arrow for bisector
        blue_marker = ET.Element('defs')
        marker = ET.Element('marker', {
            'id': 'arrowhead-blue',
            'markerWidth': '10',
            'markerHeight': '10',
            'refX': '9',
            'refY': '3',
            'orient': 'auto'
        })
        polygon = ET.Element('polygon', {
            'points': '0 0, 10 3, 0 6',
            'fill': '#0088FF'
        })
        marker.append(polygon)
        blue_marker.append(marker)
        root.insert(0, blue_marker)

        # Orange arrow for potential frame corners
        orange_marker = ET.Element('defs')
        marker = ET.Element('marker', {
            'id': 'arrowhead-orange',
            'markerWidth': '10',
            'markerHeight': '10',
            'refX': '9',
            'refY': '3',
            'orient': 'auto'
        })
        polygon = ET.Element('polygon', {
            'points': '0 0, 10 3, 0 6',
            'fill': '#FFA500'
        })
        marker.append(polygon)
        orange_marker.append(marker)
        root.insert(0, orange_marker)

        # Green arrow for inward direction to centroid
        green_marker = ET.Element('defs')
        marker = ET.Element('marker', {
            'id': 'arrowhead-green',
            'markerWidth': '10',
            'markerHeight': '10',
            'refX': '9',
            'refY': '3',
            'orient': 'auto'
        })
        polygon = ET.Element('polygon', {
            'points': '0 0, 10 3, 0 6',
            'fill': '#00CC00'
        })
        marker.append(polygon)
        green_marker.append(marker)
        root.insert(0, green_marker)

        # Add straight edges (only with helpers)
        if include_helpers:
            for path_idx, (path_elem, corner_info) in enumerate(zip(paths, corners_list)):
                if corner_info is None:
                    continue

                straight_segments = corner_info.get('straight_segments', [])

                # Draw straight edges as magenta/purple lines
                for seg in straight_segments:
                    p1 = seg['p1']
                    p2 = seg['p2']
                    line = ET.Element('line', {
                        'x1': str(p1[0]),
                        'y1': str(p1[1]),
                        'x2': str(p2[0]),
                        'y2': str(p2[1]),
                        'stroke': '#FF00FF',
                        'stroke-width': '2',
                        'opacity': '0.7'
                    })
                    corners_group.append(line)

        # Add frame corners and segment data to path elements
        # After removing contour-filled paths, all remaining paths are contour paths
        for piece_idx, (path_elem, corner_info) in enumerate(zip(paths, corners_list)):
            if corner_info is None:
                continue

            segments = corner_info.get('segments', [])

            if not include_helpers:
                # Add frame corners data to path element (for solver)
                if piece_idx in valid_frame_corners_by_piece:
                    frame_corners = valid_frame_corners_by_piece[piece_idx]
                    frame_corners_str = ';'.join([f"{c[0]},{c[1]}" for c in frame_corners])
                    path_elem.set('data-frame-corners', frame_corners_str)

                # Add segment data to path element (for solver)
                if segments:
                    segments_data = []
                    for seg in segments:
                        seg_str = f"{seg['id']}:{seg['corner_start'][0]},{seg['corner_start'][1]}-{seg['corner_end'][0]},{seg['corner_end'][1]}:{1 if seg['is_straight'] else 0}"
                        segments_data.append(seg_str)
                    path_elem.set('data-segments', ';'.join(segments_data))

            # Add segment labels (visual)
            for segment in segments:
                corner_start = segment['corner_start']
                corner_end = segment['corner_end']
                is_straight = segment['is_straight']
                segment_id = segment['id']

                # Calculate midpoint between the two corners
                mid_x = (corner_start[0] + corner_end[0]) / 2.0
                mid_y = (corner_start[1] + corner_end[1]) / 2.0

                # Draw segment ID label
                # With helpers: colorful (magenta for straight, gray for curved)
                # Without helpers: simple black text
                if include_helpers:
                    fill_color = '#FF00FF' if is_straight else '#888888'
                    font_weight = 'bold' if is_straight else 'normal'
                else:
                    fill_color = '#000000'
                    font_weight = 'normal'

                segment_text = ET.Element('text', {
                    'x': str(mid_x),
                    'y': str(mid_y),
                    'fill': fill_color,
                    'font-size': '12',
                    'font-family': 'Arial',
                    'font-weight': font_weight,
                    'text-anchor': 'middle',
                    'dominant-baseline': 'middle',
                    'opacity': '0.9',
                    'background': 'white'
                })
                segment_text.text = f"S{segment_id}"
                corners_group.append(segment_text)

        # Add potential frame corners (only with helpers)
        if include_helpers:
            for path_idx, (path_elem, corner_info) in enumerate(zip(paths, corners_list)):
                if corner_info is None:
                    continue

                frame_corners_for_arrows = corner_info.get('frame_corners', [])
                outer_corners = corner_info.get('outer_corners', [])
                segments = corner_info.get('segments', [])

            # Use centroid from corner_info (calculated by corner detector)
            centroid_tuple = corner_info.get('centroid')
            if centroid_tuple is not None:
                centroid = np.array(centroid_tuple)
            else:
                centroid = None

            # Draw arrows for potential frame corners
            for result_item in frame_corners_for_arrows:
                # Handle both dict and tuple formats
                if isinstance(result_item, dict) and result_item.get('potential', False):
                    corner = result_item.get('corner')
                    angle = result_item.get('angle', 90)
                    corner_num = result_item.get('corner_num')

                    if corner and corner_num is not None and corner_num < len(segments):
                        p_curr = np.array(corner)

                        # Get the two segments forming the corner
                        prev_segment_idx = (corner_num - 1) % len(segments)
                        next_segment_idx = corner_num

                        prev_segment = segments[prev_segment_idx]
                        next_segment = segments[next_segment_idx]

                        # Get corner positions
                        p_prev = np.array(prev_segment['corner_start'])
                        p_next = np.array(next_segment['corner_end'])

                        # Calculate vectors from corner
                        v1 = p_prev - p_curr  # Inward along prev segment
                        v2 = p_next - p_curr  # Outward along next segment

                        # Normalize vectors
                        norm1 = np.linalg.norm(v1)
                        norm2 = np.linalg.norm(v2)

                        if norm1 > 1e-6 and norm2 > 1e-6:
                            v1_norm = v1 / norm1
                            v2_norm = v2 / norm2

                            # Calculate angle bisector (points outward from corner)
                            bisector = v1_norm + v2_norm
                            bisector_norm = np.linalg.norm(bisector)

                            if bisector_norm > 1e-6:
                                bisector_unit = bisector / bisector_norm

                                # Draw arrow along bisector (orange - opening direction)
                                arrow_length = 25
                                dx = bisector_unit[0] * arrow_length
                                dy = bisector_unit[1] * arrow_length

                                arrow_line = ET.Element('line', {
                                    'x1': str(corner[0]),
                                    'y1': str(corner[1]),
                                    'x2': str(corner[0] + dx),
                                    'y2': str(corner[1] + dy),
                                    'stroke': '#FFA500',
                                    'stroke-width': '2',
                                    'marker-end': 'url(#arrowhead-orange)'
                                })
                                corners_group.append(arrow_line)

                        # Draw arrow pointing toward centroid (green - inward direction)
                        if centroid is not None:
                            # Draw arrow from corner directly to centroid dot
                            arrow_line_center = ET.Element('line', {
                                'x1': str(corner[0]),
                                'y1': str(corner[1]),
                                'x2': str(int(centroid[0])),
                                'y2': str(int(centroid[1])),
                                'stroke': '#00CC00',
                                'stroke-width': '2',
                                'marker-end': 'url(#arrowhead-green)'
                            })
                            corners_group.append(arrow_line_center)

        # Add frame corners with different styling (star/cross markers)
        for path_idx, (path_elem, corner_info) in enumerate(zip(paths, corners_list)):
            if corner_info is None:
                continue

            all_frame_corners = corner_info.get('frame_corners', [])
            # Only draw markers for confirmed frame corners (not potential ones)
            frame_corners = [fc for fc in all_frame_corners if not fc.get('potential', False)]

            # Draw frame corners as larger markers (blue stars)
            for frame_corner in frame_corners:
                # Handle both old format (tuple) and new format (tuple from list)
                corner = frame_corner if isinstance(frame_corner, (tuple, list)) else frame_corner.get('corner', frame_corner)

                # Draw a larger circle for frame corners
                circle = ET.Element('circle', {
                    'cx': str(corner[0]),
                    'cy': str(corner[1]),
                    'r': '8',
                    'fill': 'none',
                    'stroke': '#0088FF',
                    'stroke-width': '2'
                })
                corners_group.append(circle)

                # Add a plus sign inside
                line1 = ET.Element('line', {
                    'x1': str(corner[0] - 4),
                    'y1': str(corner[1]),
                    'x2': str(corner[0] + 4),
                    'y2': str(corner[1]),
                    'stroke': '#0088FF',
                    'stroke-width': '1'
                })
                corners_group.append(line1)

                line2 = ET.Element('line', {
                    'x1': str(corner[0]),
                    'y1': str(corner[1] - 4),
                    'x2': str(corner[0]),
                    'y2': str(corner[1] + 4),
                    'stroke': '#0088FF',
                    'stroke-width': '1'
                })
                corners_group.append(line2)

                # Add label
                text = ET.Element('text', {
                    'x': str(corner[0] + 12),
                    'y': str(corner[1] - 12),
                    'fill': '#0088FF',
                    'font-size': '10',
                    'font-family': 'Arial',
                    'font-weight': 'bold'
                })
                text.text = 'F'
                corners_group.append(text)

        # Add legend
        legend_y = 20
        legend_group = ET.Element('g', {'id': 'legend', 'transform': f'translate(10, {legend_y})'})

        # Outer corner legend
        outer_circle = ET.Element('circle', {'cx': '10', 'cy': '0', 'r': '5',
                                            'fill': '#FF6600', 'stroke': '#FF3300'})
        legend_group.append(outer_circle)
        outer_text = ET.Element('text', {'x': '25', 'y': '5', 'fill': 'black',
                                        'font-size': '12', 'font-family': 'Arial'})
        outer_text.text = 'Outer (convex)'
        legend_group.append(outer_text)

        # Inner corner legend
        inner_circle = ET.Element('circle', {'cx': '10', 'cy': '25', 'r': '5',
                                            'fill': '#00CC00', 'stroke': '#00AA00'})
        legend_group.append(inner_circle)
        inner_text = ET.Element('text', {'x': '25', 'y': '30', 'fill': 'black',
                                        'font-size': '12', 'font-family': 'Arial'})
        inner_text.text = 'Inner (concave)'
        legend_group.append(inner_text)

        # Bisector arrow legend
        bisector_line = ET.Element('line', {'x1': '0', 'y1': '50', 'x2': '20', 'y2': '50',
                                           'stroke': '#0088FF', 'stroke-width': '2',
                                           'marker-end': 'url(#arrowhead-blue)'})
        legend_group.append(bisector_line)
        bisector_text = ET.Element('text', {'x': '25', 'y': '55', 'fill': 'black',
                                           'font-size': '12', 'font-family': 'Arial'})
        bisector_text.text = 'Bisector (angle opens toward center)'
        legend_group.append(bisector_text)

        # Forbidden zone legend
        forbidden_rect = ET.Element('rect', {'x': '0', 'y': '70', 'width': '20', 'height': '15',
                                            'fill': '#FF6666', 'opacity': '0.2', 'stroke': '#FF3333'})
        legend_group.append(forbidden_rect)
        forbidden_text = ET.Element('text', {'x': '25', 'y': '80', 'fill': 'black',
                                            'font-size': '12', 'font-family': 'Arial'})
        forbidden_text.text = 'Forbidden zone'
        legend_group.append(forbidden_text)

        # Straight edge legend
        straight_line = ET.Element('line', {'x1': '0', 'y1': '95', 'x2': '20', 'y2': '95',
                                           'stroke': '#FF00FF', 'stroke-width': '2'})
        legend_group.append(straight_line)
        straight_text = ET.Element('text', {'x': '25', 'y': '100', 'fill': 'black',
                                           'font-size': '12', 'font-family': 'Arial'})
        straight_text.text = 'Straight edge'
        legend_group.append(straight_text)

        # Frame corner legend
        frame_circle = ET.Element('circle', {'cx': '10', 'cy': '120', 'r': '5',
                                            'fill': 'none', 'stroke': '#0088FF', 'stroke-width': '2'})
        legend_group.append(frame_circle)
        frame_line1 = ET.Element('line', {'x1': '6', 'y1': '120', 'x2': '14', 'y2': '120',
                                         'stroke': '#0088FF', 'stroke-width': '1'})
        legend_group.append(frame_line1)
        frame_line2 = ET.Element('line', {'x1': '10', 'y1': '116', 'x2': '10', 'y2': '124',
                                         'stroke': '#0088FF', 'stroke-width': '1'})
        legend_group.append(frame_line2)
        frame_text = ET.Element('text', {'x': '25', 'y': '125', 'fill': 'black',
                                        'font-size': '12', 'font-family': 'Arial'})
        frame_text.text = 'Frame corner (90°)'
        legend_group.append(frame_text)

        root.append(legend_group)

        # Add corners group to SVG
        root.append(corners_group)

        # Add centroids group
        centroids_group = ET.Element('g', {
            'id': 'centroids',
            'stroke-width': '2'
        })

        # Draw centroid circles for each piece
        for path_idx, corner_info in enumerate(corners_list):
            if corner_info is None:
                continue

            centroid = corner_info.get('centroid')
            if centroid is not None:
                # Draw green circle for centroid
                circle = ET.Element('circle', {
                    'cx': str(int(centroid[0])),
                    'cy': str(int(centroid[1])),
                    'r': '8',
                    'fill': '#00CC00',
                    'stroke': '#00AA00',
                    'stroke-width': '2'
                })
                centroids_group.append(circle)

                # Add "C" label for centroid
                text = ET.Element('text', {
                    'x': str(int(centroid[0]) + 12),
                    'y': str(int(centroid[1]) - 8),
                    'fill': '#00AA00',
                    'font-size': '12',
                    'font-family': 'Arial',
                    'font-weight': 'bold'
                })
                text.text = 'C'
                centroids_group.append(text)

        root.append(centroids_group)

        # Save the modified SVG
        try:
            tree.write(output_svg, encoding='utf-8', xml_declaration=True)
            print(f"SVG with corners saved to: {output_svg}")
        except Exception as e:
            print(f"Error writing SVG: {e}")

        return corners_list
