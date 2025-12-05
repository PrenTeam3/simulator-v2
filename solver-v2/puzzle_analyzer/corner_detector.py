"""Corner detection for puzzle pieces."""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Tuple as ArrayTuple


class CornerDetector:
    """Detects and classifies corners in puzzle piece contours."""

    # Strictness presets for straight edge detection
    STRICTNESS_LEVELS = {
        'ultra_loose': {
            'median': 1.0,   # median_dev <= tolerance * 1.0
            'mean': 1.5,     # mean_dev <= tolerance * 1.5
            'p90': 2.0,      # p90_dev <= tolerance * 2.0
            'max': 3.0       # max_dev <= tolerance * 3.0
        },
        'loose': {
            'median': 0.75,
            'mean': 1.0,
            'p90': 1.5,
            'max': 2.5
        },
        'balanced': {
            'median': 0.5,
            'mean': 0.7,
            'p90': 1.0,
            'max': 2.0
        },
        'strict': {
            'median': 0.35,
            'mean': 0.5,
            'p90': 0.75,
            'max': 1.5
        },
        'strict_plus': {
            'median': 0.32,
            'mean': 0.46,
            'p90': 0.68,
            'max': 1.35
        },
        'strict_ultra': {
            'median': 0.29,
            'mean': 0.42,
            'p90': 0.61,
            'max': 1.2
        },
        'ultra_strict_minus': {
            'median': 0.27,
            'mean': 0.38,
            'p90': 0.55,
            'max': 1.1
        },
        'ultra_strict': {
            'median': 0.25,
            'mean': 0.35,
            'p90': 0.5,
            'max': 1.0
        }
    }

    @staticmethod
    def detect_corners(contour: np.ndarray, min_edge_length: int = 5, debug: bool = False,
                      strictness: str = 'ultra_strict_minus', piece_idx: int = None) -> Dict:
        """
        Detect and classify corners in a contour using proper straight edge detection.

        Args:
            contour: OpenCV contour (Nx1x2) or Nx2
            min_edge_length: Minimum edge length to consider
            debug: If True, print debug information
            strictness: Strictness level for straight edge detection (default: 'ultra_strict_minus'):
            piece_idx: Puzzle piece index for debug logging
                - 'ultra_loose': Most permissive, many edges marked as straight
                - 'loose': More permissive
                - 'balanced': Moderate strictness
                - 'strict': More strict, fewer edges marked as straight
                - 'strict_plus': Between strict and strict_ultra
                - 'strict_ultra': Between strict_plus and ultra_strict_minus
                - 'ultra_strict_minus': DEFAULT - Optimal for puzzle pieces
                - 'ultra_strict': Most strict, only nearly perfect lines

        Returns:
            Dict with keys:
                - 'all_corners': List of all corner points
                - 'outer_corners': List of convex corners (protruding)
                - 'inner_corners': List of concave corners (indented)
                - 'corner_ids': Dict mapping corner coords to IDs
                - 'straight_edges': List of straight edge indices
                - 'frame_corners': List of frame corner coordinates
        """
        if len(contour) < 3:
            return {
                'all_corners': [],
                'outer_corners': [],
                'inner_corners': [],
                'corner_ids': {},
                'straight_edges': [],
                'frame_corners': []
            }

        # Get contour as flat array
        contour_flat = contour.reshape(-1, 2).astype(np.float32)

        # Approximate the contour with proper epsilon
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.012 * perimeter  # TEST 2 Conservative setting
        approx_poly = cv2.approxPolyDP(contour, epsilon, True)
        approx_flat = approx_poly.reshape(-1, 2).astype(np.float32)

        # Get convex hull for reference
        convex_hull = cv2.convexHull(approx_poly)
        hull_indices = cv2.convexHull(approx_poly, returnPoints=False).flatten()

        outer_corners = []
        inner_corners = []
        corner_ids = {}
        corner_counter = 0

        # Detect convexity defects to identify inner corners
        inner_pts_defects = set()
        try:
            defects = cv2.convexityDefects(approx_poly, hull_indices)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    far = tuple(approx_flat[f].astype(int))
                    if d > 550:  # TEST 2 Conservative setting
                        inner_pts_defects.add(far)
        except:
            pass

        # Classify each corner
        n = len(approx_flat)
        for i in range(n):
            p_curr = approx_flat[i]
            p_tuple = tuple(p_curr.astype(int))

            # Check convexity defects first (inner corners)
            if p_tuple in inner_pts_defects:
                corner_ids[p_tuple] = corner_counter
                inner_corners.append(p_tuple)
                corner_counter += 1
                continue

            # Calculate angle at this corner
            p_prev = approx_flat[(i - 1) % n]
            p_next = approx_flat[(i + 1) % n]

            v1 = p_prev - p_curr
            v2 = p_next - p_curr

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 < 1e-6 or norm2 < 1e-6:
                continue

            # Calculate angle
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            # Check distance to convex hull
            dist = cv2.pointPolygonTest(
                convex_hull,
                (float(p_curr[0]), float(p_curr[1])),
                True
            )

            # Classification logic - TEST 2 Conservative settings
            if dist < -2.8:  # Clearly inside (concave)
                corner_ids[p_tuple] = corner_counter
                inner_corners.append(p_tuple)
                corner_counter += 1
            elif angle < 166:  # angle_threshold
                # Detect corners with suitable sensitivity
                if angle < 132:  # sharp_angle
                    if dist < 0:
                        # Very sharp and inside
                        corner_ids[p_tuple] = corner_counter
                        inner_corners.append(p_tuple)
                        corner_counter += 1
                    else:
                        # Very sharp and outside (outer)
                        corner_ids[p_tuple] = corner_counter
                        outer_corners.append(p_tuple)
                        corner_counter += 1
                elif angle < 166:  # Medium sharpness
                    # Outer corners
                    corner_ids[p_tuple] = corner_counter
                    outer_corners.append(p_tuple)
                    corner_counter += 1

        # Detect straight edges using proper contour-based detection
        straight_segments = CornerDetector._detect_straight_segments(
            approx_flat, contour_flat, min_edge_length, debug=debug, strictness=strictness
        )

        # Extract indices of straight segments
        straight_edges = [seg['approx_index'] for seg in straight_segments]

        # Build segment information: segments connect consecutive corners
        segments = []
        if len(outer_corners) > 0:
            num_corners = len(outer_corners)
            for segment_idx in range(num_corners):
                next_corner_idx = (segment_idx + 1) % num_corners
                corner_start = outer_corners[segment_idx]
                corner_end = outer_corners[next_corner_idx]

                # Check if any straight segment connects or is between these corners
                is_straight = segment_idx in straight_edges

                segments.append({
                    'id': segment_idx,
                    'from_corner': segment_idx,
                    'to_corner': next_corner_idx,
                    'corner_start': corner_start,
                    'corner_end': corner_end,
                    'is_straight': is_straight
                })

        # Calculate centroid for bisector direction check
        moments = cv2.moments(contour.reshape(-1, 1, 2) if len(contour.shape) == 1 else contour)
        if moments["m00"] != 0:
            centroid = (
                int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"])
            )
        else:
            centroid = None

        # Detect frame corners (90-degree corners where two straight segments meet)
        frame_corner_info = CornerDetector._detect_frame_corners(
            approx_flat, outer_corners, segments,
            contour_flat=contour_flat, centroid=centroid, debug=debug, piece_idx=piece_idx
        )
        frame_corners = frame_corner_info  # Keep full result objects (including potential flag)
        forbidden_zones = [info for info in frame_corner_info if not info.get('potential', False)]  # Only actual corners for zones

        if debug:
            print(f"  Corners: {corner_counter} total ({len(outer_corners)} outer, {len(inner_corners)} inner)")
            print(f"  Straight edges: {len(straight_segments)}")
            print(f"  Frame corners: {len(frame_corners)}")

        return {
            'all_corners': outer_corners + inner_corners,
            'outer_corners': outer_corners,
            'inner_corners': inner_corners,
            'corner_ids': corner_ids,
            'total': corner_counter,
            'straight_edges': straight_edges,
            'straight_segments': straight_segments,
            'frame_corners': frame_corners,
            'forbidden_zones': forbidden_zones,
            'segments': segments,
            'centroid': centroid
        }

    @staticmethod
    def _detect_straight_segments(approx_poly_flat: np.ndarray, contour_flat: np.ndarray,
                                 min_edge_length: int = 5, straightness_tol: float = 5.0,
                                 debug: bool = False, strictness: str = 'strict') -> List[Dict]:
        """
        Identify all straight line segments along the contour using configurable strictness criteria.
        Uses the original contour points to validate straightness, not just the approximated polygon.

        Args:
            approx_poly_flat: Approximated polygon points
            contour_flat: Original contour points
            min_edge_length: Minimum edge length to consider
            straightness_tol: Base tolerance for straightness detection (default 5.0)
            debug: If True, print debug information
            strictness: Strictness level ('ultra_loose', 'loose', 'balanced', 'strict', 'ultra_strict')

        Returns:
            List of dicts with straight segment info: {
                'approx_index': index in approx_poly,
                'p1': start point,
                'p2': end point,
                'length': segment length,
                'median_dev': median deviation,
                'max_dev': max deviation
            }
        """
        straight_segments = []
        approx = approx_poly_flat
        n = len(approx)

        if n < 2:
            return straight_segments

        for i in range(n):
            p1 = approx[i]
            p2 = approx[(i + 1) % n]

            length = np.linalg.norm(p2 - p1)

            # Check minimum length
            if length < min_edge_length:
                continue

            # Check straightness using the original contour points
            is_straight, median_dev, max_dev = CornerDetector._is_segment_straight(
                p1, p2, contour_flat, straightness_tol, strictness=strictness
            )

            if is_straight:
                segment = {
                    'approx_index': i,
                    'p1': tuple(p1.astype(int)),
                    'p2': tuple(p2.astype(int)),
                    'length': length,
                    'median_dev': median_dev,
                    'max_dev': max_dev
                }
                straight_segments.append(segment)
                if debug:
                    print(f"    [STRAIGHT] Index {i}: len={length:.1f}, median={median_dev:.2f}, max={max_dev:.2f}")
            else:
                if debug:
                    print(f"    [CURVED] Index {i}: len={length:.1f}, median={median_dev:.2f}, max={max_dev:.2f}")

        return straight_segments

    @staticmethod
    def _is_segment_straight(p1: np.ndarray, p2: np.ndarray, contour_flat: np.ndarray,
                            tolerance: float = 5.0, strictness: str = 'strict') -> Tuple[bool, float, float]:
        """
        Check if contour segment between two points is straight.
        Uses configurable strictness criteria.

        Args:
            p1: Start point of segment
            p2: End point of segment
            contour_flat: Original contour points
            tolerance: Base deviation tolerance (default 5.0)
            strictness: Strictness level ('ultra_loose', 'loose', 'balanced', 'strict', 'ultra_strict')

        Returns:
            Tuple of (is_straight, median_deviation, max_deviation)
        """
        # Find nearest contour points to p1 and p2
        idx1 = CornerDetector._find_nearest_contour_index(p1, contour_flat)
        idx2 = CornerDetector._find_nearest_contour_index(p2, contour_flat)

        if idx1 == idx2:
            return True, 0.0, 0.0

        # Get the slice of contour points for this segment
        if idx1 <= idx2:
            segment_pts = contour_flat[idx1:idx2 + 1]
        else:  # Wrap-around case
            segment_pts = np.concatenate((contour_flat[idx1:], contour_flat[0:idx2 + 1]))

        # Need enough points to make a judgment
        if len(segment_pts) < 3:
            return True, 0.0, 0.0

        line_vec = p2 - p1
        line_length = np.linalg.norm(line_vec)

        if line_length < 1:
            return True, 0.0, 0.0

        line_unit = line_vec / line_length

        deviations = []

        # Calculate deviation of all contour points from the straight line
        for pt in segment_pts:
            pt_vec = pt - p1
            projection = np.dot(pt_vec, line_unit)
            perpendicular = pt_vec - projection * line_unit
            dist = np.linalg.norm(perpendicular)
            deviations.append(dist)

        if len(deviations) < 3:
            return True, 0.0, 0.0

        # Calculate metrics
        deviations = np.array(deviations)
        median_dev = np.median(deviations)
        mean_dev = np.mean(deviations)
        max_dev = np.max(deviations)
        p90_dev = np.percentile(deviations, 90)

        # Get strictness criteria
        if strictness not in CornerDetector.STRICTNESS_LEVELS:
            strictness = 'balanced'

        criteria = CornerDetector.STRICTNESS_LEVELS[strictness]

        # Apply criteria - all must pass
        is_straight = (median_dev <= tolerance * criteria['median'] and
                      mean_dev <= tolerance * criteria['mean'] and
                      p90_dev <= tolerance * criteria['p90'] and
                      max_dev <= tolerance * criteria['max'])

        return is_straight, median_dev, max_dev

    @staticmethod
    def _find_nearest_contour_index(point: np.ndarray, contour_flat: np.ndarray) -> int:
        """Find the index of the nearest point in the contour."""
        distances = np.linalg.norm(contour_flat - point, axis=1)
        return int(np.argmin(distances))

    @staticmethod
    def _detect_frame_corners(points: np.ndarray, outer_corners: list, segments: list,
                             contour_flat: np.ndarray = None, centroid: tuple = None,
                             angle_tolerance: float = 5.0, debug: bool = False, piece_idx: int = None) -> list:
        """
        Detect frame corners using official specification:
        - Connection: Two edge lines share common endpoint (within tolerance)
        - Angle: ~90° between the two lines (85°-95° tolerance)
        - Inward Arrow: Vector toward piece center falls within the 90° opening angle
        - Convexity: Corner must be on outer convex boundary

        Args:
            points: Polygon points
            outer_corners: List of detected outer corners
            segments: List of segments with straight/curved classification
            contour_flat: Original contour points (for convexity check)
            centroid: Center of piece (for inward arrow check)
            angle_tolerance: Tolerance for 90-degree detection (default 15°, range 75°-105°)
            debug: If True, print detailed decision logging
            piece_idx: Puzzle piece index for debug logging

        Returns:
            List of frame corner info dicts with 'corner', 'potential', 'corner_num', 'angle' keys
        """
        frame_corners = []
        potential_frame_corners = []  # Corners that pass Criterion 1 and 2
        n = len(points)

        if n < 3 or not outer_corners or not segments:
            if debug:
                print(f"  [FRAME CORNERS] Skipped: n={n}, outer_corners={len(outer_corners)}, segments={len(segments)}")
            return frame_corners

        if debug:
            piece_label = f"PIECE {piece_idx}" if piece_idx is not None else "PIECE ?"
            print(f"\n  ================================================================")
            print(f"  [{piece_label}] [FRAME CORNERS] Checking {len(outer_corners)} outer corners")
            print(f"  ================================================================")
            print(f"  Criteria per framecornern.txt:")
            print(f"    1. Connection: Two edge lines share common endpoint")
            print(f"    2. Angle: ~90° (tolerance: {90-angle_tolerance:.1f}° - {90+angle_tolerance:.1f}°)")
            print(f"    3. Inward Arrow: Arrow toward piece center falls within 90° opening angle")
            print(f"    4. Convexity: Corner on outer convex boundary")

        # Check each outer corner to see if it's a frame corner
        for corner_num, corner in enumerate(outer_corners):
            corner_array = np.array(corner)

            # Find indices of this corner in points
            corner_indices = []
            for i, pt in enumerate(points):
                if np.allclose(pt, corner_array, atol=1.5):
                    corner_indices.append(i)

            if not corner_indices:
                if debug:
                    print(f"\n  [{corner_num}] Corner {corner}: SKIPPED (not found in polygon)")
                continue

            corner_idx = corner_indices[0]

            if debug:
                print(f"\n  ---------------------------------------------------------------")
                print(f"  [{corner_num}] Testing Corner at {corner}...")

            # Criterion 1: Connection - Check if corner is between two straight segments (Segment before and Segment after)
            prev_segment_idx = (corner_idx - 1) % len(segments)  # Segment from prev corner to this corner
            next_segment_idx = corner_idx  # Segment from this corner to next corner

            # Get the segments
            prev_segment = segments[prev_segment_idx] if prev_segment_idx < len(segments) else None
            next_segment = segments[next_segment_idx] if next_segment_idx < len(segments) else None

            prev_is_straight = prev_segment['is_straight'] if prev_segment else False
            next_is_straight = next_segment['is_straight'] if next_segment else False

            if not (prev_is_straight and next_is_straight):
                if debug:
                    print(f"  [FAIL] CRITERION 1 (Connection): FAILED")
                    print(f"    Not between two straight segments")
                    print(f"    Segment {prev_segment_idx} (from prev corner): {'straight' if prev_is_straight else 'NOT straight'}")
                    print(f"    Segment {next_segment_idx} (to next corner): {'straight' if next_is_straight else 'NOT straight'}")
                continue

            if debug:
                print(f"  [OK] CRITERION 1 (Connection): PASSED")
                print(f"    Segment {prev_segment_idx}: straight segment")
                print(f"    Segment {next_segment_idx}: straight segment")

            # Get the corner and adjacent points
            # Corner positions from segments
            p_prev_corner = np.array(prev_segment['corner_start'])  # Previous corner
            p_curr = np.array(corner)                                # Current corner
            p_next_corner = np.array(next_segment['corner_end'])     # Next corner

            # For earlier and later corners (for continuity check)
            prev_prev_segment_idx = (prev_segment_idx - 1) % len(segments)
            next_next_segment_idx = (next_segment_idx + 1) % len(segments)
            p_prev_prev_corner = np.array(segments[prev_prev_segment_idx]['corner_start']) if prev_prev_segment_idx < len(segments) else None
            p_next_next_corner = np.array(segments[next_next_segment_idx]['corner_end']) if next_next_segment_idx < len(segments) else None

            # Criterion 2: Angle - Check if angle is approximately 90°
            v2 = p_prev_corner - p_curr       # Vector from current corner to previous corner
            v3 = p_next_corner - p_curr       # Vector from current corner to next corner

            norm2 = np.linalg.norm(v2)
            norm3 = np.linalg.norm(v3)

            if norm2 < 1e-6 or norm3 < 1e-6:
                if debug:
                    print(f"  [FAIL] CRITERION 2 (Angle): SKIPPED (vectors too small)")
                continue

            cos_angle = np.dot(v2, v3) / (norm2 * norm3)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            if not (90 - angle_tolerance <= angle <= 90 + angle_tolerance):
                if debug:
                    print(f"  [FAIL] CRITERION 2 (Angle): FAILED")
                    print(f"    Measured angle: {angle:.2f}°")
                    print(f"    Required: {90-angle_tolerance:.1f}° to {90+angle_tolerance:.1f}°")
                continue

            if debug:
                print(f"  [OK] CRITERION 2 (Angle): PASSED")
                print(f"    Measured angle: {angle:.2f}°")

            # Add to potential frame corners (passed Criterion 1 and 2)
            potential_frame_corners.append({
                'corner': corner,
                'corner_num': corner_num,
                'angle': angle
            })

            # Criterion 3: Arrow toward center falls within the 90-degree opening angle
            if centroid is not None:
                centroid_arr = np.array(centroid)

                # Vector from corner toward piece center
                to_center = centroid_arr - p_curr
                to_center_norm = np.linalg.norm(to_center)

                if to_center_norm > 1e-6:
                    to_center = to_center / to_center_norm

                    # The opening angle is formed by extending v2 and v3 directions
                    # Check if to_center is inside the 90-degree angle
                    # by calculating dot products with both edge vectors (normalized)
                    v2_norm = v2 / np.linalg.norm(v2)
                    v3_norm = v3 / np.linalg.norm(v3)

                    # Both dot products should be positive if inside the angle
                    dot_v2 = np.dot(to_center, v2_norm)
                    dot_v3 = np.dot(to_center, v3_norm)

                    inside_angle = dot_v2 > 0 and dot_v3 > 0

                    if not inside_angle:
                        if debug:
                            print(f"  [FAIL] CRITERION 3 (Inward Arrow in 90° Angle): FAILED")
                            print(f"    Arrow to center vs Edge 1: dot={dot_v2:.3f} (positive={dot_v2 > 0})")
                            print(f"    Arrow to center vs Edge 2: dot={dot_v3:.3f} (positive={dot_v3 > 0})")
                        continue

                    if debug:
                        print(f"  [OK] CRITERION 3 (Inward Arrow in 90° Angle): PASSED")
                        print(f"    Arrow to center vs Edge 1: dot={dot_v2:.3f}")
                        print(f"    Arrow to center vs Edge 2: dot={dot_v3:.3f}")
                else:
                    if debug:
                        print(f"  [FAIL] CRITERION 3 (Inward Arrow in 90° Angle): FAILED (centroid too close)")
                    continue

            # Criterion 4: Convexity - Corner must be on outer convex boundary
            if contour_flat is not None:
                convex_hull = cv2.convexHull(contour_flat.reshape(-1, 1, 2))
                dist = cv2.pointPolygonTest(convex_hull, (float(p_curr[0]), float(p_curr[1])), True)

                if dist < -1:  # Point is clearly inside
                    if debug:
                        print(f"  [FAIL] CRITERION 4 (Convexity): FAILED")
                        print(f"    Point is inside convex hull (distance: {dist:.2f})")
                    continue

                if debug:
                    print(f"  [OK] CRITERION 4 (Convexity): PASSED")
                    print(f"    Point is on/near convex hull (distance: {dist:.2f})")

            # All criteria passed - Frame corner confirmed!
            if debug:
                print(f"\n  [ACCEPTED] Frame corner confirmed!")

            frame_info = {
                'corner': tuple(p_curr.astype(int)),
                'p1': tuple(p_prev_corner.astype(int)),
                'p2': tuple(p_next_corner.astype(int)),
                'angle': angle,
                'potential': False
            }
            frame_corners.append(frame_info)

        if debug:
            piece_label = f"PIECE {piece_idx}" if piece_idx is not None else "PIECE ?"
            print(f"\n  ================================================================")
            print(f"  [{piece_label}] [FRAME CORNERS] Result: {len(frame_corners)} frame corners detected ({len(potential_frame_corners)} potential)")
            print(f"  ================================================================\n")

        # Return both frame corners and potential frame corners
        result = frame_corners.copy()
        for result_item in result:
            result_item['potential'] = False

        for potential in potential_frame_corners:
            result.append({
                'corner': potential['corner'],
                'potential': True,
                'corner_num': potential['corner_num'],
                'angle': potential['angle']
            })

        return result

    @staticmethod
    def _check_bisector_direction(p1: np.ndarray, p2: np.ndarray, corner: np.ndarray,
                                  centroid: tuple, threshold: float = 0.05) -> bool:
        """
        Check if the bisector of the angle between two edges points toward the centroid.

        Args:
            p1: First point along edge 1
            p2: Second point along edge 2
            corner: The corner point
            centroid: Center of piece
            threshold: Minimum dot product (default 0.05)

        Returns:
            True if bisector points toward centroid
        """
        corner_arr = np.array(corner)
        centroid_arr = np.array(centroid)

        # Vectors pointing FROM corner TOWARD the edge endpoints
        v1 = p1 - corner_arr
        v2 = p2 - corner_arr

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1 or norm2 < 1:
            return False

        v1 = v1 / norm1
        v2 = v2 / norm2

        # Bisector: average of the two edge vectors
        bisector = v1 + v2
        bisector_norm = np.linalg.norm(bisector)

        if bisector_norm < 0.1:
            return False

        bisector = bisector / bisector_norm

        # Vector from corner to center
        to_center = centroid_arr - corner_arr
        center_norm = np.linalg.norm(to_center)

        if center_norm < 1:
            return False

        to_center = to_center / center_norm

        # Check dot product
        dot_product = np.dot(bisector, to_center)
        return dot_product > threshold

    @staticmethod
    def _check_forbidden_zones(p1: np.ndarray, corner: np.ndarray, p2: np.ndarray,
                               contour_flat: np.ndarray, extension_length: float = 200.0,
                               zone_width: float = 30.0, tolerance: float = 8.0,
                               debug: bool = False) -> bool:
        """
        Check that forbidden zones beyond the edges are clear of contour points.
        A forbidden zone is the area beyond the far end of each edge.

        Args:
            p1: First point (p1->corner is edge 1)
            corner: The corner point where edges meet
            p2: Second point (p2->corner is edge 2)
            contour_flat: Original contour points
            extension_length: How far to extend beyond the edges
            zone_width: Width of forbidden zone perpendicular to edge
            tolerance: Distance tolerance for detecting contour in zone
            debug: If True, print debug info

        Returns:
            True if both forbidden zones are clear
        """
        corner_arr = np.array(corner)

        # Check forbidden zone for edge 1 (away from corner beyond p1)
        zone1_clear = CornerDetector._check_single_forbidden_zone(
            corner_arr, p1, contour_flat, extension_length, zone_width, tolerance, debug
        )

        # Check forbidden zone for edge 2 (away from corner beyond p2)
        zone2_clear = CornerDetector._check_single_forbidden_zone(
            corner_arr, p2, contour_flat, extension_length, zone_width, tolerance, debug
        )

        return zone1_clear and zone2_clear

    @staticmethod
    def _check_single_forbidden_zone(corner: np.ndarray, far_endpoint: np.ndarray,
                                    contour_flat: np.ndarray, extension_length: float = 200.0,
                                    zone_width: float = 30.0, tolerance: float = 8.0,
                                    debug: bool = False) -> bool:
        """
        Check if the forbidden zone beyond far_endpoint is clear of contour.

        Args:
            corner: The corner point
            far_endpoint: The far end of the edge (where zone extends from)
            contour_flat: All contour points
            extension_length: How far forward to check
            zone_width: Width perpendicular to extension
            tolerance: Distance tolerance for violation
            debug: If True, print debug info

        Returns:
            True if zone is clear
        """
        # Direction from far_endpoint back toward corner
        to_corner = corner - far_endpoint
        dist_to_corner = np.linalg.norm(to_corner)

        if dist_to_corner < 1:
            return True

        to_corner_normalized = to_corner / dist_to_corner
        away_from_corner = -to_corner_normalized  # Direction away from corner

        # Check sample points along the extended line
        num_samples = int(extension_length / 10)

        for sample_i in range(num_samples):
            sample_dist = sample_i * 10
            sample_point = far_endpoint + away_from_corner * sample_dist

            # Find closest contour point
            distances = np.linalg.norm(contour_flat - sample_point, axis=1)
            min_dist = np.min(distances)

            # If contour is close, check if it's beyond the endpoint
            if min_dist < tolerance:
                closest_idx = np.argmin(distances)
                closest_pt = contour_flat[closest_idx]

                # Check if point is actually beyond the far endpoint
                dist_from_corner = np.linalg.norm(closest_pt - corner)
                dist_from_far = np.linalg.norm(closest_pt - far_endpoint)

                # If significantly beyond the far endpoint, zone is violated
                if dist_from_far > 20 and dist_from_corner > 20:
                    if debug:
                        print(f"        Forbidden zone violated at {sample_dist:.0f}px from endpoint")
                    return False

        return True
