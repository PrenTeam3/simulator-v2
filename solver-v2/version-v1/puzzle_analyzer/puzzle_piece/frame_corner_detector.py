import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from puzzle_analyzer.geometry import LineSegment


class FrameCornerDetector:
    """Handles frame corner detection for puzzle pieces with strict validation."""

    @staticmethod
    def identify_frame_corners(border_edges: List[LineSegment],
                               convex_hull,
                               contour,
                               contour_flat: np.ndarray,
                               centroid: Tuple[int, int],
                               inner_corners: List[Tuple[int, int]],
                               outer_corners: List[Tuple[int, int]],
                               corner_ids: Dict[Tuple[int, int], int],
                               angle_tolerance: float = 7.5,
                               debug_image=None,
                               verbose: bool = False) -> List[Tuple[int, int]]:
        """
        Identify frame corners where two border edges meet at ~90 degrees.

        Currently implements:
        - Test 1: Angle check (~90deg)
        """
        frame_corners = []
        n = len(border_edges)

        if n < 2:
            print("  Not enough border edges to form corners")
            return frame_corners

        if verbose:
            print(f"  Checking {n} border edges for frame corners...")

        # Check all pairs of border edges
        for i in range(n):
            for j in range(i + 1, n):
                edge1 = border_edges[i]
                edge2 = border_edges[j]

                # Check if edges share a common point
                corner = FrameCornerDetector._find_common_point(edge1, edge2)
                if corner is None:
                    if verbose:
                        print(f"    Edges {i},{j}: No common point found (endpoints too far apart)")
                    continue

                # Get corner ID if available
                corner_id = corner_ids.get(corner, None)
                corner_label = f"C{corner_id}" if corner_id is not None else "unknown"

                if verbose:
                    print(f"\n    === Evaluating edges {i},{j} meeting at corner {corner} (Corner {corner_label}) ===")
                    print(f"        Edge {i}: {edge1.p1} -> {edge1.p2} (length: {edge1.length:.1f})")
                    print(f"        Edge {j}: {edge2.p1} -> {edge2.p2} (length: {edge2.length:.1f})")

                # TEST 1: Check angle (~90deg)
                angle = edge1.angle_with(edge2)
                if verbose:
                    print(f"\n      [TEST 1] Angle Check:")
                    print(f"        Measured angle between edges: {angle:.1f}deg")
                    print(
                        f"        Required: 90deg ± {angle_tolerance}deg (range: {90 - angle_tolerance}deg to {90 + angle_tolerance}deg)")

                if not (90 - angle_tolerance <= angle <= 90 + angle_tolerance):
                    if verbose:
                        print(f"        [NO] FAILED: Angle {angle:.1f}deg is outside acceptable range")
                    continue

                if verbose:
                    print(f"        [OK] PASSED: Angle is within tolerance")

                # TEST 2: Check that angle opens inward (toward center)
                if verbose:
                    print(f"\n      [TEST 2] Angle Direction Check:")
                    print(f"        Checking if the 90deg angle opens toward piece center at {centroid}")

                # Calculate the bisector between the two edges
                bisector_result = FrameCornerDetector._calculate_bisector(edge1, edge2, corner, centroid)

                if bisector_result is None:
                    if verbose:
                        print(f"        [NO] FAILED: Could not calculate bisector (edges too short or parallel)")
                    continue

                bisector, to_center, dot_product = bisector_result

                if verbose:
                    print(f"        Bisector vector: [{bisector[0]:.3f}, {bisector[1]:.3f}]")
                    print(f"        To-center vector: [{to_center[0]:.3f}, {to_center[1]:.3f}]")
                    print(f"        Dot product: {dot_product:.3f}")
                    print(
                        f"        Interpretation: Bisector points {'TOWARD center' if dot_product > 0 else 'AWAY from center'}")

                # Draw bisector if debug_image is provided
                if debug_image is not None:
                    corner_arr = np.array(corner)
                    centroid_arr = np.array(centroid)

                    # Draw bisector (blue line)
                    bisector_end = corner_arr + bisector * 80
                    cv2.arrowedLine(debug_image,
                                   tuple(corner_arr.astype(int)),
                                   tuple(bisector_end.astype(int)),
                                   (255, 128, 0), 2, tipLength=0.3)  # Blue arrow

                    # Draw to-center vector (yellow line)
                    center_end = corner_arr + to_center * 80
                    cv2.arrowedLine(debug_image,
                                   tuple(corner_arr.astype(int)),
                                   tuple(center_end.astype(int)),
                                   (0, 255, 255), 2, tipLength=0.3)  # Yellow arrow

                # For frame corners: bisector should point TOWARD center (positive dot product)
                if dot_product <= 0.05:
                    if verbose:
                        print(f"        [NO] FAILED: Bisector points away from center (not a frame corner)")
                    continue

                if verbose:
                    print(f"        [OK] PASSED: Bisector points toward center (angle opens inward)")

                # TEST 3: Check forbidden zone - no contour should exist beyond corner
                if verbose:
                    print(f"\n      [TEST 3] Forbidden Zone Check:")
                    print(f"        Checking that no contour exists in the area beyond the corner")

                other1 = FrameCornerDetector._get_far_endpoint(edge1, corner)
                other2 = FrameCornerDetector._get_far_endpoint(edge2, corner)

                # Check forbidden zone for edge 1
                if verbose:
                    print(f"        Checking forbidden zone beyond edge {i}...")
                zone1_clear = FrameCornerDetector._check_forbidden_zone_clear(
                    edge1, corner, other1, centroid, contour_flat, bisector, debug_image, verbose
                )
                if verbose:
                    print(f"          Result: {'Zone is CLEAR [OK]' if zone1_clear else 'Contour found in zone [NO]'}")

                if not zone1_clear:
                    if verbose:
                        print(f"        [NO] FAILED: Edge {i} has contour in forbidden zone")
                    continue

                # Check forbidden zone for edge 2
                if verbose:
                    print(f"        Checking forbidden zone beyond edge {j}...")
                zone2_clear = FrameCornerDetector._check_forbidden_zone_clear(
                    edge2, corner, other2, centroid, contour_flat, bisector, debug_image, verbose
                )
                if verbose:
                    print(f"          Result: {'Zone is CLEAR [OK]' if zone2_clear else 'Contour found in zone [NO]'}")

                if not zone2_clear:
                    if verbose:
                        print(f"        [NO] FAILED: Edge {j} has contour in forbidden zone")
                    continue

                if verbose:
                    print(f"        [OK] PASSED: Both forbidden zones are clear")

                # Success - accepted as frame corner!
                if corner not in frame_corners:
                    frame_corners.append(corner)
                    print(f"  [OK] ACCEPTED AS FRAME CORNER {corner_label} at {corner}")

        return frame_corners

    @staticmethod
    def _calculate_bisector(edge1: LineSegment,
                            edge2: LineSegment,
                            corner: Tuple[int, int],
                            centroid: Tuple[int, int]) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Calculate the bisector of the angle between two edges at their common corner.

        Returns: (bisector_vector, to_center_vector, dot_product) or None if calculation fails
        """
        corner_arr = np.array(corner)
        centroid_arr = np.array(centroid)

        # Get the far endpoints (the other end of each edge, not the corner)
        other1 = FrameCornerDetector._get_far_endpoint(edge1, corner)
        other2 = FrameCornerDetector._get_far_endpoint(edge2, corner)

        # Vectors pointing FROM corner TOWARD the far endpoints (along the edges)
        v1 = other1 - corner_arr
        v2 = other2 - corner_arr

        # Normalize
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1 or norm2 < 1:
            return None

        v1 = v1 / norm1
        v2 = v2 / norm2

        # Bisector: the average of the two edge vectors
        # This points "into" the opening between the edges
        bisector = v1 + v2
        bisector_norm = np.linalg.norm(bisector)

        if bisector_norm < 0.1:
            # Edges are opposite directions (180deg), not valid
            return None

        bisector = bisector / bisector_norm

        # Vector from corner to center
        to_center = centroid_arr - corner_arr
        center_norm = np.linalg.norm(to_center)

        if center_norm < 1:
            return None

        to_center = to_center / center_norm

        # Calculate dot product
        dot_product = np.dot(bisector, to_center)

        return (bisector, to_center, dot_product)

    @staticmethod
    def _check_forbidden_zone_clear(edge: LineSegment,
                                    corner: Tuple[int, int],
                                    far_endpoint: np.ndarray,
                                    centroid: Tuple[int, int],
                                    contour_flat: np.ndarray,
                                    bisector: np.ndarray,
                                    debug_image=None,
                                    verbose: bool = False,
                                    extension_length: float = 200,
                                    backward_extension: float = 50,
                                    zone_width: float = 30) -> bool:
        """
        Check if the 'forbidden zone' beyond the far endpoint is clear of contour points.

        The forbidden zone extends FROM the far endpoint (opposite end from corner)
        AWAY from the corner (bent slightly toward bisector), and also extends
        backwards from the far endpoint. If any contour exists in this zone,
        it means the edge continues (so it's not a frame corner).

        Args:
            edge: The border edge
            corner: The corner point
            far_endpoint: The other end of the edge (where we check from)
            centroid: Center of the piece
            contour_flat: All contour points (Nx2 array)
            bisector: The bisector direction (toward interior of 90deg angle)
            debug_image: Optional image for debug visualization
            extension_length: How far to extend forward beyond far_endpoint
            backward_extension: How far to extend backward from far_endpoint
            zone_width: Width of the search zone (perpendicular to line)
        """
        corner_arr = np.array(corner)
        far_endpoint_arr = np.array(far_endpoint)
        centroid_arr = np.array(centroid)

        # Direction from far_endpoint toward corner
        to_corner_from_far = corner_arr - far_endpoint_arr
        dist_to_corner = np.linalg.norm(to_corner_from_far)

        if dist_to_corner < 1:
            return True

        to_corner_from_far = to_corner_from_far / dist_to_corner

        # The extended direction should:
        # 1. Go AWAY from the corner (opposite of to_corner_from_far)
        # 2. Bend toward the bisector direction (interior of 90deg angle)

        away_from_corner = -to_corner_from_far  # Direction away from corner
        extended_dir = away_from_corner + bisector * 0.3  # Add bisector influence
        extended_dir = extended_dir / np.linalg.norm(extended_dir)

        if verbose:
            print(f"          Line extends toward bisector direction (interior of 90deg angle)")

        # Define the forbidden zone as points along the extended direction
        # Check multiple sample points (both forward and backward)
        num_samples_forward = int(extension_length / 5)
        num_samples_backward = int(backward_extension / 5)

        if verbose:
            print(f"          Checking zone: {num_samples_backward} samples backward, {num_samples_forward} samples forward")
            print(f"          Note: Only checking contour points from THIS piece (total: {len(contour_flat)} points)")
            print(f"          Direction: Away from corner, bent toward interior of 90deg angle")

        # Calculate perpendicular direction (for the zone width)
        # The zone width should be on the side OPPOSITE from where the bisector points
        perp_dir = np.array([-extended_dir[1], extended_dir[0]])

        # Choose the perpendicular direction that points AWAY from bisector
        # (has negative or minimal dot product with bisector)
        dot_perp1 = np.dot(perp_dir, bisector)
        dot_perp2 = np.dot(-perp_dir, bisector)

        # Use the perpendicular that points more away from bisector (smaller dot product)
        if dot_perp1 < dot_perp2:
            zone_perp = perp_dir
        else:
            zone_perp = -perp_dir

        if verbose:
            print(f"          Zone width extends on side OPPOSITE from bisector")

        # Draw the forbidden zone if debug_image is provided
        if debug_image is not None:
            # Draw the extended direction line (forward and backward)
            forward_end = far_endpoint_arr + extended_dir * extension_length
            backward_end = far_endpoint_arr - extended_dir * backward_extension

            cv2.line(debug_image,
                    tuple(backward_end.astype(int)),
                    tuple(forward_end.astype(int)),
                    (0, 165, 255), 2)  # Orange line

            # Draw zone boundary (perpendicular to line, opposite side from bisector)
            boundary_points = []
            # Backward samples
            for sample_i in range(-num_samples_backward, 0):
                sample_pt = far_endpoint_arr + extended_dir * (sample_i * 5)
                outer_pt = sample_pt + zone_perp * zone_width
                boundary_points.append(outer_pt)
            # Forward samples
            for sample_i in range(num_samples_forward + 1):
                sample_pt = far_endpoint_arr + extended_dir * (sample_i * 5)
                outer_pt = sample_pt + zone_perp * zone_width
                boundary_points.append(outer_pt)

            # Draw the boundary line connecting all outer points
            if len(boundary_points) > 1:
                for k in range(len(boundary_points) - 1):
                    cv2.line(debug_image,
                            tuple(boundary_points[k].astype(int)),
                            tuple(boundary_points[k+1].astype(int)),
                            (0, 165, 255), 1)  # Orange boundary line

        zone_violated = False
        violation_point = None

        # Get all points on the current edge to exclude them from checks
        # Calculate a line between corner and far_endpoint, and find all contour points near it
        edge_points_mask = np.zeros(len(contour_flat), dtype=bool)
        for idx, pt in enumerate(contour_flat):
            # Distance from point to line (corner to far_endpoint)
            line_vec = far_endpoint_arr - corner_arr
            line_len = np.linalg.norm(line_vec)
            if line_len > 0:
                t = np.clip(np.dot(pt - corner_arr, line_vec) / (line_len * line_len), 0, 1)
                projection = corner_arr + t * line_vec
                dist_to_line = np.linalg.norm(pt - projection)

                # Mark as edge point if close to the line between corner and far_endpoint
                if dist_to_line < 10 and 0 <= t <= 1:
                    edge_points_mask[idx] = True

        # Check both backward and forward samples (including at far_endpoint to close gap)
        for i in range(-num_samples_backward, num_samples_forward + 1):
            sample_dist = i * 5
            # Center line sample point
            sample_point_center = far_endpoint_arr + extended_dir * sample_dist

            # Sample points along the zone perpendicular (opposite side from bisector)
            # Check multiple points along this perpendicular direction
            num_perp_samples = int(zone_width / 5)
            for j in range(num_perp_samples + 1):
                perp_offset = j * 5
                sample_point = sample_point_center + zone_perp * perp_offset

                # Draw sample point
                if debug_image is not None:
                    cv2.circle(debug_image, tuple(sample_point.astype(int)), 2, (0, 165, 255), -1)

                # Find closest contour point to this sample (excluding edge points)
                distances = np.linalg.norm(contour_flat - sample_point, axis=1)

                # Filter out edge points
                distances_filtered = distances.copy()
                distances_filtered[edge_points_mask] = np.inf

                if np.all(np.isinf(distances_filtered)):
                    continue  # No valid points to check

                min_dist = np.min(distances_filtered)

                # If we find contour within close proximity (tighter tolerance)
                if min_dist < 8:  # Tighter tolerance since we're sampling the zone
                    closest_idx = np.argmin(distances_filtered)
                    closest_pt = contour_flat[closest_idx]

                    # Double-check: Make sure it's not part of the original edge
                    dist_from_corner = np.linalg.norm(closest_pt - corner_arr)
                    dist_from_far = np.linalg.norm(closest_pt - far_endpoint_arr)

                    # If it's beyond the far endpoint (not part of the original edge)
                    if dist_from_corner > 20 and dist_from_far > 20:
                        if verbose:
                            print(f"          Found contour point at {sample_dist:.1f}px from far endpoint (forbidden zone violated)")
                            print(f"          Point is {dist_from_far:.1f}px from far endpoint, {dist_from_corner:.1f}px from corner")
                        zone_violated = True
                        violation_point = closest_pt

                        # Draw the violation
                        if debug_image is not None:
                            cv2.circle(debug_image, tuple(closest_pt.astype(int)), 8, (0, 0, 255), 2)  # Red circle
                            cv2.line(debug_image, tuple(sample_point.astype(int)),
                                    tuple(closest_pt.astype(int)), (0, 0, 255), 1)  # Red line to violation

                        return False

        if verbose:
            print(f"          No contour found in forbidden zone")

        # Draw success indicator
        if debug_image is not None and not zone_violated:
            end_point = far_endpoint_arr + extended_dir * extension_length
            cv2.circle(debug_image, tuple(end_point.astype(int)), 10, (0, 255, 0), 2)  # Green circle at end

        return True

    @staticmethod
    def _get_far_endpoint(edge: LineSegment, corner_pt: Tuple[int, int]) -> np.ndarray:
        """
        Given an edge and one of its endpoints (corner_pt),
        return the *other* endpoint as a numpy array.
        """
        corner_arr = np.array(corner_pt)
        p1_arr = np.array(edge.p1)
        p2_arr = np.array(edge.p2)

        if np.linalg.norm(p1_arr - corner_arr) > np.linalg.norm(p2_arr - corner_arr):
            return p1_arr
        else:
            return p2_arr

    @staticmethod
    def _find_common_point(seg1: LineSegment, seg2: LineSegment,
                           tolerance: float = 25.0) -> Optional[Tuple[int, int]]:
        """Find if two segments share a common endpoint within tolerance."""
        points1 = [seg1.p1, seg1.p2]
        points2 = [seg2.p1, seg2.p2]

        min_dist = float('inf')
        best_point = None

        for p1 in points1:
            for p2 in points2:
                dist = np.linalg.norm(np.array(p1) - np.array(p2))
                if dist <= tolerance and dist < min_dist:
                    min_dist = dist
                    best_point = p1

        return best_point