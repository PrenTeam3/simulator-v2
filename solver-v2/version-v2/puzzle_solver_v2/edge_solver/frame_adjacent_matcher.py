

class FrameAdjacentMatcher:
    """Handles matching of frame-adjacent segments using length and shape similarity."""

    @staticmethod
    def _find_frame_touching_segments(piece, all_segments):
        """Find segments that directly touch frame corners.

        Returns:
            dict: Mapping from frame corner index to list of segment IDs that touch it
        """
        frame_touching = {}
        tolerance = 1.0

        for frame_idx, frame_corner in enumerate(piece.frame_corners):
            frame_pos = (frame_corner.x, frame_corner.y)
            touching_segs = []

            for seg in all_segments:
                seg_start = (seg.start_corner.x, seg.start_corner.y)
                seg_end = (seg.end_corner.x, seg.end_corner.y)

                if (abs(seg_start[0] - frame_pos[0]) < tolerance and
                    abs(seg_start[1] - frame_pos[1]) < tolerance):
                    touching_segs.append(seg.segment_id)
                elif (abs(seg_end[0] - frame_pos[0]) < tolerance and
                      abs(seg_end[1] - frame_pos[1]) < tolerance):
                    touching_segs.append(seg.segment_id)

            if touching_segs:
                frame_touching[frame_idx] = touching_segs

        return frame_touching

    @staticmethod
    def _get_next_segment_away_from_frame(current_segment, all_segments, frame_adjacent_segments, piece, direction=None):
        """Find the next segment in sequence moving away from the frame corner using modulo arithmetic.

        Args:
            current_segment: The current segment to find the neighbor of
            all_segments: List of all segments for this piece
            frame_adjacent_segments: List of frame-adjacent segments (neighbors of frame-touching)
            piece: The puzzle piece object (for frame corner information)
            direction: Optional; if specified (+1 or -1), use this direction instead of auto-detecting

        Returns:
            Tuple of (next_segment, direction_used) or (None, None) if not found
        """
        num_segments = len(all_segments)
        current_id = current_segment.segment_id

        # If direction is already determined, use it
        if direction is not None:
            next_id = (current_id + direction) % num_segments
            print(f"          [Direction Check] Maintaining direction ({'+1' if direction == 1 else '-1'}): {current_id} -> {next_id}")

            # Find and return the segment with this ID
            for seg in all_segments:
                if seg.segment_id == next_id:
                    return seg, direction
            return None, None

        # First time - need to determine the direction
        # Find which segments directly touch frame corners
        frame_touching = FrameAdjacentMatcher._find_frame_touching_segments(piece, all_segments)

        # Get all frame-touching segment IDs (flattened from all corners)
        all_frame_touching_ids = set()
        for seg_ids in frame_touching.values():
            all_frame_touching_ids.update(seg_ids)

        print(f"          [Direction Check] Current seg: {current_id}, Total segs: {num_segments}")
        print(f"          [Direction Check] Frame-touching segments: {sorted(all_frame_touching_ids)}")

        # Determine which direction leads away from the frame corner
        # by checking if forward/backward leads to frame-touching segments
        next_id_forward = (current_id + 1) % num_segments
        next_id_backward = (current_id - 1) % num_segments

        # The "away" direction is the one that does NOT lead to frame-touching segments
        forward_hits_frame = next_id_forward in all_frame_touching_ids
        backward_hits_frame = next_id_backward in all_frame_touching_ids

        print(f"          [Direction Check] Forward to {next_id_forward}: {'-> FRAME' if forward_hits_frame else '-> interior'}")
        print(f"          [Direction Check] Backward to {next_id_backward}: {'-> FRAME' if backward_hits_frame else '-> interior'}")

        # Choose the direction away from frame
        if not forward_hits_frame and backward_hits_frame:
            # Forward leads away, backward leads to frame
            next_id = next_id_forward
            chosen_direction = +1
            print(f"          [Direction Check] [OK] Going FORWARD (+1, away from frame) -> segment {next_id}")
        elif forward_hits_frame and not backward_hits_frame:
            # Backward leads away, forward leads to frame
            next_id = next_id_backward
            chosen_direction = -1
            print(f"          [Direction Check] [OK] Going BACKWARD (-1, away from frame) -> segment {next_id}")
        elif not forward_hits_frame and not backward_hits_frame:
            # Both directions are away from frame (we're already deep in interior)
            # Prefer forward
            next_id = next_id_forward
            chosen_direction = +1
            print(f"          [Direction Check] Both directions away from frame, choosing FORWARD (+1) -> segment {next_id}")
        else:
            # Both hit frame - we're between two frame corners, stop
            print(f"          [Direction Check] [X] Both directions hit frame - stopping")
            return None, None

        # Find and return the segment with this ID
        for seg in all_segments:
            if seg.segment_id == next_id:
                return seg, chosen_direction

        return None, None

    @staticmethod
    def _determine_segment_orientation(segment, piece):
        """Determine if segment faces outward or inward relative to piece center.

        For frame-adjacent segments:
        - Compute the normal vector at the midpoint of the segment
        - Check if normal points away from or toward piece center

        Returns:
            bool: True if segment faces outward (toward frame), False if inward
        """
        import numpy as np

        # Convert segment contour points to numpy array if they're Point objects
        if hasattr(segment.contour_points[0], 'x'):
            # Points are Point objects with x, y attributes
            points = np.array([[p.x, p.y] for p in segment.contour_points])
        else:
            # Already numpy array
            points = np.array(segment.contour_points)

        # Get segment midpoint
        midpoint_idx = len(points) // 2
        midpoint = points[midpoint_idx]

        # Calculate normal vector at midpoint
        # Use neighboring points to compute tangent, then perpendicular
        prev_idx = max(0, midpoint_idx - 5)
        next_idx = min(len(points) - 1, midpoint_idx + 5)

        tangent = points[next_idx] - points[prev_idx]
        # Normal is perpendicular to tangent (rotate 90 degrees)
        normal = np.array([-tangent[1], tangent[0]])
        normal_length = np.linalg.norm(normal)
        if normal_length > 0:
            normal = normal / normal_length
        else:
            # Fallback if tangent is zero
            normal = np.array([1, 0])

        # Get piece center from piece.contour_points
        if hasattr(piece.contour_points[0], 'x'):
            piece_contour = np.array([[p.x, p.y] for p in piece.contour_points])
        else:
            piece_contour = np.array(piece.contour_points)

        piece_center = np.mean(piece_contour, axis=0)
        to_segment = midpoint - piece_center
        to_segment_length = np.linalg.norm(to_segment)
        if to_segment_length > 0:
            to_segment = to_segment / to_segment_length
        else:
            # Segment is at piece center (shouldn't happen for edge segments)
            return True  # Default to outward

        # If normal and to_segment point in same direction, segment faces outward
        dot_product = np.dot(normal, to_segment)

        # For frame-adjacent segments, we expect them to face outward
        # So return True if the normal points away from center
        return dot_product > 0

    @staticmethod
    def _check_orientation_compatibility(seg1_outward, seg2_outward):
        """Check if two segments can match based on their orientations.

        For frame segments, both should face outward to match properly.

        Args:
            seg1_outward: bool, True if segment 1 faces outward
            seg2_outward: bool, True if segment 2 faces outward

        Returns:
            bool: True if segments can match
        """
        # Both segments should face outward for valid frame matching
        return seg1_outward and seg2_outward

    @staticmethod
    def match_frame_adjacent_segments(piece1, segments1, piece2, segments2):
        """Match frame-adjacent segments with orientation checking."""
        from ..matrix_solver.segment_matcher import SegmentMatcher
        from ..common.data_classes import SegmentMatch
        from ..common.utils import normalize_and_center, find_min_area_rotation
        import numpy as np

        matches = []

        # Pre-compute orientations for all segments
        seg1_orientations = {}
        for seg1 in segments1:
            seg1_orientations[seg1.segment_id] = (
                FrameAdjacentMatcher._determine_segment_orientation(seg1, piece1)
            )

        seg2_orientations = {}
        for seg2 in segments2:
            seg2_orientations[seg2.segment_id] = (
                FrameAdjacentMatcher._determine_segment_orientation(seg2, piece2)
            )

        # Compare each segment pair
        for seg1 in segments1:
            for seg2 in segments2:
                # Check orientation compatibility first
                seg1_outward = seg1_orientations[seg1.segment_id]
                seg2_outward = seg2_orientations[seg2.segment_id]

                if not FrameAdjacentMatcher._check_orientation_compatibility(
                        seg1_outward, seg2_outward
                ):
                    # Skip incompatible orientations
                    continue

                # Calculate length similarity
                length_score = SegmentMatcher._length_similarity(seg1, seg2)

                # Only calculate shape similarity if length is reasonably similar
                shape_score = 0.0
                optimal_rotation = 0.0
                if length_score > 0.75:
                    # Resample and normalize segments
                    target_points = 50
                    pts1_resampled = SegmentMatcher._resample_points(
                        seg1.contour_points, target_points
                    )
                    pts2_resampled = SegmentMatcher._resample_points(
                        seg2.contour_points, target_points
                    )

                    pts1_norm = normalize_and_center(pts1_resampled)
                    pts2_norm = normalize_and_center(pts2_resampled)

                    # When matching, we need to consider that segments should be
                    # aligned with their outside faces touching
                    # This might require flipping one segment before comparison
                    if seg1_outward and seg2_outward:
                        # Flip seg2 for proper outside-to-outside matching
                        pts2_norm = pts2_norm[::-1]

                    # Find optimal rotation angle
                    optimal_rotation = find_min_area_rotation(pts1_norm, pts2_norm)

                    # Calculate shape similarity using RMSD
                    rmsd = SegmentMatcher._calculate_shape_similarity_rmsd(seg1, seg2)
                    # Convert RMSD to similarity score (0-1, where 1 is best)
                    shape_score = max(0.0, 1.0 - (rmsd * 2.0))

                # Combined match score (weighted average)
                match_score = 0.5 * length_score + 0.5 * shape_score

                matches.append(SegmentMatch(
                    piece1_id=piece1.piece_id,
                    piece2_id=piece2.piece_id,
                    seg1_id=seg1.segment_id,
                    seg2_id=seg2.segment_id,
                    match_score=match_score,
                    optimal_rotation=optimal_rotation,
                    shape_score=shape_score,
                    length_score=length_score,
                    description=f"P{piece1.piece_id}-S{seg1.segment_id} (out:{seg1_outward}) <-> "
                                f"P{piece2.piece_id}-S{seg2.segment_id} (out:{seg2_outward})"
                ))

        # Sort by match score (descending)
        matches.sort(key=lambda m: m.match_score, reverse=True)

        return matches

    @staticmethod
    def extend_segment_matches(initial_matches, piece1, segments1, frame_segs1, piece2, segments2, frame_segs2, threshold=0.8):
        """Extend segment matches along the contour away from frame corners.

        Args:
            initial_matches: List of initial SegmentMatch objects
            piece1, piece2: Piece objects
            segments1, segments2: All segments for each piece
            frame_segs1, frame_segs2: Frame-adjacent segments for each piece
            threshold: Minimum match score to continue extending (default: 0.75)

        Returns:
            List of ExtendedSegmentMatch objects
        """
        from ..common.data_classes import ExtendedSegmentMatch, Point
        from ..matrix_solver.segment_matcher import SegmentMatcher
        from ..common.utils import normalize_and_center, find_min_area_rotation
        import numpy as np
        from scipy.spatial.distance import cdist

        extended_matches = []

        for initial_match in initial_matches:
            # Only extend good initial matches
            if initial_match.match_score < threshold:
                continue

            print(f"\n    Extending match: P{initial_match.piece1_id}-S{initial_match.seg1_id} <-> "
                  f"P{initial_match.piece2_id}-S{initial_match.seg2_id} "
                  f"(initial score: {initial_match.match_score:.3f})")

            extended_match_list = []

            # Find the initial segments
            seg1 = next(s for s in frame_segs1 if s.segment_id == initial_match.seg1_id)
            seg2 = next(s for s in frame_segs2 if s.segment_id == initial_match.seg2_id)

            # Find the frame connection points (where straight frame segments end)
            # This is the point on the frame-adjacent segment that touches the frame-touching segment
            frame_touching1 = FrameAdjacentMatcher._find_frame_touching_segments(piece1, segments1)
            frame_touching2 = FrameAdjacentMatcher._find_frame_touching_segments(piece2, segments2)

            all_frame_touching_ids1 = set()
            for seg_ids in frame_touching1.values():
                all_frame_touching_ids1.update(seg_ids)
            all_frame_touching_ids2 = set()
            for seg_ids in frame_touching2.values():
                all_frame_touching_ids2.update(seg_ids)

            # Find which end of the frame-adjacent segment connects to frame
            # Check if start or end corner connects to a frame-touching segment
            tolerance = 1.0
            frame_connection_point1 = None
            for frame_touching_seg in segments1:
                if frame_touching_seg.segment_id not in all_frame_touching_ids1:
                    continue
                # Check if seg1 connects to this frame-touching segment
                if (abs(seg1.start_corner.x - frame_touching_seg.end_corner.x) < tolerance and
                    abs(seg1.start_corner.y - frame_touching_seg.end_corner.y) < tolerance):
                    frame_connection_point1 = seg1.start_corner
                    break
                elif (abs(seg1.end_corner.x - frame_touching_seg.end_corner.x) < tolerance and
                      abs(seg1.end_corner.y - frame_touching_seg.end_corner.y) < tolerance):
                    frame_connection_point1 = seg1.end_corner
                    break
                elif (abs(seg1.start_corner.x - frame_touching_seg.start_corner.x) < tolerance and
                      abs(seg1.start_corner.y - frame_touching_seg.start_corner.y) < tolerance):
                    frame_connection_point1 = seg1.start_corner
                    break
                elif (abs(seg1.end_corner.x - frame_touching_seg.start_corner.x) < tolerance and
                      abs(seg1.end_corner.y - frame_touching_seg.start_corner.y) < tolerance):
                    frame_connection_point1 = seg1.end_corner
                    break

            frame_connection_point2 = None
            for frame_touching_seg in segments2:
                if frame_touching_seg.segment_id not in all_frame_touching_ids2:
                    continue
                # Check if seg2 connects to this frame-touching segment
                if (abs(seg2.start_corner.x - frame_touching_seg.end_corner.x) < tolerance and
                    abs(seg2.start_corner.y - frame_touching_seg.end_corner.y) < tolerance):
                    frame_connection_point2 = seg2.start_corner
                    break
                elif (abs(seg2.end_corner.x - frame_touching_seg.end_corner.x) < tolerance and
                      abs(seg2.end_corner.y - frame_touching_seg.end_corner.y) < tolerance):
                    frame_connection_point2 = seg2.end_corner
                    break
                elif (abs(seg2.start_corner.x - frame_touching_seg.start_corner.x) < tolerance and
                      abs(seg2.start_corner.y - frame_touching_seg.start_corner.y) < tolerance):
                    frame_connection_point2 = seg2.start_corner
                    break
                elif (abs(seg2.end_corner.x - frame_touching_seg.start_corner.x) < tolerance and
                      abs(seg2.end_corner.y - frame_touching_seg.start_corner.y) < tolerance):
                    frame_connection_point2 = seg2.end_corner
                    break

            print(f"    Frame connection points: P{piece1.piece_id} at ({frame_connection_point1.x:.1f}, {frame_connection_point1.y:.1f}), "
                  f"P{piece2.piece_id} at ({frame_connection_point2.x:.1f}, {frame_connection_point2.y:.1f})")

            # Track the accumulated chain of segments for each piece
            chain_segments1 = [seg1]
            chain_segments2 = [seg2]

            # Track segments we've already used to avoid duplicates
            used_seg1_ids = {seg1.segment_id}
            used_seg2_ids = {seg2.segment_id}

            current_seg1 = seg1
            current_seg2 = seg2

            # Track the direction for each piece (determined on first step, maintained thereafter)
            direction1 = None
            direction2 = None

            # Try to extend the match
            extension_step = 0
            while True:
                extension_step += 1
                print(f"      Extension step {extension_step}:")
                # Get next segments moving away from frame
                # First call determines direction, subsequent calls maintain it
                next_seg1, direction1 = FrameAdjacentMatcher._get_next_segment_away_from_frame(
                    current_seg1, segments1, frame_segs1, piece1, direction1
                )
                next_seg2, direction2 = FrameAdjacentMatcher._get_next_segment_away_from_frame(
                    current_seg2, segments2, frame_segs2, piece2, direction2
                )

                print(f"        Current segments: P{piece1.piece_id}-S{current_seg1.segment_id} <-> "
                      f"P{piece2.piece_id}-S{current_seg2.segment_id}")
                print(f"        Next segments: P{piece1.piece_id}-S{next_seg1.segment_id if next_seg1 else 'None'} <-> "
                      f"P{piece2.piece_id}-S{next_seg2.segment_id if next_seg2 else 'None'}")

                # Stop if we can't find next segments on both pieces
                if next_seg1 is None or next_seg2 is None:
                    print(f"        STOP: Cannot find next segment on both pieces")
                    break

                # Stop if we've already used these segments (circular contour case)
                if next_seg1.segment_id in used_seg1_ids or next_seg2.segment_id in used_seg2_ids:
                    print(f"        STOP: Segment already used (circular contour)")
                    break

                # Add the new segments to the chains
                chain_segments1.append(next_seg1)
                chain_segments2.append(next_seg2)

                # Now compare the ENTIRE accumulated chains, not just the new segments
                print(f"        Comparing entire chains: {len(chain_segments1)} segments each")

                # Combine all points from the chain for each piece
                chain_points1 = []
                for seg in chain_segments1:
                    chain_points1.extend(seg.contour_points)

                chain_points2 = []
                for seg in chain_segments2:
                    chain_points2.extend(seg.contour_points)

                # Calculate length similarity for the entire chain
                chain_length1 = sum(SegmentMatcher._get_segment_length(s) for s in chain_segments1)
                chain_length2 = sum(SegmentMatcher._get_segment_length(s) for s in chain_segments2)

                if chain_length1 == 0 or chain_length2 == 0:
                    length_score = 0.0
                else:
                    avg_length = (chain_length1 + chain_length2) / 2
                    length_diff = abs(chain_length1 - chain_length2)
                    length_score = max(0.0, 1.0 - (length_diff / avg_length))

                print(f"        Chain length score: {length_score:.3f}")

                shape_score = 0.0
                optimal_rotation = 0.0

                if length_score > 0.75:
                    # Resample and normalize the entire chains
                    target_points = 50 * len(chain_segments1)  # Scale with chain length
                    pts1_resampled = SegmentMatcher._resample_points(chain_points1, target_points)
                    pts2_resampled = SegmentMatcher._resample_points(chain_points2, target_points)

                    pts1_norm = normalize_and_center(pts1_resampled)
                    pts2_norm = normalize_and_center(pts2_resampled)

                    # Find optimal rotation angle for the entire chain
                    optimal_rotation = find_min_area_rotation(pts1_norm, pts2_norm)

                    # Calculate shape similarity using RMSD for the entire chain
                    pts1_array = np.array([[p.x, p.y] for p in pts1_norm])
                    pts2_rotated = SegmentMatcher._resample_points(
                        [Point(p.x * np.cos(optimal_rotation) - p.y * np.sin(optimal_rotation),
                               p.x * np.sin(optimal_rotation) + p.y * np.cos(optimal_rotation))
                         for p in pts2_norm], target_points
                    )
                    pts2_array = np.array([[p.x, p.y] for p in pts2_rotated])

                    distance_matrix = cdist(pts1_array, pts2_array)
                    min_distances_1_to_2 = np.min(distance_matrix, axis=1)
                    min_distances_2_to_1 = np.min(distance_matrix, axis=0)
                    all_distances = np.concatenate([min_distances_1_to_2, min_distances_2_to_1])
                    rmsd = np.sqrt(np.mean(all_distances ** 2))

                    shape_score = max(0.0, 1.0 - (rmsd * 2.0))

                    # Check connection point alignment
                    # Transform frame connection points using the same normalization and rotation
                    if frame_connection_point1 and frame_connection_point2:
                        # Get centroid of original chain for normalization
                        chain1_centroid_x = sum(p.x for p in chain_points1) / len(chain_points1)
                        chain1_centroid_y = sum(p.y for p in chain_points1) / len(chain_points1)
                        chain2_centroid_x = sum(p.x for p in chain_points2) / len(chain_points2)
                        chain2_centroid_y = sum(p.y for p in chain_points2) / len(chain_points2)

                        # Normalize connection points (center and scale like we did for the chains)
                        # Get max distance for scaling (same as normalize_and_center does)
                        max_dist1 = max(np.sqrt((p.x - chain1_centroid_x)**2 + (p.y - chain1_centroid_y)**2) for p in chain_points1)
                        max_dist2 = max(np.sqrt((p.x - chain2_centroid_x)**2 + (p.y - chain2_centroid_y)**2) for p in chain_points2)

                        # Normalize frame connection point 1
                        conn1_x = (frame_connection_point1.x - chain1_centroid_x) / max_dist1 if max_dist1 > 0 else 0
                        conn1_y = (frame_connection_point1.y - chain1_centroid_y) / max_dist1 if max_dist1 > 0 else 0

                        # Normalize and rotate frame connection point 2
                        conn2_x_centered = (frame_connection_point2.x - chain2_centroid_x) / max_dist2 if max_dist2 > 0 else 0
                        conn2_y_centered = (frame_connection_point2.y - chain2_centroid_y) / max_dist2 if max_dist2 > 0 else 0

                        # Apply same rotation as used for chain2
                        conn2_x_rot = conn2_x_centered * np.cos(optimal_rotation) - conn2_y_centered * np.sin(optimal_rotation)
                        conn2_y_rot = conn2_x_centered * np.sin(optimal_rotation) + conn2_y_centered * np.cos(optimal_rotation)

                        # Calculate distance between transformed connection points
                        connection_distance = np.sqrt((conn1_x - conn2_x_rot)**2 + (conn1_y - conn2_y_rot)**2)

                        # Convert to alignment score (0 = far apart, 1 = perfectly aligned)
                        # Assume normalized space, so distance > 0.1 is bad
                        connection_alignment_score = max(0.0, 1.0 - (connection_distance / 0.2))

                        print(f"        Connection point distance: {connection_distance:.4f}, alignment score: {connection_alignment_score:.3f}")

                        # Include connection alignment in the shape score
                        # Give it significant weight since it's critical for proper fitting
                        shape_score = 0.7 * shape_score + 0.3 * connection_alignment_score
                    else:
                        connection_alignment_score = 1.0  # Default if we couldn't find connection points
                        print(f"        Warning: Could not find frame connection points for alignment check")

                # Dynamic weighting: as chain grows, shape matters more than length
                # Start with 50/50, gradually shift to favor shape
                # Chain length 1-2: 50% length, 50% shape
                # Chain length 3-4: 40% length, 60% shape
                # Chain length 5+:  30% length, 70% shape
                chain_length = len(chain_segments1)
                if chain_length <= 2:
                    length_weight = 0.5
                    shape_weight = 0.5
                elif chain_length <= 4:
                    length_weight = 0.4
                    shape_weight = 0.6
                else:
                    length_weight = 0.3
                    shape_weight = 0.7

                # Combined match score for the entire chain
                match_score = length_weight * length_score + shape_weight * shape_score

                # ============================================================
                # PROGRESSIVE SHAPE THRESHOLD - IMPORTANT FOR QUALITY CONTROL
                # ============================================================
                # As chains get longer, shape similarity must be stricter
                # This prevents accumulation of poor-quality extensions
                # Chain length 1-2: shape must be >= 0.70
                # Chain length 3-4: shape must be >= 0.80
                # Chain length 5-6: shape must be >= 0.85
                # Chain length 7+:  shape must be >= 0.90
                if chain_length <= 2:
                    shape_threshold = 0.70
                elif chain_length <= 4:
                    shape_threshold = 0.80
                elif chain_length <= 6:
                    shape_threshold = 0.85
                else:
                    shape_threshold = 0.90
                # ============================================================

                print(f"        Chain shape score: {shape_score:.3f} (threshold: {shape_threshold:.2f})")
                print(f"        Chain weights: length={length_weight:.1f}, shape={shape_weight:.1f}")
                print(f"        Chain combined match score: {match_score:.3f} (threshold: {threshold:.2f})")

                # Stop if shape quality drops below threshold OR combined score below threshold
                if shape_score < shape_threshold or match_score < threshold:
                    if shape_score < shape_threshold:
                        print(f"        STOP: Chain shape score {shape_score:.3f} below shape threshold {shape_threshold:.2f}")
                    else:
                        print(f"        STOP: Chain match score {match_score:.3f} below combined threshold {threshold}")
                    # Remove the last segments we added since they failed
                    chain_segments1.pop()
                    chain_segments2.pop()
                    break

                print(f"        [OK] Extended match accepted!")

                # Create match object for this extension
                from ..common.data_classes import SegmentMatch
                extension_match = SegmentMatch(
                    piece1_id=piece1.piece_id,
                    piece2_id=piece2.piece_id,
                    seg1_id=next_seg1.segment_id,
                    seg2_id=next_seg2.segment_id,
                    match_score=match_score,
                    optimal_rotation=optimal_rotation,
                    shape_score=shape_score,
                    length_score=length_score,
                    description=f"Extended: P{piece1.piece_id}-S{next_seg1.segment_id} <-> "
                                f"P{piece2.piece_id}-S{next_seg2.segment_id}"
                )

                extended_match_list.append(extension_match)
                used_seg1_ids.add(next_seg1.segment_id)
                used_seg2_ids.add(next_seg2.segment_id)

                # Move to next segments
                current_seg1 = next_seg1
                current_seg2 = next_seg2

            # Create ExtendedSegmentMatch object
            total_matched = 1 + len(extended_match_list)

            # Calculate average match score
            all_scores = [initial_match.match_score] + [m.match_score for m in extended_match_list]
            avg_score = sum(all_scores) / len(all_scores)

            # Combined score: 50% quality + 50% length bonus
            # Length bonus: normalize by max possible segments (e.g., 10)
            max_segments = 10
            length_bonus = min(1.0, total_matched / max_segments)
            combined_score = 0.5 * avg_score + 0.5 * length_bonus

            print(f"    Final result: {total_matched} segments matched, "
                  f"avg_score={avg_score:.3f}, combined_score={combined_score:.3f}")

            extended_match = ExtendedSegmentMatch(
                initial_match=initial_match,
                extended_matches=extended_match_list,
                total_segments_matched=total_matched,
                combined_score=combined_score,
                average_match_score=avg_score
            )

            extended_matches.append(extended_match)

        # Sort by combined score (descending)
        extended_matches.sort(key=lambda m: m.combined_score, reverse=True)

        return extended_matches