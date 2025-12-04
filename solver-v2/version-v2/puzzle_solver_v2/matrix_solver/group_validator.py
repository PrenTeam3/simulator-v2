"""Validate diagonal groups by combining segments and expanding until match quality degrades."""
import random
import numpy as np
from typing import Dict, List, Tuple
from ..common.data_classes import ContourSegment
from .segment_matcher import SegmentMatcher
from ..common.utils import normalize_and_center, find_min_area_rotation


class GroupValidator:
    """Validate and expand groups by iteratively adding adjacent segments until quality degrades."""

    @staticmethod
    def _combine_segments_list(segments: List[ContourSegment]) -> ContourSegment:
        """Combine multiple adjacent segments into a single group segment.

        Args:
            segments: List of segments to combine (in order)

        Returns:
            A new ContourSegment representing the combined group
        """
        if not segments:
            return None

        # Combine contour points from all segments
        combined_points = []
        for seg in segments:
            combined_points.extend(seg.contour_points)

        # Create a new segment with combined properties
        combined_segment = ContourSegment(
            segment_id=-1,  # Temporary ID for combined segment
            piece_id=segments[0].piece_id,
            start_corner=segments[0].start_corner,
            end_corner=segments[-1].end_corner,
            contour_points=combined_points,
            piece_centroid=segments[0].piece_centroid,
            is_border_edge=any(seg.is_border_edge for seg in segments)
        )

        return combined_segment

    @staticmethod
    def _calculate_group_scores(group_segs1: List[ContourSegment],
                               group_segs2: List[ContourSegment]) -> Tuple[float, float, float, str]:
        """Calculate length, shape, and angle similarity scores for two group segments.

        Args:
            group_segs1: List of segments from piece 1 to combine
            group_segs2: List of segments from piece 2 to combine

        Returns:
            Tuple of (length_score, shape_score, angle_score, quality_text)
        """
        # Combine segments to create group segments
        group_seg1 = GroupValidator._combine_segments_list(group_segs1)
        group_seg2 = GroupValidator._combine_segments_list(group_segs2)

        if not group_seg1 or not group_seg2:
            return -1, -1, np.nan, "N/A"

        # Calculate length similarity (based on contour point counts)
        len1 = len(group_seg1.contour_points)
        len2 = len(group_seg2.contour_points)
        max_len = max(len1, len2)
        if max_len > 0:
            length_score = 1.0 - (abs(len1 - len2) / max_len)
        else:
            length_score = 0.0

        # Calculate shape similarity using RMSD
        shape_score = SegmentMatcher._calculate_shape_similarity_rmsd(group_seg1, group_seg2)

        # Calculate rotation angle that minimizes bounding box area
        try:
            target_points = 50
            pts1_resampled = SegmentMatcher._resample_points(group_seg1.contour_points, target_points)
            pts2_resampled = SegmentMatcher._resample_points(group_seg2.contour_points, target_points)

            # Normalize and center both contours
            pts1_norm = normalize_and_center(pts1_resampled)
            pts2_norm = normalize_and_center(pts2_resampled)

            # Find the rotation angle that minimizes bounding box area
            optimal_rotation_rad = find_min_area_rotation(pts1_norm, pts2_norm)

            # Convert from radians to degrees
            angle_score = np.degrees(optimal_rotation_rad)

            # Normalize to 0-180 range (180° and 360° are equivalent)
            angle_score = angle_score % 360
            if angle_score > 180:
                angle_score = 360 - angle_score
        except Exception:
            angle_score = np.nan

        # Determine quality based on shape RMSD
        if shape_score < 0:
            quality_text = "N/A"
        elif shape_score <= 0.05:
            quality_text = "EXCELLENT"
        elif shape_score <= 0.10:
            quality_text = "GOOD"
        elif shape_score <= 0.15:
            quality_text = "MODERATE"
        else:
            quality_text = "POOR"

        return length_score, shape_score, angle_score, quality_text

    @staticmethod
    def _is_good_quality(quality_text: str) -> bool:
        """Check if quality is good (not medium or bad)."""
        return quality_text in ["EXCELLENT", "GOOD"]

    @staticmethod
    def expand_group_iteratively(group_segs_p1: List[ContourSegment], group_segs_p2: List[ContourSegment],
                                all_segments_p1: List[ContourSegment], all_segments_p2: List[ContourSegment]) -> Tuple:
        """Expand groups by adding adjacent segments until quality degrades.

        Args:
            group_segs_p1: Initial group segments from piece 1
            group_segs_p2: Initial group segments from piece 2
            all_segments_p1: All segments from piece 1 (for wraparound)
            all_segments_p2: All segments from piece 2 (for wraparound)

        Returns:
            Tuple of (final_group_segs_p1, final_group_segs_p2, test_results)
            where test_results is list of (state, length_score, shape_score, angle_score, quality_text)
        """
        current_p1 = group_segs_p1.copy()
        current_p2 = group_segs_p2.copy()
        test_results = []

        # Initial score
        length, shape, angle, quality = GroupValidator._calculate_group_scores(current_p1, current_p2)
        test_results.append(("Initial", length, shape, angle, quality))

        if not GroupValidator._is_good_quality(quality):
            # Initial match is not good, return as is
            return current_p1, current_p2, test_results

        # Try expanding - attempt adding segments from both sides
        while True:
            expanded = False

            # Try adding segment to the right of piece 1
            p1_right_idx = (group_segs_p1[-1].segment_id + 1) % len(all_segments_p1)
            right_seg_p1 = all_segments_p1[p1_right_idx]
            if right_seg_p1 not in current_p1:
                test_p1 = current_p1 + [right_seg_p1]

                # Also try adding segment to right of piece 2
                p2_right_idx = (group_segs_p2[-1].segment_id + 1) % len(all_segments_p2)
                right_seg_p2 = all_segments_p2[p2_right_idx]
                if right_seg_p2 not in current_p2:
                    test_p2 = current_p2 + [right_seg_p2]
                    length, shape, angle, quality = GroupValidator._calculate_group_scores(test_p1, test_p2)
                    test_results.append(("Add Right-Right", length, shape, angle, quality))

                    if GroupValidator._is_good_quality(quality):
                        current_p1 = test_p1
                        current_p2 = test_p2
                        expanded = True
                    else:
                        # Try adding left on p2 instead
                        p2_left_idx = (group_segs_p2[0].segment_id - 1) % len(all_segments_p2)
                        left_seg_p2 = all_segments_p2[p2_left_idx]
                        if left_seg_p2 not in current_p2:
                            test_p2_left = [left_seg_p2] + current_p2
                            length, shape, angle, quality = GroupValidator._calculate_group_scores(test_p1, test_p2_left)
                            test_results.append(("Add Right-Left", length, shape, angle, quality))

                            if GroupValidator._is_good_quality(quality):
                                current_p1 = test_p1
                                current_p2 = test_p2_left
                                expanded = True

            # Try adding segment to the left of piece 1
            if not expanded:
                p1_left_idx = (group_segs_p1[0].segment_id - 1) % len(all_segments_p1)
                left_seg_p1 = all_segments_p1[p1_left_idx]
                if left_seg_p1 not in current_p1:
                    test_p1 = [left_seg_p1] + current_p1

                    # Try adding right on piece 2
                    p2_right_idx = (group_segs_p2[-1].segment_id + 1) % len(all_segments_p2)
                    right_seg_p2 = all_segments_p2[p2_right_idx]
                    if right_seg_p2 not in current_p2:
                        test_p2 = current_p2 + [right_seg_p2]
                        length, shape, angle, quality = GroupValidator._calculate_group_scores(test_p1, test_p2)
                        test_results.append(("Add Left-Right", length, shape, angle, quality))

                        if GroupValidator._is_good_quality(quality):
                            current_p1 = test_p1
                            current_p2 = test_p2
                            expanded = True
                        else:
                            # Try adding left on p2 instead
                            p2_left_idx = (group_segs_p2[0].segment_id - 1) % len(all_segments_p2)
                            left_seg_p2 = all_segments_p2[p2_left_idx]
                            if left_seg_p2 not in current_p2:
                                test_p2_left = [left_seg_p2] + current_p2
                                length, shape, angle, quality = GroupValidator._calculate_group_scores(test_p1, test_p2_left)
                                test_results.append(("Add Left-Left", length, shape, angle, quality))

                                if GroupValidator._is_good_quality(quality):
                                    current_p1 = test_p1
                                    current_p2 = test_p2_left
                                    expanded = True

            if not expanded:
                # No more good expansions possible
                break

        return current_p1, current_p2, test_results

    @staticmethod
    def test_group_pair(length_matrix: np.ndarray, shape_matrix: np.ndarray,
                        rotation_matrix: np.ndarray, group_indices_piece1: List[int],
                        group_indices_piece2: List[int], col_idx: int,
                        piece1_id: int, piece2_id: int, group_id: int,
                        segments1: List[ContourSegment], segments2: List[ContourSegment]) -> Tuple:
        """Test a pair of groups and expand them iteratively.

        Args:
            length_matrix: Length similarity matrix
            shape_matrix: Shape similarity matrix
            rotation_matrix: Rotation angle matrix
            group_indices_piece1: List of segment indices in the group from piece 1
            group_indices_piece2: List of segment indices in the group from piece 2
            col_idx: Column index for piece 2
            piece1_id: ID of piece 1
            piece2_id: ID of piece 2
            group_id: ID of the group being tested
            segments1: List of segments from piece 1
            segments2: List of segments from piece 2

        Returns:
            Tuple of (final_group_segs_p1, final_group_segs_p2, expansion_results)
        """
        if len(group_indices_piece1) < 2 or len(group_indices_piece2) < 2:
            return None

        # Get indices from middle of piece 1 group
        if len(group_indices_piece1) <= 3:
            p1_test_pairs = [(group_indices_piece1[0], group_indices_piece1[1])]
        else:
            middle_start = len(group_indices_piece1) // 3
            middle_end = 2 * len(group_indices_piece1) // 3
            middle_indices = group_indices_piece1[middle_start:middle_end]
            if len(middle_indices) < 2:
                p1_test_pairs = [(group_indices_piece1[0], group_indices_piece1[1])]
            else:
                start_pos = random.randint(0, len(middle_indices) - 2)
                p1_test_pairs = [(middle_indices[start_pos], middle_indices[start_pos + 1])]

        # Get indices from middle of piece 2 group
        if len(group_indices_piece2) <= 3:
            p2_test_pairs = [(group_indices_piece2[0], group_indices_piece2[1])]
        else:
            middle_start = len(group_indices_piece2) // 3
            middle_end = 2 * len(group_indices_piece2) // 3
            middle_indices = group_indices_piece2[middle_start:middle_end]
            if len(middle_indices) < 2:
                p2_test_pairs = [(group_indices_piece2[0], group_indices_piece2[1])]
            else:
                start_pos = random.randint(0, len(middle_indices) - 2)
                p2_test_pairs = [(middle_indices[start_pos], middle_indices[start_pos + 1])]

        # Test combinations and expand
        for (p1_seg_a, p1_seg_b) in p1_test_pairs:
            for (p2_seg_a, p2_seg_b) in p2_test_pairs:
                # Get the initial segments
                seg1a = next((s for s in segments1 if s.segment_id == p1_seg_a), None)
                seg1b = next((s for s in segments1 if s.segment_id == p1_seg_b), None)
                seg2a = next((s for s in segments2 if s.segment_id == p2_seg_a), None)
                seg2b = next((s for s in segments2 if s.segment_id == p2_seg_b), None)

                if not (seg1a and seg1b and seg2a and seg2b):
                    continue

                # Start with the initial pair
                initial_group_p1 = [seg1a, seg1b]
                initial_group_p2 = [seg2a, seg2b]

                # Expand iteratively
                expanded_p1, expanded_p2, expansion_results = GroupValidator.expand_group_iteratively(
                    initial_group_p1, initial_group_p2, segments1, segments2
                )

                # Return the expansion results
                return (expanded_p1, expanded_p2, expansion_results)

        return None

    @staticmethod
    def validate_all_groups(length_matrix: np.ndarray, shape_matrix: np.ndarray,
                           rotation_matrix: np.ndarray,
                           length_groups_dict: Dict[int, List[Tuple[int, int]]],
                           piece1_id: int, piece2_id: int,
                           segments1: List[ContourSegment], segments2: List[ContourSegment],
                           num_segments: int) -> Tuple[List[Tuple], Dict]:
        """Validate all groups by iteratively expanding them.

        Args:
            length_matrix: Length similarity matrix
            shape_matrix: Shape similarity matrix
            rotation_matrix: Rotation angle matrix
            length_groups_dict: Dictionary mapping column -> list of (start_idx, length) tuples for piece 1
            piece1_id: ID of piece 1
            piece2_id: ID of piece 2
            segments1: List of segments from piece 1
            segments2: List of segments from piece 2
            num_segments: Number of segments in piece 2

        Returns:
            Tuple of (group_tests, best_group_match)
            where best_group_match contains the group with highest quality and longest length
        """
        group_tests = []
        best_group_match = None
        best_score = -1
        best_length = 0

        for col_idx in sorted(length_groups_dict.keys()):
            groups_piece1 = length_groups_dict[col_idx]

            for group_id, (group_start_p1, group_len_p1) in enumerate(groups_piece1):
                # Expand group indices for piece 1 with wraparound
                group_indices_p1 = [(group_start_p1 + i) % num_segments for i in range(group_len_p1)]

                # For piece 2, the group should be around the col_idx position
                group_indices_p2 = [col_idx]

                # Try to extend the group in piece 2 to match piece 1's group length
                if len(segments2) > 0:
                    for i in range(1, max(2, group_len_p1)):
                        next_idx = (col_idx + i) % len(segments2)
                        if next_idx not in group_indices_p2:
                            group_indices_p2.append(next_idx)

                    # Also try extending backwards
                    for i in range(1, max(2, group_len_p1)):
                        prev_idx = (col_idx - i) % len(segments2)
                        if prev_idx not in group_indices_p2:
                            group_indices_p2.insert(0, prev_idx)

                test_result = GroupValidator.test_group_pair(
                    length_matrix, shape_matrix, rotation_matrix,
                    group_indices_p1, group_indices_p2, col_idx,
                    piece1_id, piece2_id, group_id,
                    segments1, segments2
                )

                if test_result:
                    group_tests.append(test_result)

                    # Track the best group match
                    expanded_p1, expanded_p2, expansion_results = test_result
                    if expansion_results:
                        final_state, final_length, final_shape, final_angle, final_quality = expansion_results[-1]
                        total_size = len(expanded_p1) + len(expanded_p2)

                        # Score combining length and quality (prefer excellent, then good, then by shape score)
                        quality_priority = {"EXCELLENT": 3, "GOOD": 2, "MODERATE": 1, "POOR": 0}.get(final_quality, 0)
                        # Lower shape score is better, so we invert it (higher is better)
                        shape_quality = 1.0 / (1.0 + final_shape)
                        combined_score = quality_priority + final_length * shape_quality

                        if combined_score > best_score or (combined_score == best_score and total_size > best_length):
                            best_score = combined_score
                            best_length = total_size
                            best_group_match = {
                                'seg_ids_p1': [s.segment_id for s in expanded_p1],
                                'seg_ids_p2': [s.segment_id for s in expanded_p2],
                                'total_size': total_size,
                                'length_score': final_length,
                                'shape_score': final_shape,
                                'angle': final_angle,
                                'quality': final_quality,
                                'expansion_steps': len(expansion_results)
                            }

        return group_tests, best_group_match
