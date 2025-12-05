"""Chain matching module for finding sequential segment matches."""

import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from .geometry_utils import GeometryUtils


@dataclass
class SegmentMatch:
    """Represents a single segment-to-segment match."""
    piece1_id: int
    piece2_id: int
    seg1_id: int
    seg2_id: int
    length_score: float
    direction_score: float
    shape_score: float
    is_valid: bool


@dataclass
class ChainMatch:
    """Represents a chain of sequential segment matches."""
    piece1_id: int
    piece2_id: int
    segment_ids_p1: List[int]  # Sequential segment IDs from piece 1
    segment_ids_p2: List[int]  # Sequential segment IDs from piece 2
    length_score: float
    direction_score: float
    shape_score: float
    is_valid: bool
    chain_length: int  # Number of segments in the chain
    additional_metrics: Optional[Dict] = None  # Additional shape metrics for evaluation
    # Chain endpoint markers (blue = frame connection, red = interior)
    blue_dot_p1: Optional[np.ndarray] = None  # Blue dot for piece 1 chain
    red_dot_p1: Optional[np.ndarray] = None   # Red dot for piece 1 chain
    blue_dot_p2: Optional[np.ndarray] = None  # Blue dot for piece 2 chain
    red_dot_p2: Optional[np.ndarray] = None   # Red dot for piece 2 chain


class ChainMatcher:
    """Finds and validates chains of sequential segment matches."""

    # Thresholds for valid matches
    LENGTH_THRESHOLD = 80.0
    DIRECTION_DEVIATION_THRESHOLD = 60.0  # degrees
    SHAPE_THRESHOLD = 80.0

    @staticmethod
    def _get_next_segment_id(current_seg_id: int, num_segments: int, direction: int) -> int:
        """Get the next segment ID in the given direction with wraparound.

        Args:
            current_seg_id: Current segment ID
            num_segments: Total number of segments
            direction: +1 for forward, -1 for backward

        Returns:
            Next segment ID
        """
        return (current_seg_id + direction) % num_segments

    @staticmethod
    def _calculate_combined_segment_scores(seg1_points_list: List[np.ndarray],
                                           seg2_points_list: List[np.ndarray],
                                           seg1_normals: List[np.ndarray],
                                           seg2_normals: List[np.ndarray]) -> Tuple[float, float, float]:
        """Calculate scores for combined segments.

        Args:
            seg1_points_list: List of point arrays for segments from piece 1
            seg2_points_list: List of point arrays for segments from piece 2 (already aligned)
            seg1_normals: List of normal vectors for segments from piece 1
            seg2_normals: List of normal vectors for segments from piece 2 (already rotated)

        Returns:
            Tuple of (length_score, direction_score, shape_score)
        """
        # Combine all points from the segments
        combined_seg1 = np.vstack(seg1_points_list)
        combined_seg2 = np.vstack(seg2_points_list)

        # 1. Calculate length score
        seg1_length = np.sum(np.linalg.norm(np.diff(combined_seg1, axis=0), axis=1))
        seg2_length = np.sum(np.linalg.norm(np.diff(combined_seg2, axis=0), axis=1))
        length_diff = abs(seg1_length - seg2_length)
        length_score = max(0, 100 - (length_diff / max(seg1_length, seg2_length) * 100))

        # 2. Calculate average direction score
        # Average the normal vectors
        avg_seg1_normal = np.mean(seg1_normals, axis=0)
        avg_seg2_normal = np.mean(seg2_normals, axis=0)

        # Normalize the averaged normals
        if np.linalg.norm(avg_seg1_normal) > 0:
            avg_seg1_normal = avg_seg1_normal / np.linalg.norm(avg_seg1_normal)
        if np.linalg.norm(avg_seg2_normal) > 0:
            avg_seg2_normal = avg_seg2_normal / np.linalg.norm(avg_seg2_normal)

        # Calculate angle between averaged normals
        dot_product = np.dot(avg_seg1_normal, avg_seg2_normal)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_between = np.degrees(np.arccos(dot_product))

        # Calculate deviation from 180 degrees (opposite directions)
        deviation_from_180 = abs(180 - angle_between)
        direction_score = max(0, 100 - deviation_from_180)

        # 3. Calculate shape similarity using bidirectional RMSD
        distance_matrix = cdist(combined_seg1.astype(np.float64), combined_seg2.astype(np.float64))

        # Bidirectional distances
        min_distances_1_to_2 = np.min(distance_matrix, axis=1)
        min_distances_2_to_1 = np.min(distance_matrix, axis=0)
        all_distances = np.concatenate([min_distances_1_to_2, min_distances_2_to_1])

        # Calculate RMSD
        rmsd = np.sqrt(np.mean(all_distances ** 2))

        # Normalize by average segment length
        avg_length = (seg1_length + seg2_length) / 2
        rmsd_percentage = (rmsd / avg_length) * 100 if avg_length > 0 else 100

        # Convert to shape similarity score
        shape_score = max(0, 100 - rmsd_percentage)

        return length_score, direction_score, shape_score

    @staticmethod
    def _score_chain(chain_segs1, chain_segs2, piece1, piece2, all_segments1, all_segments2):
        """Calculate scoring metrics for a chain.

        Args:
            chain_segs1: List of segments in chain for piece 1
            chain_segs2: List of segments in chain for piece 2
            piece1: Piece 1 object
            piece2: Piece 2 object
            all_segments1: All segments for piece 1
            all_segments2: All segments for piece 2

        Returns:
            Tuple of (length_score, direction_score, shape_score, is_valid)
        """
        from scipy.spatial.distance import cdist
        from math import atan2, cos, sin

        # Combine all points from the chain segments
        chain1_points = []
        for seg in chain_segs1:
            chain1_points.extend([[p.x, p.y] for p in seg.contour_points])

        chain2_points = []
        for seg in chain_segs2:
            chain2_points.extend([[p.x, p.y] for p in seg.contour_points])

        chain1_points = np.array(chain1_points, dtype=np.float64)
        chain2_points = np.array(chain2_points, dtype=np.float64)

        # Calculate arrows for chains using utility function
        mid1, normal1 = GeometryUtils.calculate_arrow_for_segment(chain1_points, piece1)
        mid2, normal2 = GeometryUtils.calculate_arrow_for_segment(chain2_points, piece2)

        # Need to align the chains before comparing shape similarity
        # Determine frame endpoints (EXACT same logic as visualizer)
        from .visualizer_helpers import VisualizerHelpers

        # Find frame-touching segments
        frame_touching_ids1 = VisualizerHelpers._find_frame_touching_segment_ids(piece1, all_segments1)
        frame_touching_ids2 = VisualizerHelpers._find_frame_touching_segment_ids(piece2, all_segments2)

        # Chain 1: Check which end has frame-touching neighbor outside the chain
        first_seg1 = chain_segs1[0]
        num_segments1 = len(all_segments1)
        first_id1 = first_seg1.segment_id
        first_prev_id1 = (first_id1 - 1) % num_segments1
        first_next_id1 = (first_id1 + 1) % num_segments1
        first_prev_is_frame1 = first_prev_id1 in frame_touching_ids1
        first_next_is_frame1 = first_next_id1 in frame_touching_ids1

        if len(chain_segs1) > 1:
            second_seg_id1 = chain_segs1[1].segment_id
            if first_prev_is_frame1 and second_seg_id1 != first_prev_id1:
                frame_at_start1 = True
            elif first_next_is_frame1 and second_seg_id1 != first_next_id1:
                frame_at_start1 = True
            else:
                frame_at_start1 = False
        else:
            frame_at_start1 = first_prev_is_frame1

        if frame_at_start1:
            B1 = chain1_points[0]
            R1 = chain1_points[-1]
        else:
            B1 = chain1_points[-1]
            R1 = chain1_points[0]

        # Chain 2: Check which end has frame-touching neighbor outside the chain
        first_seg2 = chain_segs2[0]
        num_segments2 = len(all_segments2)
        first_id2 = first_seg2.segment_id
        first_prev_id2 = (first_id2 - 1) % num_segments2
        first_next_id2 = (first_id2 + 1) % num_segments2
        first_prev_is_frame2 = first_prev_id2 in frame_touching_ids2
        first_next_is_frame2 = first_next_id2 in frame_touching_ids2

        if len(chain_segs2) > 1:
            second_seg_id2 = chain_segs2[1].segment_id
            if first_prev_is_frame2 and second_seg_id2 != first_prev_id2:
                frame_at_start2 = True
            elif first_next_is_frame2 and second_seg_id2 != first_next_id2:
                frame_at_start2 = True
            else:
                frame_at_start2 = False
        else:
            frame_at_start2 = first_prev_is_frame2

        if frame_at_start2:
            B2 = chain2_points[0]
            R2 = chain2_points[-1]
        else:
            B2 = chain2_points[-1]
            R2 = chain2_points[0]

        # Align chain2 to chain1 using GeometryUtils
        chain2_aligned, R2_aligned = GeometryUtils.align_chains(
            chain1_points, chain2_points, B1, B2, R1, R2
        )

        # Calculate rotation angle for normal vector transformation
        from math import atan2, cos, sin
        translation = B1 - B2
        R2_translated = R2 + translation
        v1 = R1 - B1
        v2 = R2_translated - B1
        angle1 = atan2(v1[1], v1[0])
        angle2 = atan2(v2[1], v2[0])
        rotation_angle = angle1 - angle2
        cos_r = cos(rotation_angle)
        sin_r = sin(rotation_angle)

        # 1. Length score
        chain1_length = np.sum(np.linalg.norm(np.diff(chain1_points, axis=0), axis=1))
        chain2_length = np.sum(np.linalg.norm(np.diff(chain2_points, axis=0), axis=1))
        length_diff = abs(chain1_length - chain2_length)
        length_score = max(0, 100 - (length_diff / max(chain1_length, chain2_length) * 100))

        # 2. Direction score (arrows should point in opposite directions)
        # Rotate normal2 to match chain2 rotation
        normal2_rotated = np.array([
            cos_r * normal2[0] - sin_r * normal2[1],
            sin_r * normal2[0] + cos_r * normal2[1]
        ])

        dot_product = np.dot(normal1, normal2_rotated)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_between = np.degrees(np.arccos(dot_product))
        deviation_from_180 = abs(180 - angle_between)
        direction_score = max(0, 100 - deviation_from_180)

        # 3. Calculate base shape metrics (used for additional metrics)
        distance_matrix = cdist(chain1_points, chain2_aligned)
        min_distances_1_to_2 = np.min(distance_matrix, axis=1)
        min_distances_2_to_1 = np.min(distance_matrix, axis=0)
        all_distances = np.concatenate([min_distances_1_to_2, min_distances_2_to_1])
        rmsd = np.sqrt(np.mean(all_distances ** 2))
        avg_length = (chain1_length + chain2_length) / 2
        rmsd_percentage = (rmsd / avg_length) * 100 if avg_length > 0 else 100
        rmsd_shape_score = max(0, 100 - rmsd_percentage)  # OLD metric, now additional

        # ========================================================================
        # ADDITIONAL SHAPE METRICS (for comparison - not used for validation)
        # ========================================================================

        # Metric 1: RMSD-based (OLD PRIMARY, now additional)
        rmsd_score = rmsd_shape_score

        # Metric 2: Absolute RMSD with Per-Segment Normalization
        num_segments = len(chain_segs1)
        rmsd_per_segment = rmsd / num_segments if num_segments > 0 else rmsd
        if rmsd_per_segment > 15.0:
            absolute_shape_score = 0.0
        else:
            absolute_shape_score = max(0, 100 - (rmsd_per_segment / 15.0) * 100)

        # Metric 3: Maximum Point Deviation (Hausdorff-inspired)
        max_dist_1_to_2 = np.max(min_distances_1_to_2)
        max_dist_2_to_1 = np.max(min_distances_2_to_1)
        max_deviation = max(max_dist_1_to_2, max_dist_2_to_1)

        if max_deviation <= 10.0:
            hausdorff_shape_score = 100 - (max_deviation / 10.0) * 20
        elif max_deviation <= 30.0:
            hausdorff_shape_score = 80 - ((max_deviation - 10.0) / 20.0) * 80
        else:
            hausdorff_shape_score = 0.0

        # Metric 4: Progressive Consistency Score
        # Note: This requires history tracking, so we'll calculate a placeholder here
        # The actual progressive consistency will be calculated in extend_match_progressively
        consistency_shape_score = rmsd_shape_score  # Placeholder - will be overridden later

        # ========================================================================
        # PRIMARY SHAPE SCORE: Enclosed Area Difference (NEW)
        # ========================================================================
        # Build closed polygon: chain1 → red connecting line → chain2 (reversed) → back to start
        import cv2

        # Create the polygon points (R2_aligned was already calculated by align_chains)
        polygon_points = []

        # Add chain1 points (from blue to red)
        if frame_at_start1:
            polygon_points.extend(chain1_points)  # Already goes blue → red
        else:
            polygon_points.extend(chain1_points[::-1])  # Reverse to go blue → red

        # Add red connecting line (from chain1's red to chain2's red)
        polygon_points.append(R2_aligned)

        # Add chain2 points in reverse (from red back to blue)
        if frame_at_start2:
            polygon_points.extend(chain2_aligned[::-1])  # Reverse to go red → blue
        else:
            polygon_points.extend(chain2_aligned)  # Already goes red → blue

        # Calculate area
        polygon_array = np.array(polygon_points, dtype=np.float32)
        area = abs(cv2.contourArea(polygon_array))

        # Normalize by chain length squared (area scales with length²)
        normalized_area = area / (avg_length ** 2) if avg_length > 0 else 1.0

        # Score based on normalized area (THIS IS NOW THE PRIMARY SHAPE SCORE)
        if normalized_area <= 0.05:
            area_shape_score = 100 - (normalized_area / 0.05) * 30  # 100% to 70%
        elif normalized_area <= 0.15:
            area_shape_score = 70 - ((normalized_area - 0.05) / 0.10) * 70  # 70% to 0%
        else:
            area_shape_score = 0.0

        # Set primary shape score to area-based score
        shape_score = area_shape_score

        # Package additional metrics (for comparison)
        additional_metrics = {
            'rmsd_score': rmsd_score,  # OLD primary, now additional
            'rmsd': rmsd,
            'rmsd_percentage': rmsd_percentage,
            'absolute_rmsd_score': absolute_shape_score,
            'rmsd_per_segment': rmsd_per_segment,
            'hausdorff_score': hausdorff_shape_score,
            'max_deviation': max_deviation,
            'consistency_score': consistency_shape_score,  # Placeholder
            'area_score': area_shape_score,  # Same as primary shape_score now
            'enclosed_area': area,
            'normalized_area': normalized_area
        }

        # Validate using chain thresholds (80% for length, 70% for shape)
        # NOTE: Direction is calculated and displayed but NOT used for validation
        # NOW using area-based shape_score for validation
        is_valid = (length_score >= 80.0) and (shape_score >= 70.0)

        # Calculate blue and red dots for both chains
        # Blue dot = frame end, Red dot = interior end
        # These are the ENDPOINTS used for alignment (B1, R1, B2, R2)
        blue_dot_p1 = B1.copy()
        red_dot_p1 = R1.copy()
        blue_dot_p2 = B2.copy()
        red_dot_p2 = R2.copy()

        return length_score, direction_score, shape_score, is_valid, additional_metrics, blue_dot_p1, red_dot_p1, blue_dot_p2, red_dot_p2

    @staticmethod
    def _find_frame_touching_segments(piece, all_segments):
        """Find segments that directly touch frame corners.

        Returns:
            set: Set of segment IDs that touch frame corners
        """
        frame_touching_ids = set()
        tolerance = 1.0

        for frame_corner in piece.frame_corners:
            frame_pos = (frame_corner.x, frame_corner.y)

            for seg in all_segments:
                seg_start = (seg.start_corner.x, seg.start_corner.y)
                seg_end = (seg.end_corner.x, seg.end_corner.y)

                if (abs(seg_start[0] - frame_pos[0]) < tolerance and
                    abs(seg_start[1] - frame_pos[1]) < tolerance):
                    frame_touching_ids.add(seg.segment_id)
                elif (abs(seg_end[0] - frame_pos[0]) < tolerance and
                      abs(seg_end[1] - frame_pos[1]) < tolerance):
                    frame_touching_ids.add(seg.segment_id)

        return frame_touching_ids

    @staticmethod
    def _determine_direction_away_from_frame(current_seg_id: int, num_segments: int,
                                             frame_touching_ids: set, piece_id: int) -> Optional[int]:
        """Determine which direction (+1 or -1) leads away from the frame corner.

        Args:
            current_seg_id: Current segment ID
            num_segments: Total number of segments
            frame_touching_ids: Set of segment IDs that touch frame corners
            piece_id: Piece ID for logging

        Returns:
            +1 for forward, -1 for backward, None if both directions hit frame
        """
        next_id_forward = (current_seg_id + 1) % num_segments
        next_id_backward = (current_seg_id - 1) % num_segments

        forward_hits_frame = next_id_forward in frame_touching_ids
        backward_hits_frame = next_id_backward in frame_touching_ids

        print(f"      [Direction] Current seg: P{piece_id}-S{current_seg_id}")
        print(f"      [Direction] Frame-touching segments for P{piece_id}: {sorted(frame_touching_ids)}")
        print(f"      [Direction] Forward (+1) to S{next_id_forward}: {'FRAME' if forward_hits_frame else 'interior'}")
        print(f"      [Direction] Backward (-1) to S{next_id_backward}: {'FRAME' if backward_hits_frame else 'interior'}")

        if not forward_hits_frame and backward_hits_frame:
            # Forward leads away, backward leads to frame
            print(f"      [Direction] Choosing FORWARD (+1, away from frame)")
            return +1
        elif forward_hits_frame and not backward_hits_frame:
            # Backward leads away, forward leads to frame
            print(f"      [Direction] Choosing BACKWARD (-1, away from frame)")
            return -1
        elif not forward_hits_frame and not backward_hits_frame:
            # Both directions are away from frame
            print(f"      [Direction] Both directions away from frame, choosing FORWARD (+1)")
            return +1
        else:
            # Both hit frame - between two frame corners
            print(f"      [Direction] ERROR: Both directions hit frame - cannot extend")
            return None

    @staticmethod
    def extend_match_progressively(initial_match: SegmentMatch,
                                   num_segments_p1: int,
                                   num_segments_p2: int,
                                   frame_touching_p1: set,
                                   frame_touching_p2: set,
                                   piece1,
                                   segments1,
                                   piece2,
                                   segments2) -> List[ChainMatch]:
        """Progressively extend a match until it becomes invalid.

        Returns all intermediate chain lengths as a list of ChainMatch objects.
        Stops extending when the match quality drops below thresholds.

        Args:
            initial_match: Starting match to extend
            num_segments_p1: Total number of segments in piece 1
            num_segments_p2: Total number of segments in piece 2
            frame_touching_p1: Set of segment IDs that touch frame for piece 1
            frame_touching_p2: Set of segment IDs that touch frame for piece 2
            piece1: Piece 1 object (for score calculation)
            segments1: All segments for piece 1
            piece2: Piece 2 object (for score calculation)
            segments2: All segments for piece 2

        Returns:
            List of ChainMatch objects for each progressive extension (length 1, 2, 3, ...)
        """
        chains = []
        rmsd_history = []  # Track RMSD for consistency scoring

        # Start with the initial match (chain of length 1)
        chain_seg_ids_p1 = [initial_match.seg1_id]
        chain_seg_ids_p2 = [initial_match.seg2_id]

        print(f"    Starting progressive extension:")
        print(f"      Initial match: P{initial_match.piece1_id}-S{initial_match.seg1_id} <-> P{initial_match.piece2_id}-S{initial_match.seg2_id}")
        print(f"      Initial scores: Length={initial_match.length_score:.1f}%, Direction={initial_match.direction_score:.1f}%, Shape={initial_match.shape_score:.1f}%, Valid={initial_match.is_valid}")

        # For initial match, we need to get the additional metrics and dots
        chain_segs1_initial = [s for s in segments1 if s.segment_id == initial_match.seg1_id]
        chain_segs2_initial = [s for s in segments2 if s.segment_id == initial_match.seg2_id]

        _, _, _, _, initial_additional_metrics, blue_p1, red_p1, blue_p2, red_p2 = ChainMatcher._score_chain(
            chain_segs1_initial, chain_segs2_initial, piece1, piece2, segments1, segments2
        )
        rmsd_history.append(initial_additional_metrics['rmsd'])

        # For chain length 1, consistency score is 100% (no history to compare)
        initial_additional_metrics['consistency_score'] = 100.0

        # Add the initial match as chain length 1
        chains.append(ChainMatch(
            piece1_id=initial_match.piece1_id,
            piece2_id=initial_match.piece2_id,
            segment_ids_p1=chain_seg_ids_p1.copy(),
            segment_ids_p2=chain_seg_ids_p2.copy(),
            length_score=initial_match.length_score,
            direction_score=initial_match.direction_score,
            shape_score=initial_match.shape_score,
            is_valid=initial_match.is_valid,
            chain_length=1,
            additional_metrics=initial_additional_metrics,
            blue_dot_p1=blue_p1,
            red_dot_p1=red_p1,
            blue_dot_p2=blue_p2,
            red_dot_p2=red_p2
        ))

        # Determine direction for extensions (only do once at the beginning)
        print(f"\n    Determining extension directions:")
        direction_p1 = ChainMatcher._determine_direction_away_from_frame(
            chain_seg_ids_p1[-1], num_segments_p1, frame_touching_p1, initial_match.piece1_id
        )
        direction_p2 = ChainMatcher._determine_direction_away_from_frame(
            chain_seg_ids_p2[-1], num_segments_p2, frame_touching_p2, initial_match.piece2_id
        )

        if direction_p1 is None or direction_p2 is None:
            print(f"      Cannot determine valid extension direction")
            print(f"      Returning only initial match (chain length 1)")
            return chains

        print(f"      P{initial_match.piece1_id} direction: {'+1' if direction_p1 == 1 else '-1'}")
        print(f"      P{initial_match.piece2_id} direction: {'+1' if direction_p2 == 1 else '-1'}")

        # Keep extending until we hit an invalid match or boundary
        max_extensions = min(num_segments_p1, num_segments_p2)  # Safety limit
        print(f"\n    Beginning extensions (max possible: {max_extensions}):")

        for extension_step in range(1, max_extensions + 1):
            print(f"\n    Extension step {extension_step} (attempting chain length {len(chain_seg_ids_p1) + 1}):")

            # Get next segments
            next_seg_id_p1 = ChainMatcher._get_next_segment_id(chain_seg_ids_p1[-1], num_segments_p1, direction_p1)
            next_seg_id_p2 = ChainMatcher._get_next_segment_id(chain_seg_ids_p2[-1], num_segments_p2, direction_p2)

            print(f"      Next segments: P{initial_match.piece1_id}-S{next_seg_id_p1}, P{initial_match.piece2_id}-S{next_seg_id_p2}")

            # Check if we've wrapped around
            if next_seg_id_p1 in chain_seg_ids_p1 or next_seg_id_p2 in chain_seg_ids_p2:
                print(f"      ERROR: Hit circular boundary (segment already in chain), stopping")
                break

            # Add the new segments temporarily
            chain_seg_ids_p1.append(next_seg_id_p1)
            chain_seg_ids_p2.append(next_seg_id_p2)

            print(f"      Extended chain: P{initial_match.piece1_id}{chain_seg_ids_p1} <-> P{initial_match.piece2_id}{chain_seg_ids_p2}")

            # Calculate scores for this extended chain
            chain_segs1 = [s for s in segments1 if s.segment_id in chain_seg_ids_p1]
            chain_segs2 = [s for s in segments2 if s.segment_id in chain_seg_ids_p2]

            print(f"      Calculating scores for chain length {len(chain_seg_ids_p1)}...")
            length_score, direction_score, shape_score, is_valid, additional_metrics, blue_p1, red_p1, blue_p2, red_p2 = ChainMatcher._score_chain(
                chain_segs1, chain_segs2, piece1, piece2, segments1, segments2
            )

            # Track RMSD for consistency scoring (still useful even though area is primary)
            current_rmsd = additional_metrics['rmsd']
            rmsd_history.append(current_rmsd)

            # Calculate progressive consistency score based on RMSD degradation
            if len(rmsd_history) > 1:
                # Calculate penalties for RMSD increases
                penalty = 0
                for i in range(len(rmsd_history) - 1):
                    rmsd_increase = rmsd_history[i + 1] - rmsd_history[i]
                    if rmsd_increase > 5.0:  # More than 5px jump
                        penalty += (rmsd_increase - 5.0) * 2  # 2% penalty per extra pixel

                # Base score from current area-based shape score
                consistency_score = max(0, shape_score - penalty)
                additional_metrics['consistency_score'] = consistency_score
            else:
                additional_metrics['consistency_score'] = 100.0

            print(f"      Scores: Length={length_score:.1f}%, Direction={direction_score:.1f}%, Shape={shape_score:.1f}%")
            print(f"      Valid: {is_valid}")

            # Add this chain to the list
            chains.append(ChainMatch(
                piece1_id=initial_match.piece1_id,
                piece2_id=initial_match.piece2_id,
                segment_ids_p1=chain_seg_ids_p1.copy(),
                segment_ids_p2=chain_seg_ids_p2.copy(),
                length_score=length_score,
                direction_score=direction_score,
                shape_score=shape_score,
                is_valid=is_valid,
                chain_length=len(chain_seg_ids_p1),
                additional_metrics=additional_metrics,
                blue_dot_p1=blue_p1,
                red_dot_p1=red_p1,
                blue_dot_p2=blue_p2,
                red_dot_p2=red_p2
            ))

            # Stop AFTER adding the first invalid match
            if not is_valid:
                print(f"      Match became INVALID - stopping (included this invalid match in results)")
                break
            else:
                print(f"      Match is VALID - continuing to next extension")

        print(f"\n    Completed: Generated {len(chains)} progressive chain(s) (lengths: {[c.chain_length for c in chains]})")
        return chains

    @staticmethod
    def extend_match_to_chain(initial_match: SegmentMatch,
                             all_segment_data: Dict,
                             num_segments_p1: int,
                             num_segments_p2: int,
                             frame_touching_p1: set,
                             frame_touching_p2: set,
                             max_chain_length: int = 2) -> Optional[ChainMatch]:
        """Try to extend a single match into a chain by adding adjacent segments.

        For now, this just logs which segments would be in the chain, without actually scoring them.
        Segments are extended moving AWAY from frame corners.

        Args:
            initial_match: Starting match to extend
            all_segment_data: Dictionary containing aligned segment data
            num_segments_p1: Total number of segments in piece 1
            num_segments_p2: Total number of segments in piece 2
            frame_touching_p1: Set of segment IDs that touch frame for piece 1
            frame_touching_p2: Set of segment IDs that touch frame for piece 2
            max_chain_length: Maximum chain length to try (default: 2)

        Returns:
            ChainMatch if extension successful, None otherwise
        """
        # Start with the initial match
        chain_seg_ids_p1 = [initial_match.seg1_id]
        chain_seg_ids_p2 = [initial_match.seg2_id]

        print(f"    Starting chain:")
        print(f"      Initial match: P{initial_match.piece1_id}-S{initial_match.seg1_id} <-> P{initial_match.piece2_id}-S{initial_match.seg2_id}")

        # Determine direction for first extension (only done once)
        direction_p1 = None
        direction_p2 = None

        # Try to extend the chain up to max_chain_length
        extension_step = 0
        while len(chain_seg_ids_p1) < max_chain_length:
            extension_step += 1
            print(f"\n    Extension step {extension_step}:")

            # Determine direction for piece 1 (only on first step)
            if direction_p1 is None:
                direction_p1 = ChainMatcher._determine_direction_away_from_frame(
                    chain_seg_ids_p1[-1], num_segments_p1, frame_touching_p1, initial_match.piece1_id
                )
                if direction_p1 is None:
                    print(f"      Cannot determine direction for P{initial_match.piece1_id}, stopping")
                    break
            else:
                print(f"      [Direction] P{initial_match.piece1_id}: Maintaining direction {'+1' if direction_p1 == 1 else '-1'}")

            # Determine direction for piece 2 (only on first step)
            if direction_p2 is None:
                direction_p2 = ChainMatcher._determine_direction_away_from_frame(
                    chain_seg_ids_p2[-1], num_segments_p2, frame_touching_p2, initial_match.piece2_id
                )
                if direction_p2 is None:
                    print(f"      Cannot determine direction for P{initial_match.piece2_id}, stopping")
                    break
            else:
                print(f"      [Direction] P{initial_match.piece2_id}: Maintaining direction {'+1' if direction_p2 == 1 else '-1'}")

            # Get next segments using determined directions
            next_seg_id_p1 = ChainMatcher._get_next_segment_id(chain_seg_ids_p1[-1], num_segments_p1, direction_p1)
            next_seg_id_p2 = ChainMatcher._get_next_segment_id(chain_seg_ids_p2[-1], num_segments_p2, direction_p2)

            print(f"      Next segments:")
            print(f"        P{initial_match.piece1_id}: S{chain_seg_ids_p1[-1]} -> S{next_seg_id_p1} (direction: {'+1' if direction_p1 == 1 else '-1'})")
            print(f"        P{initial_match.piece2_id}: S{chain_seg_ids_p2[-1]} -> S{next_seg_id_p2} (direction: {'+1' if direction_p2 == 1 else '-1'})")

            # Add the new segments to the chain
            chain_seg_ids_p1.append(next_seg_id_p1)
            chain_seg_ids_p2.append(next_seg_id_p2)

            print(f"      Added to chain: P{initial_match.piece1_id}-S{next_seg_id_p1} <-> P{initial_match.piece2_id}-S{next_seg_id_p2}")

        print(f"\n    Final chain: P{initial_match.piece1_id} segments {chain_seg_ids_p1} <-> P{initial_match.piece2_id} segments {chain_seg_ids_p2}")

        # Create a chain with the proposed segment IDs
        # For now, use the initial match scores (we're not actually scoring the chain yet)
        return ChainMatch(
            piece1_id=initial_match.piece1_id,
            piece2_id=initial_match.piece2_id,
            segment_ids_p1=chain_seg_ids_p1,
            segment_ids_p2=chain_seg_ids_p2,
            length_score=initial_match.length_score,  # Placeholder
            direction_score=initial_match.direction_score,  # Placeholder
            shape_score=initial_match.shape_score,  # Placeholder
            is_valid=initial_match.is_valid,  # Placeholder
            chain_length=len(chain_seg_ids_p1)
        )

    @staticmethod
    def find_chains_from_matches(individual_matches: List[SegmentMatch],
                                 all_segment_data: Dict,
                                 num_segments_dict: Dict[int, int],
                                 frame_touching_dict: Dict[int, set],
                                 piece_frame_segments) -> List[List[ChainMatch]]:
        """Find chains by progressively extending individual matches.

        Args:
            individual_matches: List of individual segment matches
            all_segment_data: Dictionary containing aligned segment data for scoring (not currently used)
            num_segments_dict: Dictionary mapping piece_id to number of segments
            frame_touching_dict: Dictionary mapping piece_id to frame-touching segment IDs
            piece_frame_segments: List of (piece, segments, frame_segs) tuples

        Returns:
            List of progressive chain lists. Each element is a list of ChainMatch objects
            representing progressive extensions (length 1, 2, 3, ...) until invalid.
        """
        all_progressive_chains = []

        # Create piece/segment lookup dictionaries
        piece_dict = {}
        segments_dict = {}
        for piece, segments, _ in piece_frame_segments:
            piece_dict[piece.piece_id] = piece
            segments_dict[piece.piece_id] = segments

        print("\n" + "="*70)
        print("CHAIN MATCHING DEBUG - Progressive Extension Approach")
        print("="*70)
        print(f"Total individual matches: {len(individual_matches)}")

        valid_matches = [m for m in individual_matches if m.is_valid]
        print(f"Valid individual matches: {len(valid_matches)}")

        for m in valid_matches:
            print(f"  - P{m.piece1_id}-S{m.seg1_id} <-> P{m.piece2_id}-S{m.seg2_id}")

        # Try to progressively extend each valid match
        for i, initial_match in enumerate(valid_matches):
            print(f"\nAttempting progressive extension for match {i+1}/{len(valid_matches)}:")
            print(f"  Initial: P{initial_match.piece1_id}-S{initial_match.seg1_id} <-> " +
                  f"P{initial_match.piece2_id}-S{initial_match.seg2_id}")

            num_segs_p1 = num_segments_dict.get(initial_match.piece1_id)
            num_segs_p2 = num_segments_dict.get(initial_match.piece2_id)
            frame_touching_p1 = frame_touching_dict.get(initial_match.piece1_id, set())
            frame_touching_p2 = frame_touching_dict.get(initial_match.piece2_id, set())
            piece1 = piece_dict.get(initial_match.piece1_id)
            piece2 = piece_dict.get(initial_match.piece2_id)
            segments1 = segments_dict.get(initial_match.piece1_id)
            segments2 = segments_dict.get(initial_match.piece2_id)

            if num_segs_p1 is None or num_segs_p2 is None or piece1 is None or piece2 is None:
                print(f"  ERROR: Cannot find data for pieces")
                continue

            print(f"  Piece {initial_match.piece1_id} has {num_segs_p1} segments")
            print(f"  Piece {initial_match.piece2_id} has {num_segs_p2} segments")

            progressive_chains = ChainMatcher.extend_match_progressively(
                initial_match, num_segs_p1, num_segs_p2,
                frame_touching_p1, frame_touching_p2,
                piece1, segments1, piece2, segments2
            )

            if progressive_chains:
                print(f"  [OK] Created {len(progressive_chains)} progressive chain(s)")
                for chain in progressive_chains:
                    print(f"    Length {chain.chain_length}: P{chain.piece1_id}{chain.segment_ids_p1} <-> " +
                          f"P{chain.piece2_id}{chain.segment_ids_p2}")
                all_progressive_chains.append(progressive_chains)
            else:
                print(f"  [X] Could not extend")

        print(f"\n" + "="*70)
        print(f"CHAIN MATCHING SUMMARY: Found {len(all_progressive_chains)} progressive chain set(s)")
        print("="*70)

        return all_progressive_chains

    @staticmethod
    def print_chains(chains: List[ChainMatch], max_to_print: int = 10):
        """Print chain matches in a readable format.

        Args:
            chains: List of ChainMatch objects
            max_to_print: Maximum number of chains to print
        """
        if not chains:
            print("No chains found.")
            return

        print(f"\nFound {len(chains)} chain(s):")
        print("-" * 100)

        valid_chains = [c for c in chains if c.is_valid]
        invalid_chains = [c for c in chains if not c.is_valid]

        if valid_chains:
            print(f"\nVALID CHAINS ({len(valid_chains)}):")
            for i, chain in enumerate(valid_chains[:max_to_print]):
                print(f"\n  Chain {i+1}:")
                print(f"    P{chain.piece1_id} segments {chain.segment_ids_p1} <-> "
                      f"P{chain.piece2_id} segments {chain.segment_ids_p2}")
                print(f"    Length: {chain.length_score:.1f}% | "
                      f"Direction: {chain.direction_score:.1f}% | "
                      f"Shape: {chain.shape_score:.1f}%")
                print(f"    Chain length: {chain.chain_length} segments")

        if invalid_chains and max_to_print > len(valid_chains):
            remaining = max_to_print - len(valid_chains)
            print(f"\nINVALID CHAINS ({len(invalid_chains)}) - showing first {remaining}:")
            for i, chain in enumerate(invalid_chains[:remaining]):
                print(f"\n  Chain {i+1}:")
                print(f"    P{chain.piece1_id} segments {chain.segment_ids_p1} <-> "
                      f"P{chain.piece2_id} segments {chain.segment_ids_p2}")
                print(f"    Length: {chain.length_score:.1f}% | "
                      f"Direction: {chain.direction_score:.1f}% | "
                      f"Shape: {chain.shape_score:.1f}%")
                print(f"    Chain length: {chain.chain_length} segments")

        print("-" * 100)
