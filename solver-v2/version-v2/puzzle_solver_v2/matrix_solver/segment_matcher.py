"""Match segments from two puzzle pieces with advanced shape matching."""
from typing import List, Tuple, Optional
import numpy as np
import cv2
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar
from ..preparation.data_loader import AnalyzedPuzzlePiece
from ..common.data_classes import Point, ContourSegment, SegmentMatch
from ..common.utils import normalize_vector, normalize_and_center, rotate_points, find_min_area_rotation, calculate_centroid


class SegmentMatcher:
    """Matches segments from two puzzle pieces with advanced shape analysis."""

    # Configuration parameters
    ROTATION_SEARCH_RANGE = np.radians(130)  # ±45 degrees search range
    ROTATION_STEPS = 15  # Number of rotation steps to try
    ARROW_TOLERANCE = np.radians(130)  # 60 degrees tolerance for arrow pointing
    RESAMPLE_POINTS = 50  # More points for better shape comparison

    @staticmethod
    def _get_segment_direction(segment: ContourSegment) -> Tuple[float, float]:
        """Get the direction vector of a segment (from start to end corner)."""
        dx = segment.end_corner.x - segment.start_corner.x
        dy = segment.end_corner.y - segment.start_corner.y
        return normalize_vector(dx, dy)

    @staticmethod
    def _get_segment_midpoint(segment: ContourSegment) -> Point:
        """Get the midpoint of a segment."""
        mid_x = sum(p.x for p in segment.contour_points) / len(segment.contour_points)
        mid_y = sum(p.y for p in segment.contour_points) / len(segment.contour_points)
        return Point(mid_x, mid_y)

    @staticmethod
    def _get_segment_length(segment: ContourSegment) -> float:
        """Calculate the arc length of a segment."""
        total_dist = 0.0
        for i in range(len(segment.contour_points) - 1):
            p1 = segment.contour_points[i]
            p2 = segment.contour_points[i + 1]
            dist = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            total_dist += dist
        return total_dist

    @staticmethod
    def _length_similarity(seg1: ContourSegment, seg2: ContourSegment) -> float:
        """Calculate length similarity based on percentage difference (0 to 1, where 1 is identical).

        Uses percentage difference between the two segment lengths relative to their average.
        This is a percentage-based metric independent of absolute lengths.
        """
        len1 = SegmentMatcher._get_segment_length(seg1)
        len2 = SegmentMatcher._get_segment_length(seg2)

        if len1 == 0 or len2 == 0:
            return 0.0

        # Calculate percentage difference relative to average length
        average_len = (len1 + len2) / 2
        difference = abs(len1 - len2)
        percentage_diff = difference / average_len

        # Convert percentage difference to similarity score (0 to 1)
        # 0% difference = 1.0 score, 100% difference = 0.0 score
        similarity = max(0.0, 1.0 - percentage_diff)

        return similarity

    @staticmethod
    def _resample_points(points: List[Point], num_points: int) -> List[Point]:
        """Resample points uniformly along the arc length."""
        if len(points) < 2:
            return points

        # Calculate cumulative arc length
        distances = [0.0]
        for i in range(len(points) - 1):
            dist = np.sqrt((points[i+1].x - points[i].x)**2 +
                          (points[i+1].y - points[i].y)**2)
            distances.append(distances[-1] + dist)

        total_length = distances[-1]
        if total_length == 0:
            return [points[0]] * num_points

        # Create interpolation functions
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]

        fx = interp1d(distances, x_coords, kind='linear', fill_value='extrapolate')
        fy = interp1d(distances, y_coords, kind='linear', fill_value='extrapolate')

        # Resample at uniform intervals
        new_distances = np.linspace(0, total_length, num_points)
        resampled = []
        for d in new_distances:
            resampled.append(Point(float(fx(d)), float(fy(d))))

        return resampled

    @staticmethod
    def match_segments(piece1: AnalyzedPuzzlePiece, segments1: List[ContourSegment],
                      piece2: AnalyzedPuzzlePiece, segments2: List[ContourSegment]) -> List[SegmentMatch]:
        """
        Match segments from two pieces based on length similarity only.
        """
        matches = []

        for seg1 in segments1:
            for seg2 in segments2:
                # Calculate length similarity only
                length_score = SegmentMatcher._length_similarity(seg1, seg2)

                # Keep all matches (no minimum threshold for now)
                if length_score > 0.0:
                    matches.append(SegmentMatch(
                        piece1_id=piece1.piece_id,
                        piece2_id=piece2.piece_id,
                        seg1_id=seg1.segment_id,
                        seg2_id=seg2.segment_id,
                        match_score=length_score,
                        optimal_rotation=0.0,
                        shape_score=0.0,
                        length_score=length_score,
                        description=f"P{piece1.piece_id}-S{seg1.segment_id} <-> "
                                  f"P{piece2.piece_id}-S{seg2.segment_id} "
                                  f"(length_score: {length_score:.3f})"
                    ))

        # Sort by match score (best first)
        return sorted(matches, key=lambda m: m.match_score, reverse=True)

    @staticmethod
    def generate_match_matrix(piece1: AnalyzedPuzzlePiece, segments1: List[ContourSegment],
                             piece2: AnalyzedPuzzlePiece, segments2: List[ContourSegment]) -> np.ndarray:
        """
        Generate an n x m matrix of length similarity scores.
        Rows = segments from piece1, Columns = segments from piece2.
        """
        matrix = np.zeros((len(segments1), len(segments2)))

        for i, seg1 in enumerate(segments1):
            for j, seg2 in enumerate(segments2):
                # Calculate length similarity only
                length_score = SegmentMatcher._length_similarity(seg1, seg2)
                matrix[i, j] = length_score

        return matrix

    @staticmethod
    def generate_shape_similarity_matrix(piece1: AnalyzedPuzzlePiece, segments1: List[ContourSegment],
                                         piece2: AnalyzedPuzzlePiece, segments2: List[ContourSegment]) -> np.ndarray:
        """
        Generate an n x m matrix of shape similarity scores using RMSD.
        Only calculates similarity for segment pairs where length matching is above 0.75.
        Pairs below the threshold are marked as -1 (displayed as "n/a").
        Lower values indicate better shape match.
        Rows = segments from piece1, Columns = segments from piece2.
        """
        matrix = np.full((len(segments1), len(segments2)), -1.0)  # Initialize with -1 (not calculated marker)

        for i, seg1 in enumerate(segments1):
            for j, seg2 in enumerate(segments2):
                # First check length similarity
                length_score = SegmentMatcher._length_similarity(seg1, seg2)

                # Only calculate shape similarity if length match is above threshold (0.75)
                if length_score > 0.75:
                    # Calculate shape similarity (lower = better)
                    rmsd = SegmentMatcher._calculate_shape_similarity_rmsd(seg1, seg2)
                    matrix[i, j] = rmsd
                # else: leave as -1 (not calculated)

        return matrix


    @staticmethod
    def generate_rotation_angle_matrix(piece1: AnalyzedPuzzlePiece, segments1: List[ContourSegment],
                                      piece2: AnalyzedPuzzlePiece, segments2: List[ContourSegment]) -> np.ndarray:
        """
        Generate an n x m matrix of optimal rotation angles (in degrees, normalized to 0-180).
        Only calculates rotation angles for segment pairs where length matching is above 0.75.
        Pairs below the threshold are marked as np.nan (displayed as "n/a").
        Angles are normalized to 0-180 range so that 180° and 360° are treated as equivalent.
        Rows = segments from piece1, Columns = segments from piece2.
        """
        matrix = np.full((len(segments1), len(segments2)), np.nan)

        for i, seg1 in enumerate(segments1):
            for j, seg2 in enumerate(segments2):
                # First check length similarity
                length_score = SegmentMatcher._length_similarity(seg1, seg2)

                # Only calculate rotation angle if length match is above threshold (0.75)
                if length_score > 0.75:
                    # Resample both segments to same number of points
                    target_points = 50
                    pts1_resampled = SegmentMatcher._resample_points(seg1.contour_points, target_points)
                    pts2_resampled = SegmentMatcher._resample_points(seg2.contour_points, target_points)

                    # Normalize and center both contours
                    pts1_norm = normalize_and_center(pts1_resampled)
                    pts2_norm = normalize_and_center(pts2_resampled)

                    # Find the rotation angle that minimizes bounding box area
                    optimal_rotation_rad = find_min_area_rotation(pts1_norm, pts2_norm)

                    # Convert from radians to degrees
                    optimal_rotation_deg = np.degrees(optimal_rotation_rad)

                    # Normalize to 0-180 range (180° and 360° are equivalent)
                    optimal_rotation_deg = optimal_rotation_deg % 360
                    if optimal_rotation_deg > 180:
                        optimal_rotation_deg = 360 - optimal_rotation_deg

                    matrix[i, j] = optimal_rotation_deg
                # else: leave as nan (not calculated)

        return matrix

    @staticmethod
    def get_top_matches(matches: List[SegmentMatch], top_k: int = 5) -> List[SegmentMatch]:
        """Get the top K matches, ensuring no segment is used twice."""
        used_segments = set()
        top_matches = []

        for match in matches:
            seg1_key = (match.piece1_id, match.seg1_id)
            seg2_key = (match.piece2_id, match.seg2_id)

            if seg1_key not in used_segments and seg2_key not in used_segments:
                top_matches.append(match)
                used_segments.add(seg1_key)
                used_segments.add(seg2_key)

                if len(top_matches) >= top_k:
                    break

        return top_matches

    @staticmethod
    def _calculate_shape_similarity_rmsd(seg1: ContourSegment, seg2: ContourSegment) -> float:
        """
        Calculate shape similarity using bidirectional Root Mean Square Distance (RMSD).
        Calculates minimum distances from seg1 to seg2 AND from seg2 to seg1.
        Lower values indicate better shape match (0 = perfect overlap).

        Returns:
            RMSD value normalized by segment length (0 to ~1 range, where 0 is perfect)
        """
        # Resample both segments to same number of points
        target_points = 50
        pts1_resampled = SegmentMatcher._resample_points(seg1.contour_points, target_points)
        pts2_resampled = SegmentMatcher._resample_points(seg2.contour_points, target_points)

        # Normalize and center both contours
        pts1_norm = normalize_and_center(pts1_resampled)
        pts2_norm = normalize_and_center(pts2_resampled)

        # Find the rotation angle that minimizes bounding box area
        optimal_rotation = find_min_area_rotation(pts1_norm, pts2_norm)

        # Apply the optimal rotation to seg2
        pts2_rotated = rotate_points(pts2_norm, optimal_rotation)

        # Convert to numpy arrays for efficient distance calculation
        pts1_array = np.array([[p.x, p.y] for p in pts1_norm])
        pts2_array = np.array([[p.x, p.y] for p in pts2_rotated])

        # Calculate pairwise distances between all points
        distance_matrix = cdist(pts1_array, pts2_array)

        # Bidirectional distance calculation:
        # 1. For each point on seg1, get the minimum distance to seg2
        min_distances_1_to_2 = np.min(distance_matrix, axis=1)

        # 2. For each point on seg2, get the minimum distance to seg1
        min_distances_2_to_1 = np.min(distance_matrix, axis=0)

        # Combine both sets of distances
        all_distances = np.concatenate([min_distances_1_to_2, min_distances_2_to_1])

        # Calculate combined RMSD
        rmsd = np.sqrt(np.mean(all_distances ** 2))

        return rmsd

    @staticmethod
    def create_temp_match_for_visualization(piece1_id: int, seg1_id: int,
                                           piece2_id: int, seg2_id: int,
                                           seg1: 'ContourSegment', seg2: 'ContourSegment') -> SegmentMatch:
        """Create a temporary SegmentMatch for visualization purposes.

        Calculates scores for the specific pair to display in visualization.
        """
        # Calculate individual scores - length similarity only
        length_score = SegmentMatcher._length_similarity(seg1, seg2)

        # Other scores set to 0 since we're only matching on length
        shape_score = 0.0
        direction_score = 0.0
        optimal_rotation = 0.0
        match_score = length_score

        description = (f"P{piece1_id}-S{seg1_id} <-> P{piece2_id}-S{seg2_id} "
                      f"(length_score: {length_score:.3f})")

        return SegmentMatch(
            piece1_id=piece1_id,
            piece2_id=piece2_id,
            seg1_id=seg1_id,
            seg2_id=seg2_id,
            match_score=match_score,
            optimal_rotation=optimal_rotation,
            shape_score=shape_score,
            length_score=length_score,
            description=description
        )