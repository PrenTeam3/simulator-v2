"""Diagonal extraction helper with group tracking."""
from typing import List, Tuple, Dict
import numpy as np
from .match_visualizer import MatchVisualizer


class DiagonalExtractor:
    """Helper class to extract diagonal groups and track them across matrices."""

    @staticmethod
    def extract_all_diagonals(length_matrix: np.ndarray, shape_matrix: np.ndarray,
                              rotation_matrix: np.ndarray, piece1_id: int, piece2_id: int) -> Tuple[dict, dict, dict]:
        """Extract diagonal arrays and groups from length and shape matrices only.

        Returns:
            Tuple of (length_groups_dict, shape_groups_dict, angle_groups_dict)
        """
        n, m = length_matrix.shape

        length_groups_dict = {}
        shape_groups_dict = {}
        angle_groups_dict = {}

        print(f"\n{'='*70}")
        print(f"DIAGONAL EXTRACTION AND GROUP ANALYSIS")
        print(f"{'='*70}\n")

        # Extract from length and shape matrices for each column
        for start_col in range(m):
            # Length similarity
            length_values = [length_matrix[(0 - i) % n, (start_col + i) % m] for i in range(n)]
            length_groups = MatchVisualizer._find_similarity_score_groups(length_values, threshold=0.7)
            length_groups_dict[start_col] = length_groups

            # Shape similarity
            shape_values = [shape_matrix[(0 - i) % n, (start_col + i) % m] for i in range(n)]
            shape_groups = MatchVisualizer._find_rmsd_similarity_groups(shape_values, tolerance=0.05)
            shape_groups_dict[start_col] = shape_groups

        # Print the diagonal arrays
        MatchVisualizer.extract_diagonal_arrays_from_length_similarity(length_matrix, piece1_id, piece2_id)
        MatchVisualizer.extract_diagonal_arrays_from_shape_similarity(shape_matrix, piece1_id, piece2_id)
        MatchVisualizer.extract_diagonal_arrays_from_rotation_angles(rotation_matrix, piece1_id, piece2_id)

        # Analyze cross-references (LENGTH vs SHAPE only, angle excluded)
        MatchVisualizer.analyze_cross_diagonal_groups_length_shape(length_groups_dict, shape_groups_dict,
                                                                   piece1_id, piece2_id, num_segments=n)

        return length_groups_dict, shape_groups_dict, angle_groups_dict
