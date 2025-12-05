"""Matrix-based solving module for puzzle solver - handles matching using matrix analysis."""
import cv2
from pathlib import Path
from .segment_matcher import SegmentMatcher
from .match_visualizer import MatchVisualizer
from .segment_overlay_visualizer import SegmentOverlayVisualizer
from ..common import InteractiveImageViewer
from ..common.common import SolverUtils
from .diagonal_extractor import DiagonalExtractor
from .group_validator import GroupValidator


class MatrixSolver:
    """Handles the matrix-based solving phase: matching, diagonal analysis, group validation, and visualization."""

    @staticmethod
    def solve_with_matrices(prepared_data, piece_id_1=None, piece_id_2=None, show_visualizations=True):
        """Solve puzzle using matrix-based matching approach.

        Args:
            prepared_data: Dictionary returned from PuzzlePreparer.prepare_puzzle_data()
            piece_id_1: First puzzle piece ID for comparison (optional, defaults to piece 0)
            piece_id_2: Second puzzle piece ID for comparison (optional, defaults to piece 1)
            show_visualizations: Whether to display visualization windows (default: True)

        Returns:
            dict: Solving results containing:
                - matches: List of segment matches
                - length_matrix: Length similarity matrix
                - shape_matrix: Shape similarity matrix
                - rotation_matrix: Rotation angle matrix
                - best_group_match: Best matching group found
                - group_tests: All group validation tests performed
        """
        SolverUtils.print_section_header("MATRIX-BASED SOLVING PHASE")

        # Extract prepared data
        analysis_data = prepared_data['analysis_data']
        original_image = prepared_data['original_image']
        all_segments = prepared_data['all_segments']
        project_root = prepared_data['project_root']

        # Validate we have at least 2 pieces
        if len(analysis_data.pieces) < 2:
            print("Error: Need at least 2 pieces for matching")
            return None

        # Use provided piece IDs or default to first two pieces
        p1_id = piece_id_1 if piece_id_1 is not None else 0
        p2_id = piece_id_2 if piece_id_2 is not None else 1

        # Validate piece IDs
        if p1_id < 0 or p1_id >= len(analysis_data.pieces):
            print(f"Error: Piece ID {p1_id} is out of range (0-{len(analysis_data.pieces)-1})")
            return None
        if p2_id < 0 or p2_id >= len(analysis_data.pieces):
            print(f"Error: Piece ID {p2_id} is out of range (0-{len(analysis_data.pieces)-1})")
            return None
        if p1_id == p2_id:
            print("Error: Piece IDs must be different")
            return None

        piece1 = analysis_data.pieces[p1_id]
        piece2 = analysis_data.pieces[p2_id]
        segments1 = all_segments[p1_id]
        segments2 = all_segments[p2_id]

        print(f"Using Piece {p1_id} and Piece {p2_id} for comparison")
        print(f"Piece {piece1.piece_id} has {len(segments1)} segments")
        print(f"Piece {piece2.piece_id} has {len(segments2)} segments")

        # Step 5: Generate match matrices
        print("\nStep 5: Generating match matrices...")
        length_matrix = SegmentMatcher.generate_match_matrix(piece1, segments1, piece2, segments2)
        MatchVisualizer.print_match_matrix(length_matrix, piece1.piece_id, piece2.piece_id)

        shape_matrix = SegmentMatcher.generate_shape_similarity_matrix(piece1, segments1, piece2, segments2)
        MatchVisualizer.print_shape_similarity_matrix(shape_matrix, piece1.piece_id, piece2.piece_id)

        rotation_matrix = SegmentMatcher.generate_rotation_angle_matrix(piece1, segments1, piece2, segments2)
        MatchVisualizer.print_rotation_angle_matrix(rotation_matrix, piece1.piece_id, piece2.piece_id)

        # Step 6: Extract diagonal arrays and analyze cross-references
        print("\nStep 6: Extracting diagonal groups from matrices...")
        length_groups_dict, shape_groups_dict, angle_groups_dict = DiagonalExtractor.extract_all_diagonals(
            length_matrix, shape_matrix, rotation_matrix, piece1.piece_id, piece2.piece_id
        )

        # Step 7: Validate all identified groups
        print("\nStep 7: Validating identified groups...")
        group_tests, best_group_match = GroupValidator.validate_all_groups(
            length_matrix, shape_matrix, rotation_matrix,
            length_groups_dict, piece1.piece_id, piece2.piece_id,
            segments1, segments2, len(segments2)
        )

        # Visualize group tests if any were performed
        if group_tests and show_visualizations:
            print(f"Generating group validation visualization for {len(group_tests)} tests...")
            group_overlay_image = SegmentOverlayVisualizer.create_group_validation_visualization(
                piece1, segments1, piece2, segments2,
                length_matrix, shape_matrix, rotation_matrix,
                group_tests
            )

            group_overlay_output_path = prepared_data['temp_folder'] / 'group_validation_visualization.png'
            cv2.imwrite(str(group_overlay_output_path), group_overlay_image)
            print(f"Saved group validation visualization to: {group_overlay_output_path}")

            print("Displaying group validation visualization...")
            group_overlay_viewer = InteractiveImageViewer("Group Validation Visualization")
            group_overlay_viewer.show(group_overlay_image)

        # Store best group match for later use in visualization
        best_group_segments = None
        if best_group_match:
            best_group_segments = {
                'seg_ids_p1': best_group_match['seg_ids_p1'],
                'seg_ids_p2': best_group_match['seg_ids_p2']
            }
            print(f"\nBest Group Match Found:")
            print(f"  P{piece1.piece_id} Segments: {best_group_match['seg_ids_p1']}")
            print(f"  P{piece2.piece_id} Segments: {best_group_match['seg_ids_p2']}")
            print(f"  Quality: {best_group_match['quality']} | Length: {best_group_match['length_score']:.4f} | Shape: {best_group_match['shape_score']:.4f}\n")

        # Get matches
        matches = SegmentMatcher.match_segments(piece1, segments1, piece2, segments2)

        # Step 8: Match visualization
        if matches:
            print("\nStep 8: Creating match visualizations...")
            MatchVisualizer.print_matches(matches, max_to_print=10)

            # Print specific matches requested for debugging
            print("\n" + "="*70)
            print("SPECIFIC SEGMENT MATCH REQUESTS:")
            print("="*70)
            specific_match_requests = [
                (6, 3),
                (7, 2),
                (8, 1),
                (9, 0),
            ]

            for s1_id, s2_id in specific_match_requests:
                matching_pair = next((m for m in matches
                                     if m.piece1_id == piece1.piece_id and m.seg1_id == s1_id
                                     and m.piece2_id == piece2.piece_id and m.seg2_id == s2_id), None)
                if matching_pair:
                    print(f"P{piece1.piece_id}-S{s1_id} <-> P{piece2.piece_id}-S{s2_id}: {matching_pair.description}")
                else:
                    print(f"P{piece1.piece_id}-S{s1_id} <-> P{piece2.piece_id}-S{s2_id}: NO MATCH FOUND (will visualize)")

            # Visualize matches with best group highlighted
            print(f"\nGenerating segment match visualization...")
            match_image = MatchVisualizer.draw_two_pieces_side_by_side(
                original_image, piece1, segments1, piece2, segments2, matches=None, max_matches_to_draw=0,
                best_group_segments=best_group_segments
            )

            match_output_path = prepared_data['temp_folder'] / 'segment_match_visualization.png'
            cv2.imwrite(str(match_output_path), match_image)
            print(f"Saved match visualization to: {match_output_path}")

            if show_visualizations:
                print("Displaying segment match visualization...")
                match_viewer = InteractiveImageViewer("Segment Matching Visualization - Best Group Highlighted")
                match_viewer.show(match_image)

            # Show segment overlays for top matches
            print(f"\nGenerating segment overlay visualization for top matches...")
            overlay_image = SegmentOverlayVisualizer.create_overlay_visualization(
                piece1, segments1, piece2, segments2, matches, max_overlays=3
            )

            overlay_output_path = prepared_data['temp_folder'] / 'segment_overlay_visualization.png'
            cv2.imwrite(str(overlay_output_path), overlay_image)
            print(f"Saved overlay visualization to: {overlay_output_path}")

            if show_visualizations:
                print("Displaying segment overlay visualization...")
                overlay_viewer = InteractiveImageViewer("Segment Overlay Visualization")
                overlay_viewer.show(overlay_image)

            # Show segment overlays for specific requested pairs
            print(f"\nGenerating segment overlay visualization for specific requested pairs...")
            specific_overlay_image = SegmentOverlayVisualizer.create_overlay_visualization_for_specific_pairs(
                piece1, segments1, piece2, segments2, specific_match_requests
            )

            specific_overlay_output_path = prepared_data['temp_folder'] / 'segment_overlay_specific.png'
            cv2.imwrite(str(specific_overlay_output_path), specific_overlay_image)
            print(f"Saved specific overlay visualization to: {specific_overlay_output_path}")

            if show_visualizations:
                print("Displaying segment overlay visualization for specific pairs...")
                specific_overlay_viewer = InteractiveImageViewer("Specific Segment Matches")
                specific_overlay_viewer.show(specific_overlay_image)
        else:
            print("No matches found between these two pieces.")

        SolverUtils.print_section_footer("Matrix-Based Solving Complete")

        # Return results
        return {
            'matches': matches,
            'length_matrix': length_matrix,
            'shape_matrix': shape_matrix,
            'rotation_matrix': rotation_matrix,
            'best_group_match': best_group_match,
            'group_tests': group_tests,
            'piece1': piece1,
            'piece2': piece2,
            'segments1': segments1,
            'segments2': segments2
        }
