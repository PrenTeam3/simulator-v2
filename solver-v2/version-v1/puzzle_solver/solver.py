"""Main puzzle solver function."""
import cv2
from pathlib import Path
from .data_loader import PuzzleDataLoader, print_analysis_summary
from .visualizer import SolverVisualizer
from .contour_segmenter import ContourSegmenter
from .segment_matcher import SegmentMatcher
from .match_visualizer import MatchVisualizer
from .segment_overlay_visualizer import SegmentOverlayVisualizer
from .image_viewer import InteractiveImageViewer
from .diagonal_extractor import DiagonalExtractor
from .group_validator import GroupValidator
import numpy as np


def solve_puzzle(temp_folder_name=None, piece_id_1=None, piece_id_2=None):
    """Load analysis data, segment contours, and visualize with segments.

    Args:
        temp_folder_name: Optional folder name for analysis data
        piece_id_1: First puzzle piece ID for comparison (optional, defaults to piece 0)
        piece_id_2: Second puzzle piece ID for comparison (optional, defaults to piece 1)
    """
    print("\n" + "="*70)
    print("PUZZLE SOLVER - Analysis Visualization with Segments")
    print("="*70 + "\n")

    project_root = Path(__file__).parent.parent
    temp_folder = (project_root / 'temp' / temp_folder_name) if temp_folder_name else (project_root / 'temp')

    if temp_folder_name:
        print(f"Using specified analysis folder: {temp_folder_name}")
    else:
        print("Using most recent analysis folder...")

    # Load analysis data
    try:
        print("Loading analysis data from temp folder...")
        analysis_data = PuzzleDataLoader.load_from_temp_folder(temp_folder)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print_analysis_summary(analysis_data)

    # Load and process image
    image_path = Path(analysis_data.image_path)
    if not image_path.exists():
        image_path = project_root / analysis_data.image_path

    if not image_path.exists():
        print(f"Warning: Could not load image from {image_path}")
        return

    original_image = cv2.imread(str(image_path))

    # Segment contours for each piece
    print("Segmenting contours based on corners...")
    all_segments = []
    for piece in analysis_data.pieces:
        segments = ContourSegmenter.segment_piece_contours(piece)
        all_segments.append(segments)
        print(f"  Piece {piece.piece_id}: {len(segments)} segments")

    # Visualize
    print("Visualizing puzzle pieces and segments...")
    annotated_image = SolverVisualizer.draw_puzzle_pieces(original_image, analysis_data.pieces, all_segments)
    annotated_image = SolverVisualizer.draw_legend(annotated_image)

    # Save and display
    output_path = project_root / 'temp' / 'solver_visualization_output.png'
    cv2.imwrite(str(output_path), annotated_image)
    print(f"Saved visualization to: {output_path}")

    print("Displaying annotated puzzle image with all features and segments...")
    viewer = InteractiveImageViewer("Puzzle Solver - Analysis Visualization with Segments")
    viewer.show(annotated_image)

    # Print detailed information
    print("\nDetailed piece information:")
    SolverVisualizer.print_all_pieces_summary(analysis_data.pieces)

    print("Detailed segment information:")
    SolverVisualizer.print_all_segments(all_segments)

    # Segment matching - test specified pieces or default to first two
    if len(analysis_data.pieces) >= 2:
        print("\n" + "="*70)
        print("SEGMENT MATCHING TEST")
        print("="*70)

        # Use provided piece IDs or default to first two pieces
        p1_id = piece_id_1 if piece_id_1 is not None else 0
        p2_id = piece_id_2 if piece_id_2 is not None else 1

        # Validate piece IDs
        if p1_id < 0 or p1_id >= len(analysis_data.pieces):
            print(f"Error: Piece ID {p1_id} is out of range (0-{len(analysis_data.pieces)-1})")
            return
        if p2_id < 0 or p2_id >= len(analysis_data.pieces):
            print(f"Error: Piece ID {p2_id} is out of range (0-{len(analysis_data.pieces)-1})")
            return
        if p1_id == p2_id:
            print("Error: Piece IDs must be different")
            return

        piece1 = analysis_data.pieces[p1_id]
        piece2 = analysis_data.pieces[p2_id]
        segments1 = all_segments[p1_id]
        segments2 = all_segments[p2_id]

        print(f"Using Piece {p1_id} and Piece {p2_id} for comparison")

        print(f"\nMatching segments between Piece {piece1.piece_id} and Piece {piece2.piece_id}...")
        print(f"Piece {piece1.piece_id} has {len(segments1)} segments")
        print(f"Piece {piece2.piece_id} has {len(segments2)} segments")

        # Generate match matrices
        length_matrix = SegmentMatcher.generate_match_matrix(piece1, segments1, piece2, segments2)
        MatchVisualizer.print_match_matrix(length_matrix, piece1.piece_id, piece2.piece_id)

        # Generate and print shape similarity matrix
        shape_matrix = SegmentMatcher.generate_shape_similarity_matrix(piece1, segments1, piece2, segments2)
        MatchVisualizer.print_shape_similarity_matrix(shape_matrix, piece1.piece_id, piece2.piece_id)

        # Generate and print rotation angle matrix
        rotation_matrix = SegmentMatcher.generate_rotation_angle_matrix(piece1, segments1, piece2, segments2)
        MatchVisualizer.print_rotation_angle_matrix(rotation_matrix, piece1.piece_id, piece2.piece_id)

        # Extract diagonal arrays and analyze cross-references across all three matrices
        length_groups_dict, shape_groups_dict, angle_groups_dict = DiagonalExtractor.extract_all_diagonals(
            length_matrix, shape_matrix, rotation_matrix, piece1.piece_id, piece2.piece_id
        )

        # Validate all identified groups by testing group-to-group matches
        group_tests, best_group_match = GroupValidator.validate_all_groups(length_matrix, shape_matrix, rotation_matrix,
                                                                             length_groups_dict, piece1.piece_id, piece2.piece_id,
                                                                             segments1, segments2, len(segments2))

        # Visualize group tests if any were performed
        if group_tests:
            print(f"Generating group validation visualization for {len(group_tests)} tests...")
            group_overlay_image = SegmentOverlayVisualizer.create_group_validation_visualization(
                piece1, segments1, piece2, segments2,
                length_matrix, shape_matrix, rotation_matrix,
                group_tests
            )

            group_overlay_output_path = project_root / 'temp' / 'group_validation_visualization.png'
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
        if matches:
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

            match_output_path = project_root / 'temp' / 'segment_match_visualization.png'
            cv2.imwrite(str(match_output_path), match_image)
            print(f"Saved match visualization to: {match_output_path}")

            print("Displaying segment match visualization...")
            match_viewer = InteractiveImageViewer("Segment Matching Visualization - Best Group Highlighted")
            match_viewer.show(match_image)

            # Show segment overlays for top matches
            print(f"\nGenerating segment overlay visualization for top matches...")
            overlay_image = SegmentOverlayVisualizer.create_overlay_visualization(
                piece1, segments1, piece2, segments2, matches, max_overlays=3
            )

            overlay_output_path = project_root / 'temp' / 'segment_overlay_visualization.png'
            cv2.imwrite(str(overlay_output_path), overlay_image)
            print(f"Saved overlay visualization to: {overlay_output_path}")

            print("Displaying segment overlay visualization...")
            overlay_viewer = InteractiveImageViewer("Segment Overlay Visualization")
            overlay_viewer.show(overlay_image)

            # Show segment overlays for specific requested pairs
            print(f"\nGenerating segment overlay visualization for specific requested pairs...")
            specific_overlay_image = SegmentOverlayVisualizer.create_overlay_visualization_for_specific_pairs(
                piece1, segments1, piece2, segments2, specific_match_requests
            )

            specific_overlay_output_path = project_root / 'temp' / 'segment_overlay_specific.png'
            cv2.imwrite(str(specific_overlay_output_path), specific_overlay_image)
            print(f"Saved specific overlay visualization to: {specific_overlay_output_path}")

            print("Displaying segment overlay visualization for specific pairs...")
            specific_overlay_viewer = InteractiveImageViewer("Specific Segment Matches")
            specific_overlay_viewer.show(specific_overlay_image)
        else:
            print("No matches found between these two pieces.")

        print("\n" + "="*70)
        print("Segment matching complete")
        print("="*70)

    print("\n" + "="*70)
    print("Puzzle solving complete")
    print("="*70 + "\n")

