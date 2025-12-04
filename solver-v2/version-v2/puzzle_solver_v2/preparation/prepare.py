"""Preparation module for puzzle solver - handles data loading and initial visualization."""
import cv2
from pathlib import Path
from .data_loader import PuzzleDataLoader, print_analysis_summary
from .visualizer import SolverVisualizer
from .contour_segmenter import ContourSegmenter
from ..common import InteractiveImageViewer
from ..common.common import SolverUtils
import numpy as np


class PuzzlePreparer:
    """Handles the preparation phase of puzzle solving: loading, segmentation, and initial visualization."""

    @staticmethod
    def prepare_puzzle_data(temp_folder_name=None, show_visualization=True):
        """Load analysis data, segment contours, and create initial visualization.

        Args:
            temp_folder_name: Optional folder name for analysis data
            show_visualization: Whether to display the initial visualization (default: True)

        Returns:
            dict: Prepared puzzle data containing:
                - analysis_data: Loaded analysis data
                - original_image: The image to work with
                - all_segments: List of segments for each piece
                - annotated_image: Initial visualization with all pieces and segments
                - temp_folder: Path to the temp folder
                - project_root: Path to the project root
        """
        SolverUtils.print_section_header("PUZZLE SOLVER - Preparation Phase")

        project_root = Path(__file__).parent.parent.parent
        temp_folder = (project_root / 'temp' / temp_folder_name) if temp_folder_name else (project_root / 'temp')

        if temp_folder_name:
            print(f"Using specified analysis folder: {temp_folder_name}")
        else:
            print("Using most recent analysis folder...")

        # Step 1: Load analysis data
        print("Step 1: Loading analysis data from temp folder...")
        try:
            analysis_data = PuzzleDataLoader.load_from_temp_folder(temp_folder)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None

        print_analysis_summary(analysis_data)

        # Step 2: Load SVG and convert to image with orange fill
        print("\nStep 2: Loading SVG visualization...")
        svg_path = temp_folder / 'pieces_with_corners_without_helpers.svg' if temp_folder_name else None
        if svg_path is None:
            # Find the latest analysis folder
            analysis_folders = sorted((project_root / 'temp').glob('analysis_*'))
            if analysis_folders:
                svg_path = analysis_folders[-1] / 'pieces_with_corners_without_helpers.svg'

        if svg_path and svg_path.exists():
            print(f"Loading SVG visualization from: {svg_path.name}")
            # Convert SVG to image with orange fill
            from .svg_to_image_converter import SVGToImageConverter
            original_image = SVGToImageConverter.convert_svg_to_image(svg_path, fill_color=(255, 165, 0))
        else:
            print(f"Warning: Could not find SVG file, falling back to original image")
            image_path = Path(analysis_data.image_path)
            if not image_path.exists():
                image_path = project_root / analysis_data.image_path
            if not image_path.exists():
                print(f"Warning: Could not load image from {image_path}")
                return None
            original_image = cv2.imread(str(image_path))

        # Step 3: Get pre-computed segments for each piece from SVG analysis
        print("\nStep 3: Loading pre-computed segments from SVG analysis...")
        all_segments = []
        for piece in analysis_data.pieces:
            segments = ContourSegmenter.segment_piece_contours(piece)
            all_segments.append(segments)
            print(f"  Piece {piece.piece_id}: {len(segments)} segments")

        # Step 4: Visualize
        print("\nStep 4: Creating initial visualization...")
        annotated_image = SolverVisualizer.draw_puzzle_pieces(original_image, analysis_data.pieces, all_segments)
        annotated_image = SolverVisualizer.draw_legend(annotated_image)

        # Save and optionally display (save to analysis temp folder)
        output_path = temp_folder / 'solver_visualization_output.png'
        cv2.imwrite(str(output_path), annotated_image)
        print(f"Saved visualization to: {output_path}")

        if show_visualization:
            print("Displaying annotated puzzle image with all features and segments...")
            viewer = InteractiveImageViewer("Puzzle Solver - Analysis Visualization with Segments")
            viewer.show(annotated_image)

        # Print detailed information
        print("\nDetailed piece information:")
        SolverVisualizer.print_all_pieces_summary(analysis_data.pieces)

        print("\nDetailed segment information:")
        SolverVisualizer.print_all_segments(all_segments)

        SolverUtils.print_section_footer("Preparation Phase Complete")

        # Return prepared data
        return {
            'analysis_data': analysis_data,
            'original_image': original_image,
            'all_segments': all_segments,
            'annotated_image': annotated_image,
            'temp_folder': temp_folder,
            'project_root': project_root
        }
