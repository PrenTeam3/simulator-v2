import cv2
import numpy as np
from puzzle_analyzer.puzzle_piece import PuzzlePiece
from puzzle_analyzer.results_saver import save_analysis_results
from puzzle_analyzer.svg_visualizer import SVGVisualizer
from puzzle_solver.image_viewer import InteractiveImageViewer


def analyze_puzzle_pieces(image_path: str, debug_visualization: bool = False, verbose_logging: bool = False, save_results: bool = True):
    """
    Main function to detect, segment, and analyze puzzle pieces.
    Automatically detects light/dark background.

    Args:
        image_path: Path to the puzzle image
        debug_visualization: If True, shows forbidden zones, arrows, and other debug info
        verbose_logging: If True, shows detailed logging from frame corner detection
        save_results: If True, saves analysis results to a temporary folder
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    print(f"Debug visualization: {'ENABLED' if debug_visualization else 'DISABLED'}")
    if debug_visualization:
        print("  (Showing forbidden zones, bisector arrows, and other debug info)")
    else:
        print("  (Clean output - showing only analysis results)")

    output_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhanced preprocessing
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Automatic background detection
    h, w = gray.shape
    border_width = min(30, min(h, w) // 10)
    border_pixels = np.concatenate([
        gray[0:border_width, :].flatten(),
        gray[-border_width:, :].flatten(),
        gray[:, 0:border_width].flatten(),
        gray[:, -border_width:].flatten()
    ])
    avg_border_brightness = np.median(border_pixels)

    # Apply thresholding
    if avg_border_brightness > 127:
        _, thresh = cv2.threshold(blur, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        print(f"Light background detected (brightness: {avg_border_brightness:.1f})")
    else:
        _, thresh = cv2.threshold(blur, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Dark background detected (brightness: {avg_border_brightness:.1f})")

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    viewer = InteractiveImageViewer("Segmentation Mask")
    # Convert grayscale to BGR for viewer compatibility
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    viewer.show(thresh_bgr)

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    print(f"\nTotal {len(contours)} contours found.")

    # Analyze each piece
    puzzle_pieces = []

    for i, contour in enumerate(contours):
        piece = PuzzlePiece(contour, min_edge_length=30, verbose_logging=verbose_logging)

        # Filter out noise
        if piece.area < 500:
            continue

        puzzle_pieces.append(piece)
        piece.print_info(i)

        # Re-run frame corner detection with visualization enabled (if debug mode)
        if debug_visualization:
            piece.reanalyze_with_visualization(output_image)

        piece.draw_analysis(output_image, i)

    print(f"\n\n{'#' * 60}")
    print(f"FINAL SUMMARY: {len(puzzle_pieces)} valid puzzle pieces analyzed.")
    total_border_edges = sum(len(p.border_edges) for p in puzzle_pieces)
    total_corners = sum(len(p.frame_corners) for p in puzzle_pieces)
    print(f"Total border edges found: {total_border_edges}")
    print(f"Total frame corners found: {total_corners}")
    print(f"{'#' * 60}\n")

    # Save results to temporary folder if requested
    if save_results:
        temp_dir_info = save_analysis_results(output_image, puzzle_pieces, image_path, debug_visualization)

        # Generate SVG vector graphics in the same temp directory
        import os
        import glob as glob_module
        temp_dirs = sorted(glob_module.glob('temp/puzzle_analysis_*'), key=os.path.getctime, reverse=True)
        if temp_dirs:
            latest_temp = temp_dirs[0]
            svg_path = f"{latest_temp}/contours_vector.svg"
            try:
                SVGVisualizer.create_interactive_svg(contours, image.shape, puzzle_pieces, svg_path)
            except Exception as e:
                print(f"[WARNING] SVG generation failed: {e}")

    # Display legend
    legend_y = 30
    cv2.putText(output_image, "Legend:", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(output_image, "Orange: Straight segments", (10, legend_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    cv2.putText(output_image, "Blue: Border edges", (10, legend_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2)
    cv2.putText(output_image, "Yellow: Frame corners", (10, legend_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(output_image, "Red X: Inner corners", (10, legend_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(output_image, "Pink: Outer corners", (10, legend_y + 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Show debug legend items only when debug visualization is enabled
    if debug_visualization:
        cv2.putText(output_image, "Orange: Forbidden zones", (10, legend_y + 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        cv2.putText(output_image, "Blue arrow: Bisector", (10, legend_y + 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
        cv2.putText(output_image, "Yellow arrow: To center", (10, legend_y + 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Display analysis results with zoomable viewer
    viewer = InteractiveImageViewer("Analysis Results")
    viewer.show(output_image)

    # Return pieces for further processing
    return puzzle_pieces


def analyze_puzzle_pieces_get_pieces(image_path: str, debug_visualization: bool = False, verbose_logging: bool = False, save_results: bool = True):
    """
    Main function to detect, segment, and analyze puzzle pieces.
    Returns the list of analyzed pieces.

    Args:
        image_path: Path to the puzzle image
        debug_visualization: If True, shows forbidden zones, arrows, and other debug info
        verbose_logging: If True, shows detailed logging from frame corner detection
        save_results: If True, saves analysis results to a temporary folder

    Returns:
        List of PuzzlePiece objects, or None if analysis fails
    """
    return analyze_puzzle_pieces(image_path, debug_visualization, verbose_logging, save_results)
