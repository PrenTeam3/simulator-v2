"""Core puzzle piece detection and analysis."""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from puzzle_analyzer_v2.svg_visualizer import SVGVisualizer
from puzzle_analyzer_v2.svg_smoother import SVGSmoother
from puzzle_analyzer_v2.svg_corner_drawer import SVGCornerDrawer


class PuzzleAnalyzer:
    """Detects and analyzes puzzle pieces in an image."""

    def __init__(self, image_path: str, temp_dir: str = None, strictness: str = 'ultra_strict_minus', debug: bool = False):
        """
        Initialize analyzer with image path.

        Args:
            image_path: Path to puzzle image
            temp_dir: Temporary directory for results (auto-generated if None)
            strictness: Strictness level for straight edge detection
            debug: Enable debug logging for corner detection
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image at {image_path}")

        self.height, self.width = self.image.shape[:2]
        self.output_image = self.image.copy()
        self.contours = []
        self.pieces = []
        self.smoothed_svg_path = None  # Path to smoothed SVG
        self.corners_list = []  # Store corner detection results from smoothed SVG
        self.strictness = strictness  # Store strictness level
        self.debug = debug  # Store debug flag

        # Setup temp directory
        if temp_dir is None:
            temp_dir = self._create_temp_directory()
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

    def _create_temp_directory(self) -> str:
        """
        Create a timestamped temporary directory for results.

        Returns:
            Path to created temp directory
        """
        temp_base = Path("temp")
        temp_base.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = str(temp_base / f"analysis_{timestamp}")

        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def detect_pieces(self, min_area: int = 500) -> list:
        """
        Detect puzzle pieces in the image.

        Args:
            min_area: Minimum contour area to consider as valid piece

        Returns:
            List of detected contours
        """
        # Preprocessing
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Automatic background detection
        border_brightness = self._detect_background_brightness(gray)

        # Apply thresholding based on background
        if border_brightness > 127:
            _, thresh = cv2.threshold(blur, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            print(f"Light background detected (brightness: {border_brightness:.1f})")
        else:
            _, thresh = cv2.threshold(blur, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print(f"Dark background detected (brightness: {border_brightness:.1f})")

        # Morphological operations to clean up noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area
        self.contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        print(f"Found {len(contours)} total contours, {len(self.contours)} valid pieces")

        return self.contours

    def draw_pieces_with_green_marks(self) -> np.ndarray:
        """
        Draw detected pieces on output image with green marks at centroids.

        Returns:
            Image with drawn pieces and centroid marks
        """
        for idx, contour in enumerate(self.contours):
            # Draw contour outline in blue
            cv2.drawContours(self.output_image, [contour], 0, (255, 0, 0), 2)

            # Draw filled contour with transparency
            overlay = self.output_image.copy()
            cv2.drawContours(overlay, [contour], 0, (200, 255, 200), -1)
            cv2.addWeighted(overlay, 0.2, self.output_image, 0.8, 0, self.output_image)

            # Calculate and mark centroid with green circle
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])

                # Draw green circle at centroid
                cv2.circle(self.output_image, (cx, cy), 8, (0, 255, 0), -1)
                cv2.circle(self.output_image, (cx, cy), 10, (0, 255, 0), 2)

                # Add piece label
                cv2.putText(self.output_image, f"{idx}", (cx - 10, cy + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return self.output_image

    def save_output_image(self, filename: str = "output.png"):
        """
        Save the output image with green marks.

        Args:
            filename: Filename to save (saved in temp directory)
        """
        output_path = os.path.join(self.temp_dir, filename)
        cv2.imwrite(output_path, self.output_image)
        print(f"Output image saved to: {output_path}")
        return output_path

    def generate_smoothed_svg(self, filename: str = "pieces_smoothed.svg"):
        """
        Generate smoothed SVG visualization of detected pieces.
        This becomes the primary SVG for all further processing.

        Args:
            filename: Filename to save (saved in temp directory)
        """
        # Create temporary unsmoothed SVG
        temp_svg = os.path.join(self.temp_dir, "_temp_pieces.svg")
        SVGVisualizer.create_svg_from_contours(
            self.contours,
            self.image.shape,
            filename=temp_svg
        )

        # Smooth it
        output_path = os.path.join(self.temp_dir, filename)
        SVGSmoother.smooth_svg_file(temp_svg, output_path)

        # Remove temporary unsmoothed SVG
        os.remove(temp_svg)

        # Store path for later use
        self.smoothed_svg_path = output_path

        return output_path

    def generate_svg_with_corners(self, filename: str = "pieces_with_corners.svg"):
        """
        Add corner markings to the smoothed SVG.
        Uses the smoothed SVG as input (most accurate representation).

        Args:
            filename: Filename to save (saved in temp directory)
        """
        if self.smoothed_svg_path is None:
            print("Error: Smoothed SVG not generated yet")
            return None

        output_path = os.path.join(self.temp_dir, filename)

        # Detect corners from smoothed SVG and draw them
        corners_list = SVGCornerDrawer.add_corners_to_smoothed_svg(
            self.smoothed_svg_path,
            output_path,
            strictness=self.strictness,
            debug=self.debug
        )

        # Store corners info for later use
        self.corners_list = corners_list

        return output_path

    def save_analysis_data(self, filename: str = "analysis_data.json"):
        """
        Save analysis data as JSON including corner information.

        Args:
            filename: Filename to save (saved in temp directory)
        """
        output_path = os.path.join(self.temp_dir, filename)
        pieces_info = self.get_piece_info()

        data = {
            'timestamp': datetime.now().isoformat(),
            'image_path': self.image_path,
            'image_width': self.width,
            'image_height': self.height,
            'num_pieces': len(pieces_info),
            'pieces': []
        }

        for idx, info in enumerate(pieces_info):
            piece_data = {
                'id': info['id'],
                'area': float(info['area']),
                'perimeter': float(info['perimeter']),
                'centroid': {
                    'x': float(info['centroid'][0]),
                    'y': float(info['centroid'][1])
                }
            }

            # Add corner information if available
            if self.corners_list and idx < len(self.corners_list) and self.corners_list[idx]:
                corners_info = self.corners_list[idx]
                piece_data['corners'] = {
                    'total': corners_info.get('total', 0),
                    'outer_count': len(corners_info.get('outer_corners', [])),
                    'inner_count': len(corners_info.get('inner_corners', []))
                }

            data['pieces'].append(piece_data)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Analysis data saved to: {output_path}")
        return output_path

    def _detect_background_brightness(self, gray: np.ndarray) -> float:
        """
        Detect background brightness by sampling border pixels.

        Args:
            gray: Grayscale image

        Returns:
            Median brightness of border pixels
        """
        h, w = gray.shape
        border_width = min(30, min(h, w) // 10)
        border_pixels = np.concatenate([
            gray[0:border_width, :].flatten(),
            gray[-border_width:, :].flatten(),
            gray[:, 0:border_width].flatten(),
            gray[:, -border_width:].flatten()
        ])
        return np.median(border_pixels)

    def get_piece_info(self) -> list:
        """
        Get information about detected pieces.

        Returns:
            List of dicts with piece info
        """
        pieces_info = []
        for idx, contour in enumerate(self.contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            moments = cv2.moments(contour)

            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
            else:
                cx, cy = 0, 0

            pieces_info.append({
                'id': idx,
                'area': area,
                'perimeter': perimeter,
                'centroid': (cx, cy),
                'contour': contour
            })

        return pieces_info


def analyze_puzzle(image_path: str, temp_dir: str = None, strictness: str = 'ultra_strict_minus', debug: bool = False, target_frame_corners: int = 4):
    """
    Main entry point: detect puzzle pieces and generate visualizations.
    Automatically retries with different strictness levels until target frame corners are found.

    Args:
        image_path: Path to puzzle image
        temp_dir: Temporary directory for results (auto-generated if None)
        strictness: Initial strictness level for straight edge detection
                   (default: 'ultra_strict_minus' - optimal for puzzle pieces)
                   Options: 'ultra_loose', 'loose', 'balanced', 'strict', 'strict_plus',
                            'strict_ultra', 'ultra_strict_minus', 'ultra_strict'
        debug: Enable debug logging for corner detection
        target_frame_corners: Target number of frame corners to find (default: 4)

    Returns:
        PuzzleAnalyzer instance
    """
    print(f"Analyzing puzzle image: {image_path}")
    print(f"Target frame corners: {target_frame_corners}")
    if debug:
        print("Debug logging: ENABLED")

    analyzer = PuzzleAnalyzer(image_path, temp_dir=temp_dir, strictness=strictness, debug=debug)
    print(f"Results will be saved to: {analyzer.temp_dir}\n")

    # Detect pieces
    analyzer.detect_pieces(min_area=500)

    # Draw with green marks
    analyzer.draw_pieces_with_green_marks()
    analyzer.save_output_image("output.png")

    # Generate SVG files (in order: smooth first, then add corners)
    print("\nGenerating and processing SVG files...")
    analyzer.generate_smoothed_svg("pieces_smoothed.svg")
    print("  [OK] Smoothed SVG generated")

    # Define strictness progression (from strictest to loosest)
    # STRICT = fewer straight segments detected = fewer frame corners
    # LOOSE = more straight segments detected = more frame corners
    strictness_levels = [
        'ultra_strict',
        'ultra_strict_minus',
        'strict_ultra',
        'strict_plus',
        'strict',
        'balanced',
        'loose',
        'ultra_loose'
    ]

    # Start with the strictest level
    start_idx = 0

    # Get number of pieces
    num_pieces = len(analyzer.contours)

    # Track which pieces have found their frame corner
    pieces_with_frame_corner = set()
    piece_frame_corner_status = {}  # piece_id -> (strictness_level, frame_corner_count)

    print(f"\n{'=' * 70}")
    print(f"ADAPTIVE STRICTNESS: Searching for {target_frame_corners} frame corners...")
    print(f"Target: 1 frame corner per piece ({num_pieces} pieces total)")
    print(f"Starting from STRICTEST (fewest straight segments) and loosening...")
    print(f"{'=' * 70}\n")

    # Start strict and gradually loosen until we find enough frame corners
    for attempt, level_idx in enumerate(range(start_idx, len(strictness_levels))):
        current_strictness = strictness_levels[level_idx]
        print(f"[Attempt {attempt + 1}] Trying strictness level: {current_strictness}")

        # Update analyzer strictness
        analyzer.strictness = current_strictness

        # Generate corners with current strictness
        analyzer.generate_svg_with_corners("pieces_with_corners.svg")

        # Count frame corners per piece
        total_frame_corners = 0
        newly_found_pieces = []
        current_piece_status = {}  # Current frame corner count for each piece

        if analyzer.corners_list:
            for piece_idx, corner_info in enumerate(analyzer.corners_list):
                frame_count = 0
                if corner_info:
                    all_frame_corners = corner_info.get('frame_corners', [])
                    # Only count frame corners that:
                    # 1. Are not potential (passed all 4 criteria)
                    # 2. Have clear forbidden zones (no puzzle piece continuation)
                    confirmed_frame_corners = [
                        fc for fc in all_frame_corners
                        if not fc.get('potential', False) and fc.get('forbidden_zone_clear', False)
                    ]
                    frame_count = len(confirmed_frame_corners)
                    total_frame_corners += frame_count

                current_piece_status[piece_idx] = frame_count

                # Check if this piece just found its first frame corner
                if frame_count > 0 and piece_idx not in pieces_with_frame_corner:
                    pieces_with_frame_corner.add(piece_idx)
                    piece_frame_corner_status[piece_idx] = (current_strictness, frame_count)
                    newly_found_pieces.append(piece_idx)
                # Update existing piece if it now has more frame corners
                elif frame_count > 0 and piece_idx in pieces_with_frame_corner:
                    old_strictness, old_count = piece_frame_corner_status[piece_idx]
                    if frame_count != old_count:
                        piece_frame_corner_status[piece_idx] = (current_strictness, frame_count)

        print(f"  Found {total_frame_corners} frame corners across all pieces")
        print(f"  Pieces with frame corners: {len(pieces_with_frame_corner)}/{num_pieces} {sorted(list(pieces_with_frame_corner))}")

        # Show current status for all pieces (iterate through all piece indices)
        for piece_idx in range(num_pieces):
            count = current_piece_status.get(piece_idx, 0)
            if piece_idx in newly_found_pieces:
                print(f"    [NEW] Piece {piece_idx}: Found {count} frame corner(s)")
            elif count > 0:
                print(f"          Piece {piece_idx}: {count} frame corner(s)")
            else:
                print(f"          Piece {piece_idx}: 0 frame corners")

        # Check if we found at least one frame corner per piece
        if len(pieces_with_frame_corner) == num_pieces:
            # Check if total matches target
            if total_frame_corners == target_frame_corners:
                print(f"\n{'=' * 70}")
                print(f"SUCCESS! Found exactly {target_frame_corners} frame corners")
                print(f"All {num_pieces} pieces have at least one frame corner!")
                print(f"{'=' * 70}\n")
                break
            else:
                print(f"\n{'=' * 70}")
                print(f"All {num_pieces} pieces have frame corners, but total is {total_frame_corners} (target: {target_frame_corners})")
                print(f"Some pieces may have multiple frame corners. Using this result.")
                print(f"{'=' * 70}\n")
                break
        elif total_frame_corners < target_frame_corners:
            print(f"  Too few corners, loosening strictness to detect more straight edges...")

        # Stop if we reached the loosest level
        if level_idx == len(strictness_levels) - 1:
            print(f"\n{'=' * 70}")
            print(f"Reached loosest strictness level.")
            print(f"Final result: {total_frame_corners} frame corners, {len(pieces_with_frame_corner)}/{num_pieces} pieces covered")
            print(f"{'=' * 70}\n")
            break

    # Recalculate final total from stored status
    final_total_frame_corners = sum(count for _, count in piece_frame_corner_status.values())

    # Final report
    print(f"\n{'=' * 70}")
    print(f"FRAME CORNER DETECTION REPORT")
    print(f"{'=' * 70}")
    print(f"Final strictness: {analyzer.strictness}")
    print(f"Total frame corners: {final_total_frame_corners}")
    print(f"\nPer-piece results:")

    for piece_idx in range(num_pieces):
        if piece_idx in piece_frame_corner_status:
            strictness, count = piece_frame_corner_status[piece_idx]
            print(f"  [OK] Piece {piece_idx}: {count} frame corner(s) - Found at strictness '{strictness}'")
        else:
            print(f"  [MISSING] Piece {piece_idx}: NO frame corners found")

    print(f"{'=' * 70}\n")

    print(f"  [OK] Corners detected and drawn on smoothed SVG (strictness: {analyzer.strictness})")

    # Save analysis data
    analyzer.save_analysis_data("analysis_data.json")

    # Print info
    pieces_info = analyzer.get_piece_info()
    print(f"\n{'-' * 70}")
    print(f"Detected {len(pieces_info)} puzzle pieces:")
    for idx, info in enumerate(pieces_info):
        print(f"  Piece {info['id']}: Area={info['area']:.0f}, Centroid={info['centroid']}")
        if analyzer.corners_list and idx < len(analyzer.corners_list) and analyzer.corners_list[idx]:
            corner_info = analyzer.corners_list[idx]
            outer = len(corner_info.get('outer_corners', []))
            inner = len(corner_info.get('inner_corners', []))
            straight = len(corner_info.get('straight_edges', []))
            # Count only confirmed frame corners with clear forbidden zones
            all_frame_corners = corner_info.get('frame_corners', [])
            confirmed_frame_corners = [
                fc for fc in all_frame_corners
                if not fc.get('potential', False) and fc.get('forbidden_zone_clear', False)
            ]
            frame = len(confirmed_frame_corners)
            print(f"    Corners: {outer} outer, {inner} inner | Straight edges: {straight} | Frame corners: {frame}")

    print(f"\n{'=' * 70}")
    print(f"FINAL RESULT: {total_frame_corners} frame corners found (target: {target_frame_corners})")
    print(f"{'=' * 70}\n")

    return analyzer
