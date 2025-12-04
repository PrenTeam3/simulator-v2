import cv2
import numpy as np
from typing import List, Tuple, Dict

from puzzle_analyzer.geometry import LineSegment
from puzzle_analyzer.puzzle_piece.corner_detector import CornerDetector
from puzzle_analyzer.puzzle_piece.segment_detector import SegmentDetector
from puzzle_analyzer.puzzle_piece.border_detector import BorderDetector
from puzzle_analyzer.puzzle_piece.frame_corner_detector import FrameCornerDetector
from puzzle_analyzer.puzzle_piece.visualizer import PuzzlePieceVisualizer


class PuzzlePiece:
    """
    A class representing a single puzzle piece with comprehensive analysis.
    Detects corners, curves, straight edges, and potential border edges.
    """

    def __init__(self, contour: np.ndarray, min_edge_length: int = 30, verbose_logging: bool = False):
        """
        Initialize and analyze the puzzle piece based on its contour.

        Args:
            contour: OpenCV contour (Nx1x2 array)
            min_edge_length: Minimum length for a segment to be considered
            verbose_logging: If True, shows detailed logging from frame corner detection
        """
        self.contour = contour
        self.area = cv2.contourArea(self.contour)
        self.min_edge_length = min_edge_length
        self.verbose_logging = verbose_logging

        # Calculate centroid (magnetic grip point)
        M = cv2.moments(self.contour)
        if M["m00"] != 0:
            self.centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            self.centroid = (0, 0)

        # Convex hull
        self.convex_hull = cv2.convexHull(self.contour)

        # Bounding box for reference
        self.bbox = cv2.boundingRect(self.contour)

        # Polygon approximation for corner detection
        perimeter = cv2.arcLength(self.contour, True)
        epsilon = 0.015 * perimeter  # More aggressive to reduce segments
        self.approx_poly = cv2.approxPolyDP(self.contour, epsilon, True)

        # Flat format for easier processing
        self.contour_flat = self.contour[:, 0, :]
        self.approx_poly_flat = self.approx_poly[:, 0, :]

        # Find the indices of the polygon approximation points in the original contour
        self.approx_indices = []
        for approx_pt in self.approx_poly_flat:
            # Find the index of the closest point in the original contour
            diffs = np.linalg.norm(self.contour_flat - approx_pt, axis=1)
            idx = np.argmin(diffs)
            self.approx_indices.append(idx)

        # Analysis results
        self.outer_corners: List[Tuple[int, int]] = []
        self.inner_corners: List[Tuple[int, int]] = []
        self.curved_points: List[Tuple[int, int]] = []
        self.corner_ids: Dict[Tuple[int, int], int] = {}  # Maps corner coordinates to ID
        self.straight_segments: List[LineSegment] = []
        self.border_edges: List[LineSegment] = []
        self.frame_corners: List[Tuple[int, int]] = []

        # Perform analysis
        self._analyze()

    def _analyze(self, debug_image=None):
        """Run all analysis steps."""
        # Step 1: Classify corners
        self.outer_corners, self.inner_corners, self.curved_points, self.corner_ids = \
            CornerDetector.classify_corners(self.approx_poly, self.approx_poly_flat, self.convex_hull, self.contour_flat)

        # Step 2: Detect straight segments
        self.straight_segments = SegmentDetector.detect_straight_segments(
            self.approx_poly_flat, self.approx_indices, self.contour_flat, self.min_edge_length
        )

        # Step 3: Identify border edges
        self.border_edges = BorderDetector.identify_border_edges(
            self.straight_segments, self.convex_hull, self.contour, self.centroid
        )

        # Step 4: Identify frame corners (with optional visualization)
        self.frame_corners = FrameCornerDetector.identify_frame_corners(
            self.border_edges, self.convex_hull, self.contour,
            self.contour_flat, self.centroid,
            self.inner_corners, self.outer_corners, self.corner_ids,
            debug_image=debug_image,
            verbose=self.verbose_logging
        )

    def reanalyze_with_visualization(self, debug_image):
        """Re-run frame corner detection with visualization enabled."""
        self.frame_corners = FrameCornerDetector.identify_frame_corners(
            self.border_edges, self.convex_hull, self.contour,
            self.contour_flat, self.centroid,
            self.inner_corners, self.outer_corners, self.corner_ids,
            debug_image=debug_image,
            verbose=self.verbose_logging
        )

    def draw_analysis(self, image: np.ndarray, piece_id: int):
        """Draw all analysis results on the image."""
        PuzzlePieceVisualizer.draw_analysis(
            image, self.contour, self.centroid, self.straight_segments,
            self.border_edges, self.inner_corners, self.outer_corners,
            self.curved_points, self.frame_corners, self.corner_ids, piece_id
        )

    def print_info(self, piece_id: int):
        """Print analysis information to console."""
        PuzzlePieceVisualizer.print_info(
            self.area, self.centroid, self.outer_corners, self.inner_corners,
            self.curved_points, self.straight_segments, self.border_edges,
            self.frame_corners, piece_id
        )
