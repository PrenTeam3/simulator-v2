"""Access pre-computed puzzle piece segments from SVG analysis."""
from typing import List
from .data_loader import AnalyzedPuzzlePiece
from ..common.data_classes import Point, Corner, Segment, ContourSegment
from ..common.utils import distance_point_to_corner


class ContourSegmenter:
    """
    Provides access to pre-computed segments from SVG analysis.

    Segments are now computed during SVG analysis by puzzle_analyzer,
    so this class simply converts them to the ContourSegment format used by the solver.
    """

    @staticmethod
    def _find_corner_index_on_contour(corner: Corner, contour_points: List[Point]) -> int:
        """Find index of contour point closest to corner."""
        return min(range(len(contour_points)),
                  key=lambda j: distance_point_to_corner(contour_points[j], corner))

    @staticmethod
    def segment_piece_contours(piece: AnalyzedPuzzlePiece) -> List[ContourSegment]:
        """
        Get pre-computed segments for a puzzle piece from SVG analysis.

        Args:
            piece: AnalyzedPuzzlePiece with pre-loaded segments from SVG

        Returns:
            List of ContourSegment objects with contour points extracted
        """
        if not piece.segments:
            return []

        segments = []

        for seg in piece.segments:
            # Convert Segment Points to Corners for compatibility
            start_corner = Corner(seg.corner_start.x, seg.corner_start.y, 'outer')
            end_corner = Corner(seg.corner_end.x, seg.corner_end.y, 'outer')

            # Find indices on contour
            idx1 = ContourSegmenter._find_corner_index_on_contour(start_corner, piece.contour_points)
            idx2 = ContourSegmenter._find_corner_index_on_contour(end_corner, piece.contour_points)

            # Extract segment points with wrap-around handling
            if idx1 < idx2:
                segment_points = piece.contour_points[idx1:idx2+1]
            else:
                segment_points = piece.contour_points[idx1:] + piece.contour_points[:idx2+1]

            # Create ContourSegment
            segments.append(ContourSegment(
                segment_id=seg.id,
                piece_id=piece.piece_id,
                start_corner=start_corner,
                end_corner=end_corner,
                contour_points=segment_points,
                piece_centroid=piece.centroid,
                is_border_edge=seg.is_straight  # Straight segments from frame edges
            ))

        return segments

