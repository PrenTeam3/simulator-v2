"""Segment puzzle piece contours between corners."""
from dataclasses import dataclass
from typing import List
from .data_loader import AnalyzedPuzzlePiece, Point, Corner
from .utils import distance_point_to_corner


@dataclass
class ContourSegment:
    segment_id: int
    piece_id: int
    start_corner: Corner
    end_corner: Corner
    contour_points: List[Point]
    piece_centroid: Point
    is_border_edge: bool = False


class ContourSegmenter:
    """Segments puzzle piece contours based on corner positions."""

    @staticmethod
    def _min_distance_to_contour(corner: Corner, contour_points: List[Point]) -> float:
        """Find minimum distance from corner to any contour point."""
        return min((distance_point_to_corner(p, corner) for p in contour_points), default=float('inf'))

    @staticmethod
    def _find_corner_index_on_contour(corner: Corner, contour_points: List[Point]) -> int:
        """Find index of contour point closest to corner."""
        return min(range(len(contour_points)),
                  key=lambda j: distance_point_to_corner(contour_points[j], corner))

    @staticmethod
    def _deduplicate_corners(corners: List[tuple], tolerance: float = 5) -> List[tuple]:
        """Remove duplicate corners within tolerance distance."""
        if not corners:
            return []
        deduplicated = [corners[0]]
        for corner, corner_type in corners[1:]:
            if not any(distance_point_to_corner(corner, kc) <= tolerance for kc, _ in deduplicated):
                deduplicated.append((corner, corner_type))
        return deduplicated

    @staticmethod
    def _is_corner_on_contour(corner: Corner, contour_points: List[Point], tolerance: float = 25) -> bool:
        """Check if corner is on contour within tolerance."""
        return ContourSegmenter._min_distance_to_contour(corner, contour_points) <= tolerance

    @staticmethod
    def _is_border_edge(c1: Corner, c2: Corner, border_edges: List, tolerance: float = 5) -> bool:
        """Check if segment corners match a border edge."""
        for p1, p2 in border_edges:
            c1_match = (abs(p1.x - c1.x) < tolerance and abs(p1.y - c1.y) < tolerance) or \
                       (abs(p2.x - c1.x) < tolerance and abs(p2.y - c1.y) < tolerance)
            c2_match = (abs(p1.x - c2.x) < tolerance and abs(p1.y - c2.y) < tolerance) or \
                       (abs(p2.x - c2.x) < tolerance and abs(p2.y - c2.y) < tolerance)
            if c1_match and c2_match:
                return True
        return False

    @staticmethod
    def segment_piece_contours(piece: AnalyzedPuzzlePiece, contour_tolerance: float = 25) -> List[ContourSegment]:
        """Segment contour into segments between consecutive corners on the contour."""
        # Collect all corners
        all_corners = [(c, 'outer') for c in piece.outer_corners] + \
                      [(c, 'inner') for c in piece.inner_corners] + \
                      [(c, 'curved') for c in piece.curved_points] + \
                      [(c, 'frame') for c in piece.frame_corners]

        if not all_corners:
            return []

        # Filter to corners on contour
        corners_on_contour = [(c, t) for c, t in all_corners
                              if ContourSegmenter._is_corner_on_contour(c, piece.contour_points, contour_tolerance)]

        if len(corners_on_contour) < 2:
            return []

        # Sort by position along contour and deduplicate
        sorted_corners = sorted(corners_on_contour,
                               key=lambda x: ContourSegmenter._find_corner_index_on_contour(x[0], piece.contour_points))
        sorted_corners = ContourSegmenter._deduplicate_corners(sorted_corners)

        if len(sorted_corners) < 2:
            return []

        segments = []
        for i in range(len(sorted_corners)):
            c1, _ = sorted_corners[i]
            c2, _ = sorted_corners[(i + 1) % len(sorted_corners)]

            idx1 = ContourSegmenter._find_corner_index_on_contour(c1, piece.contour_points)
            idx2 = ContourSegmenter._find_corner_index_on_contour(c2, piece.contour_points)

            # Extract segment points with wrap-around handling
            if idx1 < idx2:
                segment_points = piece.contour_points[idx1:idx2+1]
            else:
                segment_points = piece.contour_points[idx1:] + piece.contour_points[:idx2+1]

            # Skip segments wrapping entire contour
            if len(segment_points) > len(piece.contour_points) / 2:
                continue

            segments.append(ContourSegment(
                segment_id=len(segments),
                piece_id=piece.piece_id,
                start_corner=c1,
                end_corner=c2,
                contour_points=segment_points,
                piece_centroid=piece.centroid,
                is_border_edge=ContourSegmenter._is_border_edge(c1, c2, piece.border_edges)
            ))

        return segments

