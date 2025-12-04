"""Visualization of puzzle pieces from analysis."""
import cv2
import numpy as np
from typing import List, Tuple
from .data_loader import AnalyzedPuzzlePiece
from ..common.data_classes import Point, ContourSegment
from ..common.utils import normalize_vector


class SolverVisualizer:
    """Visualizes puzzle pieces and their features."""

    @staticmethod
    def _calculate_outward_direction(segment: ContourSegment, piece_centroid: Point) -> Tuple[float, float]:
        """Calculate perpendicular direction to segment, pointing outward from piece."""
        # Get first and last points of segment
        p1 = segment.contour_points[0]
        p2 = segment.contour_points[-1]

        # Segment direction vector
        seg_dx = p2.x - p1.x
        seg_dy = p2.y - p1.y

        # Perpendicular vector (rotate 90 degrees counterclockwise)
        perp_dx = -seg_dy
        perp_dy = seg_dx

        # Midpoint of segment
        mid_x = sum(p.x for p in segment.contour_points) / len(segment.contour_points)
        mid_y = sum(p.y for p in segment.contour_points) / len(segment.contour_points)

        # Vector from centroid to midpoint (to determine outward direction)
        to_mid_x = mid_x - piece_centroid.x
        to_mid_y = mid_y - piece_centroid.y

        # Check if perpendicular points outward; if not, flip it
        dot_product = perp_dx * to_mid_x + perp_dy * to_mid_y
        if dot_product < 0:
            perp_dx = -perp_dx
            perp_dy = -perp_dy

        return normalize_vector(perp_dx, perp_dy)

    @staticmethod
    def _is_point_in_contour(point: Tuple[int, int], contour_array: np.ndarray) -> bool:
        """Check if point is inside contour using cv2.pointPolygonTest."""
        return cv2.pointPolygonTest(contour_array, point, False) >= 0

    @staticmethod
    def _draw_arrow(image: np.ndarray, start: Tuple[int, int], direction: Tuple[float, float],
                    length: int = 30, color: Tuple[int, int, int] = (255, 255, 255),
                    thickness: int = 2, tip_size: int = 10, contour: np.ndarray = None) -> None:
        """Draw an arrow with triangular tip, flipped if it points into the contour."""
        dx, dy = direction
        end = (int(start[0] + dx * length), int(start[1] + dy * length))

        # Check if arrow end point is inside contour; if so, flip direction
        if contour is not None and SolverVisualizer._is_point_in_contour(end, contour):
            dx, dy = -dx, -dy
            end = (int(start[0] + dx * length), int(start[1] + dy * length))

        cv2.line(image, start, end, color, thickness)

        # Draw arrowhead
        angle = np.arctan2(dy, dx)
        tip1 = (int(end[0] + tip_size * np.cos(angle + np.pi * 0.75)),
                int(end[1] + tip_size * np.sin(angle + np.pi * 0.75)))
        tip2 = (int(end[0] + tip_size * np.cos(angle - np.pi * 0.75)),
                int(end[1] + tip_size * np.sin(angle - np.pi * 0.75)))
        cv2.fillPoly(image, [np.array([end, tip1, tip2], dtype=np.int32)], color)

    @staticmethod
    def draw_puzzle_pieces(image: np.ndarray, pieces: List[AnalyzedPuzzlePiece],
                          segments: List[List[ContourSegment]] = None) -> np.ndarray:
        """Draw all puzzle pieces with features."""
        legend_width = 350
        height, width = image.shape[:2]
        canvas = np.ones((height, width + legend_width, 3), dtype=np.uint8) * 40
        canvas[:, legend_width:] = image.copy()

        # Draw all pieces with orange fill to ensure they're all visible
        for idx, piece in enumerate(pieces):
            # Draw filled piece in orange
            contour_array = np.array([[int(p.x) + legend_width, int(p.y)] for p in piece.contour_points], dtype=np.int32)
            cv2.fillPoly(canvas, [contour_array], (0, 165, 255))  # Orange fill in BGR
            cv2.polylines(canvas, [contour_array], True, (0, 0, 0), 2)  # Black outline

            # Draw centroid
            cent = (int(piece.centroid.x) + legend_width, int(piece.centroid.y))
            cv2.circle(canvas, cent, 7, (255, 0, 0), -1)
            cv2.putText(canvas, f"P{piece.piece_id}", (cent[0]-30, cent[1]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw corners
            offset_x = legend_width
            for c in piece.outer_corners:
                cv2.circle(canvas, (int(c.x) + offset_x, int(c.y)), 6, (255, 0, 255), -1)
            for c in piece.inner_corners:
                cv2.drawMarker(canvas, (int(c.x) + offset_x, int(c.y)), (0, 0, 255),
                              markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=3)
            for c in piece.frame_corners:
                pt = (int(c.x) + offset_x, int(c.y))
                cv2.drawMarker(canvas, pt, (0, 255, 255), markerType=cv2.MARKER_SQUARE, markerSize=24, thickness=5)
                cv2.circle(canvas, pt, 30, (0, 255, 255), 3)

            # Draw segments with IDs and arrows
            if segments and idx < len(segments):
                # Create contour array for point-in-contour test
                contour_array = np.array([[int(p.x) + offset_x, int(p.y)] for p in piece.contour_points], dtype=np.int32)

                for seg in segments[idx]:
                    if seg.contour_points and len(seg.contour_points) > 1:
                        seg_points = np.array([[int(p.x) + offset_x, int(p.y)] for p in seg.contour_points], dtype=np.int32)
                        cv2.polylines(canvas, [seg_points], False, (255, 0, 0), 3)

                        mid_x = sum(p.x for p in seg.contour_points) / len(seg.contour_points) + offset_x
                        mid_y = sum(p.y for p in seg.contour_points) / len(seg.contour_points)
                        mid_pt = (int(mid_x), int(mid_y))

                        direction = SolverVisualizer._calculate_outward_direction(seg, piece.centroid)
                        SolverVisualizer._draw_arrow(canvas, mid_pt, direction, length=30, color=(255, 255, 255), thickness=2, tip_size=10, contour=contour_array)

                        label = f"S{seg.segment_id}"
                        cv2.rectangle(canvas, (mid_pt[0] - 15, mid_pt[1] - 10), (mid_pt[0] + 15, mid_pt[1] + 10), (0, 255, 255), -1)
                        cv2.putText(canvas, label, (mid_pt[0] - 12, mid_pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return canvas

    @staticmethod
    def draw_legend(image: np.ndarray) -> np.ndarray:
        """Draw legend on the left side of the image."""
        result = image.copy()
        cv2.rectangle(result, (10, 10), (340, result.shape[0] - 10), (40, 40, 40), -1)
        cv2.rectangle(result, (10, 10), (340, result.shape[0] - 10), (200, 200, 200), 2)

        legend = [
            ((0, 255, 0), "Green: Contours"),
            ((255, 0, 0), "Blue: Centroid"),
            ((255, 0, 255), "Pink: Outer Corners"),
            ((0, 0, 255), "Red: Inner Corners"),
            ((255, 255, 0), "Cyan: Curved Points"),
            ((0, 255, 255), "Yellow: Frame Corners"),
            ((255, 0, 0), "Blue: Segment Line"),
            ((0, 255, 255), "Cyan: Normal Segment"),
            ((255, 255, 0), "Yellow: Border Segment"),
            ((255, 255, 255), "White: Arrow (Outward)"),
        ]

        for i, (color, text) in enumerate(legend):
            y = 30 + i * 30
            cv2.rectangle(result, (20, y), (40, y + 15), color, -1)
            cv2.putText(result, text, (50, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return result

    @staticmethod
    def print_all_pieces_summary(pieces: List[AnalyzedPuzzlePiece]) -> None:
        """Print piece summaries."""
        print(f"\n{'='*60}")
        for piece in pieces:
            print(f"Piece {piece.piece_id}: {piece.area:.0f}px² | Centroid: ({piece.centroid.x:.0f}, {piece.centroid.y:.0f}) | "
                  f"Corners: {len(piece.outer_corners)} outer, {len(piece.inner_corners)} inner, "
                  f"{len(piece.frame_corners)} frame | Segments: {len(piece.segments)}")
        print(f"{'='*60}\n")

    @staticmethod
    def print_all_segments(all_segments: List[List[ContourSegment]]) -> None:
        """Print detailed segment information."""
        print(f"\n{'='*60}")
        for piece_idx, segments in enumerate(all_segments):
            print(f"Piece {piece_idx}: {len(segments)} segments")
            for seg in segments:
                seg_type = "BORDER" if seg.is_border_edge else "NORMAL"
                print(f"  S{seg.segment_id}: {seg_type} | {len(seg.contour_points)} points | "
                      f"From ({seg.start_corner.x:.0f},{seg.start_corner.y:.0f}) To ({seg.end_corner.x:.0f},{seg.end_corner.y:.0f})")
        print(f"{'='*60}\n")

