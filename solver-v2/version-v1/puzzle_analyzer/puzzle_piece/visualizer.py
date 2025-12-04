import cv2
import numpy as np
from typing import List, Tuple, Dict
from puzzle_analyzer.geometry import LineSegment


class PuzzlePieceVisualizer:
    """Handles visualization and information display for puzzle pieces."""

    @staticmethod
    def draw_analysis(image: np.ndarray, contour, centroid: Tuple[int, int],
                      straight_segments: List[LineSegment], border_edges: List[LineSegment],
                      inner_corners: List[Tuple[int, int]], outer_corners: List[Tuple[int, int]],
                      curved_points: List[Tuple[int, int]], frame_corners: List[Tuple[int, int]],
                      corner_ids: Dict[Tuple[int, int], int],
                      piece_id: int):
        """Draw all analysis results on the image."""
        # 1. Draw contour
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        # 2. Draw centroid
        cv2.circle(image, centroid, 7, (255, 0, 0), -1)
        cv2.putText(image, f"Piece {piece_id}",
                    (centroid[0] - 30, centroid[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 3. Draw straight segments (orange)
        for segment in straight_segments:
            if not segment.is_border_edge:
                cv2.line(image, segment.p1, segment.p2, (0, 165, 255), 3)

        # 4. Draw border edges (blue - thicker)
        for segment in border_edges:
            cv2.line(image, segment.p1, segment.p2, (255, 140, 0), 6)
            # Draw midpoint marker
            mid = segment.midpoint()
            cv2.circle(image, (int(mid[0]), int(mid[1])), 5, (255, 200, 0), -1)

        # 5. Draw inner corners (red X) with IDs
        for (x, y) in inner_corners:
            cv2.drawMarker(image, (x, y), (0, 0, 255),
                           markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=12, thickness=3)
            # Draw corner ID
            corner_id = corner_ids.get((x, y), -1)
            cv2.putText(image, f"C{corner_id}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 6. Draw outer corners (pink circles) with IDs
        for (x, y) in outer_corners:
            cv2.circle(image, (x, y), 6, (255, 0, 255), -1)
            # Draw corner ID
            corner_id = corner_ids.get((x, y), -1)
            cv2.putText(image, f"C{corner_id}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # 7. Draw curved points (cyan) with IDs
        for (x, y) in curved_points:
            cv2.circle(image, (x, y), 4, (255, 255, 0), 2)
            # Draw corner ID
            corner_id = corner_ids.get((x, y), -1)
            cv2.putText(image, f"C{corner_id}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # 8. Draw frame corners (yellow squares) - LARGEST
        for (x, y) in frame_corners:
            cv2.drawMarker(image, (x, y), (0, 255, 255),
                           markerType=cv2.MARKER_SQUARE,
                           markerSize=24, thickness=5)
            cv2.circle(image, (x, y), 30, (0, 255, 255), 3)
            cv2.putText(image, "FRAME", (x + 15, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    @staticmethod
    def print_info(area: float, centroid: Tuple[int, int],
                   outer_corners: List[Tuple[int, int]], inner_corners: List[Tuple[int, int]],
                   curved_points: List[Tuple[int, int]], straight_segments: List[LineSegment],
                   border_edges: List[LineSegment], frame_corners: List[Tuple[int, int]],
                   piece_id: int):
        """Print analysis information to console."""
        print(f"\n{'=' * 60}")
        print(f"--- Piece {piece_id} ---")
        print(f"Area: {area:.0f} px²")
        print(f"Centroid (Magnetic Position): {centroid}")
        print(f"  > {len(outer_corners)} outer corners found")
        print(f"  > {len(inner_corners)} inner corners found")
        print(f"  > {len(curved_points)} curved transition points found")
        print(f"  > {len(straight_segments)} straight segments found")
        print(f"  > {len(border_edges)} BORDER EDGES identified")
        print(f"  > {len(frame_corners)} FRAME CORNERS (90°) found")
        if frame_corners:
            print(f"     Frame corner locations: {frame_corners}")
        print(f"{'=' * 60}")
