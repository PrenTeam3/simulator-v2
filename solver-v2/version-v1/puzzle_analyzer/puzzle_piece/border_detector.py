import cv2
import numpy as np
from typing import List, Tuple
from puzzle_analyzer.geometry import LineSegment


class BorderDetector:
    """Handles border edge detection for puzzle pieces."""

    @staticmethod
    def identify_border_edges(straight_segments: List[LineSegment], convex_hull,
                              contour, centroid: Tuple[int, int]) -> List[LineSegment]:
        """
        Identify potential border edges from straight segments.
        SIMPLIFIED: A straight segment is a border edge if it's long enough
        and on the outer boundary of the piece.
        """
        border_edges = []
        print(f"  Evaluating {len(straight_segments)} straight segments for border edges...")

        for idx, segment in enumerate(straight_segments):
            # Check if segment is on outer boundary
            on_boundary = BorderDetector._is_on_outer_boundary(segment, convex_hull)

            if on_boundary:
                # Check orientation relative to centroid
                points_inward = BorderDetector._check_inward_orientation(segment, contour, centroid)

                if points_inward:
                    segment.is_border_edge = True
                    border_edges.append(segment)
                    print(f"    [OK] Segment {idx} is BORDER EDGE: {segment.p1} to {segment.p2}")
                else:
                    print(f"    [NO] Segment {idx} rejected: points outward")
            else:
                print(f"    [NO] Segment {idx} rejected: not on outer boundary")

        return border_edges

    @staticmethod
    def _is_on_outer_boundary(segment: LineSegment, convex_hull) -> bool:
        """Check if segment endpoints are on the convex hull."""
        for pt in [segment.p1, segment.p2]:
            dist = cv2.pointPolygonTest(
                convex_hull,
                (float(pt[0]), float(pt[1])),
                True
            )
            # Must be very close to or on the hull
            if dist < -2:
                return False
        return True

    @staticmethod
    def _check_inward_orientation(segment: LineSegment, contour, centroid: Tuple[int, int]) -> bool:
        """
        Check if the piece bulk is on the inward side of the segment.
        Tests multiple points perpendicular to the line.
        """
        mid = segment.midpoint()
        normal = segment.normal_vector()

        # Test both directions
        distances = [15, 30, 45]

        # Direction 1: normal direction
        dir1_inside = 0
        for d in distances:
            test_pt = (mid[0] + normal[0] * d, mid[1] + normal[1] * d)
            result = cv2.pointPolygonTest(
                contour,
                (float(test_pt[0]), float(test_pt[1])),
                False
            )
            if result >= 0:
                dir1_inside += 1

        # Direction 2: opposite direction
        dir2_inside = 0
        for d in distances:
            test_pt = (mid[0] - normal[0] * d, mid[1] - normal[1] * d)
            result = cv2.pointPolygonTest(
                contour,
                (float(test_pt[0]), float(test_pt[1])),
                False
            )
            if result >= 0:
                dir2_inside += 1

        # One side should be mostly inside, other mostly outside
        # For a border edge: one direction has piece mass, other doesn't
        if dir1_inside >= 2 and dir2_inside <= 1:
            return True
        if dir2_inside >= 2 and dir1_inside <= 1:
            return True

        # Additional check: centroid side
        to_centroid = np.array(centroid) - np.array(mid)
        dot = np.dot(normal, to_centroid)

        # If tests are ambiguous, use centroid as tiebreaker
        if abs(dir1_inside - dir2_inside) <= 1:
            # Piece should be on centroid side
            return True

        return False
