"""Helper functions for edge solver v2 visualization."""

import cv2
import numpy as np


class VisualizerHelpers:
    """Helper utility functions for visualization."""

    @staticmethod
    def _find_frame_touching_segment_ids(piece, all_segments):
        """Find segment IDs that directly touch frame corners.

        Returns:
            set: Set of segment IDs that touch frame corners
        """
        frame_touching_ids = set()
        tolerance = 1.0

        for frame_corner in piece.frame_corners:
            frame_pos = (frame_corner.x, frame_corner.y)

            for seg in all_segments:
                seg_start = (seg.start_corner.x, seg.start_corner.y)
                seg_end = (seg.end_corner.x, seg.end_corner.y)

                if (abs(seg_start[0] - frame_pos[0]) < tolerance and
                    abs(seg_start[1] - frame_pos[1]) < tolerance):
                    frame_touching_ids.add(seg.segment_id)
                elif (abs(seg_end[0] - frame_pos[0]) < tolerance and
                      abs(seg_end[1] - frame_pos[1]) < tolerance):
                    frame_touching_ids.add(seg.segment_id)

        return frame_touching_ids

    @staticmethod
    def _calculate_segment_arrow(segment, piece):
        """Calculate arrow direction for a segment (normal vector pointing outward).

        Args:
            segment: Segment object
            piece: Piece object

        Returns:
            Tuple of (midpoint, normal_vector)
        """
        seg_points = np.array([[p.x, p.y] for p in segment.contour_points], dtype=np.float64)

        # Calculate at midpoint
        mid_idx = len(seg_points) // 2
        mid = seg_points[mid_idx]

        if mid_idx == 0:
            tangent = seg_points[mid_idx + 1] - seg_points[mid_idx]
        elif mid_idx == len(seg_points) - 1:
            tangent = seg_points[mid_idx] - seg_points[mid_idx - 1]
        else:
            tangent = seg_points[mid_idx + 1] - seg_points[mid_idx - 1]

        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        if np.linalg.norm(normal) > 0:
            normal = normal / np.linalg.norm(normal)

            # Verify arrow points OUTSIDE the piece
            piece_contour = np.array([[p.x, p.y] for p in piece.contour_points], dtype=np.int32)
            test_arrow_end = mid + normal * 25

            is_inside = cv2.pointPolygonTest(piece_contour, tuple(test_arrow_end.astype(float)), False)

            # If arrow points inside, flip it
            if is_inside > 0:
                normal = -normal

        return mid, normal

    @staticmethod
    def _draw_outward_arrow(image, segment_points, piece, offset_x, offset_y, color=(255, 0, 0)):
        """Draw an arrow pointing outward from the segment.

        The arrow shows which side of the segment faces outward (toward frame).

        Args:
            image: Image to draw on
            segment_points: Segment contour points (already offset)
            piece: Piece object
            offset_x: X offset for coordinate conversion
            offset_y: Y offset for coordinate conversion
            color: Arrow color in BGR format (default: red)
        """
        if len(segment_points) < 2:
            return

        # Get piece center
        piece_contour = np.array([[p.x, p.y] for p in piece.contour_points])
        piece_center = np.mean(piece_contour, axis=0)

        # Convert to image coordinates
        piece_center_img = piece_center - np.array([offset_x, offset_y])

        # Find the middle point of the segment
        mid_idx = len(segment_points) // 2
        mid_point = segment_points[mid_idx]

        # Calculate tangent at the middle point
        if mid_idx == 0:
            tangent = segment_points[mid_idx + 1] - segment_points[mid_idx]
        elif mid_idx == len(segment_points) - 1:
            tangent = segment_points[mid_idx] - segment_points[mid_idx - 1]
        else:
            tangent = segment_points[mid_idx + 1] - segment_points[mid_idx - 1]

        # Normal is perpendicular to tangent (rotate 90 degrees)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        normal_length = np.linalg.norm(normal)
        if normal_length > 0:
            normal = normal / normal_length

            # Verify arrow points OUTSIDE by checking if arrow endpoint is inside contour
            arrow_length = 30
            piece_contour_int = np.array([[p.x, p.y] for p in piece.contour_points], dtype=np.int32)

            # Convert mid_point back to original coordinates for testing
            mid_point_original = mid_point + np.array([offset_x, offset_y])
            test_arrow_end = mid_point_original + normal * arrow_length

            # Use cv2.pointPolygonTest: positive if inside, 0 if on edge, negative if outside
            is_inside = cv2.pointPolygonTest(piece_contour_int, tuple(test_arrow_end.astype(float)), False)

            # If arrow points inside the piece, flip it
            if is_inside > 0:
                normal = -normal

            # Draw arrow from mid_point outward
            arrow_end = mid_point + normal * arrow_length

            cv2.arrowedLine(
                image,
                tuple(mid_point.astype(np.int32)),
                tuple(arrow_end.astype(np.int32)),
                color,
                2,
                tipLength=0.3
            )

    @staticmethod
    def _draw_forbidden_zone(image, segment_points, piece, offset_x, offset_y):
        """Draw the forbidden zone (outside area) for a segment.

        The forbidden zone shows which side of the segment faces outward (toward frame).
        This is calculated using the normal vector pointing away from the piece center.

        Args:
            image: Image to draw on
            segment_points: Segment contour points (already offset)
            piece: Piece object
            offset_x: X offset for coordinate conversion
            offset_y: Y offset for coordinate conversion
        """
        if len(segment_points) < 2:
            return

        # Get piece center
        piece_contour = np.array([[p.x, p.y] for p in piece.contour_points])
        piece_center = np.mean(piece_contour, axis=0)

        # Convert to image coordinates
        piece_center_img = piece_center - np.array([offset_x, offset_y])

        # Create offset points along the normal (outward direction)
        offset_distance = 15  # Distance to offset for forbidden zone visualization

        forbidden_points = []
        for i in range(len(segment_points)):
            # Calculate normal at this point
            if i == 0:
                # Use next point for tangent
                tangent = segment_points[i + 1] - segment_points[i]
            elif i == len(segment_points) - 1:
                # Use previous point for tangent
                tangent = segment_points[i] - segment_points[i - 1]
            else:
                # Use neighboring points for smoother tangent
                tangent = segment_points[i + 1] - segment_points[i - 1]

            # Normal is perpendicular to tangent (rotate 90 degrees)
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
            normal_length = np.linalg.norm(normal)
            if normal_length > 0:
                normal = normal / normal_length
            else:
                continue

            # Determine if normal points toward or away from piece center
            to_center = piece_center_img - segment_points[i]
            dot_product = np.dot(normal, to_center)

            # If normal points toward center, flip it to point outward
            if dot_product > 0:
                normal = -normal

            # Create offset point
            offset_point = segment_points[i] + normal * offset_distance
            forbidden_points.append(offset_point)

        # Draw forbidden zone as a semi-transparent red polygon
        if len(forbidden_points) >= 2:
            # Create polygon: original segment + offset segment (reversed)
            polygon_points = np.vstack([
                segment_points,
                np.array(forbidden_points[::-1])
            ]).astype(np.int32)

            # Create overlay for transparency
            overlay = image.copy()
            cv2.fillPoly(overlay, [polygon_points], (0, 0, 255))  # Red color

            # Blend with original image
            alpha = 0.3  # Transparency level
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            # Draw outline of forbidden zone
            cv2.polylines(image, [np.array(forbidden_points, dtype=np.int32)], False, (0, 0, 200), 1)
