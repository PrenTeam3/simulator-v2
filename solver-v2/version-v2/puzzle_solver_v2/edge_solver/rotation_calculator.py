"""Module for calculating piece rotation angles."""

import numpy as np


class RotationCalculator:
    """Handles calculating rotation angles for puzzle pieces."""

    @staticmethod
    def calculate_piece_rotation_angles(piece_frame_segments):
        """Calculate rotation angle for each piece based on frame corner edges.

        The rotation angle is determined by making:
        - Edge A parallel to the horizontal axis
        - Edge B parallel to the vertical axis

        Args:
            piece_frame_segments: List of tuples (piece, segments, frame_adjacent_segments)

        Returns:
            dict: piece_id -> rotation_angle_degrees
        """
        rotation_angles = {}

        for piece, segments, frame_segs in piece_frame_segments:
            if len(piece.frame_corners) == 0:
                rotation_angles[piece.piece_id] = 0.0
                continue

            frame_corner = piece.frame_corners[0]

            # Find the two border segments that meet at the frame corner
            border_segments = [seg for seg in segments if seg.is_border_edge]

            # Find segments that touch the frame corner
            frame_edges = []
            tolerance = 2.0

            for seg in border_segments:
                seg_start = (seg.start_corner.x, seg.start_corner.y)
                seg_end = (seg.end_corner.x, seg.end_corner.y)
                frame_pos = (frame_corner.x, frame_corner.y)

                # Check if segment starts or ends at frame corner
                if (abs(seg_start[0] - frame_pos[0]) < tolerance and
                    abs(seg_start[1] - frame_pos[1]) < tolerance):
                    frame_edges.append((seg, 'start'))
                elif (abs(seg_end[0] - frame_pos[0]) < tolerance and
                      abs(seg_end[1] - frame_pos[1]) < tolerance):
                    frame_edges.append((seg, 'end'))

            if len(frame_edges) < 2:
                print(f"Warning: Piece {piece.piece_id} - Found only {len(frame_edges)} frame edges")
                rotation_angles[piece.piece_id] = 0.0
                continue

            # Get the two frame edges
            seg1, point1 = frame_edges[0]
            seg2, point2 = frame_edges[1]

            # Get direction vectors for each edge (away from frame corner)
            if point1 == 'start':
                # Direction is from start to end
                vec1_x = seg1.end_corner.x - seg1.start_corner.x
                vec1_y = seg1.end_corner.y - seg1.start_corner.y
            else:
                # Direction is from end to start
                vec1_x = seg1.start_corner.x - seg1.end_corner.x
                vec1_y = seg1.start_corner.y - seg1.end_corner.y

            if point2 == 'start':
                vec2_x = seg2.end_corner.x - seg2.start_corner.x
                vec2_y = seg2.end_corner.y - seg2.start_corner.y
            else:
                vec2_x = seg2.start_corner.x - seg2.end_corner.x
                vec2_y = seg2.start_corner.y - seg2.end_corner.y

            # Calculate angles of both vectors
            angle1 = np.degrees(np.arctan2(vec1_y, vec1_x))
            angle2 = np.degrees(np.arctan2(vec2_y, vec2_x))

            # Normalize angles to 0-360 range
            angle1 = angle1 % 360
            angle2 = angle2 % 360

            # Determine which edge should be horizontal (edge A) and which vertical (edge B)
            # The horizontal edge should point roughly right (0° or 180°)
            # The vertical edge should point roughly down (90° or 270°)

            # Calculate distance from each angle to horizontal (0° or 180°)
            dist1_to_horizontal = min(abs(angle1 - 0), abs(angle1 - 180), abs(angle1 - 360))
            dist2_to_horizontal = min(abs(angle2 - 0), abs(angle2 - 180), abs(angle2 - 360))

            if dist1_to_horizontal < dist2_to_horizontal:
                # vec1 should be horizontal, vec2 should be vertical
                horizontal_angle = angle1
                vertical_angle = angle2
            else:
                # vec2 should be horizontal, vec1 should be vertical
                horizontal_angle = angle2
                vertical_angle = angle1

            # Calculate rotation needed to align horizontal edge to 0° (pointing right)
            # We want to rotate so horizontal_angle becomes 0°
            rotation_needed = -horizontal_angle

            # Normalize to -180 to 180 range
            while rotation_needed > 180:
                rotation_needed -= 360
            while rotation_needed < -180:
                rotation_needed += 360

            rotation_angles[piece.piece_id] = rotation_needed

            print(f"Piece {piece.piece_id}: Edge angles = {angle1:.1f}°, {angle2:.1f}° | Rotation needed = {rotation_needed:.1f}°")

        return rotation_angles
