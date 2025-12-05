"""Module for finding frame-adjacent segments in puzzle pieces."""


class SegmentFinder:
    """Handles finding segments that are adjacent to frame corners."""

    @staticmethod
    def find_frame_adjacent_segments(piece, segments):
        """Find segments that touch the segments that are touching the frame corner.

        Args:
            piece: AnalyzedPuzzlePiece object
            segments: List of ContourSegment objects for this piece

        Returns:
            List of ContourSegment objects that are neighbors of frame-touching segments
        """
        # Step 1: Find segments directly touching the frame corner
        frame_touching_segments = []

        for frame_corner in piece.frame_corners:
            frame_pos = (frame_corner.x, frame_corner.y)

            for seg in segments:
                seg_start = (seg.start_corner.x, seg.start_corner.y)
                seg_end = (seg.end_corner.x, seg.end_corner.y)

                tolerance = 1.0
                if (abs(seg_start[0] - frame_pos[0]) < tolerance and
                    abs(seg_start[1] - frame_pos[1]) < tolerance):
                    if seg not in frame_touching_segments:
                        frame_touching_segments.append(seg)
                elif (abs(seg_end[0] - frame_pos[0]) < tolerance and
                      abs(seg_end[1] - frame_pos[1]) < tolerance):
                    if seg not in frame_touching_segments:
                        frame_touching_segments.append(seg)

        # Step 2: Find segments that touch the frame-touching segments
        second_layer_segments = []

        for frame_seg in frame_touching_segments:
            # Get the endpoints of the frame-touching segment
            frame_seg_start = (frame_seg.start_corner.x, frame_seg.start_corner.y)
            frame_seg_end = (frame_seg.end_corner.x, frame_seg.end_corner.y)

            # Find segments that share an endpoint with this frame-touching segment
            for seg in segments:
                if seg in frame_touching_segments:
                    continue  # Skip the frame-touching segments themselves

                seg_start = (seg.start_corner.x, seg.start_corner.y)
                seg_end = (seg.end_corner.x, seg.end_corner.y)

                tolerance = 1.0
                # Check if this segment shares an endpoint with frame_seg
                if ((abs(seg_start[0] - frame_seg_start[0]) < tolerance and
                     abs(seg_start[1] - frame_seg_start[1]) < tolerance) or
                    (abs(seg_start[0] - frame_seg_end[0]) < tolerance and
                     abs(seg_start[1] - frame_seg_end[1]) < tolerance) or
                    (abs(seg_end[0] - frame_seg_start[0]) < tolerance and
                     abs(seg_end[1] - frame_seg_start[1]) < tolerance) or
                    (abs(seg_end[0] - frame_seg_end[0]) < tolerance and
                     abs(seg_end[1] - frame_seg_end[1]) < tolerance)):
                    if seg not in second_layer_segments:
                        second_layer_segments.append(seg)

        return second_layer_segments
