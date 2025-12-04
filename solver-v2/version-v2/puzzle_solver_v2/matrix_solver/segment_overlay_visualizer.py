"""Visualize segment overlays for debugging matching."""
import cv2
import numpy as np
from typing import List, Tuple
from ..preparation.data_loader import AnalyzedPuzzlePiece
from ..common.data_classes import Point, ContourSegment, SegmentMatch
from .segment_matcher import SegmentMatcher
from ..common.utils import normalize_and_center, rotate_points, find_min_area_rotation


class SegmentOverlayVisualizer:
    """Visualize segment overlays to understand matching."""

    @staticmethod
    def create_overlay_visualization(piece1: AnalyzedPuzzlePiece,
                                     segments1: List[ContourSegment],
                                     piece2: AnalyzedPuzzlePiece,
                                     segments2: List[ContourSegment],
                                     matches: List[SegmentMatch],
                                     max_overlays: int = 3) -> np.ndarray:
        """
        Create visualization showing segment overlays for top matches.
        """
        # Canvas for overlays
        overlay_height = 300 * min(max_overlays, len(matches))
        overlay_width = 800
        overlay_canvas = np.ones((overlay_height, overlay_width, 3), dtype=np.uint8) * 40

        for idx, match in enumerate(matches[:max_overlays]):
            seg1 = next((s for s in segments1 if s.segment_id == match.seg1_id), None)
            seg2 = next((s for s in segments2 if s.segment_id == match.seg2_id), None)

            if seg1 and seg2:
                y_offset = idx * 300
                SegmentOverlayVisualizer._draw_segment_overlay(
                    overlay_canvas, seg1, seg2, match, y_offset
                )

        return overlay_canvas

    @staticmethod
    def create_overlay_visualization_for_specific_pairs(piece1: AnalyzedPuzzlePiece,
                                                         segments1: List[ContourSegment],
                                                         piece2: AnalyzedPuzzlePiece,
                                                         segments2: List[ContourSegment],
                                                         specific_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Create visualization for specific segment pairs (even if not in matches).
        Args:
            specific_pairs: List of (seg1_id, seg2_id) tuples to visualize
        """
        # Canvas for overlays
        overlay_height = 300 * len(specific_pairs)
        overlay_width = 800
        overlay_canvas = np.ones((overlay_height, overlay_width, 3), dtype=np.uint8) * 40

        for idx, (seg1_id, seg2_id) in enumerate(specific_pairs):
            seg1 = next((s for s in segments1 if s.segment_id == seg1_id), None)
            seg2 = next((s for s in segments2 if s.segment_id == seg2_id), None)

            if seg1 and seg2:
                y_offset = idx * 300
                # Create a temporary SegmentMatch object for visualization purposes
                # Use rotation of 0 and calculate basic scores on the fly
                temp_match = SegmentMatcher.create_temp_match_for_visualization(
                    piece1.piece_id, seg1_id, piece2.piece_id, seg2_id, seg1, seg2
                )
                SegmentOverlayVisualizer._draw_segment_overlay(
                    overlay_canvas, seg1, seg2, temp_match, y_offset
                )

        return overlay_canvas

    @staticmethod
    def _draw_segment_overlay(canvas: np.ndarray, seg1: ContourSegment, seg2: ContourSegment,
                             match: SegmentMatch, y_offset: int) -> None:
        """Draw a single segment overlay with proper alignment, padding, and bounding box area."""
        # Resample both segments to same number of points for better alignment
        target_points = 50
        pts1_resampled = SegmentMatcher._resample_points(seg1.contour_points, target_points)
        pts2_resampled = SegmentMatcher._resample_points(seg2.contour_points, target_points)

        # Normalize both contours to align them
        pts1_norm = normalize_and_center(pts1_resampled)
        pts2_norm = normalize_and_center(pts2_resampled)

        # Find the rotation angle that minimizes bounding box area
        optimal_area_rotation = find_min_area_rotation(
            pts1_norm, pts2_norm, precision=1
        )

        # Apply the optimal rotation to minimize bounding box area
        pts2_rotated = rotate_points(
            pts2_norm, optimal_area_rotation
        )

        # Store the optimal rotation for display
        area_minimizing_rotation = optimal_area_rotation

        # Find bounding box of both normalized segments (for proper scaling)
        all_points = pts1_norm + pts2_rotated
        min_x = min(p.x for p in all_points)
        max_x = max(p.x for p in all_points)
        min_y = min(p.y for p in all_points)
        max_y = max(p.y for p in all_points)

        # Calculate axis-aligned bounding box dimensions for scaling/padding
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y

        # Add padding
        padding = 0.3
        width = bbox_width if bbox_width > 0 else 2
        height = bbox_height if bbox_height > 0 else 2
        min_x -= width * padding / 2
        max_x += width * padding / 2
        min_y -= height * padding / 2
        max_y += height * padding / 2

        # Will calculate actual rotated bounding box area later after scaling
        rotated_bbox_area = None

        # Scale to fit in display area (350x220 pixels with margins)
        display_width = 350
        display_height = 220
        scale_x = display_width / (max_x - min_x) if max_x > min_x else 1
        scale_y = display_height / (max_y - min_y) if max_y > min_y else 1
        scale = min(scale_x, scale_y) * 0.95

        # Center in display area with padding
        center_x = 50 + display_width / 2
        center_y = y_offset + 50 + display_height / 2
        offset_x = center_x - (min_x + max_x) / 2 * scale
        offset_y = center_y - (min_y + max_y) / 2 * scale

        # Draw seg1 (blue)
        pts1_scaled = np.array([
            (int(p.x * scale + offset_x), int(p.y * scale + offset_y))
            for p in pts1_norm
        ], dtype=np.int32)
        cv2.polylines(canvas, [pts1_scaled], False, (255, 0, 0), 3)  # Blue, thicker

        # Draw seg2 (red) overlaid on top
        pts2_scaled = np.array([
            (int(p.x * scale + offset_x), int(p.y * scale + offset_y))
            for p in pts2_rotated
        ], dtype=np.int32)
        cv2.polylines(canvas, [pts2_scaled], False, (0, 0, 255), 2)  # Red, thinner

        # Draw corner markers
        # Seg1 start/end (blue circles)
        if len(pts1_scaled) > 0:
            cv2.circle(canvas, tuple(pts1_scaled[0]), 6, (255, 0, 0), -1)
            cv2.circle(canvas, tuple(pts1_scaled[-1]), 6, (255, 100, 100), -1)

        # Seg2 start/end (red circles)
        if len(pts2_scaled) > 0:
            cv2.circle(canvas, tuple(pts2_scaled[0]), 6, (0, 0, 255), -1)
            cv2.circle(canvas, tuple(pts2_scaled[-1]), 6, (100, 0, 255), -1)

        # Calculate minimal-area rotated bounding box around both segments
        scaled_points = list(pts1_scaled) + list(pts2_scaled)
        rotated_bbox_area = None
        if scaled_points and len(scaled_points) >= 3:
            # Convert to numpy array for contour operations
            points_array = np.array(scaled_points, dtype=np.float32)

            # Find the minimal-area rotated bounding box
            rotated_rect = cv2.minAreaRect(points_array)
            box_points = cv2.boxPoints(rotated_rect)
            box_points = np.int32(box_points)

            # Calculate the area of the rotated bounding box
            # rotated_rect returns (center, (width, height), angle)
            bbox_width_rot = rotated_rect[1][0]
            bbox_height_rot = rotated_rect[1][1]
            rotated_bbox_area = bbox_width_rot * bbox_height_rot

            # Draw the rotated bounding box
            cv2.polylines(canvas, [box_points], True, (0, 255, 0), 2)

        # Draw visualization area border
        box_x1 = int(50)
        box_y1 = int(y_offset + 50)
        box_x2 = int(50 + display_width)
        box_y2 = int(y_offset + 50 + display_height)
        cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (100, 100, 100), 1)

        # Draw info
        info_y = y_offset + 10
        info_text = f"P{match.piece1_id}-S{match.seg1_id} (Blue) vs P{match.piece2_id}-S{match.seg2_id} (Red)"
        cv2.putText(canvas, info_text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Calculate shape similarity RMSD after optimal rotation
        shape_rmsd = SegmentMatcher._calculate_shape_similarity_rmsd(seg1, seg2)
        # Convert RMSD to a similarity score (0-1, where 1 is perfect, 0 is bad)
        # Using exponential decay: higher RMSD = lower score
        shape_similarity_score = np.exp(-shape_rmsd)

        # Draw scores
        score_text = f"Score: {match.match_score:.3f} | Length: {match.length_score:.3f} | RMSD: {shape_rmsd:.4f} | Similarity: {shape_similarity_score:.3f}"
        cv2.putText(canvas, score_text, (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Draw similarity score bar (visual feedback)
        bar_x_start = 10
        bar_y = info_y + 45
        bar_width = 350
        bar_height = 8

        # Background bar (gray)
        cv2.rectangle(canvas, (bar_x_start, bar_y), (bar_x_start + bar_width, bar_y + bar_height), (100, 100, 100), -1)

        # Foreground bar colored by similarity (red = bad, yellow = medium, green = good)
        filled_width = int(bar_width * shape_similarity_score)
        if shape_similarity_score < 0.33:
            color = (0, 0, 255)  # Red - poor match
        elif shape_similarity_score < 0.66:
            color = (0, 165, 255)  # Orange - moderate match
        else:
            color = (0, 255, 0)  # Green - good match

        if filled_width > 0:
            cv2.rectangle(canvas, (bar_x_start, bar_y), (bar_x_start + filled_width, bar_y + bar_height), color, -1)

        # Label for the bar
        cv2.putText(canvas, "Shape Similarity", (bar_x_start, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        # Draw bounding box area and area-minimizing rotation
        if rotated_bbox_area is not None:
            area_text = f"Area: {rotated_bbox_area:.3f} | Min-Area Rot: {np.degrees(area_minimizing_rotation):.1f}°"
        else:
            area_text = f"Area: N/A"
        cv2.putText(canvas, area_text, (410, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw legend
        cv2.putText(canvas, "Blue = Segment 1", (10, y_offset + 280), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
        cv2.putText(canvas, "Red = Segment 2 (rotated)", (200, y_offset + 280), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.putText(canvas, "Green = Bounding Box", (410, y_offset + 280), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    @staticmethod
    def create_chain_overlay_visualization(piece1: AnalyzedPuzzlePiece,
                                          combined_seg1: ContourSegment,
                                          piece2: AnalyzedPuzzlePiece,
                                          combined_seg2: ContourSegment,
                                          match: SegmentMatch,
                                          chain_length: int,
                                          frame_connection_point1: Point = None,
                                          frame_connection_point2: Point = None) -> np.ndarray:
        """
        Create visualization for chain segments with endpoint alignment.
        The start points (frame connection points) should be exactly on top of each other,
        and the end points should also be aligned.

        Args:
            frame_connection_point1: Where chain1 touches the frame straight edge
            frame_connection_point2: Where chain2 touches the frame straight edge
        """
        # Create a taller canvas with 3 sections: chain1 separate, chain2 separate, overlay
        overlay_height = 1000
        overlay_width = 800
        overlay_canvas = np.ones((overlay_height, overlay_width, 3), dtype=np.uint8) * 40

        # Get points from both chains
        pts1 = combined_seg1.contour_points
        pts2 = combined_seg2.contour_points

        # Resample to same number of points
        target_points = max(len(pts1), len(pts2), 100)
        pts1_resampled = SegmentMatcher._resample_points(pts1, target_points)
        pts2_resampled = SegmentMatcher._resample_points(pts2, target_points)

        # Convert to numpy arrays
        pts1_np = np.array([(p.x, p.y) for p in pts1_resampled])
        pts2_np = np.array([(p.x, p.y) for p in pts2_resampled])

        # Determine which end of each chain is the frame connection point
        # The frame connection point should be closer to the actual frame connection
        if frame_connection_point1 and frame_connection_point2:
            # Calculate distances to determine which end is the start
            dist1_start = np.linalg.norm(pts1_np[0] - np.array([frame_connection_point1.x, frame_connection_point1.y]))
            dist1_end = np.linalg.norm(pts1_np[-1] - np.array([frame_connection_point1.x, frame_connection_point1.y]))

            dist2_start = np.linalg.norm(pts2_np[0] - np.array([frame_connection_point2.x, frame_connection_point2.y]))
            dist2_end = np.linalg.norm(pts2_np[-1] - np.array([frame_connection_point2.x, frame_connection_point2.y]))

            # Reverse chain1 if needed (end is closer to frame connection)
            if dist1_end < dist1_start:
                pts1_np = pts1_np[::-1]

            # Reverse chain2 if needed (end is closer to frame connection)
            if dist2_end < dist2_start:
                pts2_np = pts2_np[::-1]

        # Now both chains have their frame connection point at index 0
        # Align by translating chain2 so its start point matches chain1's start point
        start_point = pts1_np[0]
        offset = start_point - pts2_np[0]
        pts2_translated = pts2_np + offset

        # Find rotation around start point that aligns the end points
        end_point1 = pts1_np[-1]
        end_point2_translated = pts2_translated[-1]

        # Calculate angle to rotate pts2's endpoint to align with pts1's endpoint
        vec1 = end_point1 - start_point
        vec2 = end_point2_translated - start_point
        angle1 = np.arctan2(vec1[1], vec1[0])
        angle2 = np.arctan2(vec2[1], vec2[0])
        rotation_angle = angle1 - angle2

        # Rotate pts2 around the start point
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        pts2_centered = pts2_translated - start_point
        pts2_rotated = pts2_centered @ rotation_matrix.T + start_point

        # Convert back to Point objects
        pts1_aligned = [Point(p[0], p[1]) for p in pts1_np]
        pts2_aligned = [Point(p[0], p[1]) for p in pts2_rotated]

        # =================================================================
        # SECTION 1: Draw Chain 1 separately (top section)
        # =================================================================
        section1_y = 10
        section_height = 280

        # Draw separator line
        cv2.line(overlay_canvas, (0, section1_y + section_height),
                (overlay_width, section1_y + section_height), (100, 100, 100), 2)

        # Calculate bounding box for chain 1 only
        pts1_x = [p.x for p in pts1_aligned]
        pts1_y = [p.y for p in pts1_aligned]
        min_x1, max_x1 = min(pts1_x), max(pts1_x)
        min_y1, max_y1 = min(pts1_y), max(pts1_y)

        padding = 0.2
        width1 = max_x1 - min_x1
        height1 = max_y1 - min_y1
        min_x1 -= width1 * padding
        max_x1 += width1 * padding
        min_y1 -= height1 * padding
        max_y1 += height1 * padding

        display_width1 = 700
        display_height1 = 200
        scale1_x = display_width1 / (max_x1 - min_x1) if max_x1 > min_x1 else 1
        scale1_y = display_height1 / (max_y1 - min_y1) if max_y1 > min_y1 else 1
        scale1 = min(scale1_x, scale1_y)

        center1_x = overlay_width / 2
        center1_y = section1_y + 50 + display_height1 / 2
        offset1_x = center1_x - (min_x1 + max_x1) / 2 * scale1
        offset1_y = center1_y - (min_y1 + max_y1) / 2 * scale1

        pts1_section = np.array([
            (int(p.x * scale1 + offset1_x), int(p.y * scale1 + offset1_y))
            for p in pts1_aligned
        ], dtype=np.int32)
        cv2.polylines(overlay_canvas, [pts1_section], False, (255, 100, 0), 3)

        # Mark start (green) and end (yellow)
        if len(pts1_section) > 0:
            cv2.circle(overlay_canvas, tuple(pts1_section[0]), 8, (0, 255, 0), -1)
            cv2.circle(overlay_canvas, tuple(pts1_section[-1]), 8, (0, 255, 255), -1)

        cv2.putText(overlay_canvas, f"Chain 1 (P{piece1.piece_id})",
                   (10, section1_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # =================================================================
        # SECTION 2: Draw Chain 2 separately (middle section)
        # =================================================================
        section2_y = section1_y + section_height + 10

        # Draw separator line
        cv2.line(overlay_canvas, (0, section2_y + section_height),
                (overlay_width, section2_y + section_height), (100, 100, 100), 2)

        # Calculate bounding box for chain 2 only
        pts2_x = [p.x for p in pts2_aligned]
        pts2_y = [p.y for p in pts2_aligned]
        min_x2, max_x2 = min(pts2_x), max(pts2_x)
        min_y2, max_y2 = min(pts2_y), max(pts2_y)

        width2 = max_x2 - min_x2
        height2 = max_y2 - min_y2
        min_x2 -= width2 * padding
        max_x2 += width2 * padding
        min_y2 -= height2 * padding
        max_y2 += height2 * padding

        scale2_x = display_width1 / (max_x2 - min_x2) if max_x2 > min_x2 else 1
        scale2_y = display_height1 / (max_y2 - min_y2) if max_y2 > min_y2 else 1
        scale2 = min(scale2_x, scale2_y)

        center2_x = overlay_width / 2
        center2_y = section2_y + 50 + display_height1 / 2
        offset2_x = center2_x - (min_x2 + max_x2) / 2 * scale2
        offset2_y = center2_y - (min_y2 + max_y2) / 2 * scale2

        pts2_section = np.array([
            (int(p.x * scale2 + offset2_x), int(p.y * scale2 + offset2_y))
            for p in pts2_aligned
        ], dtype=np.int32)
        cv2.polylines(overlay_canvas, [pts2_section], False, (0, 100, 255), 3)

        # Mark start (green) and end (yellow)
        if len(pts2_section) > 0:
            cv2.circle(overlay_canvas, tuple(pts2_section[0]), 8, (0, 255, 0), -1)
            cv2.circle(overlay_canvas, tuple(pts2_section[-1]), 8, (0, 255, 255), -1)

        cv2.putText(overlay_canvas, f"Chain 2 (P{piece2.piece_id})",
                   (10, section2_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # =================================================================
        # SECTION 3: Draw overlay (bottom section)
        # =================================================================
        section3_y = section2_y + section_height + 10

        # Find bounding box for overlay
        all_x = pts1_x + pts2_x
        all_y = pts1_y + pts2_y
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # Add padding
        width = max_x - min_x
        height = max_y - min_y
        min_x -= width * padding
        max_x += width * padding
        min_y -= height * padding
        max_y += height * padding

        # Scale to fit display
        display_width = 700
        display_height = 300
        scale_x = display_width / (max_x - min_x) if max_x > min_x else 1
        scale_y = display_height / (max_y - min_y) if max_y > min_y else 1
        scale = min(scale_x, scale_y)

        # Center in canvas
        center_x = overlay_width / 2
        center_y = section3_y + 50 + display_height / 2
        offset_x = center_x - (min_x + max_x) / 2 * scale
        offset_y = center_y - (min_y + max_y) / 2 * scale

        # Draw seg1 (blue - thicker)
        pts1_scaled = np.array([
            (int(p.x * scale + offset_x), int(p.y * scale + offset_y))
            for p in pts1_aligned
        ], dtype=np.int32)
        cv2.polylines(overlay_canvas, [pts1_scaled], False, (255, 100, 0), 4)

        # Draw seg2 (red - thinner, on top)
        pts2_scaled = np.array([
            (int(p.x * scale + offset_x), int(p.y * scale + offset_y))
            for p in pts2_aligned
        ], dtype=np.int32)
        cv2.polylines(overlay_canvas, [pts2_scaled], False, (0, 100, 255), 2)

        # Draw start point marker (green circle - should overlap for both)
        if len(pts1_scaled) > 0:
            cv2.circle(overlay_canvas, tuple(pts1_scaled[0]), 8, (0, 255, 0), -1)

        # Draw end point markers (yellow circle - should overlap for both)
        if len(pts1_scaled) > 0:
            cv2.circle(overlay_canvas, tuple(pts1_scaled[-1]), 8, (0, 255, 255), -1)

        # Add text info for overlay section
        cv2.putText(overlay_canvas, f"Overlay (Aligned)",
                   (10, section3_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Legend at bottom
        info_y = 960
        cv2.putText(overlay_canvas, "Legend:", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay_canvas, "Green = Start (Frame Connection)", (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(overlay_canvas, "Yellow = End", (350, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(overlay_canvas, f"Total: {chain_length} segments matched", (550, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return overlay_canvas

    @staticmethod
    def create_progressive_chain_visualization(piece1: AnalyzedPuzzlePiece,
                                               chain_segs1: List[ContourSegment],
                                               piece2: AnalyzedPuzzlePiece,
                                               chain_segs2: List[ContourSegment],
                                               match,
                                               all_matches_in_chain: List,
                                               frame_connection_point1: Point = None,
                                               frame_connection_point2: Point = None) -> np.ndarray:
        """
        Create progressive visualization showing chain building step by step.
        Shows overlay for: 1 segment, 2 segments, 3 segments, ... up to full chain.
        """
        num_segments = len(all_matches_in_chain)

        # Calculate canvas size: show up to 6 stages per row
        stages_per_row = min(6, num_segments)
        num_rows = (num_segments + stages_per_row - 1) // stages_per_row

        stage_width = 400
        stage_height = 300
        canvas_width = stage_width * stages_per_row
        canvas_height = stage_height * num_rows + 100  # Extra space for legend

        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 40

        # For each progressive stage (1 segment, 2 segments, etc.)
        for stage in range(1, num_segments + 1):
            # Calculate position in grid
            row = (stage - 1) // stages_per_row
            col = (stage - 1) % stages_per_row
            x_offset = col * stage_width
            y_offset = row * stage_height

            # Get segments for this stage
            stage_segs1 = chain_segs1[:stage]
            stage_segs2 = chain_segs2[:stage]

            # Combine points for this stage
            stage_points1 = []
            for i, seg in enumerate(stage_segs1):
                if i == 0:
                    stage_points1.extend(seg.contour_points)
                else:
                    stage_points1.extend(seg.contour_points[1:])

            stage_points2 = []
            for i, seg in enumerate(stage_segs2):
                if i == 0:
                    stage_points2.extend(seg.contour_points)
                else:
                    stage_points2.extend(seg.contour_points[1:])

            # Convert to numpy
            pts1_np = np.array([(p.x, p.y) for p in stage_points1])
            pts2_np = np.array([(p.x, p.y) for p in stage_points2])

            # Align: translate pts2 so first point matches pts1
            if len(pts1_np) > 0 and len(pts2_np) > 0:
                start_point = pts1_np[0]
                offset = start_point - pts2_np[0]
                pts2_translated = pts2_np + offset

                # Rotate to align end points
                if len(pts1_np) > 1 and len(pts2_translated) > 1:
                    end_point1 = pts1_np[-1]
                    end_point2_translated = pts2_translated[-1]

                    vec1 = end_point1 - start_point
                    vec2 = end_point2_translated - start_point
                    angle1 = np.arctan2(vec1[1], vec1[0])
                    angle2 = np.arctan2(vec2[1], vec2[0])
                    rotation_angle = angle1 - angle2

                    cos_a = np.cos(rotation_angle)
                    sin_a = np.sin(rotation_angle)
                    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

                    pts2_centered = pts2_translated - start_point
                    pts2_rotated = pts2_centered @ rotation_matrix.T + start_point
                else:
                    pts2_rotated = pts2_translated

                # Find bounding box
                all_x = list(pts1_np[:, 0]) + list(pts2_rotated[:, 0])
                all_y = list(pts1_np[:, 1]) + list(pts2_rotated[:, 1])
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)

                # Add padding
                padding = 0.2
                width = max_x - min_x
                height = max_y - min_y
                min_x -= width * padding
                max_x += width * padding
                min_y -= height * padding
                max_y += height * padding

                # Scale to fit in stage area
                display_width = stage_width - 40
                display_height = stage_height - 60
                scale_x = display_width / (max_x - min_x) if max_x > min_x else 1
                scale_y = display_height / (max_y - min_y) if max_y > min_y else 1
                scale = min(scale_x, scale_y)

                # Center in stage area
                center_x = x_offset + stage_width / 2
                center_y = y_offset + 30 + display_height / 2
                offset_x = center_x - (min_x + max_x) / 2 * scale
                offset_y = center_y - (min_y + max_y) / 2 * scale

                # Draw chain 1 (blue/orange - thick)
                pts1_scaled = np.array([
                    (int(p[0] * scale + offset_x), int(p[1] * scale + offset_y))
                    for p in pts1_np
                ], dtype=np.int32)
                cv2.polylines(canvas, [pts1_scaled], False, (255, 100, 0), 3)

                # Draw chain 2 (red - thin)
                pts2_scaled = np.array([
                    (int(p[0] * scale + offset_x), int(p[1] * scale + offset_y))
                    for p in pts2_rotated
                ], dtype=np.int32)
                cv2.polylines(canvas, [pts2_scaled], False, (0, 100, 255), 2)

                # Mark start and end points
                if len(pts1_scaled) > 0:
                    cv2.circle(canvas, tuple(pts1_scaled[0]), 5, (0, 255, 0), -1)  # Green start
                    cv2.circle(canvas, tuple(pts1_scaled[-1]), 5, (0, 255, 255), -1)  # Yellow end

            # Add stage label
            label = f"Stage {stage}/{num_segments}"
            cv2.putText(canvas, label, (x_offset + 10, y_offset + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw border around stage
            cv2.rectangle(canvas, (x_offset, y_offset),
                         (x_offset + stage_width, y_offset + stage_height),
                         (100, 100, 100), 1)

        # Add legend at bottom
        legend_y = canvas_height - 80
        cv2.putText(canvas, f"Progressive Chain Building: P{piece1.piece_id} <-> P{piece2.piece_id}",
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, "Blue/Orange = Chain 1  |  Red = Chain 2  |  Green = Start  |  Yellow = End",
                   (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return canvas

    @staticmethod
    def create_group_validation_visualization(piece1: AnalyzedPuzzlePiece,
                                              segments1: List[ContourSegment],
                                              piece2: AnalyzedPuzzlePiece,
                                              segments2: List[ContourSegment],
                                              length_matrix: np.ndarray,
                                              shape_matrix: np.ndarray,
                                              rotation_matrix: np.ndarray,
                                              group_tests: List[Tuple]) -> np.ndarray:
        """
        Create visualization for group validation tests with expansion results.

        Args:
            piece1: First puzzle piece
            segments1: Segments from piece 1
            piece2: Second puzzle piece
            segments2: Segments from piece 2
            length_matrix: Length similarity matrix
            shape_matrix: Shape similarity matrix
            rotation_matrix: Rotation angle matrix
            group_tests: List of test tuples (expanded_segs_p1, expanded_segs_p2, expansion_results)
        """
        # Calculate total height based on expansion steps
        total_height = 0
        for test_data in group_tests:
            expanded_p1, expanded_p2, expansion_results = test_data
            total_height += 300 * len(expansion_results)

        overlay_width = 900
        overlay_canvas = np.ones((max(300, total_height), overlay_width, 3), dtype=np.uint8) * 40

        y_offset = 0
        for test_data in group_tests:
            expanded_p1, expanded_p2, expansion_results = test_data

            # Display each expansion step
            for step_idx, expansion_step in enumerate(expansion_results):
                state, length_score, shape_score, angle_score, quality_text = expansion_step

                # Combine segments from both pieces for this step
                from .group_validator import GroupValidator
                group_seg1 = GroupValidator._combine_segments_list(expanded_p1)
                group_seg2 = GroupValidator._combine_segments_list(expanded_p2)

                if group_seg1 and group_seg2:
                    seg_ids_p1 = [s.segment_id for s in expanded_p1]
                    seg_ids_p2 = [s.segment_id for s in expanded_p2]

                    SegmentOverlayVisualizer._draw_group_expansion_overlay(
                        overlay_canvas, group_seg1, group_seg2, piece1.piece_id, piece2.piece_id,
                        seg_ids_p1, seg_ids_p2, state,
                        length_score, shape_score, angle_score, quality_text, y_offset
                    )

                    y_offset += 300

        return overlay_canvas

    @staticmethod
    def _draw_group_test_overlay(canvas: np.ndarray, seg1: ContourSegment, seg2: ContourSegment,
                                piece1_id: int, piece2_id: int,
                                seg1_id: int, seg2_id: int, col_idx: int, group_id: int,
                                length_score: float, shape_score: float, angle_score: float,
                                y_offset: int) -> None:
        """Draw a single group test overlay with segment comparison and test details."""
        # Resample both segments to same number of points for better alignment
        target_points = 50
        pts1_resampled = SegmentMatcher._resample_points(seg1.contour_points, target_points)
        pts2_resampled = SegmentMatcher._resample_points(seg2.contour_points, target_points)

        # Normalize both contours to align them
        pts1_norm = normalize_and_center(pts1_resampled)
        pts2_norm = normalize_and_center(pts2_resampled)

        # Find the rotation angle that minimizes bounding box area
        optimal_area_rotation = find_min_area_rotation(
            pts1_norm, pts2_norm, precision=1
        )

        # Apply the optimal rotation to minimize bounding box area
        pts2_rotated = rotate_points(
            pts2_norm, optimal_area_rotation
        )

        # Find bounding box of both normalized segments (for proper scaling)
        all_points = pts1_norm + pts2_rotated
        min_x = min(p.x for p in all_points)
        max_x = max(p.x for p in all_points)
        min_y = min(p.y for p in all_points)
        max_y = max(p.y for p in all_points)

        # Calculate axis-aligned bounding box dimensions for scaling/padding
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y

        # Add padding
        padding = 0.3
        width = bbox_width if bbox_width > 0 else 2
        height = bbox_height if bbox_height > 0 else 2
        min_x -= width * padding / 2
        max_x += width * padding / 2
        min_y -= height * padding / 2
        max_y += height * padding / 2

        # Scale to fit in display area (350x220 pixels with margins)
        display_width = 350
        display_height = 220
        scale_x = display_width / (max_x - min_x) if max_x > min_x else 1
        scale_y = display_height / (max_y - min_y) if max_y > min_y else 1
        scale = min(scale_x, scale_y) * 0.95

        # Center in display area with padding
        center_x = 50 + display_width / 2
        center_y = y_offset + 50 + display_height / 2
        offset_x = center_x - (min_x + max_x) / 2 * scale
        offset_y = center_y - (min_y + max_y) / 2 * scale

        # Draw seg1 (blue)
        pts1_scaled = np.array([
            (int(p.x * scale + offset_x), int(p.y * scale + offset_y))
            for p in pts1_norm
        ], dtype=np.int32)
        cv2.polylines(canvas, [pts1_scaled], False, (255, 0, 0), 3)  # Blue, thicker

        # Draw seg2 (red) overlaid on top
        pts2_scaled = np.array([
            (int(p.x * scale + offset_x), int(p.y * scale + offset_y))
            for p in pts2_rotated
        ], dtype=np.int32)
        cv2.polylines(canvas, [pts2_scaled], False, (0, 0, 255), 2)  # Red, thinner

        # Draw corner markers
        # Seg1 start/end (blue circles)
        if len(pts1_scaled) > 0:
            cv2.circle(canvas, tuple(pts1_scaled[0]), 6, (255, 0, 0), -1)
            cv2.circle(canvas, tuple(pts1_scaled[-1]), 6, (255, 100, 100), -1)

        # Seg2 start/end (red circles)
        if len(pts2_scaled) > 0:
            cv2.circle(canvas, tuple(pts2_scaled[0]), 6, (0, 0, 255), -1)
            cv2.circle(canvas, tuple(pts2_scaled[-1]), 6, (100, 0, 255), -1)

        # Calculate minimal-area rotated bounding box around both segments
        scaled_points = list(pts1_scaled) + list(pts2_scaled)
        rotated_bbox_area = None
        if scaled_points and len(scaled_points) >= 3:
            # Convert to numpy array for contour operations
            points_array = np.array(scaled_points, dtype=np.float32)

            # Find the minimal-area rotated bounding box
            rotated_rect = cv2.minAreaRect(points_array)
            box_points = cv2.boxPoints(rotated_rect)
            box_points = np.int32(box_points)

            # Calculate the area of the rotated bounding box
            bbox_width_rot = rotated_rect[1][0]
            bbox_height_rot = rotated_rect[1][1]
            rotated_bbox_area = bbox_width_rot * bbox_height_rot

            # Draw the rotated bounding box
            cv2.polylines(canvas, [box_points], True, (0, 255, 0), 2)

        # Draw visualization area border
        box_x1 = int(50)
        box_y1 = int(y_offset + 50)
        box_x2 = int(50 + display_width)
        box_y2 = int(y_offset + 50 + display_height)
        cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (100, 100, 100), 1)

        # Draw header info with piece and segment details
        info_y = y_offset + 10
        info_text = f"GROUP TEST: P{piece1_id}-S{seg1_id}+S{seg2_id} (Blue Group) <-> P{piece2_id}-S{col_idx} (Red) | Diagonal S{col_idx}, Group {group_id}"
        cv2.putText(canvas, info_text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Format score strings with n/a handling
        length_str = f"{length_score:.3f}" if length_score >= 0 else "n/a"
        shape_str = f"{shape_score:.4f}" if shape_score >= 0 else "n/a"
        angle_str = f"{angle_score:.2f}°" if not np.isnan(angle_score) else "n/a"

        # Draw scores
        score_text = f"Length: {length_str} | Shape RMSD: {shape_str} | Rotation: {angle_str}"
        cv2.putText(canvas, score_text, (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Determine overall quality color
        quality_color = (100, 100, 100)  # Default gray
        if shape_score >= 0:
            if shape_score <= 0.05:
                quality_color = (0, 255, 0)  # Green - excellent
                quality_text = "EXCELLENT"
            elif shape_score <= 0.10:
                quality_color = (0, 200, 100)  # Light green - good
                quality_text = "GOOD"
            elif shape_score <= 0.15:
                quality_color = (0, 165, 255)  # Orange - moderate
                quality_text = "MODERATE"
            else:
                quality_color = (0, 0, 255)  # Red - poor
                quality_text = "POOR"
        else:
            quality_text = "N/A"

        # Draw quality indicator
        quality_bar_x = 410
        quality_bar_y = info_y
        cv2.rectangle(canvas, (quality_bar_x, quality_bar_y - 5), (quality_bar_x + 80, quality_bar_y + 20), quality_color, -1)
        cv2.putText(canvas, f"Quality", (quality_bar_x + 5, quality_bar_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # Draw quality text
        quality_y = info_y + 25
        cv2.putText(canvas, f"Match Quality: {quality_text}", (410, quality_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, quality_color, 1)

        # Draw bounding box area
        if rotated_bbox_area is not None:
            area_text = f"Area: {rotated_bbox_area:.3f} | Rotation: {np.degrees(optimal_area_rotation):.1f}°"
        else:
            area_text = f"Area: N/A"
        cv2.putText(canvas, area_text, (410, quality_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw legend
        cv2.putText(canvas, "Blue = Combined Group Segment (Piece 1)", (10, y_offset + 280), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
        cv2.putText(canvas, "Red = Piece 2 Segment (rotated)", (320, y_offset + 280), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.putText(canvas, "Green = Bounding Box", (650, y_offset + 280), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    @staticmethod
    def _draw_group_test_overlay_both_groups(canvas: np.ndarray, seg1: ContourSegment, seg2: ContourSegment,
                                            piece1_id: int, piece2_id: int,
                                            seg1a_id: int, seg1b_id: int, seg2a_id: int, seg2b_id: int,
                                            length_score: float, shape_score: float, angle_score: float,
                                            y_offset: int) -> None:
        """Draw a single group test overlay comparing two group segments from both pieces."""
        # Resample both segments to same number of points for better alignment
        target_points = 50
        pts1_resampled = SegmentMatcher._resample_points(seg1.contour_points, target_points)
        pts2_resampled = SegmentMatcher._resample_points(seg2.contour_points, target_points)

        # Normalize both contours to align them
        pts1_norm = normalize_and_center(pts1_resampled)
        pts2_norm = normalize_and_center(pts2_resampled)

        # Find the rotation angle that minimizes bounding box area
        optimal_area_rotation = find_min_area_rotation(
            pts1_norm, pts2_norm, precision=1
        )

        # Apply the optimal rotation to minimize bounding box area
        pts2_rotated = rotate_points(
            pts2_norm, optimal_area_rotation
        )

        # Find bounding box of both normalized segments (for proper scaling)
        all_points = pts1_norm + pts2_rotated
        min_x = min(p.x for p in all_points)
        max_x = max(p.x for p in all_points)
        min_y = min(p.y for p in all_points)
        max_y = max(p.y for p in all_points)

        # Calculate axis-aligned bounding box dimensions for scaling/padding
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y

        # Add padding
        padding = 0.3
        width = bbox_width if bbox_width > 0 else 2
        height = bbox_height if bbox_height > 0 else 2
        min_x -= width * padding / 2
        max_x += width * padding / 2
        min_y -= height * padding / 2
        max_y += height * padding / 2

        # Scale to fit in display area (350x220 pixels with margins)
        display_width = 350
        display_height = 220
        scale_x = display_width / (max_x - min_x) if max_x > min_x else 1
        scale_y = display_height / (max_y - min_y) if max_y > min_y else 1
        scale = min(scale_x, scale_y) * 0.95

        # Center in display area with padding
        center_x = 50 + display_width / 2
        center_y = y_offset + 50 + display_height / 2
        offset_x = center_x - (min_x + max_x) / 2 * scale
        offset_y = center_y - (min_y + max_y) / 2 * scale

        # Draw seg1 (blue)
        pts1_scaled = np.array([
            (int(p.x * scale + offset_x), int(p.y * scale + offset_y))
            for p in pts1_norm
        ], dtype=np.int32)
        cv2.polylines(canvas, [pts1_scaled], False, (255, 0, 0), 3)  # Blue, thicker

        # Draw seg2 (red) overlaid on top
        pts2_scaled = np.array([
            (int(p.x * scale + offset_x), int(p.y * scale + offset_y))
            for p in pts2_rotated
        ], dtype=np.int32)
        cv2.polylines(canvas, [pts2_scaled], False, (0, 0, 255), 2)  # Red, thinner

        # Draw corner markers
        # Seg1 start/end (blue circles)
        if len(pts1_scaled) > 0:
            cv2.circle(canvas, tuple(pts1_scaled[0]), 6, (255, 0, 0), -1)
            cv2.circle(canvas, tuple(pts1_scaled[-1]), 6, (255, 100, 100), -1)

        # Seg2 start/end (red circles)
        if len(pts2_scaled) > 0:
            cv2.circle(canvas, tuple(pts2_scaled[0]), 6, (0, 0, 255), -1)
            cv2.circle(canvas, tuple(pts2_scaled[-1]), 6, (100, 0, 255), -1)

        # Calculate minimal-area rotated bounding box around both segments
        scaled_points = list(pts1_scaled) + list(pts2_scaled)
        rotated_bbox_area = None
        if scaled_points and len(scaled_points) >= 3:
            # Convert to numpy array for contour operations
            points_array = np.array(scaled_points, dtype=np.float32)

            # Find the minimal-area rotated bounding box
            rotated_rect = cv2.minAreaRect(points_array)
            box_points = cv2.boxPoints(rotated_rect)
            box_points = np.int32(box_points)

            # Calculate the area of the rotated bounding box
            bbox_width_rot = rotated_rect[1][0]
            bbox_height_rot = rotated_rect[1][1]
            rotated_bbox_area = bbox_width_rot * bbox_height_rot

            # Draw the rotated bounding box
            cv2.polylines(canvas, [box_points], True, (0, 255, 0), 2)

        # Draw visualization area border
        box_x1 = int(50)
        box_y1 = int(y_offset + 50)
        box_x2 = int(50 + display_width)
        box_y2 = int(y_offset + 50 + display_height)
        cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (100, 100, 100), 1)

        # Draw header info with piece and segment details
        info_y = y_offset + 10
        info_text = f"GROUP-TO-GROUP: P{piece1_id}-(S{seg1a_id}+S{seg1b_id}) (Blue) <-> P{piece2_id}-(S{seg2a_id}+S{seg2b_id}) (Red)"
        cv2.putText(canvas, info_text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Format score strings with n/a handling
        length_str = f"{length_score:.3f}" if length_score >= 0 else "n/a"
        shape_str = f"{shape_score:.4f}" if shape_score >= 0 else "n/a"
        angle_str = f"{angle_score:.2f}°" if not np.isnan(angle_score) else "n/a"

        # Draw scores
        score_text = f"Length: {length_str} | Shape RMSD: {shape_str} | Rotation: {angle_str}"
        cv2.putText(canvas, score_text, (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Determine overall quality color
        quality_color = (100, 100, 100)  # Default gray
        if shape_score >= 0:
            if shape_score <= 0.05:
                quality_color = (0, 255, 0)  # Green - excellent
                quality_text = "EXCELLENT"
            elif shape_score <= 0.10:
                quality_color = (0, 200, 100)  # Light green - good
                quality_text = "GOOD"
            elif shape_score <= 0.15:
                quality_color = (0, 165, 255)  # Orange - moderate
                quality_text = "MODERATE"
            else:
                quality_color = (0, 0, 255)  # Red - poor
                quality_text = "POOR"
        else:
            quality_text = "N/A"

        # Draw quality indicator
        quality_bar_x = 410
        quality_bar_y = info_y
        cv2.rectangle(canvas, (quality_bar_x, quality_bar_y - 5), (quality_bar_x + 80, quality_bar_y + 20), quality_color, -1)
        cv2.putText(canvas, f"Quality", (quality_bar_x + 5, quality_bar_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # Draw quality text
        quality_y = info_y + 25
        cv2.putText(canvas, f"Match Quality: {quality_text}", (410, quality_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, quality_color, 1)

        # Draw bounding box area
        if rotated_bbox_area is not None:
            area_text = f"Area: {rotated_bbox_area:.3f} | Rotation: {np.degrees(optimal_area_rotation):.1f}°"
        else:
            area_text = f"Area: N/A"
        cv2.putText(canvas, area_text, (410, quality_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw legend
        cv2.putText(canvas, "Blue = Group Seg (Piece 1: S"+str(seg1a_id)+"+S"+str(seg1b_id)+")", (10, y_offset + 280), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 0, 0), 1)
        cv2.putText(canvas, "Red = Group Seg (Piece 2: S"+str(seg2a_id)+"+S"+str(seg2b_id)+") (rotated)", (320, y_offset + 280), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
        cv2.putText(canvas, "Green = Bounding Box", (650, y_offset + 280), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    @staticmethod
    def _draw_group_expansion_overlay(canvas: np.ndarray, seg1: ContourSegment, seg2: ContourSegment,
                                     piece1_id: int, piece2_id: int,
                                     seg_ids_p1: List[int], seg_ids_p2: List[int], expansion_state: str,
                                     length_score: float, shape_score: float, angle_score: float, quality_text: str,
                                     y_offset: int) -> None:
        """Draw a group expansion step visualization."""
        # Resample both segments to same number of points for better alignment
        target_points = 50
        pts1_resampled = SegmentMatcher._resample_points(seg1.contour_points, target_points)
        pts2_resampled = SegmentMatcher._resample_points(seg2.contour_points, target_points)

        # Normalize both contours to align them
        pts1_norm = normalize_and_center(pts1_resampled)
        pts2_norm = normalize_and_center(pts2_resampled)

        # Find the rotation angle that minimizes bounding box area
        optimal_area_rotation = find_min_area_rotation(
            pts1_norm, pts2_norm, precision=1
        )

        # Apply the optimal rotation to minimize bounding box area
        pts2_rotated = rotate_points(
            pts2_norm, optimal_area_rotation
        )

        # Find bounding box of both normalized segments
        all_points = pts1_norm + pts2_rotated
        min_x = min(p.x for p in all_points)
        max_x = max(p.x for p in all_points)
        min_y = min(p.y for p in all_points)
        max_y = max(p.y for p in all_points)

        # Calculate bounding box dimensions
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y

        # Add padding
        padding = 0.3
        width = bbox_width if bbox_width > 0 else 2
        height = bbox_height if bbox_height > 0 else 2
        min_x -= width * padding / 2
        max_x += width * padding / 2
        min_y -= height * padding / 2
        max_y += height * padding / 2

        # Scale to fit in display area
        display_width = 350
        display_height = 220
        scale_x = display_width / (max_x - min_x) if max_x > min_x else 1
        scale_y = display_height / (max_y - min_y) if max_y > min_y else 1
        scale = min(scale_x, scale_y) * 0.95

        # Center in display area
        center_x = 50 + display_width / 2
        center_y = y_offset + 50 + display_height / 2
        offset_x = center_x - (min_x + max_x) / 2 * scale
        offset_y = center_y - (min_y + max_y) / 2 * scale

        # Draw seg1 (blue)
        pts1_scaled = np.array([
            (int(p.x * scale + offset_x), int(p.y * scale + offset_y))
            for p in pts1_norm
        ], dtype=np.int32)
        cv2.polylines(canvas, [pts1_scaled], False, (255, 0, 0), 3)

        # Draw seg2 (red)
        pts2_scaled = np.array([
            (int(p.x * scale + offset_x), int(p.y * scale + offset_y))
            for p in pts2_rotated
        ], dtype=np.int32)
        cv2.polylines(canvas, [pts2_scaled], False, (0, 0, 255), 2)

        # Draw corner markers
        if len(pts1_scaled) > 0:
            cv2.circle(canvas, tuple(pts1_scaled[0]), 6, (255, 0, 0), -1)
            cv2.circle(canvas, tuple(pts1_scaled[-1]), 6, (255, 100, 100), -1)

        if len(pts2_scaled) > 0:
            cv2.circle(canvas, tuple(pts2_scaled[0]), 6, (0, 0, 255), -1)
            cv2.circle(canvas, tuple(pts2_scaled[-1]), 6, (100, 0, 255), -1)

        # Calculate bounding box
        scaled_points = list(pts1_scaled) + list(pts2_scaled)
        rotated_bbox_area = None
        if scaled_points and len(scaled_points) >= 3:
            points_array = np.array(scaled_points, dtype=np.float32)
            rotated_rect = cv2.minAreaRect(points_array)
            box_points = cv2.boxPoints(rotated_rect)
            box_points = np.int32(box_points)
            bbox_width_rot = rotated_rect[1][0]
            bbox_height_rot = rotated_rect[1][1]
            rotated_bbox_area = bbox_width_rot * bbox_height_rot
            cv2.polylines(canvas, [box_points], True, (0, 255, 0), 2)

        # Draw visualization area border
        box_x1 = int(50)
        box_y1 = int(y_offset + 50)
        box_x2 = int(50 + display_width)
        box_y2 = int(y_offset + 50 + display_height)
        cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (100, 100, 100), 1)

        # Draw header with expansion state
        info_y = y_offset + 10
        seg_str_p1 = "+".join(str(s) for s in seg_ids_p1)
        seg_str_p2 = "+".join(str(s) for s in seg_ids_p2)
        info_text = f"EXPANSION [{expansion_state}]: P{piece1_id}-(S{seg_str_p1}) (Blue) <-> P{piece2_id}-(S{seg_str_p2}) (Red)"
        cv2.putText(canvas, info_text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Format scores
        length_str = f"{length_score:.3f}" if length_score >= 0 else "n/a"
        shape_str = f"{shape_score:.4f}" if shape_score >= 0 else "n/a"
        angle_str = f"{angle_score:.2f}°" if not np.isnan(angle_score) else "n/a"

        # Draw scores
        score_text = f"Length: {length_str} | Shape RMSD: {shape_str} | Rotation: {angle_str}"
        cv2.putText(canvas, score_text, (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Determine quality color
        quality_color = (100, 100, 100)
        if quality_text == "EXCELLENT":
            quality_color = (0, 255, 0)
        elif quality_text == "GOOD":
            quality_color = (0, 200, 100)
        elif quality_text == "MODERATE":
            quality_color = (0, 165, 255)
        elif quality_text == "POOR":
            quality_color = (0, 0, 255)

        # Draw quality indicator
        quality_bar_x = 410
        quality_bar_y = info_y
        cv2.rectangle(canvas, (quality_bar_x, quality_bar_y - 5), (quality_bar_x + 80, quality_bar_y + 20), quality_color, -1)
        cv2.putText(canvas, "Quality", (quality_bar_x + 5, quality_bar_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # Draw quality text
        quality_y = info_y + 25
        cv2.putText(canvas, f"Match Quality: {quality_text}", (410, quality_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, quality_color, 1)

        # Draw bounding box area
        if rotated_bbox_area is not None:
            area_text = f"Area: {rotated_bbox_area:.3f} | Rotation: {np.degrees(optimal_area_rotation):.1f}°"
        else:
            area_text = "Area: N/A"
        cv2.putText(canvas, area_text, (410, quality_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw legend
        cv2.putText(canvas, f"Blue = Piece 1 Group (S{seg_str_p1})", (10, y_offset + 280), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 0, 0), 1)
        cv2.putText(canvas, f"Red = Piece 2 Group (S{seg_str_p2}) (rotated)", (320, y_offset + 280), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
        cv2.putText(canvas, "Green = Bounding Box", (650, y_offset + 280), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
