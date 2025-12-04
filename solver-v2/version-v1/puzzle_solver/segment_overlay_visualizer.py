"""Visualize segment overlays for debugging matching."""
import cv2
import numpy as np
from typing import List, Tuple
from .data_loader import AnalyzedPuzzlePiece, Point
from .contour_segmenter import ContourSegment
from .segment_matcher import SegmentMatcher, SegmentMatch
from .utils import normalize_and_center, rotate_points, find_min_area_rotation


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
