"""Segment visualization module for edge solver v2."""

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from .visualizer_helpers import VisualizerHelpers
from .geometry_utils import GeometryUtils


class SegmentVisualizer:
    """Handles segment pair visualizations."""

    @staticmethod
    def visualize_segment_pairs(piece1, segments1, frame_segs1, piece2, segments2, frame_segs2, original_image):
        """Create a visualization showing all frame-adjacent segment pairs between two pieces.

        Args:
            piece1: First piece object
            segments1: All segments for piece1
            frame_segs1: Frame-adjacent segments for piece1
            piece2: Second piece object
            segments2: All segments for piece2
            frame_segs2: Frame-adjacent segments for piece2
            original_image: Original puzzle image

        Returns:
            Tuple of (grid_image, matches, segment_data) where:
                - grid_image: Image showing all segment pair combinations
                - matches: List of SegmentMatch objects
                - segment_data: Dict with aligned segment data for chain matching
        """
        # Calculate grid dimensions
        num_pairs = len(frame_segs1) * len(frame_segs2)
        cols = len(frame_segs2)
        rows = len(frame_segs1)

        # Create individual pair visualizations and collect match data
        pair_images = []
        matches = []
        segment_data = {}

        for seg1 in frame_segs1:
            for seg2 in frame_segs2:
                pair_img, match, seg_data = SegmentVisualizer._visualize_single_pair(
                    piece1, segments1, seg1, piece2, segments2, seg2, original_image
                )
                pair_images.append(pair_img)
                if match:
                    matches.append(match)
                if seg_data:
                    segment_data.update(seg_data)

        # Get dimensions of individual pair images
        if not pair_images:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        # Find max dimensions to create uniform grid cells
        max_pair_h = max(img.shape[0] for img in pair_images)
        max_pair_w = max(img.shape[1] for img in pair_images)

        # Add margins and labels
        margin = 20
        label_height = 40
        legend_height = 105

        # Create grid image
        grid_h = rows * (max_pair_h + margin) + margin + label_height + legend_height
        grid_w = cols * (max_pair_w + margin) + margin
        grid_image = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240

        # Add title
        title = f"Segment Pairs: Piece {piece1.piece_id} vs Piece {piece2.piece_id}"
        cv2.putText(grid_image, title, (margin, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        # Add legend
        legend_y = label_height + 10
        legend_x = margin
        cv2.putText(grid_image, "Legend:", (legend_x, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Blue dot legend
        cv2.circle(grid_image, (legend_x + 10, legend_y + 25), 6, (255, 0, 0), -1)
        cv2.putText(grid_image, "Frame connection point (touches frame corner)",
                    (legend_x + 25, legend_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Red dot legend
        cv2.circle(grid_image, (legend_x + 10, legend_y + 50), 6, (0, 0, 255), -1)
        cv2.putText(grid_image, "Interior end point (away from frame)",
                    (legend_x + 25, legend_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Outward arrow legend
        arrow_start = (legend_x + 10, legend_y + 75)
        arrow_end = (legend_x + 25, legend_y + 75)
        cv2.arrowedLine(grid_image, arrow_start, arrow_end, (255, 0, 0), 2, tipLength=0.4)
        cv2.putText(grid_image, "Outward arrow (shows outside/frame-facing direction)",
                    (legend_x + 30, legend_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Place pair images in grid (centered in their cells if smaller than max)
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx < len(pair_images):
                    pair_img = pair_images[idx]
                    pair_h, pair_w = pair_img.shape[:2]

                    # Calculate position (center in cell)
                    y_cell = row * (max_pair_h + margin) + margin + label_height + legend_height
                    x_cell = col * (max_pair_w + margin) + margin

                    # Center the image in the cell
                    y_offset = (max_pair_h - pair_h) // 2
                    x_offset = (max_pair_w - pair_w) // 2

                    y = y_cell + y_offset
                    x = x_cell + x_offset

                    grid_image[y:y+pair_h, x:x+pair_w] = pair_img
                    idx += 1

        return grid_image, matches, segment_data

    @staticmethod
    def _visualize_single_pair(piece1, segments1, seg1, piece2, segments2, seg2, original_image):
        """Create visualization for a single segment pair.

        Shows both segments highlighted on their respective pieces side by side,
        plus an overlay showing them aligned by their endpoints.
        """
        # Create side-by-side visualization
        img1 = SegmentVisualizer._draw_segment_on_piece(piece1, segments1, seg1, original_image)
        img2 = SegmentVisualizer._draw_segment_on_piece(piece2, segments2, seg2, original_image)

        # Resize to same height if needed
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        target_h = min(h1, h2, 200)  # Reduced height to fit overlay below

        scale1 = target_h / h1
        scale2 = target_h / h2

        img1_resized = cv2.resize(img1, (int(w1 * scale1), target_h))
        img2_resized = cv2.resize(img2, (int(w2 * scale2), target_h))

        # Add separator
        separator = np.ones((target_h, 10, 3), dtype=np.uint8) * 200

        # Combine side by side
        side_by_side = np.hstack([img1_resized, separator, img2_resized])

        # Create raw segment visualization (just the segments without pieces)
        raw_segments_img = SegmentVisualizer._visualize_raw_segments(
            piece1, segments1, seg1, piece2, segments2, seg2
        )

        # Create overlay visualization and get match data
        overlay_img, match, segment_data = SegmentVisualizer._create_segment_overlay(
            piece1, segments1, seg1, piece2, segments2, seg2
        )

        # Resize raw segments and overlay to match width of side-by-side
        overlay_target_w = side_by_side.shape[1]

        # Resize raw segments
        raw_h, raw_w = raw_segments_img.shape[:2]
        raw_scale = overlay_target_w / raw_w
        raw_target_h = int(raw_h * raw_scale)
        raw_resized = cv2.resize(raw_segments_img, (overlay_target_w, raw_target_h))

        # Resize overlay
        overlay_h, overlay_w = overlay_img.shape[:2]
        overlay_scale = overlay_target_w / overlay_w
        overlay_target_h = int(overlay_h * overlay_scale)
        overlay_resized = cv2.resize(overlay_img, (overlay_target_w, overlay_target_h))

        # Add labels
        label_height = 25
        raw_label_height = 20
        overlay_label_height = 20
        separator_height = 5

        total_height = (label_height + target_h + separator_height +
                       raw_label_height + raw_target_h + separator_height +
                       overlay_label_height + overlay_target_h)
        labeled = np.ones((total_height, side_by_side.shape[1], 3), dtype=np.uint8) * 255

        # Add main label
        label_text = f"P{piece1.piece_id}-S{seg1.segment_id}  vs  P{piece2.piece_id}-S{seg2.segment_id}"
        cv2.putText(labeled, label_text, (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Place side-by-side view
        y_offset = label_height
        labeled[y_offset:y_offset + target_h, :] = side_by_side

        # Add raw segments label
        y_offset += target_h + separator_height
        cv2.putText(labeled, "Raw segments (verification):", (10, y_offset + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Place raw segments view
        y_offset += raw_label_height
        labeled[y_offset:y_offset + raw_target_h, :] = raw_resized

        # Add overlay label
        y_offset += raw_target_h + separator_height
        cv2.putText(labeled, "Overlay (aligned by endpoints):", (10, y_offset + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Place overlay view
        y_offset += overlay_label_height
        labeled[y_offset:y_offset + overlay_target_h, :] = overlay_resized

        return labeled, match, segment_data

    @staticmethod
    def _visualize_raw_segments(piece1, segments1, seg1, piece2, segments2, seg2):
        """Visualize just the raw segments side by side to verify endpoint marking.

        Shows the segments extracted from their pieces with blue (frame) and red (interior) dots.
        """
        # Find frame connection points for both segments
        frame_touching_ids1 = VisualizerHelpers._find_frame_touching_segment_ids(piece1, segments1)
        frame_touching_ids2 = VisualizerHelpers._find_frame_touching_segment_ids(piece2, segments2)

        # Determine which end connects to frame for seg1
        current_id1 = seg1.segment_id
        num_segments1 = len(segments1)
        prev_id1 = (current_id1 - 1) % num_segments1
        next_id1 = (current_id1 + 1) % num_segments1
        prev_is_frame1 = prev_id1 in frame_touching_ids1
        next_is_frame1 = next_id1 in frame_touching_ids1

        if prev_is_frame1:
            seg1_frame_idx = 0
            seg1_interior_idx = -1
        elif next_is_frame1:
            seg1_frame_idx = -1
            seg1_interior_idx = 0
        else:
            seg1_frame_idx = 0
            seg1_interior_idx = -1

        # Determine which end connects to frame for seg2
        current_id2 = seg2.segment_id
        num_segments2 = len(segments2)
        prev_id2 = (current_id2 - 1) % num_segments2
        next_id2 = (current_id2 + 1) % num_segments2
        prev_is_frame2 = prev_id2 in frame_touching_ids2
        next_is_frame2 = next_id2 in frame_touching_ids2

        if prev_is_frame2:
            seg2_frame_idx = 0
            seg2_interior_idx = -1
        elif next_is_frame2:
            seg2_frame_idx = -1
            seg2_interior_idx = 0
        else:
            seg2_frame_idx = 0
            seg2_interior_idx = -1

        # Get segment points
        seg1_points = np.array([[p.x, p.y] for p in seg1.contour_points], dtype=np.float64)
        seg2_points = np.array([[p.x, p.y] for p in seg2.contour_points], dtype=np.float64)

        # Create individual segment visualizations
        seg1_img = SegmentVisualizer._draw_raw_segment(seg1_points, seg1_frame_idx, seg1_interior_idx, f"P{piece1.piece_id}-S{seg1.segment_id}", piece1)
        seg2_img = SegmentVisualizer._draw_raw_segment(seg2_points, seg2_frame_idx, seg2_interior_idx, f"P{piece2.piece_id}-S{seg2.segment_id}", piece2)

        # Combine side by side
        max_h = max(seg1_img.shape[0], seg2_img.shape[0])

        # Pad to same height
        if seg1_img.shape[0] < max_h:
            pad = np.ones((max_h - seg1_img.shape[0], seg1_img.shape[1], 3), dtype=np.uint8) * 255
            seg1_img = np.vstack([seg1_img, pad])
        if seg2_img.shape[0] < max_h:
            pad = np.ones((max_h - seg2_img.shape[0], seg2_img.shape[1], 3), dtype=np.uint8) * 255
            seg2_img = np.vstack([seg2_img, pad])

        separator = np.ones((max_h, 10, 3), dtype=np.uint8) * 200
        combined = np.hstack([seg1_img, separator, seg2_img])

        return combined

    @staticmethod
    def _draw_raw_segment(seg_points, frame_idx, interior_idx, label, piece=None):
        """Draw a single raw segment with frame and interior markers.

        Args:
            seg_points: Segment points
            frame_idx: Index of frame endpoint
            interior_idx: Index of interior endpoint
            label: Label text
            piece: Piece object (optional, for calculating correct arrow direction)
        """
        # Get bounding box
        min_x, min_y = seg_points.min(axis=0) - 20
        max_x, max_y = seg_points.max(axis=0) + 20

        canvas_w = int(max_x - min_x)
        canvas_h = int(max_y - min_y)
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        # Offset for drawing
        offset = np.array([-min_x, -min_y])
        seg_canvas = (seg_points + offset).astype(np.int32)

        # Draw segment in green
        cv2.polylines(canvas, [seg_canvas], False, (0, 200, 0), 2)

        # Draw outward arrow at segment midpoint
        mid_idx = len(seg_canvas) // 2
        mid_point = seg_canvas[mid_idx]
        mid_point_original = seg_points[mid_idx]

        # Calculate tangent
        if mid_idx == 0:
            tangent = seg_canvas[mid_idx + 1] - seg_canvas[mid_idx]
        elif mid_idx == len(seg_canvas) - 1:
            tangent = seg_canvas[mid_idx] - seg_canvas[mid_idx - 1]
        else:
            tangent = seg_canvas[mid_idx + 1] - seg_canvas[mid_idx - 1]

        # Normal perpendicular to tangent
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        normal_length = np.linalg.norm(normal)
        if normal_length > 0:
            normal = normal / normal_length

            # If we have piece info, determine correct outward direction
            if piece is not None:
                piece_contour = np.array([[p.x, p.y] for p in piece.contour_points], dtype=np.int32)

                # Verify arrow points OUTSIDE by checking if arrow endpoint is inside contour
                arrow_length = 25
                test_arrow_end = mid_point_original + normal * arrow_length

                # Use cv2.pointPolygonTest: positive if inside, 0 if on edge, negative if outside
                is_inside = cv2.pointPolygonTest(piece_contour, tuple(test_arrow_end.astype(float)), False)

                # If arrow points inside the piece, flip it
                if is_inside > 0:
                    normal = -normal
            else:
                arrow_length = 25

            arrow_end = mid_point + normal * arrow_length
            cv2.arrowedLine(canvas, tuple(mid_point.astype(np.int32)),
                          tuple(arrow_end.astype(np.int32)), (0, 0, 255), 2, tipLength=0.3)

        # Mark frame connection point (blue)
        frame_point = seg_canvas[frame_idx]
        cv2.circle(canvas, tuple(frame_point), 7, (255, 0, 0), -1)
        cv2.circle(canvas, tuple(frame_point), 9, (255, 0, 0), 2)

        # Mark interior endpoint (red)
        interior_point = seg_canvas[interior_idx]
        cv2.circle(canvas, tuple(interior_point), 7, (0, 0, 255), -1)
        cv2.circle(canvas, tuple(interior_point), 9, (0, 0, 255), 2)

        # Add label
        cv2.putText(canvas, label, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, f"Blue=frame, Red=interior", (10, canvas_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        return canvas

    @staticmethod
    def _create_segment_overlay(piece1, segments1, seg1, piece2, segments2, seg2):
        """Create an overlay visualization showing two segments aligned by their endpoints.

        Args:
            piece1: First piece object
            segments1: All segments for piece1
            seg1: First segment (frame-adjacent)
            piece2: Second piece object
            segments2: All segments for piece2
            seg2: Second segment (frame-adjacent)

        Returns:
            Image showing both segments overlaid and aligned by endpoints
        """
        # Find frame connection points for both segments
        frame_touching_ids1 = VisualizerHelpers._find_frame_touching_segment_ids(piece1, segments1)
        frame_touching_ids2 = VisualizerHelpers._find_frame_touching_segment_ids(piece2, segments2)

        # Determine which end connects to frame for seg1
        current_id1 = seg1.segment_id
        num_segments1 = len(segments1)
        prev_id1 = (current_id1 - 1) % num_segments1
        next_id1 = (current_id1 + 1) % num_segments1
        prev_is_frame1 = prev_id1 in frame_touching_ids1
        next_is_frame1 = next_id1 in frame_touching_ids1

        # Get segment points
        seg1_points = np.array([[p.x, p.y] for p in seg1.contour_points], dtype=np.float64)
        seg2_points = np.array([[p.x, p.y] for p in seg2.contour_points], dtype=np.float64)

        # Determine frame and interior points for seg1
        # If previous segment touches frame, then start_corner (index 0) is the frame point
        # If next segment touches frame, then end_corner (index -1) is the frame point
        if prev_is_frame1:
            seg1_frame_idx = 0  # Start is frame connection
            seg1_interior_idx = -1  # End is interior
        elif next_is_frame1:
            seg1_frame_idx = -1  # End is frame connection
            seg1_interior_idx = 0  # Start is interior
        else:
            # Fallback
            seg1_frame_idx = 0
            seg1_interior_idx = -1

        # Determine which end connects to frame for seg2
        current_id2 = seg2.segment_id
        num_segments2 = len(segments2)
        prev_id2 = (current_id2 - 1) % num_segments2
        next_id2 = (current_id2 + 1) % num_segments2
        prev_is_frame2 = prev_id2 in frame_touching_ids2
        next_is_frame2 = next_id2 in frame_touching_ids2

        # Determine frame and interior points for seg2
        if prev_is_frame2:
            seg2_frame_idx = 0  # Start is frame connection
            seg2_interior_idx = -1  # End is interior
        elif next_is_frame2:
            seg2_frame_idx = -1  # End is frame connection
            seg2_interior_idx = 0  # Start is interior
        else:
            # Fallback
            seg2_frame_idx = 0
            seg2_interior_idx = -1

        # Get the blue (frame) and red (interior) endpoints for both segments
        B1 = seg1_points[seg1_frame_idx]
        R1 = seg1_points[seg1_interior_idx]
        B2 = seg2_points[seg2_frame_idx]
        R2 = seg2_points[seg2_interior_idx]

        # Calculate arrow directions BEFORE any transformations using GeometryUtils
        seg1_mid, seg1_normal = GeometryUtils.calculate_arrow_for_segment(seg1_points, piece1)
        seg2_mid, seg2_normal = GeometryUtils.calculate_arrow_for_segment(seg2_points, piece2)

        # Step 1: Translate seg2 so blue endpoints match
        translation = B1 - B2
        seg2_translated = seg2_points + translation
        R2_translated = R2 + translation
        seg2_mid_translated = seg2_mid + translation

        # Step 2: Rotate seg2 around B1 so red points align
        v1 = R1 - B1
        v2 = R2_translated - B1

        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        rotation_angle = angle1 - angle2

        cos_r = np.cos(rotation_angle)
        sin_r = np.sin(rotation_angle)

        def rotate_point(p, origin):
            px, py = p - origin
            x = cos_r * px - sin_r * py + origin[0]
            y = sin_r * px + cos_r * py + origin[1]
            return np.array([x, y])

        seg2_aligned = np.array([rotate_point(p, B1) for p in seg2_translated])
        R2_aligned = rotate_point(R2_translated, B1)
        seg2_mid_aligned = rotate_point(seg2_mid_translated, B1)

        # Rotate the arrow direction vector as well
        seg2_normal_rotated = np.array([
            cos_r * seg2_normal[0] - sin_r * seg2_normal[1],
            sin_r * seg2_normal[0] + cos_r * seg2_normal[1]
        ])

        # Seg1 stays as-is (no translation needed)
        seg1_final = seg1_points
        seg2_final = seg2_aligned

        # Calculate scoring metrics before creating canvas
        # 1. Length difference score
        seg1_length = np.sum(np.linalg.norm(np.diff(seg1_points, axis=0), axis=1))
        seg2_length = np.sum(np.linalg.norm(np.diff(seg2_points, axis=0), axis=1))
        length_diff = abs(seg1_length - seg2_length)
        length_score = max(0, 100 - (length_diff / max(seg1_length, seg2_length) * 100))

        # 2. Arrow direction alignment score
        # Calculate angle between the two arrow direction vectors
        dot_product = np.dot(seg1_normal, seg2_normal_rotated)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Clip to avoid numerical errors
        angle_between = np.degrees(np.arccos(dot_product))

        # Good match: arrows point in OPPOSITE directions (180 degrees)
        # We want the angle to be close to 180 degrees
        # Calculate deviation from 180 degrees
        deviation_from_180 = abs(180 - angle_between)

        # Score: 100% when exactly opposite (0 deviation), 0% when deviation is large
        # Allow some error margin - the score decreases as deviation increases
        direction_score = max(0, 100 - deviation_from_180)

        # 3. Shape similarity score using bidirectional RMSD
        # Calculate pairwise distances between all points of the aligned segments
        from scipy.spatial.distance import cdist

        # seg1_final and seg2_final are already aligned, so we can directly compare them
        seg1_array = seg1_final.astype(np.float64)
        seg2_array = seg2_final.astype(np.float64)

        # Calculate pairwise distances between all points
        distance_matrix = cdist(seg1_array, seg2_array)

        # Bidirectional distance calculation:
        # 1. For each point on seg1, get the minimum distance to seg2
        min_distances_1_to_2 = np.min(distance_matrix, axis=1)

        # 2. For each point on seg2, get the minimum distance to seg1
        min_distances_2_to_1 = np.min(distance_matrix, axis=0)

        # Combine both sets of distances
        all_distances = np.concatenate([min_distances_1_to_2, min_distances_2_to_1])

        # Calculate RMSD (Root Mean Square Distance)
        rmsd = np.sqrt(np.mean(all_distances ** 2))

        # Normalize RMSD by average segment length to get a percentage
        avg_length = (seg1_length + seg2_length) / 2
        rmsd_percentage = (rmsd / avg_length) * 100 if avg_length > 0 else 100

        # Convert to shape similarity score (0-100%, where 100% is perfect match)
        # Lower RMSD = higher score
        shape_score = max(0, 100 - rmsd_percentage)

        # Create canvas for visualization with extra space at bottom for scores
        all_points = np.vstack([seg1_final, seg2_final])
        min_x, min_y = all_points.min(axis=0) - 30
        max_x, max_y = all_points.max(axis=0) + 30

        canvas_w = max(int(max_x - min_x), 400)  # Minimum width of 400px for score text
        canvas_h = int(max_y - min_y) + 105  # Add 105 pixels for score text (3 scores now)
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        # Offset to draw on canvas
        offset = np.array([-min_x, -min_y])

        # Draw seg1 in green with outward arrow (using pre-calculated direction)
        seg1_canvas = (seg1_final + offset).astype(np.int32)
        seg1_mid_canvas = (seg1_mid + offset).astype(np.int32)
        arrow_length = 25
        seg1_arrow_end = seg1_mid_canvas + (seg1_normal * arrow_length).astype(np.int32)
        cv2.arrowedLine(canvas, tuple(seg1_mid_canvas), tuple(seg1_arrow_end), (0, 200, 0), 2, tipLength=0.3)
        cv2.polylines(canvas, [seg1_canvas], False, (0, 200, 0), 2)

        # Draw seg2 in blue with outward arrow (using pre-calculated rotated direction)
        seg2_canvas = (seg2_final + offset).astype(np.int32)
        seg2_mid_canvas = (seg2_mid_aligned + offset).astype(np.int32)
        seg2_arrow_end = seg2_mid_canvas + (seg2_normal_rotated * arrow_length).astype(np.int32)
        cv2.arrowedLine(canvas, tuple(seg2_mid_canvas), tuple(seg2_arrow_end), (200, 100, 0), 2, tipLength=0.3)
        cv2.polylines(canvas, [seg2_canvas], False, (200, 100, 0), 2)

        # Mark endpoints
        # Blue (frame) points - both should be at B1
        B1_canvas = (B1 + offset).astype(np.int32)

        # Red (interior) points
        R1_canvas = (R1 + offset).astype(np.int32)
        R2_canvas = (R2_aligned + offset).astype(np.int32)

        # Draw seg1 endpoints (with green outline to match seg1 color)
        # Frame point (blue dot)
        cv2.circle(canvas, tuple(B1_canvas), 6, (255, 0, 0), -1)  # Blue fill
        cv2.circle(canvas, tuple(B1_canvas), 8, (0, 150, 0), 2)   # Green outline
        # Interior point (red dot)
        cv2.circle(canvas, tuple(R1_canvas), 6, (0, 0, 255), -1)  # Red fill
        cv2.circle(canvas, tuple(R1_canvas), 8, (0, 150, 0), 2)   # Green outline

        # Draw seg2 endpoints (with blue outline to match seg2 color)
        # Frame point (blue dot) - overlaps with seg1's frame point at B1
        cv2.circle(canvas, tuple(B1_canvas), 6, (255, 0, 0), -1)  # Blue fill
        cv2.circle(canvas, tuple(B1_canvas), 8, (150, 80, 0), 2)  # Blue outline
        # Interior point (red dot)
        cv2.circle(canvas, tuple(R2_canvas), 6, (0, 0, 255), -1)  # Red fill
        cv2.circle(canvas, tuple(R2_canvas), 8, (150, 80, 0), 2)  # Blue outline

        # Draw line between interior points to show alignment distance
        distance = np.linalg.norm(R1_canvas.astype(float) - R2_canvas.astype(float))
        if distance > 1:  # Only show if distance is significant
            cv2.line(canvas, tuple(R1_canvas), tuple(R2_canvas), (150, 150, 150), 1, cv2.LINE_AA)
            # Calculate and display distance
            mid_point = ((R1_canvas + R2_canvas) // 2).astype(np.int32)
            cv2.putText(canvas, f"{distance:.1f}px", tuple(mid_point + [5, -5]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Add labels with rotation info
        cv2.putText(canvas, f"P{piece1.piece_id}-S{seg1.segment_id} (green)", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
        cv2.putText(canvas, f"P{piece2.piece_id}-S{seg2.segment_id} (blue, rot: {np.degrees(rotation_angle):.1f}deg)", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 80, 0), 1)

        # Determine if this is a valid match based on rules
        # Rule 1: Deviation from opposite direction must be below 60 degrees
        # Rule 2: Length must be above 80%
        # Rule 3: Shape similarity must be above 80%
        is_valid_match = (deviation_from_180 < 60.0) and (length_score >= 80.0) and (shape_score >= 80.0)

        # Add scoring information at the bottom (in white background area below segments)
        score_y_start = canvas_h - 95
        cv2.putText(canvas, "Scores:", (10, score_y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(canvas, f"Length match: {length_score:.1f}% (diff: {length_diff:.1f}px)", (10, score_y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(canvas, f"Direction match: {direction_score:.1f}% (angle: {angle_between:.1f}deg)", (10, score_y_start + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(canvas, f"Shape similarity: {shape_score:.1f}% (RMSD: {rmsd:.2f}px)", (10, score_y_start + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # Add match status
        match_color = (0, 150, 0) if is_valid_match else (0, 0, 200)  # Green if valid, red if not
        match_text = "VALID MATCH" if is_valid_match else "NOT A MATCH"
        cv2.putText(canvas, match_text, (canvas_w - 200, score_y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, match_color, 2)

        # Add endpoint alignment info
        cv2.putText(canvas, f"Blue dot: Frame connection (aligned)", (10, canvas_h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        cv2.putText(canvas, f"Red dots: Interior endpoints (distance: {distance:.1f}px)", (10, canvas_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Create match data for chain matching
        from .chain_matcher import SegmentMatch

        match = SegmentMatch(
            piece1_id=piece1.piece_id,
            piece2_id=piece2.piece_id,
            seg1_id=seg1.segment_id,
            seg2_id=seg2.segment_id,
            length_score=length_score,
            direction_score=direction_score,
            shape_score=shape_score,
            is_valid=is_valid_match
        )

        # Create segment data dictionary for chain score calculation
        segment_data = {
            (piece1.piece_id, seg1.segment_id, piece2.piece_id, seg2.segment_id): {
                'seg1_points': seg1_final,
                'seg2_aligned': seg2_final,
                'seg1_normal': seg1_normal,
                'seg2_normal_rotated': seg2_normal_rotated
            }
        }

        return canvas, match, segment_data

    @staticmethod
    def _create_chain_overlay(piece1, segments1, chain_segs1, piece2, segments2, chain_segs2,
                               arrow1_mid, arrow1_normal, arrow2_mid, arrow2_normal):
        """Create an overlay visualization showing entire chain of segments aligned.

        This combines all segments in the chain into a single overlay instead of individual pairs.

        Args:
            piece1: First piece object
            segments1: All segments for piece1
            chain_segs1: List of segments in the chain for piece1 (in order)
            piece2: Second piece object
            segments2: All segments for piece2
            chain_segs2: List of segments in the chain for piece2 (in order)
            arrow1_mid: Arrow midpoint for chain1 (from first segment)
            arrow1_normal: Arrow normal vector for chain1
            arrow2_mid: Arrow midpoint for chain2 (from first segment)
            arrow2_normal: Arrow normal vector for chain2

        Returns:
            Image showing both chains overlaid and aligned
        """
        # Combine all points from the chain segments
        chain1_points = []
        for seg in chain_segs1:
            chain1_points.extend([[p.x, p.y] for p in seg.contour_points])

        chain2_points = []
        for seg in chain_segs2:
            chain2_points.extend([[p.x, p.y] for p in seg.contour_points])

        chain1_points = np.array(chain1_points, dtype=np.float64)
        chain2_points = np.array(chain2_points, dtype=np.float64)

        # Determine which end of each complete chain is the frame connection
        frame_touching_ids1 = VisualizerHelpers._find_frame_touching_segment_ids(piece1, segments1)
        frame_touching_ids2 = VisualizerHelpers._find_frame_touching_segment_ids(piece2, segments2)

        # Chain 1: Check which end has frame-touching neighbor outside the chain
        first_seg1 = chain_segs1[0]
        num_segments1 = len(segments1)
        first_id1 = first_seg1.segment_id
        first_prev_id1 = (first_id1 - 1) % num_segments1
        first_next_id1 = (first_id1 + 1) % num_segments1
        first_prev_is_frame1 = first_prev_id1 in frame_touching_ids1
        first_next_is_frame1 = first_next_id1 in frame_touching_ids1

        if len(chain_segs1) > 1:
            second_seg_id1 = chain_segs1[1].segment_id
            if first_prev_is_frame1 and second_seg_id1 != first_prev_id1:
                frame_at_start1 = True
            elif first_next_is_frame1 and second_seg_id1 != first_next_id1:
                frame_at_start1 = True
            else:
                frame_at_start1 = False
        else:
            frame_at_start1 = first_prev_is_frame1

        if frame_at_start1:
            B1 = chain1_points[0]
            R1 = chain1_points[-1]
        else:
            B1 = chain1_points[-1]
            R1 = chain1_points[0]

        # Chain 2: Check which end has frame-touching neighbor outside the chain
        first_seg2 = chain_segs2[0]
        num_segments2 = len(segments2)
        first_id2 = first_seg2.segment_id
        first_prev_id2 = (first_id2 - 1) % num_segments2
        first_next_id2 = (first_id2 + 1) % num_segments2
        first_prev_is_frame2 = first_prev_id2 in frame_touching_ids2
        first_next_is_frame2 = first_next_id2 in frame_touching_ids2

        if len(chain_segs2) > 1:
            second_seg_id2 = chain_segs2[1].segment_id
            if first_prev_is_frame2 and second_seg_id2 != first_prev_id2:
                frame_at_start2 = True
            elif first_next_is_frame2 and second_seg_id2 != first_next_id2:
                frame_at_start2 = True
            else:
                frame_at_start2 = False
        else:
            frame_at_start2 = first_prev_is_frame2

        if frame_at_start2:
            B2 = chain2_points[0]
            R2 = chain2_points[-1]
        else:
            B2 = chain2_points[-1]
            R2 = chain2_points[0]

        # Translate chain2 so blue endpoints match
        translation = B1 - B2
        chain2_translated = chain2_points + translation
        R2_translated = R2 + translation

        # Rotate chain2 so red points align
        v1 = R1 - B1
        v2 = R2_translated - B1

        from math import atan2, cos, sin
        angle1 = atan2(v1[1], v1[0])
        angle2 = atan2(v2[1], v2[0])
        rotation_angle = angle1 - angle2

        cos_r = cos(rotation_angle)
        sin_r = sin(rotation_angle)

        def rotate(p, origin):
            px, py = p - origin
            x = cos_r * px - sin_r * py + origin[0]
            y = sin_r * px + cos_r * py + origin[1]
            return np.array([x, y])

        chain2_aligned = np.array([rotate(p, B1) for p in chain2_translated])
        R2_aligned = rotate(R2_translated, B1)

        # Create canvas
        all_points = np.vstack([chain1_points, chain2_aligned])
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)

        canvas_w = max(int(max_x - min_x) + 40, 400)
        canvas_h = int(max_y - min_y) + 140  # Extra space for scores

        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        # Offset points to fit in canvas
        offset = np.array([20 - min_x, 20 - min_y])
        chain1_canvas = (chain1_points + offset).astype(np.int32)
        chain2_canvas = (chain2_aligned + offset).astype(np.int32)

        # Draw chains
        cv2.polylines(canvas, [chain1_canvas], False, (0, 200, 0), 3)  # Green for chain 1
        cv2.polylines(canvas, [chain2_canvas], False, (200, 100, 0), 3)  # Blue for chain 2

        # Draw arrows from first segments (transformed to overlay coordinates)
        # Arrow 1 (chain1) - already in correct position
        arrow1_mid_canvas = (arrow1_mid + offset).astype(np.int32)
        arrow1_end = arrow1_mid + arrow1_normal * 25
        arrow1_end_canvas = (arrow1_end + offset).astype(np.int32)
        cv2.arrowedLine(canvas, tuple(arrow1_mid_canvas), tuple(arrow1_end_canvas),
                       (0, 0, 255), 2, tipLength=0.3)

        # Arrow 2 (chain2) - needs to be transformed (translated and rotated)
        # Apply same transformation as chain2
        arrow2_mid_translated = arrow2_mid + translation
        arrow2_end_original = arrow2_mid + arrow2_normal * 25
        arrow2_end_translated = arrow2_end_original + translation

        # Rotate arrow2 around B1
        def rotate(p, origin):
            px, py = p - origin
            x = cos_r * px - sin_r * py + origin[0]
            y = sin_r * px + cos_r * py + origin[1]
            return np.array([x, y])

        arrow2_mid_aligned = rotate(arrow2_mid_translated, B1)
        arrow2_end_aligned = rotate(arrow2_end_translated, B1)

        arrow2_mid_canvas = (arrow2_mid_aligned + offset).astype(np.int32)
        arrow2_end_canvas = (arrow2_end_aligned + offset).astype(np.int32)
        cv2.arrowedLine(canvas, tuple(arrow2_mid_canvas), tuple(arrow2_end_canvas),
                       (0, 0, 255), 2, tipLength=0.3)

        # Draw endpoints
        # Blue dot: shared frame connection (beginning of both chains)
        # Red dots: ONLY the final endpoints of each complete chain (not intermediate connections)
        B1_canvas = (B1 + offset).astype(np.int32)
        R1_canvas = (R1 + offset).astype(np.int32)
        R2_canvas = (R2_aligned + offset).astype(np.int32)

        cv2.circle(canvas, tuple(B1_canvas), 8, (255, 0, 0), -1)  # Blue dot (shared frame point)
        cv2.circle(canvas, tuple(R1_canvas), 8, (0, 0, 255), -1)  # Red dot (chain1 end)
        cv2.circle(canvas, tuple(R2_canvas), 8, (0, 0, 255), -1)  # Red dot (chain2 end)

        # Add title and score information
        seg_ids_1 = [s.segment_id for s in chain_segs1]
        seg_ids_2 = [s.segment_id for s in chain_segs2]

        cv2.putText(canvas, f"P{piece1.piece_id}{seg_ids_1} (green) aligned with P{piece2.piece_id}{seg_ids_2} (blue)",
                   (10, canvas_h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, f"Blue dot: Shared frame connection point",
                   (10, canvas_h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        cv2.putText(canvas, f"Red dots: Interior endpoints (chain ends)",
                   (10, canvas_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        distance = np.linalg.norm(R1 - R2_aligned)
        cv2.putText(canvas, f"Interior endpoint distance: {distance:.1f}px",
                   (10, canvas_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        return canvas

    @staticmethod
    def _draw_segment_overlay_arrow(image, segment_points, color, piece):
        """Draw outward arrow for segment overlay visualization.

        Args:
            image: Image to draw on
            segment_points: Segment contour points (already offset for canvas)
            color: Arrow color in BGR format
            piece: Piece object (for determining outward direction)
        """
        if len(segment_points) < 2:
            return

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

            # Get piece center to determine outward direction
            piece_contour = np.array([[p.x, p.y] for p in piece.contour_points])
            piece_center = np.mean(piece_contour, axis=0)

            # Check if normal points away from piece center
            # Note: Since segment might be translated/rotated, we use a heuristic
            # If dot product is positive, normal points toward center, so flip it
            to_center = piece_center - mid_point
            if np.dot(normal, to_center) > 0:
                normal = -normal

            # Draw arrow from mid_point outward
            arrow_length = 25
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
    def _draw_segment_on_piece(piece, all_segments, segment, original_image):
        """Draw a highlighted segment on its piece with correct frame connection marking.

        Args:
            piece: Piece object
            all_segments: All segments for this piece
            segment: Segment to highlight (must be frame-adjacent)
            original_image: Original puzzle image

        Returns:
            Image with highlighted segment
        """
        # Get bounding box for the piece
        import numpy as np
        contour_points = np.array([[int(p.x), int(p.y)] for p in piece.contour_points])
        import cv2
        x, y, w, h = cv2.boundingRect(contour_points)

        # Add padding
        padding = 30
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(original_image.shape[1], x + w + padding)
        y_end = min(original_image.shape[0], y + h + padding)

        # Crop region and ensure it's contiguous
        cropped = np.ascontiguousarray(original_image[y_start:y_end, x_start:x_end])

        # Draw piece contour in light gray
        piece_contour = (contour_points - np.array([x_start, y_start])).astype(np.int32)
        cv2.polylines(cropped, [piece_contour], True, (200, 200, 200), 2)

        # Find which segments touch the frame corners (frame-touching segments)
        frame_touching_ids = VisualizerHelpers._find_frame_touching_segment_ids(piece, all_segments)

        # Determine which end of the segment connects to the frame
        # Check neighbors in the segment sequence
        current_id = segment.segment_id
        num_segments = len(all_segments)

        prev_id = (current_id - 1) % num_segments
        next_id = (current_id + 1) % num_segments

        # Check if previous or next segment is frame-touching
        prev_is_frame = prev_id in frame_touching_ids
        next_is_frame = next_id in frame_touching_ids

        # Determine frame connection point
        # If previous segment is frame-touching, start_corner connects to frame
        # If next segment is frame-touching, end_corner connects to frame
        segment_points = np.array([[int(p.x - x_start), int(p.y - y_start)] for p in segment.contour_points])

        if prev_is_frame:
            # Start point connects to frame
            frame_point = segment_points[0]
            interior_point = segment_points[-1]
        elif next_is_frame:
            # End point connects to frame
            frame_point = segment_points[-1]
            interior_point = segment_points[0]
        else:
            # Fallback: use start as frame point
            frame_point = segment_points[0]
            interior_point = segment_points[-1]

        # Draw the outward arrow before drawing the segment (bright red)
        VisualizerHelpers._draw_outward_arrow(cropped, segment_points, piece, x_start, y_start, color=(0, 0, 255))

        # Draw the highlighted segment in bright color
        cv2.polylines(cropped, [segment_points], False, (0, 255, 0), 3)

        # Draw endpoints with correct colors
        if len(segment_points) > 0:
            cv2.circle(cropped, tuple(frame_point), 6, (255, 0, 0), -1)  # Frame connection in blue
            cv2.circle(cropped, tuple(interior_point), 6, (0, 0, 255), -1)  # Interior end in red

        return cropped

