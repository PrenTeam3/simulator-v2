"""Chain visualization module for edge solver v2."""

import cv2
import numpy as np
from math import atan2, cos, sin
from .visualizer_helpers import VisualizerHelpers
from .geometry_utils import GeometryUtils


class ChainVisualizer:
    """Handles chain visualizations."""

    @staticmethod
    def visualize_progressive_chains(piece1, segments1, piece2, segments2, progressive_chains, original_image):
        """Create a visualization showing all progressive chain extensions vertically.

        Each chain length (1, 2, 3, ...) is shown as a separate row with scores,
        continuing until the match becomes invalid.

        Args:
            piece1: First piece object
            segments1: All segments for piece1
            piece2: Second piece object
            segments2: All segments for piece2
            progressive_chains: List of ChainMatch objects with increasing chain lengths
            original_image: Original puzzle image

        Returns:
            Image showing progressive chain overlays stacked vertically
        """
        from scipy.spatial.distance import cdist

        overlay_images = []

        for chain in progressive_chains:
            # Get the segment objects for this chain length
            chain_segs1 = [s for s in segments1 if s.segment_id in chain.segment_ids_p1]
            chain_segs2 = [s for s in segments2 if s.segment_id in chain.segment_ids_p2]

            # Calculate arrows for the entire chain
            arrow1_mid, arrow1_normal = ChainVisualizer._calculate_chain_arrow(chain_segs1, piece1)
            arrow2_mid, arrow2_normal = ChainVisualizer._calculate_chain_arrow(chain_segs2, piece2)

            # Create the overlay for this chain length with scoring
            overlay = ChainVisualizer._create_chain_overlay(
                piece1, segments1, chain_segs1,
                piece2, segments2, chain_segs2,
                arrow1_mid, arrow1_normal, arrow2_mid, arrow2_normal,
                chain  # Pass the chain object with additional metrics
            )

            overlay_images.append(overlay)

        # Stack all overlays vertically
        if not overlay_images:
            return np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Find max width
        max_width = max(img.shape[1] for img in overlay_images)

        # Pad images to same width and stack
        padded_images = []
        separator_height = 20
        separator = np.ones((separator_height, max_width, 3), dtype=np.uint8) * 200

        for i, img in enumerate(overlay_images):
            if img.shape[1] < max_width:
                # Pad to max width
                pad_width = max_width - img.shape[1]
                padded = np.hstack([img, np.ones((img.shape[0], pad_width, 3), dtype=np.uint8) * 255])
            else:
                padded = img

            padded_images.append(padded)
            if i < len(overlay_images) - 1:
                padded_images.append(separator)

        # Stack vertically
        result = np.vstack(padded_images)

        return result

    @staticmethod
    def _calculate_chain_arrow(chain_segments, piece):
        """Calculate arrow direction for entire chain (normal vector pointing outward).

        Args:
            chain_segments: List of segments in the chain
            piece: Piece object

        Returns:
            Tuple of (midpoint, normal_vector)
        """
        # Combine all points from the chain segments
        chain_points = []
        for seg in chain_segments:
            chain_points.extend([[p.x, p.y] for p in seg.contour_points])

        chain_points = np.array(chain_points, dtype=np.float64)

        # Calculate at midpoint of the entire chain
        mid_idx = len(chain_points) // 2
        mid = chain_points[mid_idx]

        if mid_idx == 0:
            tangent = chain_points[mid_idx + 1] - chain_points[mid_idx]
        elif mid_idx == len(chain_points) - 1:
            tangent = chain_points[mid_idx] - chain_points[mid_idx - 1]
        else:
            tangent = chain_points[mid_idx + 1] - chain_points[mid_idx - 1]

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
    def visualize_chain(piece1, segments1, chain_segs1, piece2, segments2, chain_segs2, original_image):
        """Create a visualization showing chain on pieces, raw chains, and overlay.

        Args:
            piece1: First piece object
            segments1: All segments for piece1
            chain_segs1: Chain segments for piece1 (in order)
            piece2: Second piece object
            segments2: All segments for piece2
            chain_segs2: Chain segments for piece2 (in order)
            original_image: Original puzzle image

        Returns:
            Image showing pieces with chains, raw chains, and combined overlay
        """
        # Step 1: Show both pieces with chain segments highlighted (side-by-side)
        img1 = ChainVisualizer._draw_chain_on_piece(piece1, segments1, chain_segs1, original_image)
        img2 = ChainVisualizer._draw_chain_on_piece(piece2, segments2, chain_segs2, original_image)

        # Resize to same height
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        target_h = min(h1, h2, 300)

        scale1 = target_h / h1
        scale2 = target_h / h2

        img1_resized = cv2.resize(img1, (int(w1 * scale1), target_h))
        img2_resized = cv2.resize(img2, (int(w2 * scale2), target_h))

        # Add separator
        separator = np.ones((target_h, 10, 3), dtype=np.uint8) * 200

        # Combine side by side
        pieces_side_by_side = np.hstack([img1_resized, separator, img2_resized])

        # Calculate arrows for the entire chain (not just first segment)
        arrow1_mid, arrow1_normal = ChainVisualizer._calculate_chain_arrow(chain_segs1, piece1)
        arrow2_mid, arrow2_normal = ChainVisualizer._calculate_chain_arrow(chain_segs2, piece2)

        # Step 2: Show both raw chain segments separately
        raw_chain1 = ChainVisualizer._draw_raw_chain(chain_segs1, piece1, segments1, arrow1_mid, arrow1_normal)
        raw_chain2 = ChainVisualizer._draw_raw_chain(chain_segs2, piece2, segments2, arrow2_mid, arrow2_normal)

        # Resize raw chains to same height
        h_raw1, w_raw1 = raw_chain1.shape[:2]
        h_raw2, w_raw2 = raw_chain2.shape[:2]
        target_h_raw = min(h_raw1, h_raw2, 200)

        scale_raw1 = target_h_raw / h_raw1
        scale_raw2 = target_h_raw / h_raw2

        raw1_resized = cv2.resize(raw_chain1, (int(w_raw1 * scale_raw1), target_h_raw))
        raw2_resized = cv2.resize(raw_chain2, (int(w_raw2 * scale_raw2), target_h_raw))

        # Add separator
        separator_raw = np.ones((target_h_raw, 10, 3), dtype=np.uint8) * 200

        # Combine raw chains side by side
        raw_chains_side_by_side = np.hstack([raw1_resized, separator_raw, raw2_resized])

        # Resize to match width of pieces
        if raw_chains_side_by_side.shape[1] != pieces_side_by_side.shape[1]:
            scale_w = pieces_side_by_side.shape[1] / raw_chains_side_by_side.shape[1]
            new_h = int(raw_chains_side_by_side.shape[0] * scale_w)
            raw_chains_side_by_side = cv2.resize(raw_chains_side_by_side,
                                                  (pieces_side_by_side.shape[1], new_h))

        # Step 3: Create combined chain overlay (all segments together)
        combined_overlay = ChainVisualizer._create_chain_overlay(
            piece1, segments1, chain_segs1, piece2, segments2, chain_segs2,
            arrow1_mid, arrow1_normal, arrow2_mid, arrow2_normal
        )

        # Resize overlay to match width
        overlay_h, overlay_w = combined_overlay.shape[:2]
        if overlay_w > 0:
            overlay_scale = pieces_side_by_side.shape[1] / overlay_w
            overlay_target_h = int(overlay_h * overlay_scale)
            combined_overlay_resized = cv2.resize(combined_overlay,
                                                   (pieces_side_by_side.shape[1], overlay_target_h))
        else:
            combined_overlay_resized = combined_overlay

        # Combine everything vertically with labels
        label_height = 40
        separator_height = 15

        total_height = (label_height + pieces_side_by_side.shape[0] +
                       separator_height + label_height + raw_chains_side_by_side.shape[0] +
                       separator_height + label_height + combined_overlay_resized.shape[0])
        result = np.ones((total_height, pieces_side_by_side.shape[1], 3), dtype=np.uint8) * 255

        # Add title for pieces
        seg_ids_1 = [s.segment_id for s in chain_segs1]
        seg_ids_2 = [s.segment_id for s in chain_segs2]
        title = f"Chain: P{piece1.piece_id}{seg_ids_1} <-> P{piece2.piece_id}{seg_ids_2}"
        cv2.putText(result, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Place pieces side by side
        y_offset = label_height
        result[y_offset:y_offset + pieces_side_by_side.shape[0], :] = pieces_side_by_side

        # Add label for raw chains
        y_offset += pieces_side_by_side.shape[0] + separator_height
        cv2.putText(result, "Raw chain segments:", (10, y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Place raw chains
        y_offset += label_height
        result[y_offset:y_offset + raw_chains_side_by_side.shape[0], :] = raw_chains_side_by_side

        # Add label for combined overlay
        y_offset += raw_chains_side_by_side.shape[0] + separator_height
        cv2.putText(result, "Chain overlay (aligned):", (10, y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Place combined overlay
        y_offset += label_height
        result[y_offset:y_offset + combined_overlay_resized.shape[0], :] = combined_overlay_resized

        return result
    @staticmethod
    def _create_chain_overlay(piece1, segments1, chain_segs1, piece2, segments2, chain_segs2,
                               arrow1_mid, arrow1_normal, arrow2_mid, arrow2_normal, chain_match=None):
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
            chain_match: Optional ChainMatch object with additional metrics

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

        # Align chain2 to chain1 using GeometryUtils
        chain2_aligned, R2_aligned = GeometryUtils.align_chains(
            chain1_points, chain2_points, B1, B2, R1, R2
        )

        # Calculate rotation angle for normal vector transformation
        translation = B1 - B2
        R2_translated = R2 + translation
        v1 = R1 - B1
        v2 = R2_translated - B1
        angle1 = atan2(v1[1], v1[0])
        angle2 = atan2(v2[1], v2[0])
        rotation_angle = angle1 - angle2
        cos_r = cos(rotation_angle)
        sin_r = sin(rotation_angle)

        # Calculate scoring metrics
        # 1. Length difference score
        chain1_length = np.sum(np.linalg.norm(np.diff(chain1_points, axis=0), axis=1))
        chain2_length = np.sum(np.linalg.norm(np.diff(chain2_points, axis=0), axis=1))
        length_diff = abs(chain1_length - chain2_length)
        length_score = max(0, 100 - (length_diff / max(chain1_length, chain2_length) * 100))

        # 2. Arrow direction alignment score
        # Rotate arrow2_normal to match the chain2 rotation
        arrow2_normal_rotated = np.array([
            cos_r * arrow2_normal[0] - sin_r * arrow2_normal[1],
            sin_r * arrow2_normal[0] + cos_r * arrow2_normal[1]
        ])

        # Calculate angle between the two arrow direction vectors
        dot_product = np.dot(arrow1_normal, arrow2_normal_rotated)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_between = np.degrees(np.arccos(dot_product))

        # Good match: arrows point in OPPOSITE directions (180 degrees)
        deviation_from_180 = abs(180 - angle_between)
        direction_score = max(0, 100 - deviation_from_180)

        # 3. Shape similarity score using bidirectional RMSD
        from scipy.spatial.distance import cdist

        chain1_array = chain1_points.astype(np.float64)
        chain2_array = chain2_aligned.astype(np.float64)

        # Calculate pairwise distances between all points
        distance_matrix = cdist(chain1_array, chain2_array)

        # Bidirectional distance calculation
        min_distances_1_to_2 = np.min(distance_matrix, axis=1)
        min_distances_2_to_1 = np.min(distance_matrix, axis=0)
        all_distances = np.concatenate([min_distances_1_to_2, min_distances_2_to_1])

        # Calculate RMSD
        rmsd = np.sqrt(np.mean(all_distances ** 2))

        # Normalize RMSD by average chain length to get a percentage
        avg_length = (chain1_length + chain2_length) / 2
        rmsd_percentage = (rmsd / avg_length) * 100 if avg_length > 0 else 100

        # Convert to shape similarity score
        shape_score = max(0, 100 - rmsd_percentage)

        # Determine if this is a valid match
        # Use the is_valid from chain_match if available (calculated with area-based scoring)
        # Otherwise recalculate using local scores (fallback for compatibility)
        if chain_match:
            is_valid_match = chain_match.is_valid
        else:
            # Fallback: Note: Chain matching uses 80% length, 70% shape (direction NOT used)
            is_valid_match = (length_score >= 80.0) and (shape_score >= 70.0)

        # Create canvas with extra space for scores (increased for additional metrics)
        all_points = np.vstack([chain1_points, chain2_aligned])
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)

        canvas_w = max(int(max_x - min_x) + 40, 500)  # Increased width for longer text
        canvas_h = int(max_y - min_y) + 320  # Increased space for additional metrics

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
        arrow2_mid_aligned = GeometryUtils.rotate_point(arrow2_mid_translated, B1, angle_rad=rotation_angle)
        arrow2_end_aligned = GeometryUtils.rotate_point(arrow2_end_translated, B1, angle_rad=rotation_angle)

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

        # Add scoring information
        score_y_start = canvas_h - 280
        cv2.putText(canvas, "Scores:", (10, score_y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(canvas, f"Length match: {length_score:.1f}% (diff: {length_diff:.1f}px)", (10, score_y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(canvas, f"Direction match: {direction_score:.1f}% (angle: {angle_between:.1f}deg)", (10, score_y_start + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # Display shape score based on what's actually being used
        if chain_match and chain_match.additional_metrics:
            # Area-based is now primary
            area_val = chain_match.additional_metrics['enclosed_area']
            norm_area = chain_match.additional_metrics['normalized_area']
            cv2.putText(canvas, f"Shape similarity: {shape_score:.1f}% (Enclosed Area: {area_val:.1f}px², norm: {norm_area:.4f})",
                       (10, score_y_start + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        else:
            # Fallback if no additional metrics
            cv2.putText(canvas, f"Shape similarity: {shape_score:.1f}%", (10, score_y_start + 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # Add match status
        match_color = (0, 150, 0) if is_valid_match else (0, 0, 200)
        match_text = "VALID MATCH" if is_valid_match else "NOT A MATCH"
        cv2.putText(canvas, match_text, (canvas_w - 200, score_y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, match_color, 2)

        # Add additional shape metrics if available
        if chain_match and chain_match.additional_metrics:
            metrics = chain_match.additional_metrics
            metric_y = score_y_start + 110

            cv2.putText(canvas, "Alternative Shape Metrics (for comparison):", (10, metric_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 150), 1)

            # Metric 1: RMSD-based (OLD PRIMARY)
            cv2.putText(canvas, f"1. RMSD-based (old): {metrics['rmsd_score']:.1f}% (RMSD: {metrics['rmsd']:.2f}px)",
                       (10, metric_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # Metric 2: Absolute RMSD
            cv2.putText(canvas, f"2. Absolute RMSD/seg: {metrics['absolute_rmsd_score']:.1f}% ({metrics['rmsd_per_segment']:.2f}px/seg)",
                       (10, metric_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # Metric 3: Hausdorff
            cv2.putText(canvas, f"3. Hausdorff (max dev): {metrics['hausdorff_score']:.1f}% (max: {metrics['max_deviation']:.2f}px)",
                       (10, metric_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # Metric 4: Consistency
            cv2.putText(canvas, f"4. Consistency: {metrics['consistency_score']:.1f}%",
                       (10, metric_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Add chain info
        cv2.putText(canvas, f"P{piece1.piece_id}{seg_ids_1} (green) aligned with P{piece2.piece_id}{seg_ids_2} (blue)",
                   (10, canvas_h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, f"Blue dot: Shared frame connection point",
                   (10, canvas_h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        distance = np.linalg.norm(R1 - R2_aligned)
        cv2.putText(canvas, f"Red dots: Interior endpoints (distance: {distance:.1f}px)",
                   (10, canvas_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        return canvas
    @staticmethod
    def _draw_chain_on_piece(piece, all_segments, chain_segments, original_image):
        """Draw all chain segments on a piece (modification of _draw_segment_on_piece for multiple segments).

        Args:
            piece: Piece object
            all_segments: All segments for this piece
            chain_segments: List of segments in the chain to highlight
            original_image: Original puzzle image

        Returns:
            Image with highlighted chain segments
        """
        # Get bounding box for the piece
        contour_points = np.array([[int(p.x), int(p.y)] for p in piece.contour_points])
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

        # Draw all chain segments
        for segment in chain_segments:
            # Determine which end of the segment connects to the frame
            current_id = segment.segment_id
            num_segments = len(all_segments)

            prev_id = (current_id - 1) % num_segments
            next_id = (current_id + 1) % num_segments

            # Check if previous or next segment is frame-touching
            prev_is_frame = prev_id in frame_touching_ids
            next_is_frame = next_id in frame_touching_ids

            # Get segment points
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
    @staticmethod
    def _draw_raw_chain(chain_segments, piece, all_segments, arrow_mid=None, arrow_normal=None):
        """Draw raw chain segments without any transformation.

        Args:
            chain_segments: List of segments in the chain
            piece: Piece object
            all_segments: All segments for this piece

        Returns:
            Image showing the raw chain segments
        """
        # Combine all points from chain segments
        chain_points = []
        for seg in chain_segments:
            chain_points.extend([[p.x, p.y] for p in seg.contour_points])

        if not chain_points:
            return np.ones((100, 100, 3), dtype=np.uint8) * 255

        chain_points = np.array(chain_points, dtype=np.float64)

        # Find bounding box
        min_x, min_y = np.min(chain_points, axis=0)
        max_x, max_y = np.max(chain_points, axis=0)

        # Add padding
        padding = 20
        canvas_w = int(max_x - min_x + 2 * padding)
        canvas_h = int(max_y - min_y + 2 * padding)

        # Create canvas
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        # Offset points to canvas coordinates
        offset = np.array([min_x - padding, min_y - padding])
        chain_canvas = (chain_points - offset).astype(np.int32)

        # Find frame-touching segments
        frame_touching_ids = VisualizerHelpers._find_frame_touching_segment_ids(piece, all_segments)

        # Determine which end of the COMPLETE CHAIN connects to frame
        # We need to check both the first and last segments to see which one has a frame-adjacent neighbor
        first_seg = chain_segments[0]
        last_seg = chain_segments[-1]

        num_segments = len(all_segments)

        # Check first segment
        first_id = first_seg.segment_id
        first_prev_id = (first_id - 1) % num_segments
        first_next_id = (first_id + 1) % num_segments
        first_prev_is_frame = first_prev_id in frame_touching_ids
        first_next_is_frame = first_next_id in frame_touching_ids

        # Check last segment
        last_id = last_seg.segment_id
        last_prev_id = (last_id - 1) % num_segments
        last_next_id = (last_id + 1) % num_segments
        last_prev_is_frame = last_prev_id in frame_touching_ids
        last_next_is_frame = last_next_id in frame_touching_ids

        # Determine which end of the complete chain is the frame connection
        # The frame connection is at whichever end has a frame-touching neighbor OUTSIDE the chain
        # For a chain [9, 10], if segment 8 is frame-touching, then segment 9's start is the frame connection

        # Check if first segment has a frame neighbor that's NOT in the chain
        if len(chain_segments) > 1:
            second_seg_id = chain_segments[1].segment_id
            # If the previous segment is frame-touching (and not the next segment in chain)
            if first_prev_is_frame and second_seg_id != first_prev_id:
                frame_at_start = True
            # If the next segment is frame-touching (and not the next segment in chain)
            elif first_next_is_frame and second_seg_id != first_next_id:
                frame_at_start = True
            else:
                frame_at_start = False
        else:
            # Single segment chain
            frame_at_start = first_prev_is_frame

        # Draw the chain as a polyline
        cv2.polylines(canvas, [chain_canvas], False, (0, 200, 0), 3)

        # Draw arrow if provided
        if arrow_mid is not None and arrow_normal is not None:
            # Transform arrow to canvas coordinates
            arrow_mid_canvas = (arrow_mid - offset).astype(np.int32)
            arrow_end = arrow_mid + arrow_normal * 25  # Arrow length
            arrow_end_canvas = (arrow_end - offset).astype(np.int32)

            # Draw arrow
            cv2.arrowedLine(
                canvas,
                tuple(arrow_mid_canvas),
                tuple(arrow_end_canvas),
                (0, 0, 255),  # Red arrow
                2,
                tipLength=0.3
            )

        # Draw endpoints - only at the very beginning and very end of the complete chain
        if frame_at_start:
            # Blue dot at start, red dot at end
            cv2.circle(canvas, tuple(chain_canvas[0]), 8, (255, 0, 0), -1)
            cv2.circle(canvas, tuple(chain_canvas[-1]), 8, (0, 0, 255), -1)
        else:
            # Blue dot at end, red dot at start
            cv2.circle(canvas, tuple(chain_canvas[-1]), 8, (255, 0, 0), -1)
            cv2.circle(canvas, tuple(chain_canvas[0]), 8, (0, 0, 255), -1)

        # Add label
        seg_ids = [s.segment_id for s in chain_segments]
        label = f"P{piece.piece_id} chain {seg_ids}"
        cv2.putText(canvas, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return canvas
