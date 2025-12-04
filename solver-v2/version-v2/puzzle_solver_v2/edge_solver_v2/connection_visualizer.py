"""Connection visualization module for displaying piece connections."""

import cv2
import numpy as np
from ..common import InteractiveImageViewer


class ConnectionVisualizer:
    """Handles visualization of puzzle piece connections."""

    @staticmethod
    def visualize_connections_dialog(piece_connections, piece_frame_segments, original_image, project_root, temp_folder=None):
        """Create and display visual connections diagram.

        Args:
            piece_connections: Dictionary of piece_id -> list of (connected_piece_id, chain, side) tuples
            piece_frame_segments: List of (piece, segments, frame_adjacent_segments) tuples
            original_image: Original puzzle image
            project_root: Project root directory
        """
        # Create visualization image
        img_height, img_width = original_image.shape[:2]
        viz_img = original_image.copy()

        # Create semi-transparent overlay
        overlay = viz_img.copy()

        # Draw piece contours and labels
        for piece, segments, frame_segs in piece_frame_segments:
            # Draw piece contour
            contour_points = np.array([[int(p.x), int(p.y)] for p in piece.contour_points], dtype=np.int32)
            cv2.polylines(overlay, [contour_points], True, (200, 200, 200), 2)

            # Draw centroid
            centroid_pt = (int(piece.centroid.x), int(piece.centroid.y))
            cv2.circle(overlay, centroid_pt, 8, (255, 100, 0), -1)

            # Draw piece ID label
            cv2.putText(overlay, f'P{piece.piece_id}',
                       (centroid_pt[0] + 15, centroid_pt[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Collect unique connections (avoid duplicates)
        unique_connections = {}
        for piece_id, connections in piece_connections.items():
            for connected_piece_id, chain, side in connections:
                # Create unique key
                pair_key = tuple(sorted([piece_id, connected_piece_id]))
                if pair_key not in unique_connections:
                    unique_connections[pair_key] = chain

        # Sort by shape score
        sorted_connections = sorted(unique_connections.items(),
                                   key=lambda x: x[1].shape_score,
                                   reverse=True)

        # Define distinct colors for each connection
        connection_colors_segments = [
            (0, 255, 255),    # Yellow (connection 1)
            (255, 0, 255),    # Magenta (connection 2)
            (255, 165, 0),    # Orange (connection 3)
            (0, 255, 0),      # Green (connection 4)
            (255, 0, 0),      # Blue (connection 5)
            (180, 105, 255),  # Pink (connection 6)
        ]

        # Highlight matching chain segments - SAME COLOR for both pieces in a connection
        for idx, (pair_key, chain) in enumerate(sorted_connections):
            p1_id, p2_id = pair_key

            # Get pieces and segments
            piece1, segments1, _ = next((p, s, f) for p, s, f in piece_frame_segments if p.piece_id == p1_id)
            piece2, segments2, _ = next((p, s, f) for p, s, f in piece_frame_segments if p.piece_id == p2_id)

            # Get chain segments
            chain_segs1 = [s for s in segments1 if s.segment_id in chain.segment_ids_p1]
            chain_segs2 = [s for s in segments2 if s.segment_id in chain.segment_ids_p2]

            # Use the same color for both pieces in this connection
            color = connection_colors_segments[idx % len(connection_colors_segments)]
            thickness = 5

            # Draw chain segments for piece1
            for seg in chain_segs1:
                seg_points = np.array([[int(p.x), int(p.y)] for p in seg.contour_points], dtype=np.int32)
                cv2.polylines(overlay, [seg_points], False, color, thickness)

            # Draw chain segments for piece2
            for seg in chain_segs2:
                seg_points = np.array([[int(p.x), int(p.y)] for p in seg.contour_points], dtype=np.int32)
                cv2.polylines(overlay, [seg_points], False, color, thickness)

            # Draw blue and red dots for chain endpoints
            # Blue dot = frame connection, Red dot = interior connection
            if chain.blue_dot_p1 is not None and chain.red_dot_p1 is not None:
                # Piece 1 dots
                blue_p1 = (int(chain.blue_dot_p1[0]), int(chain.blue_dot_p1[1]))
                red_p1 = (int(chain.red_dot_p1[0]), int(chain.red_dot_p1[1]))
                cv2.circle(overlay, blue_p1, 8, (255, 0, 0), -1)  # Blue dot (BGR: blue)
                cv2.circle(overlay, red_p1, 8, (0, 0, 255), -1)   # Red dot (BGR: red)

                # Piece 2 dots
                blue_p2 = (int(chain.blue_dot_p2[0]), int(chain.blue_dot_p2[1]))
                red_p2 = (int(chain.red_dot_p2[0]), int(chain.red_dot_p2[1]))
                cv2.circle(overlay, blue_p2, 8, (255, 0, 0), -1)  # Blue dot (BGR: blue)
                cv2.circle(overlay, red_p2, 8, (0, 0, 255), -1)   # Red dot (BGR: red)

        # Draw connection lines between matched pieces
        connection_colors = [
            (0, 255, 0),    # Green - best match
            (0, 255, 255),  # Yellow
            (0, 200, 255),  # Orange
            (0, 150, 255),  # Orange-red
            (0, 100, 255),  # Red
        ]

        for idx, (pair_key, chain) in enumerate(sorted_connections):
            p1_id, p2_id = pair_key

            # Get pieces
            piece1 = next(p for p, _, _ in piece_frame_segments if p.piece_id == p1_id)
            piece2 = next(p for p, _, _ in piece_frame_segments if p.piece_id == p2_id)

            # Get centroids
            pt1 = (int(piece1.centroid.x), int(piece1.centroid.y))
            pt2 = (int(piece2.centroid.x), int(piece2.centroid.y))

            # Choose color based on validity and ranking
            if chain.is_valid:
                color = connection_colors[min(idx, len(connection_colors) - 1)]
                thickness = 4 if idx == 0 else 3
            else:
                color = (100, 100, 100)  # Gray for invalid
                thickness = 2

            # Draw connection line
            cv2.line(overlay, pt1, pt2, color, thickness)

            # Draw score label at midpoint
            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2

            score_text = f"{chain.shape_score:.0f}%"
            if chain.is_valid:
                score_text += " OK"

            # Draw background rectangle for text
            text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(overlay,
                         (mid_x - 5, mid_y - text_size[1] - 5),
                         (mid_x + text_size[0] + 5, mid_y + 5),
                         (255, 255, 255), -1)

            cv2.putText(overlay, score_text,
                       (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Blend overlay with original
        cv2.addWeighted(overlay, 0.7, viz_img, 0.3, 0, viz_img)

        # Add title and legend
        title_bg_height = 120  # Increased from 100 to fit 4 connection items
        legend_img = np.ones((title_bg_height, img_width, 3), dtype=np.uint8) * 255

        # Title
        cv2.putText(legend_img, "PUZZLE ASSEMBLY - PIECE CONNECTIONS",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Legend
        num_connections = len(unique_connections)
        cv2.putText(legend_img, f"Selected connections: {num_connections}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
        cv2.putText(legend_img, f"(2 connections per piece)",
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Color legend
        legend_x = img_width - 400
        cv2.putText(legend_img, "Matching segments (same color = connected):",
                   (legend_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # Show color for each connection
        for i, (pair_key, chain) in enumerate(sorted_connections[:4]):  # Show up to 4
            y_pos = 50 + (i * 18)
            color = connection_colors_segments[i % len(connection_colors_segments)]
            cv2.line(legend_img, (legend_x, y_pos), (legend_x + 30, y_pos), color, 4)
            cv2.putText(legend_img, f"P{pair_key[0]} <-> P{pair_key[1]}", (legend_x + 40, y_pos + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Combine title and visualization
        final_img = np.vstack([legend_img, viz_img])

        # Save (use temp_folder if provided, otherwise fall back to project_root/temp)
        if temp_folder:
            output_path = temp_folder / 'puzzle_connections_v2.png'
        else:
            output_path = project_root / 'temp' / 'puzzle_connections_v2.png'
        cv2.imwrite(str(output_path), final_img)
        print(f"  Saved connections visualization to: {output_path}")

        # Display
        viewer = InteractiveImageViewer("Puzzle Connections - Edge Solver V2")
        viewer.show(final_img)
