"""Visualization module for assembly solver steps."""

import cv2
import numpy as np
from typing import Dict, List


class AssemblyVisualizer:
    """Creates visualizations for each step of the assembly process."""

    @staticmethod
    def visualize_step1_centroids(step_data: Dict) -> np.ndarray:
        """Visualize Step 1: Show all pieces with their centroids and chain endpoints marked.

        Args:
            step_data: Dictionary containing 'pieces', 'centroids', 'chain_endpoints', 'original_image'

        Returns:
            np.ndarray: Visualization image
        """
        pieces = step_data['pieces']
        centroids = step_data['centroids']
        chain_endpoints = step_data.get('chain_endpoints', {})
        original_image = step_data['original_image']

        # Create a copy of the original image
        vis_image = original_image.copy()

        # Define colors
        CENTROID_COLOR = (0, 255, 255)  # Yellow
        TEXT_COLOR = (255, 255, 255)  # White
        PIECE_OUTLINE_COLOR = (0, 255, 0)  # Green
        BLUE_DOT_COLOR = (255, 0, 0)  # Blue in BGR
        RED_DOT_COLOR = (0, 0, 255)  # Red in BGR

        # Draw each piece's contour and centroid
        for piece in pieces:
            # Draw piece contour
            contour_points = np.array([[int(p.x), int(p.y)] for p in piece.contour_points])
            cv2.polylines(vis_image, [contour_points], True, PIECE_OUTLINE_COLOR, 2)

            # Get centroid
            centroid = centroids[piece.piece_id]
            cx, cy = int(centroid[0]), int(centroid[1])

            # Draw centroid as a large circle
            cv2.circle(vis_image, (cx, cy), 8, CENTROID_COLOR, -1)
            cv2.circle(vis_image, (cx, cy), 10, (0, 0, 0), 2)  # Black outline

            # Draw crosshair
            cross_size = 20
            cv2.line(vis_image, (cx - cross_size, cy), (cx + cross_size, cy), CENTROID_COLOR, 2)
            cv2.line(vis_image, (cx, cy - cross_size), (cx, cy + cross_size), CENTROID_COLOR, 2)

            # Add label
            label = f"P{piece.piece_id}"
            cv2.putText(vis_image, label, (cx + 15, cy - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
            cv2.putText(vis_image, f"({cx},{cy})", (cx + 15, cy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

        # Draw red/blue dots for all chains
        for piece_id, endpoints_list in chain_endpoints.items():
            for endpoint_data in endpoints_list:
                blue_dot = endpoint_data['blue_dot']
                red_dot = endpoint_data['red_dot']

                # Draw blue dot
                bx, by = int(blue_dot[0]), int(blue_dot[1])
                cv2.circle(vis_image, (bx, by), 10, BLUE_DOT_COLOR, -1)
                cv2.circle(vis_image, (bx, by), 12, (0, 0, 0), 2)  # Black outline

                # Draw red dot
                rx, ry = int(red_dot[0]), int(red_dot[1])
                cv2.circle(vis_image, (rx, ry), 10, RED_DOT_COLOR, -1)
                cv2.circle(vis_image, (rx, ry), 12, (0, 0, 0), 2)  # Black outline

        # Add legend
        legend_x = 20
        legend_y = 30
        cv2.putText(vis_image, "Yellow: Centroids | Blue dots: Frame endpoints | Red dots: Interior endpoints",
                   (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

        # Add title
        title = "STEP 1: Calculate Centroids & Chain Endpoints"
        AssemblyVisualizer._add_title(vis_image, title)

        return vis_image

    @staticmethod
    def _add_title(image: np.ndarray, title: str, height: int = 60):
        """Add a title bar at the top of the image.

        Args:
            image: Image to add title to (modified in place)
            title: Title text
            height: Height of title bar in pixels
        """
        # Create title bar
        h, w = image.shape[:2]
        title_bar = np.zeros((height, w, 3), dtype=np.uint8)
        title_bar[:] = (50, 50, 50)  # Dark gray background

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        cv2.putText(title_bar, title, (text_x, text_y), font, font_scale,
                   (255, 255, 255), thickness)

        # Combine title bar with image
        combined = np.vstack([title_bar, image])

        # Replace image content
        image.resize(combined.shape, refcheck=False)
        image[:] = combined

    @staticmethod
    def visualize_step2_orientation(step_data: Dict) -> np.ndarray:
        """Visualize Step 2: Show pieces rotated to align with axes.

        Args:
            step_data: Dictionary containing 'normalized_pieces', 'piece_orientations', 'original_image'

        Returns:
            np.ndarray: Visualization image
        """
        normalized_pieces = step_data['normalized_pieces']
        piece_orientations = step_data['piece_orientations']
        original_image = step_data['original_image']

        # Create a copy of the original image
        vis_image = original_image.copy()
        vis_image[:] = 255  # White background

        # Define colors
        PIECE_COLOR = (200, 200, 255)  # Light blue fill
        PIECE_OUTLINE = (0, 0, 255)  # Blue outline
        AXIS_COLOR = (150, 150, 150)  # Gray for axes
        TEXT_COLOR = (0, 0, 0)  # Black text
        ROTATION_ARROW_COLOR = (255, 0, 0)  # Red for rotation indicator

        # Draw axis lines
        h, w = vis_image.shape[:2]
        # Horizontal axis
        cv2.line(vis_image, (0, h//2), (w, h//2), AXIS_COLOR, 1, cv2.LINE_AA)
        # Vertical axis
        cv2.line(vis_image, (w//2, 0), (w//2, h), AXIS_COLOR, 1, cv2.LINE_AA)

        # Draw each normalized piece
        for piece_id, piece_data in normalized_pieces.items():
            normalized_contour = piece_data['normalized_contour'].astype(np.int32)
            rotation = piece_data['rotation']
            centroid = piece_data['centroid']

            # Fill piece
            cv2.fillPoly(vis_image, [normalized_contour], PIECE_COLOR)

            # Draw outline
            cv2.polylines(vis_image, [normalized_contour], True, PIECE_OUTLINE, 2)

            # Draw centroid
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.circle(vis_image, (cx, cy), 5, (0, 255, 0), -1)

            # Draw rotation indicator (arc)
            if abs(rotation) > 1.0:  # Only show if significant rotation
                arc_radius = 30
                start_angle = 0
                end_angle = int(-rotation)  # Negative because cv2 uses clockwise

                # Draw arc
                cv2.ellipse(vis_image, (cx, cy), (arc_radius, arc_radius),
                           0, start_angle, end_angle, ROTATION_ARROW_COLOR, 2)

            # Add label with rotation angle
            label = f"P{piece_id}: {rotation:.1f}°"
            # Position label near piece
            label_x = cx + 40
            label_y = cy - 20

            cv2.putText(vis_image, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

        # Add axis labels
        cv2.putText(vis_image, "X-axis", (w - 80, h//2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, AXIS_COLOR, 1)
        cv2.putText(vis_image, "Y-axis", (w//2 + 10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, AXIS_COLOR, 1)

        # Add title
        title = "STEP 2: Normalize Orientations (Axis Alignment)"
        AssemblyVisualizer._add_title(vis_image, title)

        return vis_image

    @staticmethod
    def visualize_step3_anchor_selection(step_data: Dict) -> np.ndarray:
        """Visualize Step 3: Highlight the selected anchor piece.

        Args:
            step_data: Dictionary containing 'anchor_piece_id', 'normalized_pieces', 'piece_scores', 'original_image'

        Returns:
            np.ndarray: Visualization image
        """
        anchor_piece_id = step_data['anchor_piece_id']
        normalized_pieces = step_data['normalized_pieces']
        piece_scores = step_data['piece_scores']
        original_image = step_data['original_image']

        # Create a copy of the original image
        vis_image = original_image.copy()
        vis_image[:] = 255  # White background

        # Define colors
        REGULAR_PIECE_COLOR = (220, 220, 220)  # Light gray for non-anchor pieces
        REGULAR_OUTLINE = (150, 150, 150)  # Gray outline
        ANCHOR_PIECE_COLOR = (100, 255, 100)  # Bright green for anchor
        ANCHOR_OUTLINE = (0, 200, 0)  # Dark green outline
        TEXT_COLOR = (0, 0, 0)  # Black text
        SCORE_COLOR = (255, 0, 0)  # Red for scores
        CROWN_COLOR = (255, 215, 0)  # Gold for crown

        # Draw each piece
        for piece_id, piece_data in normalized_pieces.items():
            normalized_contour = piece_data['normalized_contour'].astype(np.int32)
            centroid = piece_data['centroid']

            # Choose color based on whether it's the anchor
            is_anchor = (piece_id == anchor_piece_id)
            piece_color = ANCHOR_PIECE_COLOR if is_anchor else REGULAR_PIECE_COLOR
            outline_color = ANCHOR_OUTLINE if is_anchor else REGULAR_OUTLINE
            outline_thickness = 4 if is_anchor else 2

            # Fill piece
            cv2.fillPoly(vis_image, [normalized_contour], piece_color)

            # Draw outline
            cv2.polylines(vis_image, [normalized_contour], True, outline_color, outline_thickness)

            # Draw centroid
            cx, cy = int(centroid[0]), int(centroid[1])
            centroid_color = (0, 150, 0) if is_anchor else (100, 100, 100)
            cv2.circle(vis_image, (cx, cy), 6, centroid_color, -1)

            # Add piece label
            label = f"P{piece_id}"
            if is_anchor:
                label = f"P{piece_id} (ANCHOR)"

            # Get score info
            score_info = piece_scores.get(piece_id, {})
            total_score = score_info.get('total_score', 0)

            # Position labels
            label_y = cy - 30
            cv2.putText(vis_image, label, (cx - 50, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

            # Add score
            score_text = f"Score: {total_score:.1f}"
            cv2.putText(vis_image, score_text, (cx - 50, label_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, SCORE_COLOR, 1)

            # Draw crown on anchor piece
            if is_anchor:
                # Simple crown shape (triangle with points)
                crown_y = cy - 60
                crown_size = 20
                crown_points = np.array([
                    [cx - crown_size, crown_y + crown_size],
                    [cx - crown_size//2, crown_y],
                    [cx, crown_y + crown_size],
                    [cx + crown_size//2, crown_y],
                    [cx + crown_size, crown_y + crown_size]
                ], dtype=np.int32)
                cv2.fillPoly(vis_image, [crown_points], CROWN_COLOR)
                cv2.polylines(vis_image, [crown_points], True, (200, 150, 0), 2)

        # Add legend
        legend_x = 20
        legend_y = 30
        cv2.putText(vis_image, "Anchor Piece: Highest connection scores",
                   (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
        cv2.rectangle(vis_image, (legend_x, legend_y + 10), (legend_x + 30, legend_y + 30),
                     ANCHOR_PIECE_COLOR, -1)
        cv2.rectangle(vis_image, (legend_x, legend_y + 10), (legend_x + 30, legend_y + 30),
                     ANCHOR_OUTLINE, 2)

        # Add title
        title = "STEP 3: Select Anchor Piece"
        AssemblyVisualizer._add_title(vis_image, title)

        return vis_image

    @staticmethod
    def visualize_step4_anchor_placement(step_data: Dict) -> np.ndarray:
        """Visualize Step 4: Show anchor piece positioned at top-left.

        Args:
            step_data: Dictionary containing 'placed_pieces', 'anchor_piece_id', 'normalized_pieces', 'original_image'

        Returns:
            np.ndarray: Visualization image
        """
        placed_pieces = step_data['placed_pieces']
        anchor_piece_id = step_data['anchor_piece_id']
        normalized_pieces = step_data['normalized_pieces']
        original_image = step_data['original_image']

        # Create a fresh canvas
        h, w = original_image.shape[:2]
        vis_image = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

        # Define colors
        PLACED_PIECE_COLOR = (150, 255, 150)  # Light green for placed piece
        PLACED_OUTLINE = (0, 200, 0)  # Green outline
        UNPLACED_PIECE_COLOR = (220, 220, 220)  # Light gray for unplaced pieces
        UNPLACED_OUTLINE = (150, 150, 150)  # Gray outline
        TEXT_COLOR = (0, 0, 0)  # Black text
        ORIGIN_COLOR = (255, 0, 0)  # Red for origin marker
        GRID_COLOR = (200, 200, 200)  # Light gray for grid

        # Draw light grid
        grid_spacing = 100
        for x in range(0, w, grid_spacing):
            cv2.line(vis_image, (x, 0), (x, h), GRID_COLOR, 1)
        for y in range(0, h, grid_spacing):
            cv2.line(vis_image, (0, y), (w, y), GRID_COLOR, 1)

        # Draw origin marker (top-left reference point)
        origin_x, origin_y = 100, 100
        cv2.drawMarker(vis_image, (origin_x, origin_y), ORIGIN_COLOR,
                      cv2.MARKER_CROSS, 30, 3)
        cv2.putText(vis_image, "Origin", (origin_x + 15, origin_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORIGIN_COLOR, 2)

        # Draw placed pieces only (don't show unplaced pieces in Step 4)
        for piece_id, placed_piece in placed_pieces.items():
            contour = placed_piece.contour_points.astype(np.int32)

            # Fill and outline
            cv2.fillPoly(vis_image, [contour], PLACED_PIECE_COLOR)
            cv2.polylines(vis_image, [contour], True, PLACED_OUTLINE, 3)

            # Draw centroid
            cx, cy = int(placed_piece.position[0]), int(placed_piece.position[1])
            cv2.circle(vis_image, (cx, cy), 8, (0, 150, 0), -1)

            # Add label
            label = f"P{piece_id} (PLACED)"
            cv2.putText(vis_image, label, (cx - 60, cy - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

            # Add position info
            pos_text = f"Pos: ({cx},{cy})"
            cv2.putText(vis_image, pos_text, (cx - 60, cy + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

        # Add legend
        legend_x = 20
        legend_y = h - 80
        cv2.putText(vis_image, "Green: Placed | Gray: Not yet placed",
                   (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)

        # Add title
        title = "STEP 4: Position Anchor at Origin"
        AssemblyVisualizer._add_title(vis_image, title)

        return vis_image

    @staticmethod
    def visualize_step5_second_piece(step_data: Dict) -> np.ndarray:
        """Visualize Step 5: Show second piece placed with red/blue dots.

        Args:
            step_data: Dictionary containing 'placed_pieces', 'anchor_piece_id', 'next_piece_id',
                      'normalized_pieces', 'original_image', and red/blue dots

        Returns:
            np.ndarray: Visualization image
        """
        placed_pieces = step_data['placed_pieces']
        anchor_piece_id = step_data['anchor_piece_id']
        next_piece_id = step_data['next_piece_id']
        normalized_pieces = step_data['normalized_pieces']
        original_image = step_data['original_image']

        # Get red/blue dots (after alignment)
        B1_placed = step_data.get('B1_placed')
        R1_placed = step_data.get('R1_placed')
        B2_final = step_data.get('B2_final')  # B2 after alignment (should be at B1)
        R2_final = step_data.get('R2_final')  # R2 after alignment (should be at R1)

        # Create a fresh canvas
        h, w = original_image.shape[:2]
        vis_image = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

        # Define colors
        PLACED_PIECE_COLOR = (150, 255, 150)  # Light green for placed pieces
        PLACED_OUTLINE = (0, 200, 0)  # Green outline
        ANCHOR_COLOR = (100, 200, 255)  # Light blue for anchor
        ANCHOR_OUTLINE = (0, 100, 200)  # Blue outline
        UNPLACED_PIECE_COLOR = (220, 220, 220)  # Light gray for unplaced pieces
        UNPLACED_OUTLINE = (150, 150, 150)  # Gray outline
        TEXT_COLOR = (0, 0, 0)  # Black text
        ORIGIN_COLOR = (255, 0, 0)  # Red for origin marker
        GRID_COLOR = (200, 200, 200)  # Light gray for grid
        CONNECTION_LINE_COLOR = (255, 150, 0)  # Orange for connection

        # Draw light grid
        grid_spacing = 100
        for x in range(0, w, grid_spacing):
            cv2.line(vis_image, (x, 0), (x, h), GRID_COLOR, 1)
        for y in range(0, h, grid_spacing):
            cv2.line(vis_image, (0, y), (w, y), GRID_COLOR, 1)

        # Draw origin marker
        origin_x, origin_y = 100, 100
        cv2.drawMarker(vis_image, (origin_x, origin_y), ORIGIN_COLOR,
                      cv2.MARKER_CROSS, 30, 3)
        cv2.putText(vis_image, "Origin", (origin_x + 15, origin_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORIGIN_COLOR, 2)

        # Draw unplaced pieces (dimmed)
        for piece_id, piece_data in normalized_pieces.items():
            if piece_id not in placed_pieces:
                normalized_contour = piece_data['normalized_contour'].astype(np.int32)
                cv2.fillPoly(vis_image, [normalized_contour], UNPLACED_PIECE_COLOR)
                cv2.polylines(vis_image, [normalized_contour], True, UNPLACED_OUTLINE, 1)

                centroid = piece_data['centroid']
                cx, cy = int(centroid[0]), int(centroid[1])
                cv2.putText(vis_image, f"P{piece_id}", (cx - 20, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Draw connection line between placed pieces
        if len(placed_pieces) >= 2:
            anchor_pos = placed_pieces[anchor_piece_id].position
            next_pos = placed_pieces[next_piece_id].position
            cv2.line(vis_image,
                    (int(anchor_pos[0]), int(anchor_pos[1])),
                    (int(next_pos[0]), int(next_pos[1])),
                    CONNECTION_LINE_COLOR, 2, cv2.LINE_AA)

        # Draw placed pieces
        for piece_id, placed_piece in placed_pieces.items():
            contour = placed_piece.contour_points.astype(np.int32)

            # Choose color - anchor is blue, new piece is green
            if piece_id == anchor_piece_id:
                piece_color = ANCHOR_COLOR
                outline_color = ANCHOR_OUTLINE
                label_suffix = "(ANCHOR)"
            elif piece_id == next_piece_id:
                piece_color = PLACED_PIECE_COLOR
                outline_color = PLACED_OUTLINE
                label_suffix = "(NEW)"
            else:
                piece_color = PLACED_PIECE_COLOR
                outline_color = PLACED_OUTLINE
                label_suffix = "(PLACED)"

            # Fill and outline
            cv2.fillPoly(vis_image, [contour], piece_color)
            cv2.polylines(vis_image, [contour], True, outline_color, 3)

            # Draw centroid
            cx, cy = int(placed_piece.position[0]), int(placed_piece.position[1])
            cv2.circle(vis_image, (cx, cy), 8, outline_color, -1)

            # Add label
            label = f"P{piece_id} {label_suffix}"
            cv2.putText(vis_image, label, (cx - 70, cy - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

            # Add position info
            pos_text = f"Pos: ({cx},{cy})"
            cv2.putText(vis_image, pos_text, (cx - 70, cy + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

        # Draw red/blue dots on the chains
        # Offset dots slightly so both B1/B2 and R1/R2 can be seen with equal spacing
        offset = 8  # Pixels to offset for visibility

        if B1_placed is not None and R1_placed is not None:
            # Blue dot on anchor piece (frame connection) - offset upward
            b1_x, b1_y = int(B1_placed[0]), int(B1_placed[1] - offset)
            cv2.circle(vis_image, (b1_x, b1_y), 12, (255, 0, 0), -1)  # Blue dot (BGR: blue)
            cv2.circle(vis_image, (b1_x, b1_y), 14, (0, 0, 0), 2)  # Black outline
            cv2.putText(vis_image, "B1", (b1_x + 20, b1_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Red dot on anchor piece (interior connection) - offset upward
            r1_x, r1_y = int(R1_placed[0]), int(R1_placed[1] - offset)
            cv2.circle(vis_image, (r1_x, r1_y), 12, (0, 0, 255), -1)  # Red dot (BGR: red)
            cv2.circle(vis_image, (r1_x, r1_y), 14, (0, 0, 0), 2)  # Black outline
            cv2.putText(vis_image, "R1", (r1_x + 20, r1_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if B2_final is not None and R2_final is not None:
            # Blue dot on next piece (should now be at B1 after alignment) - offset downward
            b2_x, b2_y = int(B2_final[0]), int(B2_final[1] + offset)
            cv2.circle(vis_image, (b2_x, b2_y), 10, (255, 0, 0), -1)  # Blue dot (BGR: blue)
            cv2.circle(vis_image, (b2_x, b2_y), 12, (0, 0, 0), 2)  # Black outline
            cv2.putText(vis_image, "B2", (b2_x + 20, b2_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Red dot on next piece (should now be at R1 after alignment) - offset downward
            r2_x, r2_y = int(R2_final[0]), int(R2_final[1] + offset)
            cv2.circle(vis_image, (r2_x, r2_y), 10, (0, 0, 255), -1)  # Red dot (BGR: red)
            cv2.circle(vis_image, (r2_x, r2_y), 12, (0, 0, 0), 2)  # Black outline
            cv2.putText(vis_image, "R2", (r2_x + 20, r2_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add legend
        legend_x = 20
        legend_y = h - 120
        cv2.putText(vis_image, "Blue: Anchor | Green: New piece | Gray: Not placed",
                   (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
        cv2.putText(vis_image, f"Orange line: Connection (chain match)",
                   (legend_x, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
        cv2.putText(vis_image, f"Red/Blue dots: Chain endpoints (B=frame, R=interior)",
                   (legend_x, legend_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)

        # Add title
        title = "STEP 5: Place Second Piece (with Red/Blue Dots)"
        AssemblyVisualizer._add_title(vis_image, title)

        return vis_image

    @staticmethod
    def visualize_step6_third_piece(step_data: Dict) -> np.ndarray:
        """Visualize Step 6: Show third piece placed with red/blue dots.

        Args:
            step_data: Dictionary containing 'placed_pieces', 'placed_piece_id', 'next_piece_id',
                      'normalized_pieces', 'original_image', and red/blue dots

        Returns:
            np.ndarray: Visualization image
        """
        placed_pieces = step_data['placed_pieces']
        placed_piece_id = step_data['placed_piece_id']
        next_piece_id = step_data['next_piece_id']
        normalized_pieces = step_data['normalized_pieces']
        original_image = step_data['original_image']

        # Get red/blue dots (after alignment)
        B1_placed = step_data.get('B1_placed')
        R1_placed = step_data.get('R1_placed')
        B2_final = step_data.get('B2_final')  # B2 after alignment (should be at B1)
        R2_final = step_data.get('R2_final')  # R2 after alignment (should be at R1)

        # Create a fresh canvas
        h, w = original_image.shape[:2]
        vis_image = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

        # Define colors
        PLACED_PIECE_COLOR = (150, 255, 150)  # Light green for placed pieces
        PLACED_OUTLINE = (0, 200, 0)  # Green outline
        NEW_PIECE_COLOR = (255, 200, 150)  # Light orange for new piece
        NEW_OUTLINE = (200, 100, 0)  # Orange outline
        UNPLACED_PIECE_COLOR = (220, 220, 220)  # Light gray for unplaced pieces
        UNPLACED_OUTLINE = (150, 150, 150)  # Gray outline
        TEXT_COLOR = (0, 0, 0)  # Black text
        ORIGIN_COLOR = (255, 0, 0)  # Red for origin marker
        GRID_COLOR = (200, 200, 200)  # Light gray for grid
        CONNECTION_LINE_COLOR = (255, 150, 0)  # Orange for connection

        # Draw light grid
        grid_spacing = 100
        for x in range(0, w, grid_spacing):
            cv2.line(vis_image, (x, 0), (x, h), GRID_COLOR, 1)
        for y in range(0, h, grid_spacing):
            cv2.line(vis_image, (0, y), (w, y), GRID_COLOR, 1)

        # Draw origin marker
        origin_x, origin_y = 100, 100
        cv2.drawMarker(vis_image, (origin_x, origin_y), ORIGIN_COLOR,
                      cv2.MARKER_CROSS, 30, 3)
        cv2.putText(vis_image, "Origin", (origin_x + 15, origin_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORIGIN_COLOR, 2)

        # Draw unplaced pieces (dimmed)
        for piece_id, piece_data in normalized_pieces.items():
            if piece_id not in placed_pieces:
                normalized_contour = piece_data['normalized_contour'].astype(np.int32)
                cv2.fillPoly(vis_image, [normalized_contour], UNPLACED_PIECE_COLOR)
                cv2.polylines(vis_image, [normalized_contour], True, UNPLACED_OUTLINE, 1)

                centroid = piece_data['centroid']
                cx, cy = int(centroid[0]), int(centroid[1])
                cv2.putText(vis_image, f"P{piece_id}", (cx - 20, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Draw connection line between the two pieces
        if placed_piece_id in placed_pieces and next_piece_id in placed_pieces:
            placed_pos = placed_pieces[placed_piece_id].position
            next_pos = placed_pieces[next_piece_id].position
            cv2.line(vis_image,
                    (int(placed_pos[0]), int(placed_pos[1])),
                    (int(next_pos[0]), int(next_pos[1])),
                    CONNECTION_LINE_COLOR, 2, cv2.LINE_AA)

        # Draw placed pieces
        for piece_id, placed_piece in placed_pieces.items():
            contour = placed_piece.contour_points.astype(np.int32)

            # Choose color - new piece is orange, others are green
            if piece_id == next_piece_id:
                piece_color = NEW_PIECE_COLOR
                outline_color = NEW_OUTLINE
                label_suffix = "(NEW)"
            else:
                piece_color = PLACED_PIECE_COLOR
                outline_color = PLACED_OUTLINE
                label_suffix = "(PLACED)"

            # Fill and outline
            cv2.fillPoly(vis_image, [contour], piece_color)
            cv2.polylines(vis_image, [contour], True, outline_color, 3)

            # Draw centroid
            cx, cy = int(placed_piece.position[0]), int(placed_piece.position[1])
            cv2.circle(vis_image, (cx, cy), 8, outline_color, -1)

            # Add label
            label = f"P{piece_id} {label_suffix}"
            cv2.putText(vis_image, label, (cx - 70, cy - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

            # Add position info
            pos_text = f"Pos: ({cx},{cy})"
            cv2.putText(vis_image, pos_text, (cx - 70, cy + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

        # Draw red/blue dots on the chains
        offset = 8  # Pixels to offset for visibility

        if B1_placed is not None and R1_placed is not None:
            # Blue dot - offset upward
            b1_x, b1_y = int(B1_placed[0]), int(B1_placed[1] - offset)
            cv2.circle(vis_image, (b1_x, b1_y), 12, (255, 0, 0), -1)  # Blue dot (BGR: blue)
            cv2.circle(vis_image, (b1_x, b1_y), 14, (0, 0, 0), 2)  # Black outline
            cv2.putText(vis_image, "B1", (b1_x + 20, b1_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Red dot - offset upward
            r1_x, r1_y = int(R1_placed[0]), int(R1_placed[1] - offset)
            cv2.circle(vis_image, (r1_x, r1_y), 12, (0, 0, 255), -1)  # Red dot (BGR: red)
            cv2.circle(vis_image, (r1_x, r1_y), 14, (0, 0, 0), 2)  # Black outline
            cv2.putText(vis_image, "R1", (r1_x + 20, r1_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if B2_final is not None and R2_final is not None:
            # Blue dot - offset downward
            b2_x, b2_y = int(B2_final[0]), int(B2_final[1] + offset)
            cv2.circle(vis_image, (b2_x, b2_y), 10, (255, 0, 0), -1)  # Blue dot (BGR: blue)
            cv2.circle(vis_image, (b2_x, b2_y), 12, (0, 0, 0), 2)  # Black outline
            cv2.putText(vis_image, "B2", (b2_x + 20, b2_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Red dot - offset downward
            r2_x, r2_y = int(R2_final[0]), int(R2_final[1] + offset)
            cv2.circle(vis_image, (r2_x, r2_y), 10, (0, 0, 255), -1)  # Red dot (BGR: red)
            cv2.circle(vis_image, (r2_x, r2_y), 12, (0, 0, 0), 2)  # Black outline
            cv2.putText(vis_image, "R2", (r2_x + 20, r2_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add legend
        legend_x = 20
        legend_y = h - 120
        cv2.putText(vis_image, "Green: Previously placed | Orange: New piece | Gray: Not placed",
                   (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
        cv2.putText(vis_image, f"Orange line: Connection (chain match)",
                   (legend_x, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
        cv2.putText(vis_image, f"Red/Blue dots: Chain endpoints (B=frame, R=interior)",
                   (legend_x, legend_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)

        # Add title
        title = "STEP 6: Place Third Piece (with Red/Blue Dots)"
        AssemblyVisualizer._add_title(vis_image, title)

        return vis_image

    @staticmethod
    def visualize_step7_fourth_piece(step_data: Dict) -> np.ndarray:
        """Visualize Step 7: Show fourth piece placed with red/blue dots.

        Args:
            step_data: Dictionary containing 'placed_pieces', 'placed_piece_id', 'next_piece_id',
                      'normalized_pieces', 'original_image', and red/blue dots

        Returns:
            np.ndarray: Visualization image
        """
        placed_pieces = step_data['placed_pieces']
        placed_piece_id = step_data['placed_piece_id']
        next_piece_id = step_data['next_piece_id']
        normalized_pieces = step_data['normalized_pieces']
        original_image = step_data['original_image']

        # Get red/blue dots (after alignment)
        B1_placed = step_data.get('B1_placed')
        R1_placed = step_data.get('R1_placed')
        B2_final = step_data.get('B2_final')  # B2 after alignment (should be at B1)
        R2_final = step_data.get('R2_final')  # R2 after alignment (should be at R1)

        # Create a fresh canvas
        h, w = original_image.shape[:2]
        vis_image = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

        # Define colors
        PLACED_PIECE_COLOR = (150, 255, 150)  # Light green for placed pieces
        PLACED_OUTLINE = (0, 200, 0)  # Green outline
        NEW_PIECE_COLOR = (200, 150, 255)  # Light purple for new piece
        NEW_OUTLINE = (100, 0, 200)  # Purple outline
        TEXT_COLOR = (0, 0, 0)  # Black text
        ORIGIN_COLOR = (255, 0, 0)  # Red for origin marker
        GRID_COLOR = (200, 200, 200)  # Light gray for grid
        CONNECTION_LINE_COLOR = (255, 150, 0)  # Orange for connection

        # Draw light grid
        grid_spacing = 100
        for x in range(0, w, grid_spacing):
            cv2.line(vis_image, (x, 0), (x, h), GRID_COLOR, 1)
        for y in range(0, h, grid_spacing):
            cv2.line(vis_image, (0, y), (w, y), GRID_COLOR, 1)

        # Draw origin marker
        origin_x, origin_y = 100, 100
        cv2.drawMarker(vis_image, (origin_x, origin_y), ORIGIN_COLOR,
                      cv2.MARKER_CROSS, 30, 3)
        cv2.putText(vis_image, "Origin", (origin_x + 15, origin_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORIGIN_COLOR, 2)

        # Draw connection line between the two pieces
        if placed_piece_id in placed_pieces and next_piece_id in placed_pieces:
            placed_pos = placed_pieces[placed_piece_id].position
            next_pos = placed_pieces[next_piece_id].position
            cv2.line(vis_image,
                    (int(placed_pos[0]), int(placed_pos[1])),
                    (int(next_pos[0]), int(next_pos[1])),
                    CONNECTION_LINE_COLOR, 2, cv2.LINE_AA)

        # Draw all placed pieces
        for piece_id, placed_piece in placed_pieces.items():
            contour = placed_piece.contour_points.astype(np.int32)

            # Choose color - new piece is purple, others are green
            if piece_id == next_piece_id:
                piece_color = NEW_PIECE_COLOR
                outline_color = NEW_OUTLINE
                label_suffix = "(NEW/FINAL)"
            else:
                piece_color = PLACED_PIECE_COLOR
                outline_color = PLACED_OUTLINE
                label_suffix = "(PLACED)"

            # Fill and outline
            cv2.fillPoly(vis_image, [contour], piece_color)
            cv2.polylines(vis_image, [contour], True, outline_color, 3)

            # Draw centroid
            cx, cy = int(placed_piece.position[0]), int(placed_piece.position[1])
            cv2.circle(vis_image, (cx, cy), 8, outline_color, -1)

            # Add label
            label = f"P{piece_id} {label_suffix}"
            cv2.putText(vis_image, label, (cx - 70, cy - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

            # Add position info
            pos_text = f"Pos: ({cx},{cy})"
            cv2.putText(vis_image, pos_text, (cx - 70, cy + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

        # Draw red/blue dots on the chains
        offset = 8  # Pixels to offset for visibility

        if B1_placed is not None and R1_placed is not None:
            # Blue dot - offset upward
            b1_x, b1_y = int(B1_placed[0]), int(B1_placed[1] - offset)
            cv2.circle(vis_image, (b1_x, b1_y), 12, (255, 0, 0), -1)  # Blue dot (BGR: blue)
            cv2.circle(vis_image, (b1_x, b1_y), 14, (0, 0, 0), 2)  # Black outline
            cv2.putText(vis_image, "B1", (b1_x + 20, b1_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Red dot - offset upward
            r1_x, r1_y = int(R1_placed[0]), int(R1_placed[1] - offset)
            cv2.circle(vis_image, (r1_x, r1_y), 12, (0, 0, 255), -1)  # Red dot (BGR: red)
            cv2.circle(vis_image, (r1_x, r1_y), 14, (0, 0, 0), 2)  # Black outline
            cv2.putText(vis_image, "R1", (r1_x + 20, r1_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if B2_final is not None and R2_final is not None:
            # Blue dot - offset downward
            b2_x, b2_y = int(B2_final[0]), int(B2_final[1] + offset)
            cv2.circle(vis_image, (b2_x, b2_y), 10, (255, 0, 0), -1)  # Blue dot (BGR: blue)
            cv2.circle(vis_image, (b2_x, b2_y), 12, (0, 0, 0), 2)  # Black outline
            cv2.putText(vis_image, "B2", (b2_x + 20, b2_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Red dot - offset downward
            r2_x, r2_y = int(R2_final[0]), int(R2_final[1] + offset)
            cv2.circle(vis_image, (r2_x, r2_y), 10, (0, 0, 255), -1)  # Red dot (BGR: red)
            cv2.circle(vis_image, (r2_x, r2_y), 12, (0, 0, 0), 2)  # Black outline
            cv2.putText(vis_image, "R2", (r2_x + 20, r2_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add legend
        legend_x = 20
        legend_y = h - 120
        cv2.putText(vis_image, "Green: Previously placed | Purple: Final piece",
                   (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
        cv2.putText(vis_image, f"Orange line: Connection (chain match)",
                   (legend_x, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
        cv2.putText(vis_image, f"Red/Blue dots: Chain endpoints (B=frame, R=interior)",
                   (legend_x, legend_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)

        # Add title
        title = "STEP 7: Place Fourth Piece (Final Piece with Red/Blue Dots)"
        AssemblyVisualizer._add_title(vis_image, title)

        return vis_image

    @staticmethod
    def visualize_step8_final_assembly(step_data: Dict) -> np.ndarray:
        """Visualize Step 8: Show all pieces assembled together.

        Args:
            step_data: Dictionary containing 'placed_pieces', 'original_image'

        Returns:
            np.ndarray: Visualization image
        """
        placed_pieces = step_data['placed_pieces']
        original_image = step_data['original_image']

        # Create a fresh canvas
        h, w = original_image.shape[:2]
        vis_image = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

        # Define colors
        PIECE_COLORS = [
            (150, 255, 150),  # Light green
            (255, 200, 150),  # Light orange
            (200, 150, 255),  # Light purple
            (255, 255, 150),  # Light yellow
        ]
        OUTLINE_COLOR = (0, 150, 0)  # Dark green outline
        TEXT_COLOR = (0, 0, 0)  # Black text
        GRID_COLOR = (230, 230, 230)  # Very light gray for grid

        # Draw light grid
        grid_spacing = 100
        for x in range(0, w, grid_spacing):
            cv2.line(vis_image, (x, 0), (x, h), GRID_COLOR, 1)
        for y in range(0, h, grid_spacing):
            cv2.line(vis_image, (0, y), (w, y), GRID_COLOR, 1)

        # Draw all placed pieces
        for piece_id, placed_piece in placed_pieces.items():
            contour = placed_piece.contour_points.astype(np.int32)

            # Choose color based on piece ID
            piece_color = PIECE_COLORS[piece_id % len(PIECE_COLORS)]

            # Fill and outline
            cv2.fillPoly(vis_image, [contour], piece_color)
            cv2.polylines(vis_image, [contour], True, OUTLINE_COLOR, 3)

            # Draw centroid
            cx, cy = int(placed_piece.position[0]), int(placed_piece.position[1])
            cv2.circle(vis_image, (cx, cy), 8, OUTLINE_COLOR, -1)

            # Add label
            label = f"P{piece_id}"
            cv2.putText(vis_image, label, (cx - 20, cy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

        # Add legend
        legend_x = 20
        legend_y = h - 60
        cv2.putText(vis_image, f"Assembly complete: {len(placed_pieces)} pieces placed",
                   (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

        # Add title
        title = "STEP 8: Final Assembly"
        AssemblyVisualizer._add_title(vis_image, title)

        return vis_image

    @staticmethod
    def create_combined_visualization(step_images: List[np.ndarray],
                                     spacing: int = 20) -> np.ndarray:
        """Combine multiple step visualizations into one long vertical image.

        Args:
            step_images: List of visualization images for each step
            spacing: Vertical spacing between steps in pixels

        Returns:
            np.ndarray: Combined visualization image
        """
        if not step_images:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        # Find maximum width
        max_width = max(img.shape[1] for img in step_images)

        # Resize all images to have the same width
        resized_images = []
        for img in step_images:
            if img.shape[1] < max_width:
                # Pad image to match max width
                padding = max_width - img.shape[1]
                padded = np.pad(img, ((0, 0), (0, padding), (0, 0)),
                               mode='constant', constant_values=0)
                resized_images.append(padded)
            else:
                resized_images.append(img)

        # Create spacer
        spacer = np.zeros((spacing, max_width, 3), dtype=np.uint8)
        spacer[:] = (100, 100, 100)  # Gray spacer

        # Combine images with spacers
        combined_parts = []
        for i, img in enumerate(resized_images):
            combined_parts.append(img)
            if i < len(resized_images) - 1:
                combined_parts.append(spacer)

        # Stack vertically
        combined = np.vstack(combined_parts)

        return combined
