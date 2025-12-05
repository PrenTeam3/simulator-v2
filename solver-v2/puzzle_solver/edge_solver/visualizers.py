"""Module for visualization functions for the edge solver."""

import cv2
from ..common.utils import calculate_segment_center


class EdgeSolverVisualizer:
    """Handles visualization of connections and assembly for edge solver."""

    @staticmethod
    def visualize_connections_svg(piece_connections, piece_frame_segments, all_matches, original_image, project_root):
        """Create SVG visualization showing connections between pieces.

        Args:
            piece_connections: Dictionary of piece_id -> list of connections
            piece_frame_segments: List of tuples (piece, segments, frame_adjacent_segments)
            all_matches: List of all matches
            original_image: Original image array
            project_root: Path to project root
        """
        from ..common.data_classes import ExtendedSegmentMatch
        # Get image dimensions
        img_height, img_width = original_image.shape[:2]

        # Start SVG
        svg_lines = []
        svg_lines.append(f'<svg width="{img_width}" height="{img_height}" xmlns="http://www.w3.org/2000/svg">')
        svg_lines.append(f'  <!-- Edge-based solver connections visualization -->')

        # Draw piece contours
        svg_lines.append('  <!-- Piece contours -->')
        for piece, segments, frame_segs in piece_frame_segments:
            # Draw all segments for this piece
            for seg in segments:
                points = ' '.join([f'{p.x},{p.y}' for p in seg.contour_points])
                color = "#FFA500" if seg in frame_segs else "#808080"
                svg_lines.append(f'  <polyline points="{points}" '
                               f'fill="none" stroke="{color}" stroke-width="2" opacity="0.5"/>')

            # Draw centroid
            svg_lines.append(f'  <circle cx="{piece.centroid.x}" cy="{piece.centroid.y}" r="5" '
                           f'fill="blue" opacity="0.7"/>')
            # Label piece
            svg_lines.append(f'  <text x="{piece.centroid.x + 10}" y="{piece.centroid.y + 5}" '
                           f'font-size="20" fill="blue" font-weight="bold">P{piece.piece_id}</text>')

        # Draw connection lines between matching segments
        svg_lines.append('  <!-- Connection lines -->')
        used_connections = set()

        for piece_id, connections in piece_connections.items():
            # Get piece segments
            piece, segments, _ = next((p, s, f) for p, s, f in piece_frame_segments if p.piece_id == piece_id)

            for connected_piece_id, match, side in connections:
                # Avoid drawing the same connection twice
                connection_key = tuple(sorted([piece_id, connected_piece_id]))
                if connection_key in used_connections:
                    continue
                used_connections.add(connection_key)

                # Get connected piece segments
                connected_piece, connected_segments, _ = next((p, s, f) for p, s, f in piece_frame_segments if p.piece_id == connected_piece_id)

                # Find the matching segments - handle both ExtendedSegmentMatch and regular SegmentMatch
                if isinstance(match, ExtendedSegmentMatch):
                    # For extended matches, get all matched segment IDs in the chain
                    all_matches_in_chain = [match.initial_match] + match.extended_matches
                    score = match.combined_score
                    chain_length = match.total_segments_matched

                    # Collect all segment IDs for piece1 and piece2
                    seg1_ids = []
                    seg2_ids = []
                    for m in all_matches_in_chain:
                        if m.piece1_id == piece_id:
                            seg1_ids.append(m.seg1_id)
                            seg2_ids.append(m.seg2_id)
                        else:
                            seg1_ids.append(m.seg2_id)
                            seg2_ids.append(m.seg1_id)

                    # Get all segments in the chain
                    chain_segs1 = [s for s in segments if s.segment_id in seg1_ids]
                    chain_segs2 = [s for s in connected_segments if s.segment_id in seg2_ids]

                    # Calculate center of the entire chain for each piece
                    all_points1 = []
                    for seg in chain_segs1:
                        all_points1.extend([(p.x, p.y) for p in seg.contour_points])

                    all_points2 = []
                    for seg in chain_segs2:
                        all_points2.extend([(p.x, p.y) for p in seg.contour_points])

                    # Calculate chain center
                    if all_points1 and all_points2:
                        seg1_center_x = sum(p[0] for p in all_points1) / len(all_points1)
                        seg1_center_y = sum(p[1] for p in all_points1) / len(all_points1)
                        seg2_center_x = sum(p[0] for p in all_points2) / len(all_points2)
                        seg2_center_y = sum(p[1] for p in all_points2) / len(all_points2)
                    else:
                        continue

                    # Highlight all segments in the chain with thicker stroke
                    for seg in chain_segs1:
                        points = ' '.join([f'{p.x},{p.y}' for p in seg.contour_points])
                        svg_lines.append(f'  <polyline points="{points}" '
                                       f'fill="none" stroke="#00FFFF" stroke-width="4" opacity="0.8"/>')

                    for seg in chain_segs2:
                        points = ' '.join([f'{p.x},{p.y}' for p in seg.contour_points])
                        svg_lines.append(f'  <polyline points="{points}" '
                                       f'fill="none" stroke="#FF00FF" stroke-width="4" opacity="0.8"/>')

                else:
                    # Regular single segment match
                    seg1_id = match.seg1_id if match.piece1_id == piece_id else match.seg2_id
                    seg2_id = match.seg2_id if match.piece1_id == piece_id else match.seg1_id
                    score = match.match_score
                    chain_length = 1

                    seg1 = next(s for s in segments if s.segment_id == seg1_id)
                    seg2 = next(s for s in connected_segments if s.segment_id == seg2_id)

                    # Calculate center of each segment
                    seg1_center_x, seg1_center_y = calculate_segment_center(seg1)
                    seg2_center_x, seg2_center_y = calculate_segment_center(seg2)

                # Determine color based on match score
                if score >= 0.8:
                    color = "#00FF00"  # Green for high score
                elif score >= 0.6:
                    color = "#FFFF00"  # Yellow for medium score
                else:
                    color = "#FF0000"  # Red for low score

                # Draw line between segment centers
                svg_lines.append(f'  <line x1="{seg1_center_x}" y1="{seg1_center_y}" '
                               f'x2="{seg2_center_x}" y2="{seg2_center_y}" '
                               f'stroke="{color}" stroke-width="3" opacity="0.8"/>')

                # Add score label at midpoint with chain length info
                mid_x = (seg1_center_x + seg2_center_x) / 2
                mid_y = (seg1_center_y + seg2_center_y) / 2
                label_text = f'{score:.2f} (L={chain_length})' if chain_length > 1 else f'{score:.2f}'
                svg_lines.append(f'  <text x="{mid_x}" y="{mid_y}" '
                               f'font-size="14" fill="{color}" font-weight="bold" '
                               f'text-anchor="middle">{label_text}</text>')

        # Add legend
        svg_lines.append('  <!-- Legend -->')
        legend_x = 10
        legend_y = 30
        svg_lines.append(f'  <text x="{legend_x}" y="{legend_y}" font-size="16" fill="black" font-weight="bold">Legend:</text>')
        svg_lines.append(f'  <line x1="{legend_x}" y1="{legend_y + 10}" x2="{legend_x + 50}" y2="{legend_y + 10}" '
                       f'stroke="#00FF00" stroke-width="3"/>')
        svg_lines.append(f'  <text x="{legend_x + 60}" y="{legend_y + 15}" font-size="14" fill="black">High match (0.8+)</text>')
        svg_lines.append(f'  <line x1="{legend_x}" y1="{legend_y + 30}" x2="{legend_x + 50}" y2="{legend_y + 30}" '
                       f'stroke="#FFFF00" stroke-width="3"/>')
        svg_lines.append(f'  <text x="{legend_x + 60}" y="{legend_y + 35}" font-size="14" fill="black">Medium match (0.6+)</text>')
        svg_lines.append(f'  <line x1="{legend_x}" y1="{legend_y + 50}" x2="{legend_x + 50}" y2="{legend_y + 50}" '
                       f'stroke="#FF0000" stroke-width="3"/>')
        svg_lines.append(f'  <text x="{legend_x + 60}" y="{legend_y + 55}" font-size="14" fill="black">Low match (&lt;0.6)</text>')
        svg_lines.append(f'  <line x1="{legend_x}" y1="{legend_y + 70}" x2="{legend_x + 50}" y2="{legend_y + 70}" '
                       f'stroke="#00FFFF" stroke-width="4"/>')
        svg_lines.append(f'  <text x="{legend_x + 60}" y="{legend_y + 75}" font-size="14" fill="black">Chain segments (piece 1)</text>')
        svg_lines.append(f'  <line x1="{legend_x}" y1="{legend_y + 90}" x2="{legend_x + 50}" y2="{legend_y + 90}" '
                       f'stroke="#FF00FF" stroke-width="4"/>')
        svg_lines.append(f'  <text x="{legend_x + 60}" y="{legend_y + 95}" font-size="14" fill="black">Chain segments (piece 2)</text>')

        svg_lines.append('</svg>')

        # Write SVG file
        output_path = project_root / 'temp' / 'edge_solver_connections.svg'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(svg_lines))

        print(f"\nSVG visualization saved to: {output_path}")

    @staticmethod
    def visualize_connections_dialog(piece_connections, piece_frame_segments, original_image, project_root):
        """Display connections visualization from SVG file in an interactive OpenCV window.

        Args:
            piece_connections: Dictionary of piece_id -> list of connections
            piece_frame_segments: List of tuples (piece, segments, frame_adjacent_segments)
            original_image: Original image array
            project_root: Path to project root
        """
        from ..common import InteractiveImageViewer
        from pathlib import Path

        # Load the SVG file and convert to image
        svg_path = project_root / 'temp' / 'edge_solver_connections.svg'

        # Use svglib and reportlab to convert SVG to PIL Image, then to numpy
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            import numpy as np

            # Convert SVG to ReportLab drawing
            drawing = svg2rlg(str(svg_path))

            # Render to PIL Image
            pil_img = renderPM.drawToPIL(drawing)

            # Convert PIL to numpy array (RGB)
            img_rgb = np.array(pil_img)

            # Convert RGB to BGR for OpenCV
            img_display = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        except ImportError:
            # Fallback: use original image with simple overlay
            print("Warning: svglib/reportlab not installed. Showing original image instead.")
            print("Install with: pip install svglib reportlab")
            img_display = original_image.copy()

            # Add text overlay
            cv2.putText(img_display, 'SVG saved to temp/edge_solver_connections.svg',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img_display, 'Install svglib and reportlab to display SVG in dialog',
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display using interactive viewer
        print("\nDisplaying edge solver connections visualization...")
        viewer = InteractiveImageViewer("Edge Solver Connections")
        viewer.show(img_display)

    @staticmethod
    def visualize_final_solution_dialog(svg_path):
        """Display final solution SVG in an interactive OpenCV window.

        Args:
            svg_path: Path to the SVG file
        """
        from ..common import InteractiveImageViewer

        # Load and convert SVG to image
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            import numpy as np

            # Convert SVG to ReportLab drawing
            drawing = svg2rlg(str(svg_path))

            # Render to PIL Image
            pil_img = renderPM.drawToPIL(drawing)

            # Convert PIL to numpy array (RGB)
            img_rgb = np.array(pil_img)

            # Convert RGB to BGR for OpenCV
            img_display = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        except ImportError:
            print("Warning: svglib/reportlab not installed.")
            print("Install with: pip install svglib reportlab")
            print(f"SVG saved to: {svg_path}")
            return

        # Display using interactive viewer
        print("\nDisplaying final solution...")
        viewer = InteractiveImageViewer("Final Solution - All Pieces")
        viewer.show(img_display)

    @staticmethod
    def assemble_and_display_pieces(piece_connections, piece_frame_segments, all_segments, original_image):
        """Assemble puzzle pieces by rotating one piece so frame corner is at top-left.

        Args:
            piece_connections: Dictionary of piece_id -> list of connections
            piece_frame_segments: List of tuples (piece, segments, frame_adjacent_segments)
            all_segments: List of all segments for all pieces
            original_image: Original image array
        """
        import numpy as np
        from ..common import InteractiveImageViewer

        # Pick the first piece with a frame corner
        selected_piece = None
        selected_segments = None
        for piece, segments, frame_segs in piece_frame_segments:
            if len(piece.frame_corners) > 0:
                selected_piece = piece
                selected_segments = segments
                break

        if selected_piece is None:
            print("No piece with frame corner found!")
            return

        print(f"\nSelected Piece {selected_piece.piece_id} for assembly")
        print(f"Frame corner at: ({selected_piece.frame_corners[0].x:.0f}, {selected_piece.frame_corners[0].y:.0f})")

        # Get all contour points for this piece
        all_points = []
        for seg in selected_segments:
            all_points.extend([(int(p.x), int(p.y)) for p in seg.contour_points])

        # Create mask for this piece
        contour = np.array(all_points, dtype=np.int32)
        mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)

        # Extract piece image
        piece_img = cv2.bitwise_and(original_image, original_image, mask=mask)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Crop to bounding box
        piece_cropped = piece_img[y:y+h, x:x+w].copy()
        mask_cropped = mask[y:y+h, x:x+w].copy()

        # Get frame corner position in the cropped image
        frame_corner = selected_piece.frame_corners[0]
        frame_x_in_crop = int(frame_corner.x - x)
        frame_y_in_crop = int(frame_corner.y - y)

        print(f"Frame corner in cropped image: ({frame_x_in_crop}, {frame_y_in_crop})")

        # Calculate angle to rotate frame corner to top-left
        # We want the frame corner to point towards top-left (0, 0)
        center_x = w // 2
        center_y = h // 2

        # Vector from center to frame corner
        dx = frame_x_in_crop - center_x
        dy = frame_y_in_crop - center_y

        # Calculate current angle of frame corner
        current_angle = np.degrees(np.arctan2(dy, dx))

        # Target angle for top-left is -135 degrees (or 225 degrees)
        target_angle = -135

        # Rotation needed
        rotation_angle = target_angle - current_angle

        print(f"Current angle: {current_angle:.1f}°, Target: {target_angle:.1f}°, Rotation: {rotation_angle:.1f}°")

        # Rotate the piece
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

        # Calculate new bounding box size after rotation
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust transformation matrix for new size
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Rotate both image and mask
        rotated_img = cv2.warpAffine(piece_cropped, M, (new_w, new_h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))
        rotated_mask = cv2.warpAffine(mask_cropped, M, (new_w, new_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)

        # Create final image with white background
        final_img = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255

        # Copy rotated piece
        for i in range(new_h):
            for j in range(new_w):
                if rotated_mask[i, j] > 0:
                    final_img[i, j] = rotated_img[i, j]

        # Display assembled puzzle
        print(f"\nDisplaying assembled puzzle (P{selected_piece.piece_id} rotated {rotation_angle:.1f}°)...")
        viewer = InteractiveImageViewer("Assembled Puzzle - Frame Corner at Top-Left")
        viewer.show(final_img)
