"""Module for building puzzle solutions and creating SVG outputs."""

import cv2
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from ..common.utils import calculate_segment_center


class SolutionBuilder:
    """Handles building puzzle solutions and creating SVG visualizations."""

    @staticmethod
    def create_solution_svg(piece_frame_segments, rotation_angles, original_image, project_root, piece_connections=None, all_matches=None):
        """Create SVG file with the first piece rotated and positioned in top-left corner.

        Args:
            piece_frame_segments: List of tuples (piece, segments, frame_adjacent_segments)
            rotation_angles: Dictionary of piece_id -> rotation_angle_degrees
            original_image: Original image array
            project_root: Path to project root
            piece_connections: Optional dictionary of piece connections
            all_matches: Optional list of all matches
        """
        # Select the first piece
        first_piece, first_segments, _ = piece_frame_segments[0]
        rotation_angle = rotation_angles[first_piece.piece_id]

        print(f"Creating solution with Piece {first_piece.piece_id}, rotation: {rotation_angle:.1f}°")

        # Get all contour points for this piece
        all_points = []
        for seg in first_segments:
            all_points.extend([(int(p.x), int(p.y)) for p in seg.contour_points])

        # Get bounding box
        contour = np.array(all_points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate canvas size (add some padding)
        padding = 50
        canvas_width = w + 2 * padding
        canvas_height = h + 2 * padding

        # Start SVG
        svg_lines = []
        svg_lines.append(f'<svg width="{canvas_width}" height="{canvas_height}" xmlns="http://www.w3.org/2000/svg">')
        svg_lines.append(f'  <!-- Solution SVG - Piece {first_piece.piece_id} -->')
        svg_lines.append(f'  <!-- Rotation: {rotation_angle:.1f}° -->')

        # Create a group for the rotated piece
        # Position the piece in top-left corner (with padding)
        piece_center_x = x + w / 2
        piece_center_y = y + h / 2

        # Target position (top-left with padding, accounting for piece center)
        target_x = padding + w / 2
        target_y = padding + h / 2

        # Create transform: translate to origin, rotate, translate to target position
        svg_lines.append(f'  <g transform="translate({target_x},{target_y}) rotate({rotation_angle}) translate({-piece_center_x},{-piece_center_y})">')

        # Draw all segments
        for seg in first_segments:
            points = ' '.join([f'{p.x},{p.y}' for p in seg.contour_points])
            color = "#FF6B6B" if seg.is_border_edge else "#4ECDC4"
            svg_lines.append(f'    <polyline points="{points}" '
                           f'fill="none" stroke="{color}" stroke-width="2"/>')

        # Draw centroid
        svg_lines.append(f'    <circle cx="{first_piece.centroid.x}" cy="{first_piece.centroid.y}" '
                       f'r="3" fill="blue"/>')

        # Draw frame corner if exists
        if first_piece.frame_corners:
            fc = first_piece.frame_corners[0]
            svg_lines.append(f'    <circle cx="{fc.x}" cy="{fc.y}" r="5" fill="red"/>')

        svg_lines.append('  </g>')

        # Add reference grid (optional)
        svg_lines.append('  <!-- Reference grid -->')
        for i in range(0, canvas_width, 50):
            svg_lines.append(f'  <line x1="{i}" y1="0" x2="{i}" y2="{canvas_height}" '
                           f'stroke="#E0E0E0" stroke-width="0.5" opacity="0.3"/>')
        for i in range(0, canvas_height, 50):
            svg_lines.append(f'  <line x1="0" y1="{i}" x2="{canvas_width}" y2="{i}" '
                           f'stroke="#E0E0E0" stroke-width="0.5" opacity="0.3"/>')

        svg_lines.append('</svg>')

        # Write SVG file
        output_path = project_root / 'temp' / 'edge_solver_solution.svg'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(svg_lines))

        print(f"Solution SVG saved to: {output_path}")

        # Now try to add all remaining pieces
        if piece_connections and all_matches:
            SolutionBuilder.add_all_pieces_to_solution(
                piece_frame_segments,
                rotation_angles,
                piece_connections,
                all_matches,
                project_root
            )

    @staticmethod
    def add_all_pieces_to_solution(piece_frame_segments, rotation_angles, piece_connections, all_matches, project_root):
        """Add all pieces to the solution SVG iteratively.

        Args:
            piece_frame_segments: List of tuples (piece, segments, frame_adjacent_segments)
            rotation_angles: Dictionary of piece_id -> rotation_angle_degrees
            piece_connections: Dictionary of piece_id -> list of connections
            all_matches: List of all matches
            project_root: Path to project root
        """
        from ..matrix_solver.segment_matcher import SegmentMatcher
        from ..common.utils import normalize_and_center, rotate_points

        print("\nAdding all pieces to solution...")

        # Track which pieces have been placed and their final rotations
        placed_pieces = {}

        # Start with first piece
        first_piece, first_segments, _ = piece_frame_segments[0]
        first_piece_id = first_piece.piece_id
        placed_pieces[first_piece_id] = {
            'piece': first_piece,
            'segments': first_segments,
            'rotation': rotation_angles[first_piece_id]
        }
        print(f"Starting with Piece {first_piece_id}, rotation: {rotation_angles[first_piece_id]:.1f}°")

        # Keep adding pieces until all are placed
        while len(placed_pieces) < len(piece_frame_segments):
            # Find next piece to add (connected to an already-placed piece)
            next_piece_found = False

            for placed_id in list(placed_pieces.keys()):
                if placed_id not in piece_connections:
                    continue

                for connected_id, match, side in piece_connections[placed_id]:
                    if connected_id in placed_pieces:
                        continue  # Already placed

                    # Found a piece to add
                    next_piece_found = True

                    # Find the piece data
                    next_piece = None
                    next_segments = None
                    for piece, segments, _ in piece_frame_segments:
                        if piece.piece_id == connected_id:
                            next_piece = piece
                            next_segments = segments
                            break

                    if next_piece is None:
                        continue

                    # Handle both ExtendedSegmentMatch and regular SegmentMatch
                    from ..common.data_classes import ExtendedSegmentMatch
                    if isinstance(match, ExtendedSegmentMatch):
                        seg1_id = match.initial_match.seg1_id if match.piece1_id == placed_id else match.initial_match.seg2_id
                        seg2_id = match.initial_match.seg2_id if match.piece1_id == placed_id else match.initial_match.seg1_id
                        score = match.combined_score
                        score_desc = f"combined_score={score:.3f}, segments={match.total_segments_matched}"
                    else:
                        seg1_id = match.seg1_id if match.piece1_id == placed_id else match.seg2_id
                        seg2_id = match.seg2_id if match.piece1_id == placed_id else match.seg1_id
                        score = match.match_score
                        score_desc = f"score={score:.3f}"

                    print(f"\nConnecting Piece {connected_id} to Piece {placed_id}")
                    print(f"Match: S{seg1_id} <-> S{seg2_id}, {score_desc}")

                    # Get base rotation
                    base_rotation = rotation_angles[connected_id]
                    print(f"Base rotation for Piece {connected_id}: {base_rotation:.1f}°")

                    # Find matching segments
                    placed_segments = placed_pieces[placed_id]['segments']
                    placed_rotation = placed_pieces[placed_id]['rotation']  # Use the final rotation of the placed piece
                    seg1 = next(s for s in placed_segments if s.segment_id == seg1_id)
                    seg2 = next(s for s in next_segments if s.segment_id == seg2_id)

                    # Test rotations
                    test_rotations = [0, 90, 180, 270]
                    best_rotation = base_rotation
                    best_score = 0.0

                    print("Testing rotations:")
                    for additional_rotation in test_rotations:
                        test_rotation = base_rotation + additional_rotation

                        # Resample and normalize
                        pts1_resampled = SegmentMatcher._resample_points(seg1.contour_points, 50)
                        pts2_resampled = SegmentMatcher._resample_points(seg2.contour_points, 50)

                        pts1_norm = normalize_and_center(pts1_resampled)
                        pts2_norm = normalize_and_center(pts2_resampled)

                        # Rotate pts1 by the placed piece's final rotation
                        pts1_rotated = rotate_points(pts1_norm, np.radians(placed_rotation))

                        # Rotate pts2 by test rotation (NO additional optimal rotation)
                        pts2_rotated = rotate_points(pts2_norm, np.radians(test_rotation))

                        # Calculate RMSD
                        pts1_array = np.array([[p.x, p.y] for p in pts1_rotated])
                        pts2_array = np.array([[p.x, p.y] for p in pts2_rotated])

                        distance_matrix = cdist(pts1_array, pts2_array)
                        min_distances_1_to_2 = np.min(distance_matrix, axis=1)
                        min_distances_2_to_1 = np.min(distance_matrix, axis=0)
                        all_distances = np.concatenate([min_distances_1_to_2, min_distances_2_to_1])

                        rmsd = np.sqrt(np.mean(all_distances ** 2))
                        similarity_score = max(0.0, 1.0 - (rmsd * 2.0))

                        print(f"  Rotation {test_rotation:.1f}°: similarity = {similarity_score:.3f}, RMSD = {rmsd:.3f}")

                        if similarity_score > best_score:
                            best_score = similarity_score
                            best_rotation = test_rotation

                    print(f"Best rotation: {best_rotation:.1f}° (similarity: {best_score:.3f})")

                    # Add to placed pieces with connection info
                    placed_pieces[connected_id] = {
                        'piece': next_piece,
                        'segments': next_segments,
                        'rotation': best_rotation,
                        'connected_to': placed_id,
                        'matching_segment_placed': seg1.segment_id,
                        'matching_segment_new': seg2.segment_id
                    }

                    break

                if next_piece_found:
                    break

            if not next_piece_found:
                print("\nNo more connected pieces found!")
                break

        print(f"\n\nPlaced {len(placed_pieces)} pieces total")

        # Now create the final SVG with all pieces
        SolutionBuilder.create_final_solution_svg(placed_pieces, project_root)

    @staticmethod
    def create_final_solution_svg(placed_pieces, project_root):
        """Create SVG file with all placed pieces in their final positions.

        Args:
            placed_pieces: Dictionary of piece_id -> {'piece', 'segments', 'rotation', 'connected_to', ...}
            project_root: Path to project root
        """
        from .visualizers import EdgeSolverVisualizer

        print("\nCreating final solution SVG with all pieces...")

        # Calculate positions for each piece based on connections
        piece_positions = {}  # piece_id -> (x, y)

        # Start with first piece at origin
        first_piece_id = list(placed_pieces.keys())[0]
        first_data = placed_pieces[first_piece_id]

        # Get first piece contour center
        first_points = []
        for seg in first_data['segments']:
            first_points.extend([(int(p.x), int(p.y)) for p in seg.contour_points])
        first_contour = np.array(first_points, dtype=np.int32)
        fx, fy, fw, fh = cv2.boundingRect(first_contour)

        # Position first piece at canvas center
        canvas_center_x = 500
        canvas_center_y = 500
        piece_positions[first_piece_id] = (canvas_center_x, canvas_center_y)

        print(f"Piece {first_piece_id} positioned at ({canvas_center_x:.1f}, {canvas_center_y:.1f})")

        # Position other pieces based on their connections
        for piece_id in list(placed_pieces.keys())[1:]:
            piece_data = placed_pieces[piece_id]
            connected_to = piece_data['connected_to']
            seg_placed_id = piece_data['matching_segment_placed']
            seg_new_id = piece_data['matching_segment_new']

            # Get the position of the piece we're connecting to
            parent_pos = piece_positions[connected_to]
            parent_data = placed_pieces[connected_to]
            parent_rotation = parent_data['rotation']

            # Find the matching segments
            parent_seg = next(s for s in parent_data['segments'] if s.segment_id == seg_placed_id)
            new_seg = next(s for s in piece_data['segments'] if s.segment_id == seg_new_id)

            # Calculate center of parent segment
            parent_seg_center_x, parent_seg_center_y = calculate_segment_center(parent_seg)

            # Rotate parent segment center by parent rotation
            parent_contour = []
            for seg in parent_data['segments']:
                parent_contour.extend([(p.x, p.y) for p in seg.contour_points])
            pcx = sum(p[0] for p in parent_contour) / len(parent_contour)
            pcy = sum(p[1] for p in parent_contour) / len(parent_contour)

            # Apply parent rotation to segment center
            angle_rad = np.radians(parent_rotation)
            dx = parent_seg_center_x - pcx
            dy = parent_seg_center_y - pcy
            rotated_dx = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
            rotated_dy = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
            parent_seg_world_x = parent_pos[0] + rotated_dx
            parent_seg_world_y = parent_pos[1] + rotated_dy

            # Calculate center of new segment
            new_seg_center_x, new_seg_center_y = calculate_segment_center(new_seg)

            # Calculate new piece center
            new_contour = []
            for seg in piece_data['segments']:
                new_contour.extend([(p.x, p.y) for p in seg.contour_points])
            ncx = sum(p[0] for p in new_contour) / len(new_contour)
            ncy = sum(p[1] for p in new_contour) / len(new_contour)

            # Apply new piece rotation to segment center
            new_rotation = piece_data['rotation']
            angle_rad_new = np.radians(new_rotation)
            dx_new = new_seg_center_x - ncx
            dy_new = new_seg_center_y - ncy
            rotated_dx_new = dx_new * np.cos(angle_rad_new) - dy_new * np.sin(angle_rad_new)
            rotated_dy_new = dx_new * np.sin(angle_rad_new) + dy_new * np.cos(angle_rad_new)

            # Position new piece so its segment aligns with parent segment
            new_piece_x = parent_seg_world_x - rotated_dx_new
            new_piece_y = parent_seg_world_y - rotated_dy_new

            # Calculate dynamic spacing based on piece sizes
            # Get bounding box of parent piece
            parent_bbox = cv2.boundingRect(np.array(parent_contour, dtype=np.int32))
            parent_max_dim = max(parent_bbox[2], parent_bbox[3])

            # Get bounding box of new piece
            new_bbox = cv2.boundingRect(np.array(new_contour, dtype=np.int32))
            new_max_dim = max(new_bbox[2], new_bbox[3])

            # Calculate spacing as half of combined max dimensions (ensures no overlap)
            spacing = (parent_max_dim + new_max_dim) / 5.0

            # Calculate direction vector from parent to new piece (based on segments)
            direction_x = new_piece_x - parent_pos[0]
            direction_y = new_piece_y - parent_pos[1]

            # Normalize direction
            distance = np.sqrt(direction_x**2 + direction_y**2)
            if distance > 0:
                direction_x /= distance
                direction_y /= distance

                # Apply spacing
                new_piece_x += direction_x * spacing
                new_piece_y += direction_y * spacing

            piece_positions[piece_id] = (new_piece_x, new_piece_y)
            print(f"Piece {piece_id} positioned at ({new_piece_x:.1f}, {new_piece_y:.1f}) - connected to Piece {connected_to}, spacing={spacing:.1f}px")

        # Adjust all positions to ensure they're within canvas bounds
        # Find min/max positions
        all_x = [pos[0] for pos in piece_positions.values()]
        all_y = [pos[1] for pos in piece_positions.values()]
        min_x = min(all_x)
        min_y = min(all_y)
        max_x = max(all_x)
        max_y = max(all_y)

        # Add padding
        padding = 200

        # Shift all positions so min is at padding
        offset_x = padding - min_x
        offset_y = padding - min_y

        for piece_id in piece_positions:
            old_pos = piece_positions[piece_id]
            piece_positions[piece_id] = (old_pos[0] + offset_x, old_pos[1] + offset_y)
            print(f"Adjusted Piece {piece_id} to ({piece_positions[piece_id][0]:.1f}, {piece_positions[piece_id][1]:.1f})")

        # Calculate canvas size based on adjusted positions
        adjusted_max_x = max_x + offset_x
        adjusted_max_y = max_y + offset_y
        canvas_width = int(adjusted_max_x + padding)
        canvas_height = int(adjusted_max_y + padding)

        print(f"\nCanvas size: {canvas_width} x {canvas_height}")

        # Start SVG
        svg_lines = []
        svg_lines.append(f'<svg width="{canvas_width}" height="{canvas_height}" xmlns="http://www.w3.org/2000/svg">')
        svg_lines.append(f'  <!-- Final Solution SVG - {len(placed_pieces)} pieces -->')

        # Add reference grid
        svg_lines.append('  <!-- Reference grid -->')
        for i in range(0, canvas_width, 100):
            svg_lines.append(f'  <line x1="{i}" y1="0" x2="{i}" y2="{canvas_height}" '
                           f'stroke="#E0E0E0" stroke-width="0.5" opacity="0.3"/>')
        for i in range(0, canvas_height, 100):
            svg_lines.append(f'  <line x1="0" y1="{i}" x2="{canvas_width}" y2="{i}" '
                           f'stroke="#E0E0E0" stroke-width="0.5" opacity="0.3"/>')

        # Draw each piece
        for piece_id, piece_data in sorted(placed_pieces.items()):
            piece = piece_data['piece']
            segments = piece_data['segments']
            rotation = piece_data['rotation']
            target_x, target_y = piece_positions[piece_id]

            # Get piece center in original coordinates
            piece_points = []
            for seg in segments:
                piece_points.extend([(int(p.x), int(p.y)) for p in seg.contour_points])

            piece_contour = np.array(piece_points, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(piece_contour)
            piece_center_x = x + w / 2
            piece_center_y = y + h / 2

            # Create transform: translate to target, rotate, translate back
            svg_lines.append(f'  <g id="piece_{piece_id}">')
            svg_lines.append(f'    <!-- Piece {piece_id}, rotation: {rotation:.1f}° -->')
            svg_lines.append(f'    <g transform="translate({target_x},{target_y}) rotate({rotation}) translate({-piece_center_x},{-piece_center_y})">')

            # Draw all segments
            for seg in segments:
                points = ' '.join([f'{p.x},{p.y}' for p in seg.contour_points])
                color = "#FF6B6B" if seg.is_border_edge else "#4ECDC4"
                svg_lines.append(f'      <polyline points="{points}" '
                               f'fill="none" stroke="{color}" stroke-width="2"/>')

            # Draw centroid
            svg_lines.append(f'      <circle cx="{piece.centroid.x}" cy="{piece.centroid.y}" '
                           f'r="3" fill="blue"/>')

            # Draw frame corner if exists
            if piece.frame_corners:
                fc = piece.frame_corners[0]
                svg_lines.append(f'      <circle cx="{fc.x}" cy="{fc.y}" r="5" fill="red"/>')

            # Add piece label
            svg_lines.append(f'      <text x="{piece.centroid.x + 10}" y="{piece.centroid.y - 10}" '
                           f'font-size="20" fill="white" font-weight="bold">P{piece_id}</text>')

            svg_lines.append('    </g>')
            svg_lines.append('  </g>')

        svg_lines.append('</svg>')

        # Write SVG file
        output_path = project_root / 'temp' / 'edge_solver_final_solution.svg'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(svg_lines))

        print(f"Final solution SVG saved to: {output_path}")

        # Display the final solution
        EdgeSolverVisualizer.visualize_final_solution_dialog(output_path)
