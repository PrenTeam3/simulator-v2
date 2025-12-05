"""Visualize segment matches between two pieces."""
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple
from ..preparation.data_loader import AnalyzedPuzzlePiece
from ..common.data_classes import Point, ContourSegment, SegmentMatch
from .segment_matcher import SegmentMatcher


class MatchVisualizer:
    """Visualizes segment matches between two pieces."""

    @staticmethod
    def draw_two_pieces_side_by_side(image: np.ndarray,
                                    piece1: AnalyzedPuzzlePiece,
                                    segments1: List[ContourSegment],
                                    piece2: AnalyzedPuzzlePiece,
                                    segments2: List[ContourSegment],
                                    matches: List[SegmentMatch] = None,
                                    max_matches_to_draw: int = 3,
                                    best_group_segments: dict = None) -> np.ndarray:
        """
        Draw two pieces side by side with their segments and matching connections.

        Args:
            image: Original puzzle image
            piece1: First puzzle piece
            segments1: Segments from first piece
            piece2: Second puzzle piece
            segments2: Segments from second piece
            matches: List of SegmentMatch objects (should be sorted by score)
            max_matches_to_draw: Maximum number of top matches to draw as connections
            best_group_segments: Dictionary with 'seg_ids_p1' and 'seg_ids_p2' for best group match

        Returns:
            Annotated image showing both pieces and matches
        """
        height, width = image.shape[:2]
        # Create canvas with space for two pieces side by side
        canvas = np.ones((height, width * 2 + 100, 3), dtype=np.uint8) * 40

        # Draw piece 1 on the left
        MatchVisualizer._draw_piece_on_canvas(canvas, piece1, segments1, offset_x=50, offset_y=0)

        # Draw piece 2 on the right
        MatchVisualizer._draw_piece_on_canvas(canvas, piece2, segments2, offset_x=width + 100, offset_y=0)

        # Highlight best group segments if provided
        if best_group_segments:
            seg_ids_p1 = best_group_segments.get('seg_ids_p1', [])
            seg_ids_p2 = best_group_segments.get('seg_ids_p2', [])

            # Highlight segments from piece 1 with bright green outline
            for seg_id in seg_ids_p1:
                seg = next((s for s in segments1 if s.segment_id == seg_id), None)
                if seg:
                    seg_array = np.array([[int(p.x) + 50, int(p.y)] for p in seg.contour_points], dtype=np.int32)
                    cv2.polylines(canvas, [seg_array], False, (0, 255, 0), 5)  # Green - bright highlight

            # Highlight segments from piece 2 with bright yellow outline
            for seg_id in seg_ids_p2:
                seg = next((s for s in segments2 if s.segment_id == seg_id), None)
                if seg:
                    seg_array = np.array([[int(p.x) + width + 100, int(p.y)] for p in seg.contour_points], dtype=np.int32)
                    cv2.polylines(canvas, [seg_array], False, (0, 255, 255), 5)  # Cyan - bright highlight

            # Draw connecting lines between best group segments
            for seg_id_p1 in seg_ids_p1:
                seg1 = next((s for s in segments1 if s.segment_id == seg_id_p1), None)
                if seg1:
                    # Calculate midpoint for piece 1 segment
                    mid_x1 = sum(int(p.x) for p in seg1.contour_points) / len(seg1.contour_points) + 50
                    mid_y1 = sum(int(p.y) for p in seg1.contour_points) / len(seg1.contour_points)
                    pt1 = (int(mid_x1), int(mid_y1))

                    # Draw a line to the centroid area of piece 2 for visual connection
                    for seg_id_p2 in seg_ids_p2:
                        seg2 = next((s for s in segments2 if s.segment_id == seg_id_p2), None)
                        if seg2:
                            # Calculate midpoint for piece 2 segment
                            mid_x2 = sum(int(p.x) for p in seg2.contour_points) / len(seg2.contour_points) + width + 100
                            mid_y2 = sum(int(p.y) for p in seg2.contour_points) / len(seg2.contour_points)
                            pt2 = (int(mid_x2), int(mid_y2))

                            # Draw thick line in bright white to show the best group connection
                            cv2.line(canvas, pt1, pt2, (255, 255, 255), 3)

            # Add text label for best group match
            cv2.rectangle(canvas, (width // 2 - 80, 20), (width // 2 + 80, 50), (40, 40, 40), -1)
            cv2.putText(canvas, "BEST GROUP MATCH", (width // 2 - 70, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw match connections between pieces (top matches only)
        if matches:
            for idx, match in enumerate(matches[:max_matches_to_draw]):
                seg1 = next((s for s in segments1 if s.segment_id == match.seg1_id), None)
                seg2 = next((s for s in segments2 if s.segment_id == match.seg2_id), None)

                if seg1 and seg2 and hasattr(seg1, 'midpoint_display') and hasattr(seg2, 'midpoint_display'):
                    # Use pre-calculated display midpoints
                    pt1 = seg1.midpoint_display
                    pt2 = seg2.midpoint_display

                    # Color based on match quality
                    color = MatchVisualizer._score_to_color(match.match_score)

                    # Draw curved line to avoid overlapping segment IDs
                    # Use quadratic bezier curve through control point above/below line
                    mid_x = (pt1[0] + pt2[0]) // 2
                    mid_y = (pt1[1] + pt2[1]) // 2

                    # Offset control point perpendicular to the line
                    offset = 30 + idx * 20  # Vary offset for multiple lines
                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    length = (dx**2 + dy**2)**0.5
                    if length > 0:
                        perp_x = -dy / length * offset
                        perp_y = dx / length * offset
                    else:
                        perp_x, perp_y = 0, 0

                    ctrl_pt = (int(mid_x + perp_x), int(mid_y + perp_y))

                    # Draw quadratic bezier curve
                    points = []
                    for t in np.linspace(0, 1, 50):
                        # Quadratic bezier: B(t) = (1-t)^2*P0 + 2(1-t)t*C + t^2*P1
                        x = (1-t)**2 * pt1[0] + 2*(1-t)*t * ctrl_pt[0] + t**2 * pt2[0]
                        y = (1-t)**2 * pt1[1] + 2*(1-t)*t * ctrl_pt[1] + t**2 * pt2[1]
                        points.append((int(x), int(y)))

                    points = np.array(points, dtype=np.int32)
                    cv2.polylines(canvas, [points], False, color, 2)

                    # Draw circles at endpoints
                    cv2.circle(canvas, pt1, 5, color, -1)
                    cv2.circle(canvas, pt2, 5, color, -1)

                    # Draw match info text at control point
                    text = f"M{idx+1}: {match.match_score:.2f}"
                    text_pos = (ctrl_pt[0] - 35, ctrl_pt[1] - 5)
                    cv2.rectangle(canvas, (text_pos[0] - 2, text_pos[1] - 12),
                                 (text_pos[0] + 68, text_pos[1] + 3), (40, 40, 40), -1)
                    cv2.putText(canvas, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        return canvas

    @staticmethod
    def _draw_piece_on_canvas(canvas: np.ndarray, piece: AnalyzedPuzzlePiece,
                             segments: List[ContourSegment], offset_x: int, offset_y: int) -> None:
        """Helper: Draw a single piece on the canvas at given offset."""
        # Draw contour
        contour_array = np.array([[int(p.x) + offset_x, int(p.y) + offset_y] for p in piece.contour_points], dtype=np.int32)
        cv2.fillPoly(canvas, [contour_array], (0, 165, 255))
        cv2.polylines(canvas, [contour_array], True, (0, 255, 0), 2)

        # Draw centroid
        cent = (int(piece.centroid.x) + offset_x, int(piece.centroid.y) + offset_y)
        cv2.circle(canvas, cent, 5, (255, 0, 0), -1)
        cv2.putText(canvas, f"P{piece.piece_id}", (cent[0] - 20, cent[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw segments (store midpoints for later use in drawing connections)
        for seg in segments:
            seg_points = np.array([[int(p.x) + offset_x, int(p.y) + offset_y] for p in seg.contour_points], dtype=np.int32)
            cv2.polylines(canvas, [seg_points], False, (255, 0, 0), 2)

            # Draw segment ID
            mid_x = sum(p.x for p in seg.contour_points) / len(seg.contour_points) + offset_x
            mid_y = sum(p.y for p in seg.contour_points) / len(seg.contour_points) + offset_y
            mid_pt = (int(mid_x), int(mid_y))

            cv2.rectangle(canvas, (mid_pt[0] - 12, mid_pt[1] - 8), (mid_pt[0] + 12, mid_pt[1] + 8), (0, 255, 255), -1)
            cv2.putText(canvas, f"S{seg.segment_id}", (mid_pt[0] - 10, mid_pt[1] + 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            seg.midpoint_display = mid_pt  # Store for later use

    @staticmethod
    def _score_to_color(score: float) -> Tuple[int, int, int]:
        """Convert match score (0-1) to BGR color (green = high, red = low)."""
        if score < 0.3:
            return (0, 0, 255)  # Red
        elif score < 0.6:
            return (0, 165, 255)  # Orange
        elif score < 0.85:
            return (0, 255, 255)  # Yellow
        else:
            return (0, 255, 0)  # Green

    @staticmethod
    def print_matches(matches: List[SegmentMatch], max_to_print: int = 10) -> None:
        """Print match information to console."""
        print(f"\n{'='*70}")
        print(f"Found {len(matches)} potential segment matches")
        print(f"{'='*70}")

        for idx, match in enumerate(matches[:max_to_print]):
            print(f"{idx+1}. {match.description}")

        if len(matches) > max_to_print:
            print(f"... and {len(matches) - max_to_print} more matches")

        print(f"{'='*70}\n")

    @staticmethod
    def print_match_matrix(matrix, piece1_id: int, piece2_id: int) -> None:
        """Print match matrix to console with clear labels using pandas DataFrame.

        Marks the best score for each row segment (Piece 1) and column segment (Piece 2).
        Best scores are marked with * for row max, ** for column max, *** for both.
        """
        n, m = matrix.shape

        # Create DataFrame with segment labels
        row_labels = [f"P{piece1_id}-S{i}" for i in range(n)]
        col_labels = [f"S{j}" for j in range(m)]

        # Find best score for each row and column
        row_max_indices = np.argmax(matrix, axis=1)  # Best match for each row segment
        col_max_indices = np.argmax(matrix, axis=0)  # Best match for each column segment

        # Create display matrix with annotations
        display_matrix = []
        for i in range(n):
            row = []
            for j in range(m):
                value = matrix[i, j]
                marker = ""
                # Mark best for this row segment
                if j == row_max_indices[i]:
                    marker += "*"
                # Mark best for this column segment
                if i == col_max_indices[j]:
                    marker += "*"

                row.append(f"{value:.2f}{marker}")
            display_matrix.append(row)

        df = pd.DataFrame(display_matrix, index=row_labels, columns=col_labels)

        print(f"\nLength Similarity Matrix: Piece {piece1_id} segments (rows) × Piece {piece2_id} segments (columns)")
        print(df.to_string())
        print("\n(* = best match for that segment, ** = mutual best match)")

        print(f"\nNon-zero matches: {(matrix > 0).sum()}")
        print(f"Max match score: {matrix.max():.3f}" if matrix.max() > 0 else "Max match score: 0.000")
        print()

    @staticmethod
    def print_shape_similarity_matrix(matrix, piece1_id: int, piece2_id: int) -> None:
        """Print shape similarity matrix to console with clear labels using pandas DataFrame.

        For calculated values, marks the best (lowest) score for each row segment (Piece 1)
        and column segment (Piece 2). Uncalculated values show as "n/a".
        Best scores are marked with * for row min, ** for column min, *** for both.
        """
        n, m = matrix.shape

        # Create DataFrame with segment labels, replacing -1 with "n/a" for display
        row_labels = [f"P{piece1_id}-S{i}" for i in range(n)]
        col_labels = [f"S{j}" for j in range(m)]

        # Find best (minimum) scores for each row and column among calculated values
        row_min_indices = np.full(n, -1, dtype=int)
        col_min_indices = np.full(m, -1, dtype=int)

        for i in range(n):
            valid_cols = [j for j in range(m) if matrix[i, j] >= 0]
            if valid_cols:
                row_min_indices[i] = valid_cols[np.argmin(matrix[i, valid_cols])]

        for j in range(m):
            valid_rows = [i for i in range(n) if matrix[i, j] >= 0]
            if valid_rows:
                col_min_indices[j] = valid_rows[np.argmin(matrix[valid_rows, j])]

        # Create a copy for display with "n/a" for uncalculated values and markers for best
        display_matrix = []
        for i in range(n):
            row = []
            for j in range(m):
                if matrix[i, j] < 0:
                    row.append("n/a")
                else:
                    value = matrix[i, j]
                    marker = ""
                    # Mark best (minimum) for this row segment
                    if j == row_min_indices[i]:
                        marker += "*"
                    # Mark best (minimum) for this column segment
                    if i == col_min_indices[j]:
                        marker += "*"

                    row.append(f"{value:.4f}{marker}")
            display_matrix.append(row)

        df = pd.DataFrame(display_matrix, index=row_labels, columns=col_labels)

        print(f"\nShape Similarity Matrix (RMSD): Piece {piece1_id} segments (rows) × Piece {piece2_id} segments (columns)")
        print("(Lower scores = better shape match after optimal rotation)")
        print(df.to_string())
        print("\n(* = best match for that segment, ** = mutual best match)")

        # Calculate min/max ignoring -1 (not calculated) values
        valid_values = matrix[matrix >= 0]
        if valid_values.size > 0:
            print(f"\nMin RMSD: {valid_values.min():.4f}")
            print(f"Max RMSD: {valid_values.max():.4f}")
            print(f"Calculated pairs: {valid_values.size} / {matrix.size} (others: length < 0.75)")
        else:
            print("\nMin RMSD: N/A")
            print("Max RMSD: N/A")
        print()



    @staticmethod
    def print_rotation_angle_matrix(matrix, piece1_id: int, piece2_id: int) -> None:
        """Print rotation angle matrix to console with clear labels using pandas DataFrame.

        Displays optimal rotation angles in degrees for each segment pair.
        Only calculated values are shown for pairs where length matching is above 0.75.
        Uncalculated values show as "n/a".
        """
        n, m = matrix.shape

        # Create DataFrame with segment labels
        row_labels = [f"P{piece1_id}-S{i}" for i in range(n)]
        col_labels = [f"S{j}" for j in range(m)]

        # Create display matrix with "n/a" for uncalculated values
        display_matrix = []
        for i in range(n):
            row = []
            for j in range(m):
                if np.isnan(matrix[i, j]):
                    row.append("n/a")
                else:
                    # Format angle with up to 2 decimal places
                    angle = matrix[i, j]
                    row.append(f"{angle:7.2f}°")
            display_matrix.append(row)

        df = pd.DataFrame(display_matrix, index=row_labels, columns=col_labels)

        print(f"\nRotation Angle Matrix: Piece {piece1_id} segments (rows) × Piece {piece2_id} segments (columns)")
        print("(Optimal rotation angles in degrees for minimal bounding box overlap)")
        print(df.to_string())

        # Calculate statistics ignoring nan values
        valid_angles = matrix[~np.isnan(matrix)]
        if valid_angles.size > 0:
            print(f"\nStatistics (for calculated angles):")
            print(f"  Min angle: {valid_angles.min():7.2f}°")
            print(f"  Max angle: {valid_angles.max():7.2f}°")
            print(f"  Mean angle: {valid_angles.mean():7.2f}°")
            print(f"  Std dev: {valid_angles.std():7.2f}°")
            print(f"  Calculated pairs: {valid_angles.size} / {matrix.size} (others: length < 0.75)")
        else:
            print("\nNo rotation angles calculated (insufficient length matches)")
        print()

    @staticmethod
    def _find_angle_similarity_groups(angles: List[float], tolerance: float = 15.0) -> List[Tuple[int, int]]:
        """Find groups of continuous similar angles in a circular array.

        Treats the array as circular, so groups can wrap from end to start.
        Allows one item per group to be different (outlier), but still keep them grouped.
        Returns list of (start_idx, length) tuples for groups with tolerance.
        """
        n = len(angles)
        if n == 0:
            return []

        # Filter to only valid (non-nan) angles
        valid_indices = [i for i in range(n) if not np.isnan(angles[i])]
        if len(valid_indices) < 2:
            return []

        groups = []
        visited = set()

        for start_idx in valid_indices:
            if start_idx in visited:
                continue

            current_group = [start_idx]
            visited.add(start_idx)
            base_angle = angles[start_idx]
            outlier_used = False  # Track if we've used the one allowed outlier

            # Try to extend forward
            next_idx = (start_idx + 1) % n
            while next_idx != start_idx and next_idx not in visited:
                if next_idx in valid_indices:
                    angle_diff = abs(angles[next_idx] - base_angle)
                    if angle_diff > 90:
                        angle_diff = 180 - angle_diff

                    if angle_diff <= tolerance:
                        current_group.append(next_idx)
                        visited.add(next_idx)
                    elif not outlier_used:
                        # Allow one outlier
                        current_group.append(next_idx)
                        visited.add(next_idx)
                        outlier_used = True
                    else:
                        break
                next_idx = (next_idx + 1) % n

            # Try to extend backward from start
            prev_idx = (start_idx - 1) % n
            while prev_idx != start_idx and prev_idx not in visited:
                if prev_idx in valid_indices:
                    angle_diff = abs(angles[prev_idx] - base_angle)
                    if angle_diff > 90:
                        angle_diff = 180 - angle_diff

                    if angle_diff <= tolerance:
                        current_group.insert(0, prev_idx)
                        visited.add(prev_idx)
                    elif not outlier_used:
                        # Allow one outlier
                        current_group.insert(0, prev_idx)
                        visited.add(prev_idx)
                        outlier_used = True
                    else:
                        break
                prev_idx = (prev_idx - 1) % n

            if len(current_group) >= 2:
                groups.append((current_group[0], len(current_group)))

        return groups

    @staticmethod
    def extract_diagonal_arrays_from_rotation_angles(matrix: np.ndarray, piece1_id: int, piece2_id: int) -> dict:
        """Extract diagonal values from rotation angle matrix and create arrays.

        For an n x m matrix, generates arrays where:
        - Array i contains values from diagonals starting at row i
        - Each diagonal moves north-east (up-right): (i, j), (i-1, j+1), (i-2, j+2), etc.
        - When reaching the top (row < 0), wraps around to the bottom (row += n)
        - When reaching the right edge, wraps around to the left (col = 0)
        - Continues for exactly n elements to show the full wraparound pattern

        This helps visualize matching patterns along diagonal directions in puzzle piece matching.
        """
        n, m = matrix.shape

        print(f"\n{'='*70}")
        print(f"DIAGONAL ARRAYS FROM ROTATION ANGLE MATRIX: Piece {piece1_id} <-> Piece {piece2_id}")
        print(f"(North-East diagonals with wraparound)")
        print(f"{'='*70}\n")

        # Generate diagonal arrays for each starting column position
        for start_col in range(m):
            diagonal_values = []
            angle_values = []
            diag_indices = []
            diag_row_labels = []
            segment_mappings = []

            # Start from position (0, start_col) and go diagonally north-east (up-right with wraparound)
            row, col = 0, start_col
            for step in range(n):
                angle = matrix[row, col]
                diag_indices.append((row, col))
                diag_row_labels.append(f"P{piece1_id}-S{row}")
                angle_values.append(angle)

                if np.isnan(angle):
                    diagonal_values.append("n/a")
                else:
                    diagonal_values.append(f"{angle:.2f}°")

                # Create segment mapping: P{piece1_id}-S{row}=P{piece2_id}-S{col}
                segment_mappings.append(f"P{piece1_id}-S{row}=P{piece2_id}-S{col}")

                # Move north-east: up one row (with wraparound), right one column
                row = (row - 1) % n  # Wraparound when going above 0
                col = (col + 1) % m  # Wraparound when going beyond m-1

            # Find similarity groups
            groups = MatchVisualizer._find_angle_similarity_groups(angle_values, tolerance=15.0)
            group_map = {}
            for group_start, group_len in groups:
                for i in range(group_len):
                    idx = (group_start + i) % n
                    group_map[idx] = group_start

            # ANSI color codes
            colors = [
                "\033[42m",  # Green background
                "\033[43m",  # Yellow background
                "\033[44m",  # Blue background
                "\033[45m",  # Magenta background
                "\033[46m",  # Cyan background
                "\033[41m",  # Red background
            ]
            reset = "\033[0m"

            # Create output with color coding
            print(f"Diagonal starting at S{start_col}:")
            print(f"  Segments: {' -> '.join(diag_row_labels)}")

            # Print values with colors for similarity groups
            colored_values = []
            for i, val in enumerate(diagonal_values):
                if i in group_map:
                    color = colors[group_map[i] % len(colors)]
                    colored_values.append(f"{color}{val}{reset}")
                else:
                    colored_values.append(val)

            print(f"  Values:   {' -> '.join(colored_values)}")

            # Print segment mapping array
            print(f"  Segment Map: [{', '.join(repr(m) for m in segment_mappings)}]")

            # Print legend for groups
            if groups:
                print(f"  Groups:   ", end="")
                for group_idx, (group_start, group_len) in enumerate(groups):
                    group_indices = [(group_start + i) % n for i in range(group_len)]
                    color = colors[group_start % len(colors)]
                    indices_str = ", ".join(str(idx) for idx in group_indices)
                    print(f"{color}Group {group_idx+1}: [{indices_str}]{reset}", end="  ")
                print()
            print()

    @staticmethod
    def _find_similarity_score_groups(scores: List[float], threshold: float = 0.7) -> List[Tuple[int, int]]:
        """Find groups of continuous similar scores in a circular array.

        Treats the array as circular, so groups can wrap from end to start.
        Allows one item per group to be below threshold (outlier), but still keep them grouped.
        n/a values (0.0) can only be the single outlier, and groups must have at least one valid score.
        Returns list of (start_idx, length) tuples for groups above threshold.
        """
        n = len(scores)
        if n == 0:
            return []

        # Filter to only scores above threshold
        valid_indices = [i for i in range(n) if scores[i] >= threshold]
        if len(valid_indices) < 2:
            return []

        groups = []
        visited = set()

        for start_idx in valid_indices:
            if start_idx in visited:
                continue

            current_group = [start_idx]
            visited.add(start_idx)
            outlier_used = False  # Track if we've used the one allowed outlier

            # Try to extend forward
            next_idx = (start_idx + 1) % n
            while next_idx != start_idx and next_idx not in visited:
                if next_idx in valid_indices:
                    current_group.append(next_idx)
                    visited.add(next_idx)
                elif not outlier_used and scores[next_idx] > 0:
                    # Allow one outlier below threshold (but not n/a which is 0.0)
                    current_group.append(next_idx)
                    visited.add(next_idx)
                    outlier_used = True
                else:
                    break
                next_idx = (next_idx + 1) % n

            # Try to extend backward from start
            prev_idx = (start_idx - 1) % n
            while prev_idx != start_idx and prev_idx not in visited:
                if prev_idx in valid_indices:
                    current_group.insert(0, prev_idx)
                    visited.add(prev_idx)
                elif not outlier_used and scores[prev_idx] > 0:
                    # Allow one outlier below threshold (but not n/a which is 0.0)
                    current_group.insert(0, prev_idx)
                    visited.add(prev_idx)
                    outlier_used = True
                else:
                    break
                prev_idx = (prev_idx - 1) % n

            if len(current_group) >= 2:
                groups.append((current_group[0], len(current_group)))

        return groups

    @staticmethod
    def extract_diagonal_arrays_from_length_similarity(matrix: np.ndarray, piece1_id: int, piece2_id: int) -> None:
        """Extract diagonal values from length similarity matrix and create arrays.

        For an n x m matrix, generates arrays where:
        - Each array corresponds to a starting column position
        - Each diagonal moves north-east (up-right): (0, j), (-1, j+1), (-2, j+2), etc.
        - When reaching the top (row < 0), wraps around to the bottom (row += n)
        - When reaching the right edge, wraps around to the left (col = 0)
        - Continues for exactly n elements to show the full wraparound pattern

        This helps visualize matching patterns along diagonal directions in puzzle piece matching.
        """
        n, m = matrix.shape

        print(f"\n{'='*70}")
        print(f"DIAGONAL ARRAYS FROM LENGTH SIMILARITY MATRIX: Piece {piece1_id} <-> Piece {piece2_id}")
        print(f"(North-East diagonals with wraparound)")
        print(f"{'='*70}\n")

        # ANSI color codes
        colors = [
            "\033[42m",  # Green background
            "\033[43m",  # Yellow background
            "\033[44m",  # Blue background
            "\033[45m",  # Magenta background
            "\033[46m",  # Cyan background
            "\033[41m",  # Red background
        ]
        reset = "\033[0m"

        # Generate diagonal arrays for each starting column position
        for start_col in range(m):
            diagonal_values = []
            score_values = []
            diag_row_labels = []
            segment_mappings = []

            # Start from position (0, start_col) and go diagonally north-east (up-right with wraparound)
            row, col = 0, start_col
            for step in range(n):
                similarity = matrix[row, col]
                diag_row_labels.append(f"P{piece1_id}-S{row}")
                score_values.append(similarity)

                diagonal_values.append(f"{similarity:.3f}")
                # Create segment mapping: P{piece1_id}-S{row}=P{piece2_id}-S{col}
                segment_mappings.append(f"P{piece1_id}-S{row}=P{piece2_id}-S{col}")

                # Move north-east: up one row (with wraparound), right one column
                row = (row - 1) % n  # Wraparound when going above 0
                col = (col + 1) % m  # Wraparound when going beyond m-1

            # Find similarity groups (threshold 0.7 for high matches)
            groups = MatchVisualizer._find_similarity_score_groups(score_values, threshold=0.7)
            group_map = {}
            for group_start, group_len in groups:
                for i in range(group_len):
                    idx = (group_start + i) % n
                    group_map[idx] = group_start

            # Create output
            print(f"Diagonal starting at S{start_col}:")
            print(f"  Segments: {' -> '.join(diag_row_labels)}")

            # Print values with colors for similarity groups
            colored_values = []
            for i, val in enumerate(diagonal_values):
                if i in group_map:
                    color = colors[group_map[i] % len(colors)]
                    colored_values.append(f"{color}{val}{reset}")
                else:
                    colored_values.append(val)

            print(f"  Values:   {' -> '.join(colored_values)}")

            # Print segment mapping array
            print(f"  Segment Map: [{', '.join(repr(m) for m in segment_mappings)}]")

            # Print legend for groups
            if groups:
                print(f"  Groups:   ", end="")
                for group_idx, (group_start, group_len) in enumerate(groups):
                    group_indices = [(group_start + i) % n for i in range(group_len)]
                    color = colors[group_start % len(colors)]
                    indices_str = ", ".join(str(idx) for idx in group_indices)
                    print(f"{color}Group {group_idx+1}: [{indices_str}]{reset}", end="  ")
                print()
            print()

    @staticmethod
    def _find_rmsd_similarity_groups(rmsds: List[float], tolerance: float = 0.05) -> List[Tuple[int, int]]:
        """Find groups of continuous similar RMSD values in a circular array.

        Lower RMSD = better match. Groups values below tolerance threshold.
        Treats the array as circular, so groups can wrap from end to start.
        Allows one item per group to be different (outlier), but still keep them grouped.
        n/a values (-1) can only be the single outlier, and groups must have at least one valid RMSD.
        Returns list of (start_idx, length) tuples for groups with low RMSD.
        """
        n = len(rmsds)
        if n == 0:
            return []

        # Filter to only valid RMSD values (>= 0, not n/a which is -1)
        valid_indices = [i for i in range(n) if rmsds[i] >= 0]
        if len(valid_indices) < 2:
            return []

        groups = []
        visited = set()

        for start_idx in valid_indices:
            if start_idx in visited:
                continue

            current_group = [start_idx]
            visited.add(start_idx)
            base_rmsd = rmsds[start_idx]
            outlier_used = False  # Track if we've used the one allowed outlier

            # Try to extend forward
            next_idx = (start_idx + 1) % n
            while next_idx != start_idx and next_idx not in visited:
                if next_idx in valid_indices:
                    rmsd_diff = abs(rmsds[next_idx] - base_rmsd)
                    if rmsd_diff <= tolerance:
                        current_group.append(next_idx)
                        visited.add(next_idx)
                    elif not outlier_used:
                        # Allow one outlier (but not n/a which is -1)
                        current_group.append(next_idx)
                        visited.add(next_idx)
                        outlier_used = True
                    else:
                        break
                else:
                    break
                next_idx = (next_idx + 1) % n

            # Try to extend backward from start
            prev_idx = (start_idx - 1) % n
            while prev_idx != start_idx and prev_idx not in visited:
                if prev_idx in valid_indices:
                    rmsd_diff = abs(rmsds[prev_idx] - base_rmsd)
                    if rmsd_diff <= tolerance:
                        current_group.insert(0, prev_idx)
                        visited.add(prev_idx)
                    elif not outlier_used:
                        # Allow one outlier (but not n/a which is -1)
                        current_group.insert(0, prev_idx)
                        visited.add(prev_idx)
                        outlier_used = True
                    else:
                        break
                else:
                    break
                prev_idx = (prev_idx - 1) % n

            if len(current_group) >= 2:
                groups.append((current_group[0], len(current_group)))

        return groups

    @staticmethod
    def extract_diagonal_arrays_from_shape_similarity(matrix: np.ndarray, piece1_id: int, piece2_id: int) -> None:
        """Extract diagonal values from shape similarity (RMSD) matrix and create arrays.

        For an n x m matrix, generates arrays where:
        - Each array corresponds to a starting column position
        - Each diagonal moves north-east (up-right): (0, j), (-1, j+1), (-2, j+2), etc.
        - When reaching the top (row < 0), wraps around to the bottom (row += n)
        - When reaching the right edge, wraps around to the left (col = 0)
        - Continues for exactly n elements to show the full wraparound pattern
        - Values marked as -1 are displayed as "n/a" (not calculated)

        This helps visualize matching patterns along diagonal directions in puzzle piece matching.
        """
        n, m = matrix.shape

        print(f"\n{'='*70}")
        print(f"DIAGONAL ARRAYS FROM SHAPE SIMILARITY MATRIX: Piece {piece1_id} <-> Piece {piece2_id}")
        print(f"(North-East diagonals with wraparound)")
        print(f"{'='*70}\n")

        # ANSI color codes
        colors = [
            "\033[42m",  # Green background
            "\033[43m",  # Yellow background
            "\033[44m",  # Blue background
            "\033[45m",  # Magenta background
            "\033[46m",  # Cyan background
            "\033[41m",  # Red background
        ]
        reset = "\033[0m"

        # Generate diagonal arrays for each starting column position
        for start_col in range(m):
            diagonal_values = []
            rmsd_values = []
            diag_row_labels = []
            segment_mappings = []

            # Start from position (0, start_col) and go diagonally north-east (up-right with wraparound)
            row, col = 0, start_col
            for step in range(n):
                rmsd = matrix[row, col]
                diag_row_labels.append(f"P{piece1_id}-S{row}")
                rmsd_values.append(rmsd)

                if rmsd < 0:  # -1 means not calculated
                    diagonal_values.append("n/a")
                else:
                    diagonal_values.append(f"{rmsd:.4f}")

                # Create segment mapping: P{piece1_id}-S{row}=P{piece2_id}-S{col}
                segment_mappings.append(f"P{piece1_id}-S{row}=P{piece2_id}-S{col}")

                # Move north-east: up one row (with wraparound), right one column
                row = (row - 1) % n  # Wraparound when going above 0
                col = (col + 1) % m  # Wraparound when going beyond m-1

            # Find similarity groups (low RMSD values, tolerance 0.05)
            groups = MatchVisualizer._find_rmsd_similarity_groups(rmsd_values, tolerance=0.05)
            group_map = {}
            for group_start, group_len in groups:
                for i in range(group_len):
                    idx = (group_start + i) % n
                    group_map[idx] = group_start

            # Create output
            print(f"Diagonal starting at S{start_col}:")
            print(f"  Segments: {' -> '.join(diag_row_labels)}")

            # Print values with colors for similarity groups
            colored_values = []
            for i, val in enumerate(diagonal_values):
                if i in group_map:
                    color = colors[group_map[i] % len(colors)]
                    colored_values.append(f"{color}{val}{reset}")
                else:
                    colored_values.append(val)

            print(f"  Values:   {' -> '.join(colored_values)}")

            # Print segment mapping array
            print(f"  Segment Map: [{', '.join(repr(m) for m in segment_mappings)}]")

            # Print legend for groups
            if groups:
                print(f"  Groups:   ", end="")
                for group_idx, (group_start, group_len) in enumerate(groups):
                    group_indices = [(group_start + i) % n for i in range(group_len)]
                    color = colors[group_start % len(colors)]
                    indices_str = ", ".join(str(idx) for idx in group_indices)
                    print(f"{color}Group {group_idx+1}: [{indices_str}]{reset}", end="  ")
                print()
            print()

    @staticmethod
    def analyze_cross_diagonal_groups(length_groups_dict: dict, shape_groups_dict: dict, angle_groups_dict: dict,
                                     piece1_id: int, piece2_id: int, num_segments: int = 9) -> None:
        """Analyze and highlight diagonal groups that appear in at least 2 of the 3 matrices.

        Args:
            length_groups_dict: Dictionary mapping column -> list of groups from length similarity
            shape_groups_dict: Dictionary mapping column -> list of groups from shape similarity
            angle_groups_dict: Dictionary mapping column -> list of groups from angle similarity
            piece1_id: ID of piece 1
            piece2_id: ID of piece 2
            num_segments: Number of segments in piece 1 (for modulo wraparound)
        """
        print(f"\n{'='*70}")
        print(f"CROSS-DIAGONAL GROUP ANALYSIS: Piece {piece1_id} <-> Piece {piece2_id}")
        print(f"(Groups appearing in at least 2 of 3 matrices)")
        print(f"{'='*70}\n")

        # Get all column indices
        all_cols = set()
        all_cols.update(length_groups_dict.keys())
        all_cols.update(shape_groups_dict.keys())
        all_cols.update(angle_groups_dict.keys())

        common_groups_found = False

        for col in sorted(all_cols):
            length_groups = length_groups_dict.get(col, [])
            shape_groups = shape_groups_dict.get(col, [])
            angle_groups = angle_groups_dict.get(col, [])

            # Convert groups to sets of indices for comparison
            length_group_sets = [set((col_start + i) % num_segments for i in range(col_len)) for col_start, col_len in length_groups]
            shape_group_sets = [set((col_start + i) % num_segments for i in range(col_len)) for col_start, col_len in shape_groups]
            angle_group_sets = [set((col_start + i) % num_segments for i in range(col_len)) for col_start, col_len in angle_groups]

            # Find overlapping groups (at least 50% overlap counts as common)
            for lg_idx, lg in enumerate(length_group_sets):
                for sg_idx, sg in enumerate(shape_group_sets):
                    overlap = lg & sg
                    if len(overlap) >= min(len(lg), len(sg)) * 0.5 if min(len(lg), len(sg)) > 0 else False:
                        common_groups_found = True
                        indices_str = ", ".join(str(i) for i in sorted(overlap))
                        print(f"[HIT] Diagonal S{col}: Common group in LENGTH + SHAPE: [{indices_str}]")

                for ag_idx, ag in enumerate(angle_group_sets):
                    overlap = lg & ag
                    if len(overlap) >= min(len(lg), len(ag)) * 0.5 if min(len(lg), len(ag)) > 0 else False:
                        common_groups_found = True
                        indices_str = ", ".join(str(i) for i in sorted(overlap))
                        print(f"[HIT] Diagonal S{col}: Common group in LENGTH + ANGLE: [{indices_str}]")

            for sg_idx, sg in enumerate(shape_group_sets):
                for ag_idx, ag in enumerate(angle_group_sets):
                    overlap = sg & ag
                    if len(overlap) >= min(len(sg), len(ag)) * 0.5 if min(len(sg), len(ag)) > 0 else False:
                        common_groups_found = True
                        indices_str = ", ".join(str(i) for i in sorted(overlap))
                        print(f"[HIT] Diagonal S{col}: Common group in SHAPE + ANGLE: [{indices_str}]")

        if not common_groups_found:
            print("No common groups found across matrices.")

        print()

    @staticmethod
    def analyze_cross_diagonal_groups_length_shape(length_groups_dict: dict, shape_groups_dict: dict,
                                                  piece1_id: int, piece2_id: int, num_segments: int = 9) -> None:
        """Analyze and highlight diagonal groups that appear in both LENGTH and SHAPE matrices.

        Args:
            length_groups_dict: Dictionary mapping column -> list of groups from length similarity
            shape_groups_dict: Dictionary mapping column -> list of groups from shape similarity
            piece1_id: ID of piece 1
            piece2_id: ID of piece 2
            num_segments: Number of segments in piece 1 (for modulo wraparound)
        """
        print(f"\n{'='*70}")
        print(f"CROSS-DIAGONAL GROUP ANALYSIS: Piece {piece1_id} <-> Piece {piece2_id}")
        print(f"(Groups appearing in both LENGTH and SHAPE matrices)")
        print(f"{'='*70}\n")

        # Get all column indices
        all_cols = set()
        all_cols.update(length_groups_dict.keys())
        all_cols.update(shape_groups_dict.keys())

        common_groups_found = False

        for col in sorted(all_cols):
            length_groups = length_groups_dict.get(col, [])
            shape_groups = shape_groups_dict.get(col, [])

            # Convert groups to sets of indices for comparison
            length_group_sets = [set((col_start + i) % num_segments for i in range(col_len)) for col_start, col_len in length_groups]
            shape_group_sets = [set((col_start + i) % num_segments for i in range(col_len)) for col_start, col_len in shape_groups]

            # Find overlapping groups (at least 50% overlap counts as common)
            for lg_idx, lg in enumerate(length_group_sets):
                for sg_idx, sg in enumerate(shape_group_sets):
                    overlap = lg & sg
                    if len(overlap) >= min(len(lg), len(sg)) * 0.5 if min(len(lg), len(sg)) > 0 else False:
                        common_groups_found = True
                        indices_str = ", ".join(str(i) for i in sorted(overlap))
                        print(f"[HIT] Diagonal S{col}: Common group in LENGTH + SHAPE: [{indices_str}]")

        if not common_groups_found:
            print("No common groups found between LENGTH and SHAPE matrices.")

        print()
