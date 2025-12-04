"""Geometry utility functions for edge solver v2."""

import numpy as np
import cv2
from math import cos, sin, atan2
from typing import Tuple


class GeometryUtils:
    """Shared geometry utilities for alignment and transformations."""

    @staticmethod
    def determine_chain_endpoints(chain_seg_ids, all_segments, piece, frame_touching_ids):
        """Determine which end of a chain is the blue (frame) vs red (interior) endpoint.

        This is the shared logic used by both chain_matcher and assembly_solver.

        Args:
            chain_seg_ids: List of segment IDs in the chain (in order)
            all_segments: All segments for this piece
            piece: The piece object
            frame_touching_ids: Set of segment IDs that touch the frame

        Returns:
            Tuple of (blue_point, red_point) as numpy arrays
        """
        if len(chain_seg_ids) == 0:
            return None, None

        # Get the first and last segment IDs
        first_seg_id = chain_seg_ids[0]
        last_seg_id = chain_seg_ids[-1]

        # Get the actual segment objects
        first_seg = next((s for s in all_segments if s.segment_id == first_seg_id), None)
        last_seg = next((s for s in all_segments if s.segment_id == last_seg_id), None)

        if not first_seg or not last_seg:
            return None, None

        # Determine which end is closer to frame
        # Check neighbors of first segment
        num_segments = len(all_segments)
        first_prev_id = (first_seg_id - 1) % num_segments
        first_next_id = (first_seg_id + 1) % num_segments
        first_prev_is_frame = first_prev_id in frame_touching_ids
        first_next_is_frame = first_next_id in frame_touching_ids

        # If the chain has multiple segments, check which direction it extends
        if len(chain_seg_ids) > 1:
            second_seg_id = chain_seg_ids[1]
            # If the second segment is in the "next" direction from first
            if second_seg_id == first_next_id:
                # Chain goes: first -> next direction
                # So if prev is frame, frame is at START of first segment
                frame_at_start = first_prev_is_frame
            elif second_seg_id == first_prev_id:
                # Chain goes: first -> prev direction
                # So if next is frame, frame is at START of first segment
                frame_at_start = first_next_is_frame
            else:
                # Segments are not adjacent - shouldn't happen
                frame_at_start = first_prev_is_frame
        else:
            # Single segment chain
            frame_at_start = first_prev_is_frame

        # DEBUG: Print what we determined
        print(f"      [DEBUG] Chain {chain_seg_ids}: first={first_seg_id}, last={last_seg_id}")
        print(f"      [DEBUG] first_prev={first_prev_id} (frame={first_prev_is_frame}), first_next={first_next_id} (frame={first_next_is_frame})")
        if len(chain_seg_ids) > 1:
            print(f"      [DEBUG] second_seg={second_seg_id}, frame_at_start={frame_at_start}")
        print(f"      [DEBUG] first_seg corners: start=({first_seg.start_corner.x:.1f},{first_seg.start_corner.y:.1f}), end=({first_seg.end_corner.x:.1f},{first_seg.end_corner.y:.1f})")
        print(f"      [DEBUG] last_seg corners: start=({last_seg.start_corner.x:.1f},{last_seg.start_corner.y:.1f}), end=({last_seg.end_corner.x:.1f},{last_seg.end_corner.y:.1f})")

        # The blue and red dots mark the START and END of the chain itself
        # NOT where the chain connects to the frame
        # Blue is always the end closer to frame, Red is the end farther from frame

        # For a chain [11, 10, 9, 8, 7]:
        # - Blue dot: between S11 and S10 (the shared corner = S11.end_corner or S10.start_corner)
        # - Red dot: between S7 and S6 (the end of the chain = S7.end_corner)

        # The blue and red dots mark the EXTERNAL boundaries of the chain
        # For chain [11, 10, 9, 8, 7]:
        # - Blue: junction BEFORE S11 (between S0 and S11) = S11.start_corner
        # - Red: junction AFTER S7 (between S7 and S6) = S7.end_corner

        # Blue dot: start of the first segment in the chain (external boundary)
        blue_point = np.array([first_seg.start_corner.x, first_seg.start_corner.y])

        # Red dot: end of the last segment in the chain (external boundary)
        red_point = np.array([last_seg.end_corner.x, last_seg.end_corner.y])

        print(f"      [DEBUG] Blue=first.start, Red=last.end")

        return blue_point, red_point

    @staticmethod
    def rotate_point(point, origin, angle_rad=None, target_point=None):
        """Rotate a point around an origin.

        Args:
            point: Point to rotate (x, y) or array
            origin: Origin point (x, y) or array
            angle_rad: Angle in radians to rotate (optional)
            target_point: If provided, calculates angle to align point towards target_point

        Returns:
            Rotated point as numpy array
        """
        point = np.array(point, dtype=np.float64)
        origin = np.array(origin, dtype=np.float64)

        # Calculate angle if target_point is provided
        if target_point is not None:
            target_point = np.array(target_point, dtype=np.float64)

            # Current vector from origin to point
            current_vec = point - origin
            # Desired vector from origin to target
            desired_vec = target_point - origin

            # Calculate angle between vectors
            if np.linalg.norm(current_vec) > 0 and np.linalg.norm(desired_vec) > 0:
                current_angle = atan2(current_vec[1], current_vec[0])
                desired_angle = atan2(desired_vec[1], desired_vec[0])
                angle_rad = desired_angle - current_angle
            else:
                angle_rad = 0

        if angle_rad is None:
            raise ValueError("Either angle_rad or target_point must be provided")

        # Translate to origin
        translated = point - origin

        # Rotate
        cos_a = cos(angle_rad)
        sin_a = sin(angle_rad)
        rotated = np.array([
            translated[0] * cos_a - translated[1] * sin_a,
            translated[0] * sin_a + translated[1] * cos_a
        ], dtype=np.float64)

        # Translate back
        return rotated + origin

    @staticmethod
    def calculate_arrow_for_segment(points, piece, arrow_length=25):
        """Calculate arrow (midpoint and outward normal) for a segment.

        Args:
            points: Array of segment points (N x 2)
            piece: Piece object (for checking if arrow points outward)
            arrow_length: Length of arrow for testing direction

        Returns:
            Tuple of (midpoint, normal_vector) where:
                - midpoint: (x, y) at middle of segment
                - normal_vector: (dx, dy) normalized outward-pointing normal
        """
        points = np.array(points, dtype=np.float64)

        # Find midpoint
        mid_idx = len(points) // 2
        midpoint = points[mid_idx]

        # Calculate tangent at midpoint
        if mid_idx == 0:
            tangent = points[mid_idx + 1] - points[mid_idx]
        elif mid_idx == len(points) - 1:
            tangent = points[mid_idx] - points[mid_idx - 1]
        else:
            tangent = points[mid_idx + 1] - points[mid_idx - 1]

        # Calculate normal (perpendicular to tangent)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

        # Normalize
        if np.linalg.norm(normal) > 0:
            normal = normal / np.linalg.norm(normal)

            # Ensure normal points outward (away from piece interior)
            piece_contour = np.array([[p.x, p.y] for p in piece.contour_points], dtype=np.int32)
            test_point = midpoint + normal * arrow_length

            is_inside = cv2.pointPolygonTest(piece_contour, tuple(test_point.astype(float)), False)
            if is_inside > 0:
                # Arrow points inward, flip it
                normal = -normal

        return midpoint, normal

    @staticmethod
    def align_chains(chain1_points, chain2_points, B1, B2, R1, R2):
        """Align two chains by translating and rotating.

        Aligns chain2 to chain1 by:
        1. Translating so B2 -> B1 (blue/frame endpoints match)
        2. Rotating around B1 so R2 -> R1 (red/interior endpoints match)

        Args:
            chain1_points: Points of first chain (N x 2)
            chain2_points: Points of second chain (M x 2)
            B1: Blue endpoint of chain1 (frame connection)
            B2: Blue endpoint of chain2 (frame connection)
            R1: Red endpoint of chain1 (interior)
            R2: Red endpoint of chain2 (interior)

        Returns:
            Tuple of (chain2_aligned, R2_aligned) where:
                - chain2_aligned: Aligned points of chain2
                - R2_aligned: Aligned position of R2
        """
        chain1_points = np.array(chain1_points, dtype=np.float64)
        chain2_points = np.array(chain2_points, dtype=np.float64)
        B1 = np.array(B1, dtype=np.float64)
        B2 = np.array(B2, dtype=np.float64)
        R1 = np.array(R1, dtype=np.float64)
        R2 = np.array(R2, dtype=np.float64)

        # Step 1: Translate chain2 so B2 aligns with B1
        translation = B1 - B2
        chain2_translated = chain2_points + translation
        R2_translated = R2 + translation

        # Step 2: Rotate around B1 to align R2 with R1
        R2_aligned = GeometryUtils.rotate_point(R2_translated, B1, target_point=R1)

        # Calculate the angle used for R2
        vec_current = R2_translated - B1
        vec_desired = R1 - B1

        if np.linalg.norm(vec_current) > 0 and np.linalg.norm(vec_desired) > 0:
            angle_current = atan2(vec_current[1], vec_current[0])
            angle_desired = atan2(vec_desired[1], vec_desired[0])
            rotation_angle = angle_desired - angle_current
        else:
            rotation_angle = 0

        # Apply same rotation to all points in chain2
        chain2_aligned = np.array([
            GeometryUtils.rotate_point(pt, B1, angle_rad=rotation_angle)
            for pt in chain2_translated
        ], dtype=np.float64)

        return chain2_aligned, R2_aligned

    @staticmethod
    def calculate_length(points):
        """Calculate total length of a polyline.

        Args:
            points: Array of points (N x 2)

        Returns:
            Total length in pixels
        """
        points = np.array(points, dtype=np.float64)
        if len(points) < 2:
            return 0.0

        diffs = np.diff(points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)

    @staticmethod
    def calculate_angle_between_vectors(vec1, vec2):
        """Calculate angle in degrees between two vectors.

        Args:
            vec1: First vector (2D)
            vec2: Second vector (2D)

        Returns:
            Angle in degrees (0-180)
        """
        vec1 = np.array(vec1, dtype=np.float64)
        vec2 = np.array(vec2, dtype=np.float64)

        # Normalize
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        vec1_normalized = vec1 / norm1
        vec2_normalized = vec2 / norm2

        # Calculate angle
        dot_product = np.clip(np.dot(vec1_normalized, vec2_normalized), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)

        return np.degrees(angle_rad)
