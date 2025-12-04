"""Utility functions for puzzle solver."""
from typing import List, Tuple
import numpy as np
from .data_classes import Point, Corner


# ============================================================================
# BASIC UTILITIES
# ============================================================================

def distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


def distance_point_to_corner(p: Point, c: Corner) -> float:
    """Calculate distance between a point and a corner."""
    return ((p.x - c.x) ** 2 + (p.y - c.y) ** 2) ** 0.5


def normalize_vector(dx: float, dy: float) -> Tuple[float, float]:
    """Normalize a 2D vector."""
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length > 0:
        return dx / length, dy / length
    return 1, 0


# ============================================================================
# GEOMETRY OPERATIONS
# ============================================================================

def calculate_centroid(points: List[Point]) -> Tuple[float, float]:
    """Calculate centroid (center) of a set of points."""
    if not points:
        return 0.0, 0.0
    cx = sum(p.x for p in points) / len(points)
    cy = sum(p.y for p in points) / len(points)
    return cx, cy


def calculate_segment_center(segment) -> Tuple[float, float]:
    """Calculate center point of a segment.

    Args:
        segment: ContourSegment object with contour_points attribute

    Returns:
        Tuple of (center_x, center_y)
    """
    return calculate_centroid(segment.contour_points)


def normalize_and_center(points: List[Point]) -> List[Point]:
    """Normalize and center a contour: center at origin and scale to unit size."""
    if len(points) < 2:
        return points

    # Center at origin
    cx, cy = calculate_centroid(points)
    centered = [Point(p.x - cx, p.y - cy) for p in points]

    # Scale to unit size
    max_dist = max(np.sqrt(p.x**2 + p.y**2) for p in centered)
    if max_dist > 0:
        normalized = [Point(p.x / max_dist, p.y / max_dist) for p in centered]
    else:
        normalized = centered

    return normalized


def rotate_points(points: List[Point], angle: float, center: Point = None) -> List[Point]:
    """
    Rotate points around a center by given angle.

    Args:
        points: List of points to rotate
        angle: Rotation angle in radians
        center: Center point. If None, rotates around origin (0, 0)

    Returns:
        List of rotated points
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotated = []

    if center is None:
        # Rotate around origin
        for p in points:
            new_x = p.x * cos_a - p.y * sin_a
            new_y = p.x * sin_a + p.y * cos_a
            rotated.append(Point(new_x, new_y))
    else:
        # Rotate around center
        for p in points:
            x = p.x - center.x
            y = p.y - center.y
            new_x = x * cos_a - y * sin_a
            new_y = x * sin_a + y * cos_a
            rotated.append(Point(new_x + center.x, new_y + center.y))

    return rotated


def find_min_area_rotation(pts1: List[Point], pts2: List[Point], precision: int = 2) -> float:
    """
    Find the rotation angle for pts2 that minimizes the bounding box area
    of both pts1 and pts2 combined.

    Args:
        pts1: First set of points (stays fixed)
        pts2: Second set of points (to be rotated)
        precision: Rotation search step size in degrees (default: 2)

    Returns:
        Optimal rotation angle in radians
    """
    import cv2

    best_angle = 0.0
    best_area = float('inf')

    # Search over rotation angles
    for angle_deg in range(0, 360, precision):
        angle_rad = np.radians(angle_deg)

        # Rotate pts2 by this angle around origin
        pts2_rotated = rotate_points(pts2, angle_rad)

        # Combine both point sets
        all_points = pts1 + pts2_rotated
        points_array = np.array([(p.x, p.y) for p in all_points], dtype=np.float32)

        # Calculate the minimal-area rotated bounding box
        if len(points_array) >= 3:
            rotated_rect = cv2.minAreaRect(points_array)
            bbox_width = rotated_rect[1][0]
            bbox_height = rotated_rect[1][1]
            area = bbox_width * bbox_height

            # Keep track of the best rotation
            if area < best_area:
                best_area = area
                best_angle = angle_rad

    return best_angle
