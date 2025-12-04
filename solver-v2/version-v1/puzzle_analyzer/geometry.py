import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class LineSegment:
    """Represents a straight line segment on the puzzle piece contour."""
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    length: float
    is_border_edge: bool = False

    def angle_with(self, other: 'LineSegment') -> float:
        """Calculate angle between two line segments in degrees."""
        v1 = np.array(self.p2) - np.array(self.p1)
        v2 = np.array(other.p2) - np.array(other.p1)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(np.abs(cos_angle)))

    def midpoint(self) -> Tuple[float, float]:
        """Calculate the midpoint of the line segment."""
        return ((self.p1[0] + self.p2[0]) / 2, (self.p1[1] + self.p2[1]) / 2)

    def direction_vector(self) -> np.ndarray:
        """Get normalized direction vector."""
        v = np.array(self.p2) - np.array(self.p1)
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v

    def normal_vector(self) -> np.ndarray:
        """Calculate the perpendicular normal vector."""
        v = np.array(self.p2) - np.array(self.p1)
        normal = np.array([-v[1], v[0]])
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        return normal
