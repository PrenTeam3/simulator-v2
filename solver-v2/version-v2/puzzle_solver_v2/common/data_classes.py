"""Basic data classes shared across puzzle solver modules."""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Point:
    """Represents a 2D point coordinate."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Corner:
    """Represents a corner point on a puzzle piece with type information."""
    x: float
    y: float
    corner_type: str = "unknown"

    def to_point(self) -> Point:
        return Point(self.x, self.y)


@dataclass
class Segment:
    """Represents a segment between two corners on a puzzle piece edge."""
    id: int
    corner_start: Point
    corner_end: Point
    is_straight: bool


@dataclass
class ContourSegment:
    """Represents a contour segment between two corners with detailed information."""
    segment_id: int
    piece_id: int
    start_corner: Corner
    end_corner: Corner
    contour_points: List[Point]
    piece_centroid: Point
    is_border_edge: bool = False


@dataclass
class SegmentMatch:
    """Represents a potential match between two segments."""
    piece1_id: int
    piece2_id: int
    seg1_id: int
    seg2_id: int
    match_score: float
    optimal_rotation: float  # Optimal rotation angle in radians
    shape_score: float  # Shape matching score component
    length_score: float  # Length matching score component
    description: str


@dataclass
class ExtendedSegmentMatch:
    """Represents an extended match between two segments that continues along the contour."""
    initial_match: SegmentMatch  # The initial segment match that started the extension
    extended_matches: List[SegmentMatch]  # Additional consecutive segment matches
    total_segments_matched: int  # Total number of segments matched (initial + extended)
    combined_score: float  # Combined score considering quality and length (50% avg_quality + 50% length_bonus)
    average_match_score: float  # Average match score across all matched segments

    @property
    def piece1_id(self):
        return self.initial_match.piece1_id

    @property
    def piece2_id(self):
        return self.initial_match.piece2_id
