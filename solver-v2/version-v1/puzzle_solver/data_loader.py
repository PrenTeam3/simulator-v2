"""Load and parse puzzle analysis data from JSON files."""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Point:
    x: float
    y: float
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Corner:
    x: float
    y: float
    corner_type: str = "unknown"

    def to_point(self) -> Point:
        return Point(self.x, self.y)


@dataclass
class AnalyzedPuzzlePiece:
    piece_id: int
    area: float
    centroid: Point
    contour_points: List[Point]
    outer_corners: List[Corner]
    inner_corners: List[Corner]
    curved_points: List[Corner]
    frame_corners: List[Corner]
    border_edges: List[Tuple[Point, Point]]

    @property
    def has_frame_corner(self) -> bool:
        return len(self.frame_corners) > 0


@dataclass
class PuzzleAnalysisData:
    timestamp: str
    image_path: str
    pieces: List[AnalyzedPuzzlePiece]


class PuzzleDataLoader:
    """Loads puzzle analysis data from JSON files."""

    @staticmethod
    def load_from_temp_folder(temp_folder: Path) -> PuzzleAnalysisData:
        """Load puzzle analysis from most recent temp folder or specific folder."""
        if temp_folder.name.startswith('puzzle_analysis_'):
            latest_folder = temp_folder
        else:
            latest_folder = sorted(temp_folder.glob('puzzle_analysis_*'))[-1]

        json_path = latest_folder / 'analysis_data.json'
        print(f"Loading: {latest_folder.name}")

        with open(json_path) as f:
            data = json.load(f)

        def _parse_corners(c_list: List, c_type: str) -> List[Corner]:
            return [Corner(c['x'], c['y'], c_type) for c in c_list]

        pieces = []
        for pd in data['pieces']:
            piece = AnalyzedPuzzlePiece(
                piece_id=pd['piece_id'],
                area=pd['area'],
                centroid=Point(pd['centroid']['x'], pd['centroid']['y']),
                contour_points=[Point(p['x'], p['y']) for p in pd['contour_recognition']['contour_points']],
                outer_corners=_parse_corners(pd['outer_corners'], 'outer'),
                inner_corners=_parse_corners(pd['inner_corners'], 'inner'),
                curved_points=_parse_corners(pd['curved_points'], 'curved'),
                frame_corners=_parse_corners(pd['frame_corners'], 'frame'),
                border_edges=[(Point(e['p1']['x'], e['p1']['y']), Point(e['p2']['x'], e['p2']['y']))
                              for e in pd['border_edges']]
            )
            pieces.append(piece)

        return PuzzleAnalysisData(data['timestamp'], data['image_path'], pieces)


def print_analysis_summary(data: PuzzleAnalysisData) -> None:
    """Print summary of loaded analysis data."""
    print(f"\n{'='*60}")
    print(f"PUZZLE ANALYSIS - {data.timestamp}")
    print(f"Image: {data.image_path}")
    print(f"Pieces: {len(data.pieces)}")
    for p in data.pieces:
        corners = f"({len(p.frame_corners)} frame)" if p.has_frame_corner else "(interior)"
        print(f"  P{p.piece_id}: {p.area:.0f}px² | Corners: {corners}")
    print(f"{'='*60}\n")

