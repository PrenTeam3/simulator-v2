"""_SearchState and _SearchConfig dataclasses for the tree search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from puzzle_solverv2.frame import PuzzleFrame


@dataclass
class _SearchConfig:
    """Immutable search parameters shared across all recursive calls."""
    pieces_variants: list
    corners_list:    list
    frame:           PuzzleFrame
    output_dir:      Path
    tolerance:       float
    max_depth:       int
    mode:            str
    centroid_by_idx: dict


@dataclass
class _SearchState:
    """Mutable traversal context passed between recursive calls."""
    used:           set
    side:           str          # current side being filled
    offset_mm:      float        # how far along the side the next piece starts
    target_mm:      float        # full length of the current side
    end_pos:        str          # corner at the far end of the current side
    fwd_is_horiz:   bool         # whether the current side is horizontal
    start_from_end: bool         # measure offset from the far end
    turn_side:      str          # side that a corner piece's turn seg would cover
    occ:            dict         # occupancy dict so far
    base_canvas:    Any          # np.ndarray accumulated so far (None in console_only)
    ox:             int
    oy:             int
    folder:          Path         # parent folder for this depth's output files
    prev_color:      tuple        # color of the last placed piece
    placed_contours: list         # list[np.ndarray] of already-placed contours in puzzle-mm
    folder_prefix:   str = 'P'   # 'B_P' when this is the first piece on a new side
