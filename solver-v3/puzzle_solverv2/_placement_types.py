"""Shared constants, Candidate dataclass, and occupancy helpers for placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from puzzle_solverv2.frame import PUZZLE_WIDTH_MM, PUZZLE_HEIGHT_MM


# ── Frame corner positions (mm) ───────────────────────────────────────────────

_PUZZLE_CORNERS_MM: dict[str, np.ndarray] = {
    'TL': np.array([0.0,             0.0            ]),
    'TR': np.array([PUZZLE_WIDTH_MM, 0.0            ]),
    'BR': np.array([PUZZLE_WIDTH_MM, PUZZLE_HEIGHT_MM]),
    'BL': np.array([0.0,             PUZZLE_HEIGHT_MM]),
}

# Expected horizontal direction per corner (going away along the frame edge)
_EXPECTED_HORIZ: dict[str, np.ndarray] = {
    'TL': np.array([ 1.0,  0.0]),   # right along top
    'TR': np.array([-1.0,  0.0]),   # left  along top
    'BR': np.array([-1.0,  0.0]),   # left  along bottom
    'BL': np.array([ 1.0,  0.0]),   # right along bottom
}

# Allowed length error (mm) for side-filling and full-side checks
_SIDE_TOLERANCE: float = 30.0

# For each start corner: (side_name, target_mm, closing_corner, fwd_is_horiz)
_SECOND_CONFIGS: dict[str, tuple[str, float, str, bool]] = {
    'TL': ('top',    PUZZLE_WIDTH_MM,  'TR', True),
    'BL': ('bottom', PUZZLE_WIDTH_MM,  'BR', True),
}


# ── Candidate dataclass ───────────────────────────────────────────────────────

@dataclass
class Candidate:
    """One placement candidate returned by _build_candidates."""
    pv:        Any           # PieceVariants
    variant:   Any           # Variant from pv.variants
    placed_mm: np.ndarray    # (N, 2) contour in puzzle-mm coords
    seg_len:   float         # forward segment length in mm
    valid:     bool
    reason:    str
    label:     str
    seg_h:     dict | None   # horizontal segment (corner pieces only)
    seg_v:     dict | None   # vertical  segment (corner pieces only)
    is_corner: bool


# ── Side navigation ───────────────────────────────────────────────────────────

# Maps (current_side, end_pos) → (turn_side, next_side, next_end_pos, next_fwd_is_horiz, next_start_from_end)
# when a corner piece closes the current side at end_pos.
_CORNER_TURN: dict[tuple[str, str], tuple[str, str, str, bool, bool]] = {
    # Clockwise path: TL → TR → BR → BL → TL
    ('top',    'TR'): ('right',  'right',  'BR', False, False),
    ('right',  'BR'): ('bottom', 'bottom', 'BL', True,  True ),
    ('bottom', 'BL'): ('left',   'left',   'TL', False, True ),
    ('left',   'TL'): ('top',    'top',    'TR', True,  False),
    # Counter-clockwise path: BL → BR → TR → TL → BL
    ('bottom', 'BR'): ('right',  'right',  'TR', False, True ),
    ('right',  'TR'): ('top',    'top',    'TL', True,  True ),
    ('top',    'TL'): ('left',   'left',   'BL', False, False),
    ('left',   'BL'): ('bottom', 'bottom', 'BR', True,  False),
}

# Target length (mm) for each frame side
_SIDE_TARGET: dict[str, float] = {
    'top':    PUZZLE_WIDTH_MM,
    'bottom': PUZZLE_WIDTH_MM,
    'right':  PUZZLE_HEIGHT_MM,
    'left':   PUZZLE_HEIGHT_MM,
}


# ── Occupancy helpers ─────────────────────────────────────────────────────────

def empty_occupancy() -> dict[str, list]:
    """Return a fresh occupancy dict with all four sides empty."""
    return {'top': [], 'right': [], 'bottom': [], 'left': []}


def occ_add(
    occ:       dict[str, list],
    side:      str,
    piece_idx: int,
    seg_id:    Any,
    length_mm: float,
) -> None:
    """Append one segment entry to the given side in-place."""
    occ[side].append({'piece_idx': piece_idx, 'seg_id': seg_id, 'length_mm': length_mm})


def occ_add_candidate(
    occ:           dict[str, list],
    cand:          Candidate,
    fwd_side:      str,
    turn_side:     str | None = None,
    fwd_is_horiz:  bool = True,
) -> None:
    """Record a placed candidate's segments in the occupancy dict.

    Corner candidates contribute to fwd_side and (if given) turn_side.
    Edge candidates contribute only to fwd_side.

    fwd_is_horiz controls which segment goes along the forward side:
      True  (top/bottom) → seg_h is the forward segment, seg_v is the turn segment
      False (right/left) → seg_v is the forward segment, seg_h is the turn segment
    """
    if cand.is_corner:
        seg_fwd  = cand.seg_h if fwd_is_horiz else cand.seg_v
        seg_turn = cand.seg_v if fwd_is_horiz else cand.seg_h
        occ_add(occ, fwd_side, cand.pv.piece_idx,
                seg_fwd['seg_id'], cand.seg_len)
        if turn_side is not None and seg_turn is not None:
            occ_add(occ, turn_side, cand.pv.piece_idx,
                    seg_turn['seg_id'], seg_turn['length_mm'])
    else:
        occ_add(occ, fwd_side, cand.pv.piece_idx,
                cand.variant.edges[0]['seg_id'], cand.seg_len)
