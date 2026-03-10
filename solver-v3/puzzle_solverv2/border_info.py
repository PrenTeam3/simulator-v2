"""Step 2 — Border info extraction.

For each piece, collect all outside segments (is_outside=True), convert their
lengths to mm, and build corner_pairs from the already-detected frame corners.

Output per piece (PieceBorderInfo):
    piece_idx       — index of the piece
    outside_segments— list of segment dicts {seg_id, piece_idx, length_mm, p1, p2}
    corner_pairs    — list of (seg_a, seg_b) tuples that together form a frame corner
    straight_edges  — same as outside_segments (each segment individually)
    centroid_px     — pixel centroid of the piece contour
    px_per_mm       — scale factor from frame (Step 1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from puzzle_solverv2.frame import PuzzleFrame


# ─────────────────────────────────────────────
#  Data structure
# ─────────────────────────────────────────────

@dataclass
class PieceBorderInfo:
    piece_idx:        int
    outside_segments: list[dict]          # {seg_id, piece_idx, length_mm, p1, p2}
    corner_pairs:     list[tuple]         # [(seg_a, seg_b), ...] — frame corner pairs
    straight_edges:   list[dict]          # same dicts as outside_segments, each individually
    centroid_px:      Optional[tuple]     # (cx, cy) in pixels, or None
    px_per_mm:        float


# ─────────────────────────────────────────────
#  Extraction
# ─────────────────────────────────────────────

def extract_border_info(
    corners_list:    list[dict],
    classifications: list[dict],
    frame:           PuzzleFrame,
) -> list[PieceBorderInfo]:
    """
    Extract border info for all pieces.

    Args:
        corners_list:    output of detect_corners() per piece
        classifications: output of classify_piece() per piece
        frame:           PuzzleFrame from Step 1 (provides px_per_mm)

    Returns:
        List of PieceBorderInfo, one per piece.
    """
    pieces = []

    for idx, (info, cls) in enumerate(zip(corners_list, classifications)):

        # Collect all outside segments and convert lengths to mm
        # seg_id uses format P{piece_idx}S{local_seg_idx} — matches debug.png labels
        outside_segments = []
        seg_id_map = {}  # maps (p1, p2) → seg dict, for corner_pair lookup below

        for local_seg_idx, seg in enumerate(info['all_segments']):
            if not seg.get('is_outside', False):
                continue

            length_mm = frame.px_to_mm(seg['length'])
            seg_dict = {
                'seg_id':    f"P{idx}S{local_seg_idx}",
                'piece_idx': idx,
                'length_mm': length_mm,
                'p1':        seg['p1'],
                'p2':        seg['p2'],
            }
            outside_segments.append(seg_dict)
            seg_id_map[(seg['p1'], seg['p2'])] = seg_dict

        # Build corner_pairs from already-detected frame corners
        # Each frame corner has prev_seg and next_seg — both must be outside segments
        corner_pairs = []
        for fc in cls.get('frame_corners', []):
            prev = fc['prev_seg']
            next_ = fc['next_seg']
            key_prev  = (prev['p1'],  prev['p2'])
            key_next  = (next_['p1'], next_['p2'])
            seg_a = seg_id_map.get(key_prev)
            seg_b = seg_id_map.get(key_next)
            if seg_a is not None and seg_b is not None:
                corner_pairs.append((seg_a, seg_b))

        pieces.append(PieceBorderInfo(
            piece_idx=idx,
            outside_segments=outside_segments,
            corner_pairs=corner_pairs,
            straight_edges=outside_segments,   # same list — each segment individually
            centroid_px=info.get('centroid'),
            px_per_mm=frame.px_per_mm,
        ))

    return pieces


# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────

def log_border_info(pieces: list[PieceBorderInfo]) -> None:
    """Print a structured summary of extracted border info."""
    total_length = sum(
        s['length_mm'] for p in pieces for s in p.outside_segments
    )
    expected = 2 * (190.0 + 128.0)  # full perimeter

    for p in pieces:
        piece_total = sum(s['length_mm'] for s in p.outside_segments)
        print(f"  Piece {p.piece_idx}:")
        print(f"    outside segments : {len(p.outside_segments)}")
        for s in p.outside_segments:
            print(f"      seg {s['seg_id']:>2}: {s['length_mm']:6.1f} mm")
        print(f"    corner pairs     : {len(p.corner_pairs)}")
        for i, (a, b) in enumerate(p.corner_pairs):
            print(f"      pair {i}: seg {a['seg_id']} ({a['length_mm']:.1f}mm)"
                  f" + seg {b['seg_id']} ({b['length_mm']:.1f}mm)")
        print(f"    total outside    : {piece_total:.1f} mm")

    print(f"\n  Grand total outside : {total_length:.1f} mm")
    print(f"  Expected perimeter  : {expected:.1f} mm")
