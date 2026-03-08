"""Border-first puzzle solver.

Step 1: Extract outside segment lengths (in mm) per piece and print them.
Step 2: Find dimension-valid combinations, corner-pieces-first with fallback.
"""

from __future__ import annotations

import numpy as np
from itertools import permutations, product


PUZZLE_WIDTH_MM  = 190.0
PUZZLE_HEIGHT_MM = 128.0

# Puzzle corner positions
POSITIONS = ('TL', 'TR', 'BR', 'BL')


def extract_border_info(
    corners_list: list[dict],
    classifications: list[dict],
    a4_image_width_px: int,
    a4_width_mm: float = 297.0,
) -> list[dict]:
    """
    For each piece, collect all outside segments, their lengths in mm,
    and the piece centroid in pixels.

    Returns a list of dicts, one per piece:
        {
            'piece_idx':        int,
            'type':             'corner' | 'edge' | 'inner',
            'centroid_px':      (cx, cy) in pixels,
            'px_per_mm':        float,
            'outside_segments': [{'seg_id': int, 'length_mm': float, 'p1': (x,y), 'p2': (x,y)}, ...],
        }
    """
    px_per_mm = a4_image_width_px / a4_width_mm

    pieces = []
    seg_id = 0
    for idx, (info, cls) in enumerate(zip(corners_list, classifications)):
        outside_segs = []
        for seg in info['all_segments']:
            if seg.get('is_outside', False):
                outside_segs.append({
                    'seg_id':    seg_id,
                    'piece_idx': idx,
                    'length_mm': seg['length'] / px_per_mm,
                    'p1':        seg['p1'],
                    'p2':        seg['p2'],
                })
                seg_id += 1
        pieces.append({
            'piece_idx':  idx,
            'type':       cls['type'],
            'centroid_px': info['centroid'],   # (cx, cy) in pixels, or None
            'px_per_mm':  px_per_mm,
            'outside_segments': outside_segs,
        })
    return pieces


def print_border_info(pieces: list[dict]) -> None:
    """Pretty-print the border info for validation."""
    print("\n=== Border Segment Info ===")
    for p in pieces:
        segs = p['outside_segments']
        total = sum(s['length_mm'] for s in segs)
        print(f"\nPiece {p['piece_idx']}  [{p['type']}]")
        print(f"  Outside segments: {len(segs)}")
        for s in segs:
            print(f"    seg {s['seg_id']}: {s['length_mm']:.1f} mm")
        print(f"  Total outside length: {total:.1f} mm")

    all_outside = [s for p in pieces for s in p['outside_segments']]
    grand_total  = sum(s['length_mm'] for s in all_outside)
    expected     = 2 * (PUZZLE_WIDTH_MM + PUZZLE_HEIGHT_MM)
    print(f"\nGrand total outside length : {grand_total:.1f} mm")
    print(f"Expected perimeter (190x128): {expected:.1f} mm")


# ─────────────────────────────────────────────────────────────────
#  Corner-first combination finder
# ─────────────────────────────────────────────────────────────────

def _seg_pairs(piece: dict) -> list[tuple]:
    """
    Return all pairs of outside segments from a piece to try as the two
    corner-forming segments. Invalid pairs (wrong direction, off-canvas) are
    rejected later during geometric placement in piece_placer.place_piece.
    """
    segs = piece['outside_segments']
    if len(segs) < 2:
        return []
    from itertools import combinations as _combs
    return list(_combs(segs, 2))


# Expected puzzle corner coordinates and inward horiz direction per position
_PUZZLE_CORNERS_MM = {
    'TL': np.array([0.0,            0.0            ]),
    'TR': np.array([PUZZLE_WIDTH_MM, 0.0            ]),
    'BR': np.array([PUZZLE_WIDTH_MM, PUZZLE_HEIGHT_MM]),
    'BL': np.array([0.0,            PUZZLE_HEIGHT_MM]),
}
_EXPECTED_HORIZ = {
    'TL': np.array([ 1.0,  0.0]),
    'TR': np.array([-1.0,  0.0]),
    'BR': np.array([-1.0,  0.0]),
    'BL': np.array([ 1.0,  0.0]),
}


def _centroid_in_canvas(piece: dict, seg_horiz: dict, seg_vert: dict, pos: str) -> bool:
    """
    Transform the piece centroid using the exact same logic as place_piece
    and check it lands inside the puzzle canvas.
    """
    from puzzle_solver.piece_placer import transform_point

    centroid_px = piece.get('centroid_px')
    if centroid_px is None:
        return True  # no centroid info — don't reject

    placed = transform_point(centroid_px, seg_horiz, seg_vert, pos, piece['px_per_mm'])
    if placed is None:
        return True  # degenerate segment — don't reject

    margin = 10.0
    return (
        -margin < placed[0] < PUZZLE_WIDTH_MM  + margin and
        -margin < placed[1] < PUZZLE_HEIGHT_MM + margin
    )


def _check_corner_assignment(
    assignment: dict,  # position -> (piece, seg_horiz, seg_vert)
    tolerance: float,
) -> bool:
    """
    Check if the four corner pieces placed at TL/TR/BR/BL produce:
      1. Side lengths matching the puzzle dimensions
      2. Each piece centroid landing inside the puzzle canvas
    """
    top    = assignment['TL'][1]['length_mm'] + assignment['TR'][1]['length_mm']
    bottom = assignment['BL'][1]['length_mm'] + assignment['BR'][1]['length_mm']
    left   = assignment['TL'][2]['length_mm'] + assignment['BL'][2]['length_mm']
    right  = assignment['TR'][2]['length_mm'] + assignment['BR'][2]['length_mm']

    if not (abs(top    - PUZZLE_WIDTH_MM)  <= tolerance and
            abs(bottom - PUZZLE_WIDTH_MM)  <= tolerance and
            abs(left   - PUZZLE_HEIGHT_MM) <= tolerance and
            abs(right  - PUZZLE_HEIGHT_MM) <= tolerance):
        return False

    # Centroid check: piece body must land inside the canvas
    for pos in POSITIONS:
        piece, seg_horiz, seg_vert = assignment[pos]
        if not _centroid_in_canvas(piece, seg_horiz, seg_vert, pos):
            return False

    return True


def _try_corner_pieces(corner_pieces: list[dict], tolerance: float) -> list[dict]:
    """
    Try all permutations of the given corner pieces to the 4 puzzle positions,
    and all orientations (which outside seg is horizontal vs vertical).

    Returns a list of valid assignment dicts.
    """
    valid = []

    for perm in permutations(corner_pieces):
        # perm[0]=TL, perm[1]=TR, perm[2]=BR, perm[3]=BL
        piece_pairs = [_seg_pairs(p) for p in perm]

        # Skip if any piece has no valid pairs
        if any(len(pp) == 0 for pp in piece_pairs):
            continue

        for pair_combo in product(*piece_pairs):
            # pair_combo[i] = (segA, segB) for piece at position i
            # Try both orientations per piece: (horiz, vert) or (vert, horiz)
            for orientations in product([0, 1], repeat=4):
                assignment = {}
                for i, pos in enumerate(POSITIONS):
                    seg_a, seg_b = pair_combo[i]
                    if orientations[i] == 0:
                        seg_horiz, seg_vert = seg_a, seg_b
                    else:
                        seg_horiz, seg_vert = seg_b, seg_a
                    assignment[pos] = (perm[i], seg_horiz, seg_vert)

                if _check_corner_assignment(assignment, tolerance):
                    valid.append({
                        pos: {
                            'piece_idx': assignment[pos][0]['piece_idx'],
                            'seg_horiz': assignment[pos][1],
                            'seg_vert':  assignment[pos][2],
                        }
                        for pos in POSITIONS
                    })
    return valid


def find_corner_combinations(
    pieces: list[dict],
    tolerance: float = 15.0,
) -> tuple[list[dict], int]:
    """
    Find valid corner assignments using a corner-first strategy with fallback.

    Tries in order:
      1. All 4 classified corner pieces
      2. Any 4 pieces (if < 4 corners found or step 1 yields nothing) — fallback

    Returns (list of valid assignments, num_corners_used).
    Each assignment is a dict keyed by position ('TL','TR','BR','BL'):
        {
            'piece_idx': int,
            'seg_horiz': seg dict (contributes to top/bottom),
            'seg_vert':  seg dict (contributes to left/right),
        }
    """
    corner_pieces = [p for p in pieces if p['type'] == 'corner']

    print(f"\n  Classified corner pieces: {[p['piece_idx'] for p in corner_pieces]}")

    # Step 1: try with all 4 classified corners
    if len(corner_pieces) >= 4:
        print("  Trying with all 4 classified corner pieces...")
        results = _try_corner_pieces(corner_pieces[:4], tolerance)
        if results:
            return results, 4

    # Fallback: treat all pieces as potential corners
    print("  Falling back: trying all pieces as potential corners...")
    results = _try_corner_pieces(pieces, tolerance)
    return results, 0


def print_corner_combinations(results: list[dict], num_corners_used: int) -> None:
    """Pretty-print corner combination results."""
    label = f"{num_corners_used} classified corners" if num_corners_used else "all pieces (fallback)"
    print(f"\n=== Corner-first combinations ({label}): {len(results)} found ===")

    for i, assignment in enumerate(results):
        print(f"\n--- Combination {i} ---")
        for pos in POSITIONS:
            a = assignment[pos]
            print(f"  {pos}: piece {a['piece_idx']}"
                  f"  horiz={a['seg_horiz']['length_mm']:.1f}mm"
                  f"  vert={a['seg_vert']['length_mm']:.1f}mm")
        top    = assignment['TL']['seg_horiz']['length_mm'] + assignment['TR']['seg_horiz']['length_mm']
        bottom = assignment['BL']['seg_horiz']['length_mm'] + assignment['BR']['seg_horiz']['length_mm']
        left   = assignment['TL']['seg_vert']['length_mm']  + assignment['BL']['seg_vert']['length_mm']
        right  = assignment['TR']['seg_vert']['length_mm']  + assignment['BR']['seg_vert']['length_mm']
        print(f"  → top={top:.1f}  bottom={bottom:.1f}  left={left:.1f}  right={right:.1f}")
