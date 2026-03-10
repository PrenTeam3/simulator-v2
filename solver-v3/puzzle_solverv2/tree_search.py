"""Step 6 — Tree-based combination search.

Two possible starting positions, each traversing clockwise:

  TL start:  TL -[top]-> TR -[right]-> BR -[bottom]-> BL -[left]-> TL
  BL start:  BL -[bottom]-> BR -[right]-> TR -[top]-> TL -[left]-> BL

For each side:
  - Edge pieces extend the accumulated length while there is room
  - A corner piece closes the side when accumulated + forward_seg ≈ target
  - The corner's turn_seg initialises the next side

The left side is always the closure: start_vert + edges + end_vert ≈ 128mm.

Entry point: solve()
"""

from __future__ import annotations

from puzzle_solverv2.constraints import PlacedPiece, check_c5_centroid_inside
from puzzle_solverv2.border_info import PieceBorderInfo
from puzzle_solverv2.variants import PieceVariants
from puzzle_solverv2.frame import PuzzleFrame, PUZZLE_WIDTH_MM, PUZZLE_HEIGHT_MM


# ─────────────────────────────────────────────
#  Start configurations
# ─────────────────────────────────────────────

# Each config defines:
#   position   — frame corner the starting piece occupies ('TL' or 'BL')
#   start_side — which side the starting piece's horiz_seg contributes to
#   sides      — list of (side_name, target_mm, end_position, forward_is_horiz)
#                describing the 3 sides to fill before the left closure
#
# forward_is_horiz: True  → closing corner's horiz_seg is the "forward" (closing) edge
#                   False → closing corner's vert_seg  is the "forward" (closing) edge

_START_CONFIGS = [
    {
        'position':   'TL',
        'start_side': 'top',
        'sides': [
            ('top',    PUZZLE_WIDTH_MM,  'TR', True),   # TR.horiz closes top,    TR.vert  opens right
            ('right',  PUZZLE_HEIGHT_MM, 'BR', False),  # BR.vert  closes right,  BR.horiz opens bottom
            ('bottom', PUZZLE_WIDTH_MM,  'BL', True),   # BL.horiz closes bottom, BL.vert  opens left
        ],
    },
    {
        'position':   'BL',
        'start_side': 'bottom',
        'sides': [
            ('bottom', PUZZLE_WIDTH_MM,  'BR', True),   # BR.horiz closes bottom, BR.vert  opens right
            ('right',  PUZZLE_HEIGHT_MM, 'TR', False),  # TR.vert  closes right,  TR.horiz opens top
            ('top',    PUZZLE_WIDTH_MM,  'TL', True),   # TL.horiz closes top,    TL.vert  opens left
        ],
    },
]

MAX_SOLUTIONS = 10


# ─────────────────────────────────────────────
#  Debug helpers
# ─────────────────────────────────────────────

def _d(debug: bool, indent: int, msg: str) -> None:
    if debug:
        print(f"  {'  ' * indent}{msg}")


def _side_summary(placed: list[PlacedPiece]) -> str:
    """One-line summary of which segments are assigned to each side."""
    sides:  dict[str, list[str]] = {'top': [], 'right': [], 'bottom': [], 'left': []}
    totals: dict[str, float]     = {'top': 0.0, 'right': 0.0, 'bottom': 0.0, 'left': 0.0}

    _CORNER_SIDES = {
        'TL': ('top',    'left'),
        'TR': ('top',    'right'),
        'BR': ('bottom', 'right'),
        'BL': ('bottom', 'left'),
    }

    for p in placed:
        if p.variant.type == 'corner' and p.position:
            h_side, v_side = _CORNER_SIDES[p.position]
            sides[h_side].append(f"{p.horiz_seg['seg_id']}({p.horiz_seg['length_mm']:.0f}mm)")
            sides[v_side].append(f"{p.vert_seg['seg_id']}({p.vert_seg['length_mm']:.0f}mm)")
            totals[h_side] += p.horiz_seg['length_mm']
            totals[v_side]  += p.vert_seg['length_mm']
        elif p.variant.type == 'edge' and p.side in sides:
            seg = p.variant.edges[0]
            sides[p.side].append(f"{seg['seg_id']}({seg['length_mm']:.0f}mm)")
            totals[p.side] += seg['length_mm']

    targets = {'top': PUZZLE_WIDTH_MM, 'right': PUZZLE_HEIGHT_MM,
               'bottom': PUZZLE_WIDTH_MM, 'left': PUZZLE_HEIGHT_MM}
    parts = []
    for side in ('top', 'right', 'bottom', 'left'):
        segs_str = ' + '.join(sides[side]) if sides[side] else '—'
        parts.append(f"{side}:[{segs_str}]={totals[side]:.0f}/{targets[side]:.0f}mm")
    return '  '.join(parts)


# ─────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────

def solve(
    pieces_border:   list[PieceBorderInfo],
    pieces_variants: list[PieceVariants],
    frame:           PuzzleFrame,
    tolerance:       float = 15.0,
    max_solutions:   int   = MAX_SOLUTIONS,
    debug:           bool  = False,
) -> list[list[PlacedPiece]]:
    """
    Run the tree search and return all valid solutions.

    For each piece with a corner variant, try it as starting piece at TL and BL.
    Each starting position has its own clockwise traversal order.

    Args:
        pieces_border:   Step 2 output
        pieces_variants: Step 3 output
        frame:           Step 1 PuzzleFrame
        tolerance:       ±mm per side (default 15mm)
        max_solutions:   stop after this many (default 10)
        debug:           print every decision in the tree search
    """
    total_pieces  = len(pieces_border)
    border_by_idx = {p.piece_idx: p for p in pieces_border}
    solutions: list[list[PlacedPiece]] = []

    _d(debug, 0, f"solve() — {total_pieces} pieces  tolerance={tolerance}mm  max_solutions={max_solutions}")
    _d(debug, 0, f"Frame: {frame.width_mm}x{frame.height_mm}mm")
    for pv in pieces_variants:
        cvs = [v for v in pv.variants if v.type == 'corner']
        evs = [v for v in pv.variants if v.type == 'edge']
        seg_ids = [s['seg_id'] for v in pv.variants for s in v.edges]
        _d(debug, 0, f"  P{pv.piece_idx}: {len(cvs)} corner variant(s)  "
                     f"{len(evs)} edge variant(s)  segs={seg_ids}")

    for pv in pieces_variants:
        if len(solutions) >= max_solutions:
            break

        corner_variants = [v for v in pv.variants if v.type == 'corner']
        if not corner_variants:
            _d(debug, 0, f"\nP{pv.piece_idx}: SKIP — no corner variants")
            continue

        pb = border_by_idx[pv.piece_idx]

        for v in corner_variants:
            if len(solutions) >= max_solutions:
                break

            for config in _START_CONFIGS:
                if len(solutions) >= max_solutions:
                    break

                pos        = config['position']
                start_side = config['start_side']
                sides      = config['sides']

                _d(debug, 0, f"\n{'─'*60}")
                _d(debug, 0, f"Start: P{pv.piece_idx} at {pos}  "
                             f"{seg_h['seg_id']}({seg_h['length_mm']:.1f}mm)→{start_side}  "
                             f"{seg_v['seg_id']}({seg_v['length_mm']:.1f}mm)→left")

                start_piece = PlacedPiece(
                    piece_idx=pv.piece_idx,
                    variant=v,
                    side=start_side,
                    position=pos,
                    horiz_seg=seg_h,
                    vert_seg=seg_v,
                    centroid_px=pb.centroid_px,
                    px_per_mm=frame.px_per_mm,
                )

                ok, reason = check_c5_centroid_inside([start_piece], frame)
                if not ok:
                    _d(debug, 1, f"SKIP — C5 centroid failed: {reason}")
                    continue
                _d(debug, 1, f"C5 OK")

                placed = [start_piece]
                used   = {pv.piece_idx}

                _d(debug, 1, f"State: {_side_summary(placed)}")

                _search(
                    placed, used,
                    side_idx=0,
                    current_length=seg_h['length_mm'],
                    start_vert_length=seg_v['length_mm'],
                    sides=sides,
                    border_by_idx=border_by_idx,
                    pieces_variants=pieces_variants,
                    frame=frame,
                    total_pieces=total_pieces,
                    solutions=solutions,
                    max_solutions=max_solutions,
                    tolerance=tolerance,
                    debug=debug,
                    depth=1,
                )

    _d(debug, 0, f"\nSearch complete — {len(solutions)} solution(s) found")
    return solutions


# ─────────────────────────────────────────────
#  Recursive DFS
# ─────────────────────────────────────────────

def _search(
    placed:            list[PlacedPiece],
    used:              set[int],
    side_idx:          int,             # 0, 1, 2 = regular sides; 3 = left closure
    current_length:    float,
    start_vert_length: float,           # starting piece's vert_seg (left closure)
    sides:             list[tuple],     # side sequence for this start config
    border_by_idx:     dict[int, PieceBorderInfo],
    pieces_variants:   list[PieceVariants],
    frame:             PuzzleFrame,
    total_pieces:      int,
    solutions:         list,
    max_solutions:     int,
    tolerance:         float,
    debug:             bool,
    depth:             int,
) -> None:

    if len(solutions) >= max_solutions:
        return

    # ── Left side closure ───────────────────────────────────────────────────
    if side_idx == 3:
        _close_left(
            placed, used, current_length, start_vert_length,
            sides, border_by_idx, pieces_variants, frame,
            total_pieces, solutions, max_solutions, tolerance, debug, depth,
        )
        return

    # ── Regular side ────────────────────────────────────────────────────────
    side_name, target, end_pos, forward_is_horiz = sides[side_idx]
    next_side = sides[side_idx + 1][0] if side_idx + 1 < 3 else 'left'
    unused    = [pv.piece_idx for pv in pieces_variants if pv.piece_idx not in used]

    _d(debug, depth, f"[{side_name.upper()}] current={current_length:.1f}mm  "
                     f"target={target:.1f}mm  unused=P{unused}")
    _d(debug, depth, f"  State: {_side_summary(placed)}")

    found_any = False

    for pv in pieces_variants:
        if pv.piece_idx in used:
            continue

        pb = border_by_idx[pv.piece_idx]

        for v in pv.variants:

            # ── Edge piece — extends current side ──────────────────────────
            if v.type == 'edge':
                seg     = v.edges[0]
                new_len = current_length + seg['length_mm']

                if new_len >= target - tolerance:
                    _d(debug, depth + 1,
                       f"P{pv.piece_idx} edge  {seg['seg_id']}({seg['length_mm']:.1f}mm)"
                       f"→{side_name}  total={new_len:.1f}mm — PRUNE (limit={target-tolerance:.1f}mm)")
                    continue

                _d(debug, depth + 1,
                   f"P{pv.piece_idx} edge  {seg['seg_id']}({seg['length_mm']:.1f}mm)"
                   f"→{side_name}  total={new_len:.1f}mm ✓")
                found_any = True

                pp = PlacedPiece(
                    piece_idx=pv.piece_idx, variant=v,
                    side=side_name, position=None,
                    horiz_seg=None, vert_seg=None,
                    centroid_px=pb.centroid_px, px_per_mm=frame.px_per_mm,
                )
                placed.append(pp)
                used.add(pv.piece_idx)
                _search(
                    placed, used, side_idx, new_len, start_vert_length,
                    sides, border_by_idx, pieces_variants, frame,
                    total_pieces, solutions, max_solutions, tolerance, debug, depth + 1,
                )
                placed.pop()
                used.remove(pv.piece_idx)

            # ── Corner piece — closes current side, opens next ─────────────
            elif v.type == 'corner':
                if len(v.edges) < 2:
                    continue

                for fwd, turn in [(v.edges[0], v.edges[1]), (v.edges[1], v.edges[0])]:
                    new_len = current_length + fwd['length_mm']
                    diff    = abs(new_len - target)

                    if diff > tolerance:
                        _d(debug, depth + 1,
                           f"P{pv.piece_idx} corner  {fwd['seg_id']}({fwd['length_mm']:.1f}mm)"
                           f"→{side_name} + {turn['seg_id']}({turn['length_mm']:.1f}mm)"
                           f"→{next_side}  as {end_pos}  total={new_len:.1f}mm"
                           f"  diff={diff:.1f}mm > tol — PRUNE")
                        continue

                    horiz_seg = fwd  if forward_is_horiz else turn
                    vert_seg  = turn if forward_is_horiz else fwd

                    _d(debug, depth + 1,
                       f"P{pv.piece_idx} corner  {fwd['seg_id']}({fwd['length_mm']:.1f}mm)"
                       f"→{side_name} + {turn['seg_id']}({turn['length_mm']:.1f}mm)"
                       f"→{next_side}  as {end_pos}  total={new_len:.1f}mm (diff={diff:.1f}mm ✓)")

                    pp = PlacedPiece(
                        piece_idx=pv.piece_idx, variant=v,
                        side=side_name, position=end_pos,
                        horiz_seg=horiz_seg, vert_seg=vert_seg,
                        centroid_px=pb.centroid_px, px_per_mm=frame.px_per_mm,
                    )

                    ok, reason = check_c5_centroid_inside([pp], frame)
                    if not ok:
                        _d(debug, depth + 2, f"SKIP — C5 centroid failed: {reason}")
                        continue
                    _d(debug, depth + 2,
                       f"C5 OK → P{pv.piece_idx} placed at {end_pos}, entering {next_side.upper()}")

                    found_any = True
                    placed.append(pp)
                    used.add(pv.piece_idx)
                    _search(
                        placed, used, side_idx + 1, turn['length_mm'], start_vert_length,
                        sides, border_by_idx, pieces_variants, frame,
                        total_pieces, solutions, max_solutions, tolerance, debug, depth + 1,
                    )
                    placed.pop()
                    used.remove(pv.piece_idx)

    if not found_any:
        _d(debug, depth, f"[{side_name.upper()}] dead end — no candidates fit")


# ─────────────────────────────────────────────
#  Left side closure
# ─────────────────────────────────────────────

def _close_left(
    placed:            list[PlacedPiece],
    used:              set[int],
    current_length:    float,           # vert_seg of last corner + any left edges so far
    start_vert_length: float,           # vert_seg of starting piece
    sides:             list[tuple],
    border_by_idx:     dict[int, PieceBorderInfo],
    pieces_variants:   list[PieceVariants],
    frame:             PuzzleFrame,
    total_pieces:      int,
    solutions:         list,
    max_solutions:     int,
    tolerance:         float,
    debug:             bool,
    depth:             int,
) -> None:

    if len(solutions) >= max_solutions:
        return

    total_left = current_length + start_vert_length
    remaining  = frame.height_mm - total_left
    unused     = [pv.piece_idx for pv in pieces_variants if pv.piece_idx not in used]

    _d(debug, depth,
       f"[LEFT] current={current_length:.1f}mm + start_vert={start_vert_length:.1f}mm"
       f" = {total_left:.1f}mm / {frame.height_mm}mm  remaining={remaining:.1f}mm"
       f"  unused=P{unused}  placed={len(placed)}/{total_pieces}")
    _d(debug, depth, f"  State: {_side_summary(placed)}")

    # Closure check
    if abs(total_left - frame.height_mm) <= tolerance:
        if len(placed) == total_pieces:
            _d(debug, depth, f"✅ SOLUTION FOUND (#{len(solutions)})")
            solutions.append(list(placed))
        else:
            _d(debug, depth,
               f"Left closed but {len(placed)}/{total_pieces} pieces placed — REJECT"
               f" (unused: P{unused})")
        return

    # Overshot
    if total_left > frame.height_mm + tolerance:
        _d(debug, depth, f"PRUNE — {total_left:.1f}mm > {frame.height_mm + tolerance:.1f}mm")
        return

    # Need more edge pieces
    found_any = False
    for pv in pieces_variants:
        if pv.piece_idx in used:
            continue
        for v in pv.variants:
            if v.type != 'edge':
                continue
            seg       = v.edges[0]
            new_len   = current_length + seg['length_mm']
            new_total = new_len + start_vert_length

            if new_total > frame.height_mm + tolerance:
                _d(debug, depth + 1,
                   f"P{pv.piece_idx} edge  {seg['seg_id']}({seg['length_mm']:.1f}mm)→left"
                   f"  total={new_total:.1f}mm — PRUNE (overshoot)")
                continue

            _d(debug, depth + 1,
               f"P{pv.piece_idx} edge  {seg['seg_id']}({seg['length_mm']:.1f}mm)→left"
               f"  total={new_total:.1f}mm ✓")
            found_any = True

            pb = border_by_idx[pv.piece_idx]
            pp = PlacedPiece(
                piece_idx=pv.piece_idx, variant=v,
                side='left', position=None,
                horiz_seg=None, vert_seg=None,
                centroid_px=pb.centroid_px, px_per_mm=frame.px_per_mm,
            )
            placed.append(pp)
            used.add(pv.piece_idx)
            _close_left(
                placed, used, new_len, start_vert_length,
                sides, border_by_idx, pieces_variants, frame,
                total_pieces, solutions, max_solutions, tolerance, debug, depth + 1,
            )
            placed.pop()
            used.remove(pv.piece_idx)

    if not found_any:
        _d(debug, depth, f"[LEFT] dead end — no edge candidates and closure not reached")


# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────

def log_solutions(solutions: list[list[PlacedPiece]]) -> None:
    if not solutions:
        print("  No solutions found.")
        return

    print(f"  {len(solutions)} solution(s) found:\n")
    for sol_idx, sol in enumerate(solutions):
        print(f"  ── Solution {sol_idx} ──────────────────────")
        top = bottom = right = left = 0.0
        for p in sol:
            if p.variant.type == 'corner':
                if p.position in ('TL', 'TR'): top    += p.horiz_seg['length_mm']
                if p.position in ('BR', 'BL'): bottom += p.horiz_seg['length_mm']
                if p.position in ('TR', 'BR'): right  += p.vert_seg['length_mm']
                if p.position in ('BL', 'TL'): left   += p.vert_seg['length_mm']
            else:
                if p.side == 'top':     top    += p.variant.total_length_mm
                elif p.side == 'right': right  += p.variant.total_length_mm
                elif p.side == 'bottom':bottom += p.variant.total_length_mm
                elif p.side == 'left':  left   += p.variant.total_length_mm

        for pp in sol:
            if pp.variant.type == 'corner':
                print(f"    P{pp.piece_idx} [{pp.position}]  "
                      f"horiz={pp.horiz_seg['seg_id']} ({pp.horiz_seg['length_mm']:.1f}mm)  "
                      f"vert={pp.vert_seg['seg_id']} ({pp.vert_seg['length_mm']:.1f}mm)")
            else:
                seg = pp.variant.edges[0]
                print(f"    P{pp.piece_idx} [{pp.side:>6}]  "
                      f"seg={seg['seg_id']} ({seg['length_mm']:.1f}mm)")

        print(f"    → top={top:.1f}mm  right={right:.1f}mm  "
              f"bottom={bottom:.1f}mm  left={left:.1f}mm  "
              f"(target: {PUZZLE_WIDTH_MM}×{PUZZLE_HEIGHT_MM}mm)")
