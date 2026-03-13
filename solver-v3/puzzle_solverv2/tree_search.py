"""Step 6 — public placement visualisation entry points."""

from __future__ import annotations

from pathlib import Path

import cv2

from puzzle_solverv2.frame import PuzzleFrame
from puzzle_solverv2.border_info import PieceBorderInfo
from puzzle_solverv2.variants import PieceVariants
from puzzle_solverv2.constraints import PlacedPiece
from puzzle_solverv2._placement_types import (
    _SECOND_CONFIGS,
    _SIDE_TOLERANCE,
    _CORNER_TURN,
    empty_occupancy,
    occ_add,
)
from puzzle_solverv2._placement_geometry import (
    _place_contour,
    _build_candidates,
)
from puzzle_solverv2._placement_canvas import (
    COLORS,
    _make_canvas,
    _draw_piece,
    _draw_segments,
    _mark_corner,
)
from puzzle_solverv2._placement_drawing import (
    _draw_legend,
    _side_lines,
    _candidate_lines,
)
from puzzle_solverv2._search_state import _SearchState, _SearchConfig
from puzzle_solverv2._search_step import _search_step


# ─────────────────────────────────────────────
#  Shared iteration helper
# ─────────────────────────────────────────────

def _corner_scenarios(pieces_variants: list, corners_list: list):
    """Yield one entry per corner-piece × variant × start-position combination.

    Yields: (pv, v, position, seg_h, seg_v, side_name, target, end_pos, fwd_is_horiz, contour)
    """
    for pv in pieces_variants:
        corner_variants = [v for v in pv.variants if v.type == 'corner']
        if not corner_variants:
            continue
        contour = corners_list[pv.piece_idx]['contour_flat']
        for v in corner_variants:
            for position in ('TL', 'BL'):
                seg_h = v.edges[0] if position == 'TL' else v.edges[1]
                seg_v = v.edges[1] if position == 'TL' else v.edges[0]
                side_name, target, end_pos, fwd_is_horiz = _SECOND_CONFIGS[position]
                yield pv, v, position, seg_h, seg_v, side_name, target, end_pos, fwd_is_horiz, contour


# ─────────────────────────────────────────────
#  Public visualisation functions
# ─────────────────────────────────────────────

def visualize_start_placements(
    pieces_border:   list[PieceBorderInfo],
    pieces_variants: list[PieceVariants],
    corners_list:    list[dict],
    frame:           PuzzleFrame,
    output_dir:      Path,
) -> None:
    """For each piece with a corner variant, save start.png at TL and BL."""
    border_by_idx = {p.piece_idx: p for p in pieces_border}

    for pv, v, position, seg_h, seg_v, side_name, target, end_pos, fwd_is_horiz, contour \
            in _corner_scenarios(pieces_variants, corners_list):

        pb    = border_by_idx[pv.piece_idx]
        color = COLORS[pv.piece_idx % len(COLORS)]

        occ_1 = empty_occupancy()
        occ_add(occ_1, side_name, pv.piece_idx, seg_h['seg_id'], seg_h['length_mm'])
        occ_add(occ_1, 'left',    pv.piece_idx, seg_v['seg_id'], seg_v['length_mm'])

        nc = _build_candidates(
            pieces_variants, {pv.piece_idx}, corners_list,
            frame.px_per_mm, side_name, seg_h['length_mm'],
            target, end_pos, fwd_is_horiz, _SIDE_TOLERANCE,
        )

        canvas, ox, oy = _make_canvas()
        placed_mm = _place_contour(contour, seg_h, seg_v, position, frame.px_per_mm)
        _draw_piece(canvas, placed_mm, ox, oy, color, f"P{pv.piece_idx}@{position}")
        _draw_segments(canvas, seg_h, seg_v, position, frame.px_per_mm, ox, oy)
        _mark_corner(canvas, position, ox, oy, label=f"expected {position}")

        centroid_info = f"centroid={pb.centroid_px}" if pb.centroid_px else ""
        title = (f"P{pv.piece_idx} at {position}  "
                 f"H:{seg_h['seg_id']}({seg_h['length_mm']:.1f}mm)  "
                 f"V:{seg_v['seg_id']}({seg_v['length_mm']:.1f}mm)  "
                 f"{centroid_info}")
        cv2.putText(canvas, title, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        _draw_legend(canvas, ox, oy, _side_lines(occ_1) + [''] + _candidate_lines(nc))

        folder = output_dir / "placements" / f"P{pv.piece_idx}" / position
        folder.mkdir(parents=True, exist_ok=True)
        fname = folder / "start.png"
        cv2.imwrite(str(fname), canvas)
        print(f"  [OUT] {fname.relative_to(output_dir)}")


def visualize_second_placements(
    pieces_variants: list[PieceVariants],
    corners_list:    list[dict],
    frame:           PuzzleFrame,
    output_dir:      Path,
    pieces_border:   list[PieceBorderInfo] | None = None,
    tolerance:       float = _SIDE_TOLERANCE,
    max_depth:       int   = 3,
    mode:            str   = 'all',
) -> list[tuple[str, list]]:
    """For every first-piece scenario, place subsequent pieces up to max_depth.

    mode:
      'console_only' — no images written, only valid-branch summary printed
      'valid_only'   — images written only for fully-valid branches
      'all'          — images written for every candidate, valid or not

    Returns a list of (branch_string, placed_pieces) tuples for Step 7.
    """
    centroid_by_idx: dict = (
        {pb.piece_idx: pb.centroid_px for pb in pieces_border}
        if pieces_border else {}
    )

    cfg = _SearchConfig(
        pieces_variants=pieces_variants,
        corners_list=corners_list,
        frame=frame,
        output_dir=output_dir,
        tolerance=tolerance,
        max_depth=max_depth,
        mode=mode,
        centroid_by_idx=centroid_by_idx,
    )

    results: list[tuple[str, list]] = []

    for pv, v, position, seg_h, seg_v, side_name, target, end_pos, fwd_is_horiz, contour \
            in _corner_scenarios(pieces_variants, corners_list):

        color_1  = COLORS[pv.piece_idx % len(COLORS)]
        folder   = output_dir / 'placements' / f"P{pv.piece_idx}" / position
        turn_side = _CORNER_TURN.get((side_name, end_pos), ('right',))[0]

        occ_1 = empty_occupancy()
        occ_add(occ_1, side_name, pv.piece_idx, seg_h['seg_id'], seg_h['length_mm'])
        occ_add(occ_1, 'left',    pv.piece_idx, seg_v['seg_id'], seg_v['length_mm'])

        if mode != 'console_only':
            base_1, ox, oy = _make_canvas()
            placed_mm_1 = _place_contour(contour, seg_h, seg_v, position, frame.px_per_mm)
            _draw_piece(base_1, placed_mm_1, ox, oy, color_1, f"P{pv.piece_idx}@{position}")
            _draw_segments(base_1, seg_h, seg_v, position, frame.px_per_mm, ox, oy)
            _mark_corner(base_1, position, ox, oy)
        else:
            base_1, ox, oy = None, 0, 0

        placed_1 = [PlacedPiece(
            piece_idx   = pv.piece_idx,
            variant     = v,
            side        = side_name,
            position    = position,
            horiz_seg   = seg_h,
            vert_seg    = seg_v,
            centroid_px = centroid_by_idx.get(pv.piece_idx),
            px_per_mm   = frame.px_per_mm,
        )]

        _search_step(
            _SearchState(
                used=          {pv.piece_idx},
                side=          side_name,
                offset_mm=     seg_h['length_mm'],
                target_mm=     target,
                end_pos=       end_pos,
                fwd_is_horiz=  fwd_is_horiz,
                start_from_end=False,
                turn_side=     turn_side,
                occ=           occ_1,
                base_canvas=   base_1,
                ox=ox, oy=oy,
                folder=        folder,
                prev_color=    color_1,
                folder_prefix= 'P',
            ),
            cfg, depth=2,
            path=[f"P{pv.piece_idx}@{position}"],
            placed=placed_1,
            results=results,
            pending_images=[],
        )

    print(f"\n  {'─'*56}")
    print(f"  VALID BRANCHES (depth={max_depth})  —  {len(results)} found")
    print(f"  {'─'*56}")
    for branch_str, _ in results:
        print(f"  {branch_str}")
    if not results:
        print("  (none)")
    print()

    return results
