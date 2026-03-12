"""Placement debug — visual check of TL and BL starting positions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from puzzle_solverv2.frame import PuzzleFrame
from puzzle_solverv2.border_info import PieceBorderInfo
from puzzle_solverv2.variants import PieceVariants
from puzzle_solverv2._placement_types import (
    Candidate,
    _SECOND_CONFIGS,
    _SIDE_TOLERANCE,
    _CORNER_TURN,
    _SIDE_TARGET,
    empty_occupancy,
    occ_add,
    occ_add_candidate,
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
    _draw_verdict,
    _draw_progress_bar,
    _draw_legend,
    _side_lines,
    _candidate_lines,
)


# ─────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────

def _draw_candidate(
    canvas:         np.ndarray,
    ox:             int,
    oy:             int,
    cand:           Candidate,
    end_pos:        str,
    frame:          PuzzleFrame,
    side_name:      str,
    bar_offset:     float,
    bar_color_base: tuple,
    target:         float,
) -> None:
    """Draw a candidate piece + segment markers + progress bar onto canvas."""
    color = COLORS[cand.pv.piece_idx % len(COLORS)]
    _draw_piece(canvas, cand.placed_mm, ox, oy, color, cand.label)
    if cand.seg_h is not None and cand.seg_v is not None:
        _draw_segments(canvas, cand.seg_h, cand.seg_v, end_pos, frame.px_per_mm, ox, oy)
        _mark_corner(canvas, end_pos, ox, oy)
    _draw_progress_bar(canvas, ox, oy, side_name,
                       bar_offset, cand.seg_len,
                       bar_color_base, color, target)


@dataclass
class _SearchState:
    """All context needed to continue the tree search from a given position."""
    used:           set
    side:           str          # current side being filled
    offset_mm:      float        # how far along the side the next piece starts
    target_mm:      float        # full length of the current side
    end_pos:        str          # corner at the far end of the current side
    fwd_is_horiz:   bool         # whether the current side is horizontal
    start_from_end: bool         # measure offset from the far end (bottom-up / right-to-left)
    turn_side:      str          # side that a corner piece's seg_v would cover
    occ:            dict         # occupancy dict so far
    base_canvas:    Any          # np.ndarray accumulated so far
    ox:             int
    oy:             int
    folder:         Path         # parent folder for this depth's output files
    prev_color:     tuple        # color of the last placed piece (for progress bar)
    folder_prefix:  str = 'P'   # 'B_P' when this is the first piece on a new side


# ─────────────────────────────────────────────
#  Recursive search step
# ─────────────────────────────────────────────

def _search_step(
    state:           _SearchState,
    pieces_variants: list,
    corners_list:    list,
    frame:           PuzzleFrame,
    output_dir:      Path,
    tolerance:       float,
    max_depth:       int,
    depth:           int,
    path:            list,
    results:         list,
    mode:            str,
    pending_images:  list,  # (Path, canvas) pairs accumulated along this branch
) -> None:
    """Place the next piece and recurse.

    mode:
      'console_only' — no images written, only valid-branch summary printed
      'valid_only'   — images written only for fully-valid branches (all steps valid)
      'all'          — images written for every candidate, valid or not
    """
    if depth > max_depth:
        return

    candidates = _build_candidates(
        pieces_variants, state.used, corners_list,
        frame.px_per_mm, state.side, state.offset_mm,
        state.target_mm, state.end_pos, state.fwd_is_horiz,
        tolerance, state.start_from_end,
    )

    for n, cand in enumerate(candidates, 1):
        occ_next = {k: list(v) for k, v in state.occ.items()}
        occ_add_candidate(
            occ_next, cand,
            fwd_side=state.side,
            turn_side=state.turn_side if cand.is_corner else None,
            fwd_is_horiz=state.fwd_is_horiz,
        )

        # ── Draw candidate image (skipped in console_only) ────────────────────
        piece_folder = state.folder / f"{state.folder_prefix}{cand.pv.piece_idx}"
        tag          = 'VALID' if cand.valid else 'INVALID'
        fname        = piece_folder / f"{n:03d}_{cand.variant.type}_{tag}.png"

        if mode != 'console_only':
            nc: list = []
            if cand.valid and not cand.is_corner and depth < max_depth:
                nc = _build_candidates(
                    pieces_variants, state.used | {cand.pv.piece_idx}, corners_list,
                    frame.px_per_mm, state.side, state.offset_mm + cand.seg_len,
                    state.target_mm, state.end_pos, state.fwd_is_horiz,
                    tolerance, state.start_from_end,
                )
            canvas = state.base_canvas.copy()
            _draw_candidate(
                canvas, state.ox, state.oy, cand,
                state.end_pos, frame, state.side,
                state.offset_mm, state.prev_color, state.target_mm,
            )
            title = (f"[d{depth}] {state.side}({state.offset_mm:.0f}→{state.target_mm:.0f}mm)"
                     f"  +  {cand.label}")
            cv2.putText(canvas, title, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
            _draw_verdict(canvas, cand.valid, cand.reason)
            legend = _side_lines(occ_next)
            if nc:
                legend += [''] + _candidate_lines(nc)
            _draw_legend(canvas, state.ox, state.oy, legend)

            if mode == 'all':
                piece_folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(fname), canvas)
                print(f"  [OUT] {fname.relative_to(output_dir)}")

        # ── Path bookkeeping ──────────────────────────────────────────────────
        if cand.valid:
            cand_desc = (f"P{cand.pv.piece_idx}@{state.end_pos} corner({cand.seg_len:.0f}mm)"
                         if cand.is_corner else
                         f"P{cand.pv.piece_idx} edge({cand.seg_len:.0f}mm)")
            next_path     = path + [cand_desc]
            branch_images = (pending_images + [(fname, canvas)]
                             if mode == 'valid_only' else [])

            if depth == max_depth:
                results.append(' → '.join(next_path))
                if mode == 'valid_only':
                    for f, img in branch_images:
                        f.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(f), img)
                        print(f"  [OUT] {f.relative_to(output_dir)}")
        else:
            next_path     = path
            branch_images = []

        if not cand.valid or depth >= max_depth:
            continue

        # ── Build base canvas for the next depth ─────────────────────────────
        color     = COLORS[cand.pv.piece_idx % len(COLORS)]
        used_next = state.used | {cand.pv.piece_idx}

        if mode != 'console_only':
            next_base = state.base_canvas.copy()
            _draw_piece(next_base, cand.placed_mm, state.ox, state.oy, color, cand.label)
        else:
            next_base = None

        if cand.is_corner:
            turn_info = _CORNER_TURN.get((state.side, state.end_pos))
            if turn_info is None:
                continue
            _, next_side, next_end, next_horiz, next_sfe = turn_info
            next_turn_s = _CORNER_TURN.get((next_side, next_end), ('',))[0]

            if mode != 'console_only':
                _draw_segments(next_base, cand.seg_h, cand.seg_v,
                               state.end_pos, frame.px_per_mm, state.ox, state.oy)
                _mark_corner(next_base, state.end_pos, state.ox, state.oy)

            next_state = _SearchState(
                used=used_next,
                side=next_side,
                offset_mm=(cand.seg_v if state.fwd_is_horiz else cand.seg_h)['length_mm'],
                target_mm=_SIDE_TARGET[next_side],
                end_pos=next_end,
                fwd_is_horiz=next_horiz,
                start_from_end=next_sfe,
                turn_side=next_turn_s,
                occ={k: list(v) for k, v in occ_next.items()},
                base_canvas=next_base,
                ox=state.ox, oy=state.oy,
                folder=piece_folder,
                prev_color=color,
                folder_prefix='B_P',
            )
        else:
            next_state = _SearchState(
                used=used_next,
                side=state.side,
                offset_mm=state.offset_mm + cand.seg_len,
                target_mm=state.target_mm,
                end_pos=state.end_pos,
                fwd_is_horiz=state.fwd_is_horiz,
                start_from_end=state.start_from_end,
                turn_side=state.turn_side,
                occ={k: list(v) for k, v in occ_next.items()},
                base_canvas=next_base,
                ox=state.ox, oy=state.oy,
                folder=piece_folder,
                prev_color=color,
                folder_prefix='P',
            )

        _search_step(next_state, pieces_variants, corners_list,
                     frame, output_dir, tolerance, max_depth, depth + 1,
                     next_path, results, mode, branch_images)


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

    for pv in pieces_variants:
        corner_variants = [v for v in pv.variants if v.type == 'corner']
        if not corner_variants:
            continue

        pb          = border_by_idx[pv.piece_idx]
        contour_raw = corners_list[pv.piece_idx]['contour_flat']
        color       = COLORS[pv.piece_idx % len(COLORS)]

        for v in corner_variants:
            for position in ('TL', 'BL'):
                seg_h, seg_v = (v.edges[0], v.edges[1]) if position == 'TL' else (v.edges[1], v.edges[0])
                side_name, target, end_pos, fwd_is_horiz = _SECOND_CONFIGS[position]

                occ_1 = empty_occupancy()
                occ_add(occ_1, side_name, pv.piece_idx, seg_h['seg_id'], seg_h['length_mm'])
                occ_add(occ_1, 'left',    pv.piece_idx, seg_v['seg_id'], seg_v['length_mm'])

                nc = _build_candidates(
                    pieces_variants, {pv.piece_idx}, corners_list,
                    frame.px_per_mm, side_name, seg_h['length_mm'],
                    target, end_pos, fwd_is_horiz, _SIDE_TOLERANCE,
                )

                canvas, ox, oy = _make_canvas()
                placed_mm = _place_contour(contour_raw, seg_h, seg_v, position, frame.px_per_mm)
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
    tolerance:       float = _SIDE_TOLERANCE,
    max_depth:       int   = 3,
    mode:            str   = 'all',
) -> list[str]:
    """For every first-piece scenario, place subsequent pieces up to max_depth.

    mode:
      'console_only' — no images written, only valid-branch summary printed
      'valid_only'   — images written only for fully-valid branches
      'all'          — images written for every candidate, valid or not

    Returns a list of valid branch strings for use by downstream steps.
    """
    results: list[str] = []

    for pv in pieces_variants:
        corner_variants = [v for v in pv.variants if v.type == 'corner']
        if not corner_variants:
            continue

        color_1   = COLORS[pv.piece_idx % len(COLORS)]
        contour_1 = corners_list[pv.piece_idx]['contour_flat']

        for v in corner_variants:
            for position in ('TL', 'BL'):
                seg_h, seg_v = (v.edges[0], v.edges[1]) if position == 'TL' else (v.edges[1], v.edges[0])
                side_name, target, end_pos, fwd_is_horiz = _SECOND_CONFIGS[position]
                offset_1 = seg_h['length_mm']
                folder   = output_dir / 'placements' / f"P{pv.piece_idx}" / position

                occ_1 = empty_occupancy()
                occ_add(occ_1, side_name, pv.piece_idx, seg_h['seg_id'], seg_h['length_mm'])
                occ_add(occ_1, 'left',    pv.piece_idx, seg_v['seg_id'], seg_v['length_mm'])

                if mode != 'console_only':
                    base_1, ox, oy = _make_canvas()
                    placed_1 = _place_contour(contour_1, seg_h, seg_v, position, frame.px_per_mm)
                    _draw_piece(base_1, placed_1, ox, oy, color_1, f"P{pv.piece_idx}@{position}")
                    _draw_segments(base_1, seg_h, seg_v, position, frame.px_per_mm, ox, oy)
                    _mark_corner(base_1, position, ox, oy)
                else:
                    base_1, ox, oy = None, 0, 0

                turn_side = _CORNER_TURN.get((side_name, end_pos), ('right',))[0]

                _search_step(
                    _SearchState(
                        used={pv.piece_idx},
                        side=side_name,
                        offset_mm=offset_1,
                        target_mm=target,
                        end_pos=end_pos,
                        fwd_is_horiz=fwd_is_horiz,
                        start_from_end=False,
                        turn_side=turn_side,
                        occ=occ_1,
                        base_canvas=base_1,
                        ox=ox, oy=oy,
                        folder=folder,
                        prev_color=color_1,
                        folder_prefix='P',
                    ),
                    pieces_variants, corners_list, frame,
                    output_dir, tolerance, max_depth, depth=2,
                    path=[f"P{pv.piece_idx}@{position}"],
                    results=results,
                    mode=mode,
                    pending_images=[],
                )

    print(f"\n  {'─'*56}")
    print(f"  VALID BRANCHES (depth={max_depth})  —  {len(results)} found")
    print(f"  {'─'*56}")
    if results:
        for branch in results:
            print(f"  {branch}")
    else:
        print("  (none)")
    print()

    return results
