"""Core recursive search step and supporting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from puzzle_solverv2.constraints import PlacedPiece
from puzzle_solverv2._placement_types import (
    Candidate,
    _CORNER_TURN,
    _SIDE_TARGET,
    empty_occupancy,
    occ_add_candidate,
)
from puzzle_solverv2._placement_geometry import _build_candidates
from puzzle_solverv2._placement_canvas import (
    COLORS,
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
from puzzle_solverv2._search_state import _SearchState, _SearchConfig


# ─────────────────────────────────────────────
#  Drawing helper
# ─────────────────────────────────────────────

def _draw_candidate(
    canvas:         np.ndarray,
    ox:             int,
    oy:             int,
    cand:           Candidate,
    end_pos:        str,
    frame:          Any,       # PuzzleFrame
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


# ─────────────────────────────────────────────
#  Next-state builder
# ─────────────────────────────────────────────

def _build_next_state(
    state:        _SearchState,
    cfg:          _SearchConfig,
    cand:         Candidate,
    occ_next:     dict,
    piece_folder: Path,
    color:        tuple,
    next_base:    Any,
) -> _SearchState | None:
    """Build the _SearchState for the next recursive call.

    Returns None if the corner turn lookup fails (no valid transition).
    """
    if cand.variant.type == 'corner':
        turn_info = _CORNER_TURN.get((state.side, state.end_pos))
        if turn_info is None:
            return None
        _, next_side, next_end, next_horiz, next_sfe = turn_info
        next_turn_s = _CORNER_TURN.get((next_side, next_end), ('',))[0]
        if cfg.mode != 'console_only':
            _draw_segments(next_base, cand.seg_h, cand.seg_v,
                           state.end_pos, cfg.frame.px_per_mm, state.ox, state.oy)
            _mark_corner(next_base, state.end_pos, state.ox, state.oy)
        return _SearchState(
            used=state.used | {cand.pv.piece_idx},
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
        return _SearchState(
            used=state.used | {cand.pv.piece_idx},
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


# ─────────────────────────────────────────────
#  Recursive search step
# ─────────────────────────────────────────────

def _search_step(
    state:          _SearchState,
    cfg:            _SearchConfig,
    depth:          int,
    path:           list,
    placed:         list,   # list[PlacedPiece] accumulated so far
    results:        list,   # list[tuple[str, list[PlacedPiece]]]
    pending_images: list,   # (Path, canvas) pairs accumulated along this branch
) -> None:
    """Place the next piece and recurse.

    mode:
      'console_only' — no images written, only valid-branch summary printed
      'valid_only'   — images written only for fully-valid branches
      'all'          — images written for every candidate, valid or not
    """
    if depth > cfg.max_depth:
        return

    candidates = _build_candidates(
        cfg.pieces_variants, state.used, cfg.corners_list,
        cfg.frame.px_per_mm, state.side, state.offset_mm,
        state.target_mm, state.end_pos, state.fwd_is_horiz,
        cfg.tolerance, state.start_from_end,
    )

    for n, cand in enumerate(candidates, 1):
        is_corner = cand.variant.type == 'corner'

        occ_next = {k: list(v) for k, v in state.occ.items()}
        occ_add_candidate(
            occ_next, cand,
            fwd_side=state.side,
            turn_side=state.turn_side if is_corner else None,
            fwd_is_horiz=state.fwd_is_horiz,
        )

        piece_folder = state.folder / f"{state.folder_prefix}{cand.pv.piece_idx}"
        tag          = 'VALID' if cand.valid else 'INVALID'
        fname        = piece_folder / f"{n:03d}_{cand.variant.type}_{tag}.png"

        # ── Draw candidate image (skipped in console_only) ──────────────────
        canvas = None
        if cfg.mode != 'console_only':
            nc: list = []
            if cand.valid and not is_corner and depth < cfg.max_depth:
                nc = _build_candidates(
                    cfg.pieces_variants, state.used | {cand.pv.piece_idx}, cfg.corners_list,
                    cfg.frame.px_per_mm, state.side, state.offset_mm + cand.seg_len,
                    state.target_mm, state.end_pos, state.fwd_is_horiz,
                    cfg.tolerance, state.start_from_end,
                )
            canvas = state.base_canvas.copy()
            _draw_candidate(canvas, state.ox, state.oy, cand,
                            state.end_pos, cfg.frame, state.side,
                            state.offset_mm, state.prev_color, state.target_mm)
            title = (f"[d{depth}] {state.side}({state.offset_mm:.0f}→{state.target_mm:.0f}mm)"
                     f"  +  {cand.label}")
            cv2.putText(canvas, title, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
            _draw_verdict(canvas, cand.valid, cand.reason)
            legend = _side_lines(occ_next)
            if nc:
                legend += [''] + _candidate_lines(nc)
            _draw_legend(canvas, state.ox, state.oy, legend)
            if cfg.mode == 'all':
                piece_folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(fname), canvas)
                print(f"  [OUT] {fname.relative_to(cfg.output_dir)}")

        # ── Path bookkeeping ────────────────────────────────────────────────
        if cand.valid:
            cand_desc = (f"P{cand.pv.piece_idx}@{state.end_pos} corner({cand.seg_len:.0f}mm)"
                         if is_corner else
                         f"P{cand.pv.piece_idx} edge({cand.seg_len:.0f}mm)")
            next_path   = path + [cand_desc]
            next_placed = placed + [PlacedPiece(
                piece_idx   = cand.pv.piece_idx,
                variant     = cand.variant,
                side        = state.side,
                position    = state.end_pos if is_corner else None,
                horiz_seg   = cand.seg_h,
                vert_seg    = cand.seg_v,
                centroid_px = cfg.centroid_by_idx.get(cand.pv.piece_idx),
                px_per_mm   = cfg.frame.px_per_mm,
            )]
            branch_images = (pending_images + [(fname, canvas)]
                             if cfg.mode == 'valid_only' else [])
            if depth == cfg.max_depth:
                results.append((' → '.join(next_path), next_placed))
                if cfg.mode == 'valid_only':
                    for f, img in branch_images:
                        f.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(f), img)
                        print(f"  [OUT] {f.relative_to(cfg.output_dir)}")
        else:
            next_path     = path
            next_placed   = placed
            branch_images = []

        if not cand.valid or depth >= cfg.max_depth:
            continue

        # ── Build base canvas and recurse ───────────────────────────────────
        color     = COLORS[cand.pv.piece_idx % len(COLORS)]
        next_base = None
        if cfg.mode != 'console_only':
            next_base = state.base_canvas.copy()
            _draw_piece(next_base, cand.placed_mm, state.ox, state.oy, color, cand.label)

        next_state = _build_next_state(state, cfg, cand, occ_next, piece_folder, color, next_base)
        if next_state is None:
            continue

        _search_step(next_state, cfg, depth + 1, next_path, next_placed, results, branch_images)
