"""Edge-based solving module for puzzle solver - alternative approach to matching."""

import cv2
import json
import os
from .segment_finder import SegmentFinder
from .visualizer import EdgeSolverV2Visualizer
from .connection_selector import ConnectionSelector
from .connection_visualizer import ConnectionVisualizer

from ..common import InteractiveImageViewer
from ..common.common import SolverUtils


class EdgeSolver:
    """Handles the edge-based solving phase: alternative matching approach using edges."""

    @staticmethod
    def solve_with_edges(prepared_data, show_visualizations=True):
        """Solve puzzle using edge-based matching approach.

        Args:
            prepared_data: Dictionary returned from PuzzlePreparer.prepare_puzzle_data()
            piece_id_1: Not used in edge solver (kept for API compatibility)
            piece_id_2: Not used in edge solver (kept for API compatibility)
            show_visualizations: Whether to display visualization windows (default: True)

        Returns:
            dict: Solving results containing:
                - matches: List of edge matches found between all pieces
                - algorithm_name: Name of algorithm used
                - all_pieces: List of all pieces analyzed
                - all_segments: List of all segments
        """
        SolverUtils.print_section_header("EDGE-BASED SOLVING PHASE")

        # Extract prepared data
        analysis_data = prepared_data['analysis_data']
        original_image = prepared_data['original_image']
        all_segments = prepared_data['all_segments']
        project_root = prepared_data['project_root']

        # Validate we have at least 2 pieces
        if len(analysis_data.pieces) < 2:
            print("Error: Need at least 2 pieces for matching")
            return None

        print(f"Processing all {len(analysis_data.pieces)} pieces for edge-based matching")
        for i, piece in enumerate(analysis_data.pieces):
            print(f"  - Piece {piece.piece_id}: {len(all_segments[i])} segments")

        print("\n" + "-"*70)
        print("EDGE-BASED MATCHING ALGORITHM")
        print("-"*70)

        # Step 1: Validate that all pieces have frame corners
        print("\nStep 1: Validating frame corners for all pieces...")
        pieces_with_frames = []
        for i, piece in enumerate(analysis_data.pieces):
            if len(piece.frame_corners) > 0:
                pieces_with_frames.append((piece, all_segments[i]))
                print(f"[OK] Piece {piece.piece_id} has {len(piece.frame_corners)} frame corner(s)")
            else:
                print(f"[SKIP] Piece {piece.piece_id} has no frame corners (skipping)")

        if len(pieces_with_frames) < 2:
            print("\nError: Need at least 2 pieces with frame corners for matching")
            return None

        print(f"\nFound {len(pieces_with_frames)} pieces with frame corners")

        # Step 2: Find frame-adjacent segments for all pieces
        print("\nStep 2: Finding segments adjacent to frame corners for all pieces...")
        piece_frame_segments = []
        for piece, segments in pieces_with_frames:
            frame_adjacent_segs = SegmentFinder.find_frame_adjacent_segments(piece, segments)
            piece_frame_segments.append((piece, segments, frame_adjacent_segs))
            print(f"\nPiece {piece.piece_id} - {len(frame_adjacent_segs)} frame-adjacent segments:")
            for seg in frame_adjacent_segs:
                print(f"  - Segment ID: {seg.segment_id}")

        # Step 3: Export segment pair data and visualize
        print("\n" + "-"*70)
        print("Step 3: Exporting and visualizing frame-adjacent segment pairs...")
        print("-"*70)

        # Export segment pair data to JSON
        segment_pairs_data = EdgeSolver._export_segment_pairs_data(
            piece_frame_segments, all_segments, project_root, prepared_data['temp_folder']
        )

        # Collect all matches and segment data for chain matching
        from .chain_matcher import ChainMatcher

        all_matches = []
        all_segment_data = {}

        # Iterate through all piece pairs (always do matching, visualizations optional)
        for i in range(len(piece_frame_segments)):
            for j in range(i + 1, len(piece_frame_segments)):
                piece1, segments1, frame_segs1 = piece_frame_segments[i]
                piece2, segments2, frame_segs2 = piece_frame_segments[j]

                print(f"\nProcessing segment pairs: Piece {piece1.piece_id} <-> Piece {piece2.piece_id}")
                print(f"  Piece {piece1.piece_id} has {len(frame_segs1)} frame-adjacent segments")
                print(f"  Piece {piece2.piece_id} has {len(frame_segs2)} frame-adjacent segments")
                print(f"  Total pairs to evaluate: {len(frame_segs1) * len(frame_segs2)}")

                # Create a single grid visualization showing all pairs for this piece combination
                grid_image, matches, segment_data = EdgeSolverV2Visualizer.visualize_segment_pairs(
                    piece1, segments1, frame_segs1,
                    piece2, segments2, frame_segs2,
                    original_image
                )

                # Collect matches and segment data
                all_matches.extend(matches)
                all_segment_data.update(segment_data)

                # Save the visualization
                output_path = prepared_data['temp_folder'] / f'segment_pairs_P{piece1.piece_id}_P{piece2.piece_id}.png'
                cv2.imwrite(str(output_path), grid_image)
                print(f"  Saved to: {output_path}")

                # Display the visualization only if requested
                if show_visualizations:
                    viewer = InteractiveImageViewer(
                        f"Segment Pairs: P{piece1.piece_id} <-> P{piece2.piece_id}"
                    )
                    viewer.show(grid_image)

        print(f"\n[OK] All segment pair processing complete ({len(all_matches)} matches found)")

        # Step 4: Find chains from individual matches
        print("\n" + "-"*70)
        print("Step 4: Finding segment chains (max length: 2)...")
        print("-"*70)

        print(f"\nTotal individual matches found: {len(all_matches)}")
        valid_matches = [m for m in all_matches if m.is_valid]
        print(f"Valid matches: {len(valid_matches)}")

        # Build segment count dictionary and frame-touching dictionary
        num_segments_dict = {}
        frame_touching_dict = {}
        for piece, segments, frame_segs in piece_frame_segments:
            num_segments_dict[piece.piece_id] = len(segments)
            # Find frame-touching segments for this piece
            frame_touching_dict[piece.piece_id] = ChainMatcher._find_frame_touching_segments(piece, segments)

        if valid_matches:
            all_progressive_chains = ChainMatcher.find_chains_from_matches(
                all_matches,  # Pass all matches, not just valid ones
                all_segment_data,
                num_segments_dict,
                frame_touching_dict,
                piece_frame_segments
            )
            print(f"\nFound {len(all_progressive_chains)} progressive chain set(s)")

            # Visualize progressive chains (always generate images, optionally display)
            if all_progressive_chains:
                print("\n" + "-"*70)
                print("Step 5: Processing progressive chains...")
                print("-"*70)

                for progressive_chains in all_progressive_chains:
                    # Get the first chain to find piece info
                    first_chain = progressive_chains[0]

                    # Find the pieces for this chain
                    piece1 = None
                    piece2 = None
                    segments1 = None
                    segments2 = None

                    for p, segs, _ in piece_frame_segments:
                        if p.piece_id == first_chain.piece1_id:
                            piece1 = p
                            segments1 = segs
                        if p.piece_id == first_chain.piece2_id:
                            piece2 = p
                            segments2 = segs

                    if piece1 and piece2 and segments1 and segments2:
                        print(f"\nProcessing progressive chains: P{first_chain.piece1_id} <-> P{first_chain.piece2_id}")
                        print(f"  Chain lengths: {[c.chain_length for c in progressive_chains]}")

                        # Create progressive visualization showing all chain lengths
                        progressive_image = EdgeSolverV2Visualizer.visualize_progressive_chains(
                            piece1, segments1,
                            piece2, segments2,
                            progressive_chains,
                            original_image
                        )

                        # Save the visualization
                        output_path = prepared_data['temp_folder'] / f'progressive_chain_P{first_chain.piece1_id}_P{first_chain.piece2_id}.png'
                        cv2.imwrite(str(output_path), progressive_image)
                        print(f"  Saved to: {output_path}")

                        # Display the visualization only if requested
                        if show_visualizations:
                            viewer = InteractiveImageViewer(
                                f"Progressive Chains: P{first_chain.piece1_id} <-> P{first_chain.piece2_id}"
                            )
                            viewer.show(progressive_image)

                print("\n[OK] All chain processing complete")

            # Step 6: Determine best connections for puzzle assembly
            print("\n" + "-"*70)
            print("Step 6: Determining best connections for puzzle assembly...")
            print("-"*70)

            # Collect all valid chains from progressive chains
            all_valid_chains = []
            for progressive_chains in all_progressive_chains:
                # Find the longest VALID chain in this progressive set
                valid_chains = [c for c in progressive_chains if c.is_valid]

                if valid_chains:
                    # Get the longest valid chain
                    best_chain = max(valid_chains, key=lambda c: c.chain_length)
                    all_valid_chains.append(best_chain)

            print(f"\nFound {len(all_valid_chains)} valid chain matches")

            # Use connection selector to select best 2 connections per piece
            piece_connections = ConnectionSelector.determine_best_connections(
                all_valid_chains,
                piece_frame_segments
            )

            print(f"\nSelected connections (2 per piece):")

            # Display summary of connections
            if piece_connections:
                print("\n" + "="*70)
                print("PUZZLE ASSEMBLY GUIDE - SELECTED CONNECTIONS")
                print("="*70)

                for piece_id, connections in sorted(piece_connections.items()):
                    print(f"\nPiece {piece_id}:")
                    for i, (connected_piece_id, chain, side) in enumerate(connections):
                        seg_ids_p1 = chain.segment_ids_p1
                        seg_ids_p2 = chain.segment_ids_p2

                        print(f"  Connection {i+1} (Side {side}): -> Piece {connected_piece_id}")
                        print(f"    Segments: P{chain.piece1_id}{seg_ids_p1} <-> P{chain.piece2_id}{seg_ids_p2}")
                        print(f"    Chain length: {chain.chain_length} segments")
                        print(f"    Shape match: {chain.shape_score:.1f}%")

                print("\n" + "="*70)
                total_connections = sum(len(conns) for conns in piece_connections.values()) // 2
                print(f"Summary: {total_connections} connections selected for puzzle assembly")
                print("="*70)

                # Create connections visualization (dialog shown only if visualizations enabled)
                print("\nStep 6.5: Creating connections visualization...")
                ConnectionVisualizer.visualize_connections_dialog(
                    piece_connections,
                    piece_frame_segments,
                    original_image,
                    project_root,
                    prepared_data['temp_folder'],
                    show_dialog=show_visualizations
                )

            all_best_chains = all_valid_chains
        else:
            print("No valid matches found, cannot form chains")
            all_best_chains = []

        # Return results
        return {
            'algorithm_name': 'edge_v2',
            'all_pieces': [p for p, _, _ in piece_frame_segments],
            'all_segments': all_segments,
            'piece_frame_segments': piece_frame_segments,
            'best_chains': all_best_chains,
        }

    @staticmethod
    def _export_segment_pairs_data(piece_frame_segments, all_segments, project_root, temp_folder):
        """Export all segment pair data to JSON for external debugging.

        Args:
            piece_frame_segments: List of (piece, segments, frame_adjacent_segments) tuples
            all_segments: All segments for all pieces
            project_root: Project root directory
            temp_folder: Analysis temp folder for output

        Returns:
            dict: Exported segment pair data
        """
        import numpy as np

        export_data = {
            'description': 'Segment pair matching data for debugging endpoint alignment',
            'piece_pairs': []
        }

        # Iterate through all piece pairs
        for i in range(len(piece_frame_segments)):
            for j in range(i + 1, len(piece_frame_segments)):
                piece1, segments1, frame_segs1 = piece_frame_segments[i]
                piece2, segments2, frame_segs2 = piece_frame_segments[j]

                piece_pair_data = {
                    'piece1_id': piece1.piece_id,
                    'piece2_id': piece2.piece_id,
                    'segment_pairs': []
                }

                # Find frame-touching segment IDs
                frame_touching_ids1 = EdgeSolverV2Visualizer._find_frame_touching_segment_ids(piece1, segments1)
                frame_touching_ids2 = EdgeSolverV2Visualizer._find_frame_touching_segment_ids(piece2, segments2)

                # Create all segment pairs
                for seg1 in frame_segs1:
                    for seg2 in frame_segs2:
                        # Determine endpoints for seg1
                        current_id1 = seg1.segment_id
                        num_segments1 = len(segments1)
                        prev_id1 = (current_id1 - 1) % num_segments1
                        next_id1 = (current_id1 + 1) % num_segments1
                        prev_is_frame1 = prev_id1 in frame_touching_ids1
                        next_is_frame1 = next_id1 in frame_touching_ids1

                        if prev_is_frame1:
                            seg1_frame_idx = 0
                            seg1_interior_idx = -1
                        elif next_is_frame1:
                            seg1_frame_idx = -1
                            seg1_interior_idx = 0
                        else:
                            seg1_frame_idx = 0
                            seg1_interior_idx = -1

                        # Determine endpoints for seg2
                        current_id2 = seg2.segment_id
                        num_segments2 = len(segments2)
                        prev_id2 = (current_id2 - 1) % num_segments2
                        next_id2 = (current_id2 + 1) % num_segments2
                        prev_is_frame2 = prev_id2 in frame_touching_ids2
                        next_is_frame2 = next_id2 in frame_touching_ids2

                        if prev_is_frame2:
                            seg2_frame_idx = 0
                            seg2_interior_idx = -1
                        elif next_is_frame2:
                            seg2_frame_idx = -1
                            seg2_interior_idx = 0
                        else:
                            seg2_frame_idx = 0
                            seg2_interior_idx = -1

                        # Get segment points
                        seg1_points = [[p.x, p.y] for p in seg1.contour_points]
                        seg2_points = [[p.x, p.y] for p in seg2.contour_points]

                        pair_data = {
                            'seg1': {
                                'segment_id': seg1.segment_id,
                                'points': seg1_points,
                                'frame_endpoint_index': seg1_frame_idx,
                                'interior_endpoint_index': seg1_interior_idx,
                                'frame_endpoint': seg1_points[seg1_frame_idx],
                                'interior_endpoint': seg1_points[seg1_interior_idx],
                                'prev_segment_id': prev_id1,
                                'next_segment_id': next_id1,
                                'prev_is_frame_touching': bool(prev_is_frame1),
                                'next_is_frame_touching': bool(next_is_frame1)
                            },
                            'seg2': {
                                'segment_id': seg2.segment_id,
                                'points': seg2_points,
                                'frame_endpoint_index': seg2_frame_idx,
                                'interior_endpoint_index': seg2_interior_idx,
                                'frame_endpoint': seg2_points[seg2_frame_idx],
                                'interior_endpoint': seg2_points[seg2_interior_idx],
                                'prev_segment_id': prev_id2,
                                'next_segment_id': next_id2,
                                'prev_is_frame_touching': bool(prev_is_frame2),
                                'next_is_frame_touching': bool(next_is_frame2)
                            },
                            'expected_alignment': {
                                'description': 'Blue (frame) endpoints should align, Red (interior) endpoints should align',
                                'seg1_blue_color': 'frame_endpoint',
                                'seg1_red_color': 'interior_endpoint',
                                'seg2_blue_color': 'frame_endpoint',
                                'seg2_red_color': 'interior_endpoint'
                            }
                        }

                        piece_pair_data['segment_pairs'].append(pair_data)

                export_data['piece_pairs'].append(piece_pair_data)

        # Save to JSON file in analysis temp folder
        output_path = temp_folder / 'segment_pairs_debug_data.json'
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"\n[OK] Exported segment pair data to: {output_path}")
        print(f"     Contains {len(export_data['piece_pairs'])} piece pair combinations")
        total_pairs = sum(len(pp['segment_pairs']) for pp in export_data['piece_pairs'])
        print(f"     Total segment pairs: {total_pairs}")

        return export_data

