"""Edge-based solving module for puzzle solver - alternative approach to matching."""

import cv2
from .segment_finder import SegmentFinder
from .frame_adjacent_matcher import FrameAdjacentMatcher
from .connection_manager import ConnectionManager
from .rotation_calculator import RotationCalculator
from .visualizers import EdgeSolverVisualizer
from .solution_builder import SolutionBuilder
from ..matrix_solver.segment_overlay_visualizer import SegmentOverlayVisualizer
from ..common import InteractiveImageViewer
from ..common.common import SolverUtils


class EdgeSolver:
    """Handles the edge-based solving phase: alternative matching approach using edges."""

    @staticmethod
    def solve_with_edges(prepared_data, piece_id_1=None, piece_id_2=None, show_visualizations=True):
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

        # Step 3: Match frame-adjacent segments between all pairs of pieces
        print("\nStep 3: Matching frame-adjacent segments between all piece pairs...")
        all_matches = []
        all_extended_matches = []

        for i in range(len(piece_frame_segments)):
            for j in range(i + 1, len(piece_frame_segments)):
                piece1, segments1, frame_segs1 = piece_frame_segments[i]
                piece2, segments2, frame_segs2 = piece_frame_segments[j]

                print(f"\n  Comparing Piece {piece1.piece_id} <-> Piece {piece2.piece_id}...")
                matches = FrameAdjacentMatcher.match_frame_adjacent_segments(
                    piece1, frame_segs1,
                    piece2, frame_segs2
                )

                if matches:
                    print(f"    Found {len(matches)} potential matches (top 3):")
                    for k, match in enumerate(matches[:3]):
                        print(f"      {k+1}. Seg {match.seg1_id} <-> Seg {match.seg2_id}: "
                              f"score={match.match_score:.3f} "
                              f"(length={match.length_score:.3f}, shape={match.shape_score:.3f})")
                    all_matches.extend(matches)

                    # Extend the matches along the contour
                    print(f"    Extending matches along contour (threshold=0.8)...")
                    extended_matches = FrameAdjacentMatcher.extend_segment_matches(
                        matches, piece1, segments1, frame_segs1,
                        piece2, segments2, frame_segs2, threshold=0.8
                    )

                    if extended_matches:
                        print(f"    Extended to {len(extended_matches)} extended matches (top 3):")
                        for k, ext_match in enumerate(extended_matches[:3]):
                            print(f"      {k+1}. Initial Seg {ext_match.initial_match.seg1_id} <-> "
                                  f"Seg {ext_match.initial_match.seg2_id}: "
                                  f"{ext_match.total_segments_matched} segments matched, "
                                  f"combined_score={ext_match.combined_score:.3f} "
                                  f"(avg_quality={ext_match.average_match_score:.3f})")
                        all_extended_matches.extend(extended_matches)
                else:
                    print(f"    No matches found")

        # Sort all matches by score
        all_matches.sort(key=lambda m: m.match_score, reverse=True)
        all_extended_matches.sort(key=lambda m: m.combined_score, reverse=True)

        print(f"\n\nTotal matches found across all piece pairs: {len(all_matches)}")
        print(f"Total extended matches found: {len(all_extended_matches)}")

        if all_matches:
            print("\nTop 10 overall single-segment matches:")
            for i, match in enumerate(all_matches[:10]):
                print(f"  {i+1}. P{match.piece1_id}-S{match.seg1_id} <-> P{match.piece2_id}-S{match.seg2_id}: "
                      f"score={match.match_score:.3f} "
                      f"(length={match.length_score:.3f}, shape={match.shape_score:.3f})")

        if all_extended_matches:
            print("\nTop 10 extended matches (considering segment sequences):")
            for i, ext_match in enumerate(all_extended_matches[:10]):
                print(f"  {i+1}. P{ext_match.piece1_id}-S{ext_match.initial_match.seg1_id} <-> "
                      f"P{ext_match.piece2_id}-S{ext_match.initial_match.seg2_id}: "
                      f"{ext_match.total_segments_matched} segments matched, "
                      f"combined_score={ext_match.combined_score:.3f} "
                      f"(avg_quality={ext_match.average_match_score:.3f})")

        # Step 4: Calculate rotation angles for each piece
        print("\n" + "-"*70)
        print("Step 4: Calculating rotation angles based on frame corners...")
        print("-"*70)

        rotation_angles = RotationCalculator.calculate_piece_rotation_angles(piece_frame_segments)

        print("\nRotation angles for each piece:")
        for piece_id, angle in sorted(rotation_angles.items()):
            print(f"  Piece {piece_id}: {angle:.1f}°")

        # Step 5: Determine best connections for each piece (2 connections per piece)
        print("\n" + "-"*70)
        print("Step 5: Determining best connections for each piece...")
        print("-"*70)

        # Use extended matches for better connection decisions
        piece_connections = ConnectionManager.determine_best_connections_extended(
            all_extended_matches if all_extended_matches else all_matches,
            piece_frame_segments
        )

        print("\nBest connections for each piece (each piece connects to 2 others):")
        from ..common.data_classes import ExtendedSegmentMatch
        for piece_id, connections in sorted(piece_connections.items()):
            print(f"\nPiece {piece_id}:")
            for i, (connected_piece_id, match, side) in enumerate(connections):
                if isinstance(match, ExtendedSegmentMatch):
                    seg1_id = match.initial_match.seg1_id
                    seg2_id = match.initial_match.seg2_id
                    score_str = f"combined_score={match.combined_score:.3f}, " \
                                f"segments={match.total_segments_matched}"
                else:
                    seg1_id = match.seg1_id
                    seg2_id = match.seg2_id
                    score_str = f"score={match.match_score:.3f}"

                print(f"  Connection {i+1}: -> Piece {connected_piece_id} "
                      f"(S{seg1_id if match.piece1_id == piece_id else seg2_id} <-> "
                      f"S{seg2_id if match.piece1_id == piece_id else seg1_id}, "
                      f"{score_str}, side={side})")

        # Step 5.5: Create segment overlay visualization for best connection matches
        if show_visualizations:
            print("\n" + "-"*70)
            print("Step 5.5: Creating segment overlay visualizations for best connections...")
            print("-"*70)

            # Collect all unique connection matches (use dict to avoid duplicates)
            connection_matches = {}
            for piece_id, connections in piece_connections.items():
                for connected_piece_id, match, side in connections:
                    # Create a unique key for this connection to avoid duplicates
                    pair_key = (min(match.piece1_id, match.piece2_id),
                               max(match.piece1_id, match.piece2_id))
                    # Only add if not already present
                    if pair_key not in connection_matches:
                        connection_matches[pair_key] = match

            # Visualize each unique connection match
            for i, (pair_key, match) in enumerate(connection_matches.items()):
                p1_id, p2_id = pair_key
                # Find the piece and segments data
                piece1_data = next((p, s) for p, s, _ in piece_frame_segments if p.piece_id == p1_id)
                piece2_data = next((p, s) for p, s, _ in piece_frame_segments if p.piece_id == p2_id)
                piece1, segments1 = piece1_data
                piece2, segments2 = piece2_data

                # Create visualization for this connection match
                print(f"\n  Creating overlay for connection {i+1}: P{p1_id} <-> P{p2_id}...")

                # For extended matches, create a combined segment containing the entire chain
                if isinstance(match, ExtendedSegmentMatch):
                    # Get all matches in the chain
                    all_matches_in_chain = [match.initial_match] + match.extended_matches

                    # Collect all segment IDs in the chain for each piece
                    seg1_ids = [m.seg1_id for m in all_matches_in_chain]
                    seg2_ids = [m.seg2_id for m in all_matches_in_chain]

                    # Get all segments and sort them by segment_id to maintain order
                    chain_segs1 = sorted([s for s in segments1 if s.segment_id in seg1_ids], key=lambda s: s.segment_id)
                    chain_segs2 = sorted([s for s in segments2 if s.segment_id in seg2_ids], key=lambda s: s.segment_id)

                    # Find frame connection points (where chains touch the frame straight edges)
                    # Get frame-adjacent segments
                    _, _, frame_segs1 = next((p, s, f) for p, s, f in piece_frame_segments if p.piece_id == p1_id)
                    _, _, frame_segs2 = next((p, s, f) for p, s, f in piece_frame_segments if p.piece_id == p2_id)

                    # Find which frame segments are frame-touching (straight edges connected to frame corner)
                    frame_touching_dict1 = FrameAdjacentMatcher._find_frame_touching_segments(piece1, segments1)
                    frame_touching_dict2 = FrameAdjacentMatcher._find_frame_touching_segments(piece2, segments2)

                    # Flatten the dict values to get all frame-touching segment IDs
                    frame_touching_ids1 = []
                    for seg_ids in frame_touching_dict1.values():
                        frame_touching_ids1.extend(seg_ids)

                    frame_touching_ids2 = []
                    for seg_ids in frame_touching_dict2.values():
                        frame_touching_ids2.extend(seg_ids)

                    # Determine which end of each chain is the frame connection point
                    # Check the first segment in each chain
                    first_seg1 = chain_segs1[0]
                    last_seg1 = chain_segs1[-1]

                    # For chain 1: check if start or end connects to frame
                    chain1_starts_at_frame = (first_seg1.segment_id - 1) % len(segments1) in frame_touching_ids1
                    chain1_ends_at_frame = (last_seg1.segment_id + 1) % len(segments1) in frame_touching_ids1

                    # For chain 2: check if start or end connects to frame
                    first_seg2 = chain_segs2[0]
                    last_seg2 = chain_segs2[-1]
                    chain2_starts_at_frame = (first_seg2.segment_id - 1) % len(segments2) in frame_touching_ids2
                    chain2_ends_at_frame = (last_seg2.segment_id + 1) % len(segments2) in frame_touching_ids2

                    # Reverse chains if needed so both start at the frame connection
                    if chain1_ends_at_frame and not chain1_starts_at_frame:
                        # Chain 1 needs to be reversed
                        chain_segs1 = list(reversed(chain_segs1))
                        first_seg1 = chain_segs1[0]
                        frame_connection_point1 = first_seg1.start_corner.to_point()
                    else:
                        # Chain 1 is correctly oriented
                        frame_connection_point1 = first_seg1.start_corner.to_point()

                    if chain2_ends_at_frame and not chain2_starts_at_frame:
                        # Chain 2 needs to be reversed
                        chain_segs2 = list(reversed(chain_segs2))
                        first_seg2 = chain_segs2[0]
                        frame_connection_point2 = first_seg2.start_corner.to_point()
                    else:
                        # Chain 2 is correctly oriented
                        frame_connection_point2 = first_seg2.start_corner.to_point()

                    # Now combine all points from the correctly oriented chains
                    # IMPORTANT: When we reversed the segments, we need to also reverse the points within each segment
                    # to maintain continuity
                    # Also need to check if segments are consecutive to avoid creating artificial connections
                    combined_points1 = []
                    if chain1_ends_at_frame and not chain1_starts_at_frame:
                        # Chain was reversed, so reverse points within each segment too
                        for i, seg in enumerate(chain_segs1):
                            pts = list(reversed(seg.contour_points))
                            if i == 0:
                                # First segment: add all points
                                combined_points1.extend(pts)
                            else:
                                # Skip first point to avoid duplicating the shared corner
                                combined_points1.extend(pts[1:])
                    else:
                        # Chain is in natural order
                        for i, seg in enumerate(chain_segs1):
                            if i == 0:
                                # First segment: add all points
                                combined_points1.extend(seg.contour_points)
                            else:
                                # Skip first point to avoid duplicating the shared corner
                                combined_points1.extend(seg.contour_points[1:])

                    combined_points2 = []
                    if chain2_ends_at_frame and not chain2_starts_at_frame:
                        # Chain was reversed, so reverse points within each segment too
                        for i, seg in enumerate(chain_segs2):
                            pts = list(reversed(seg.contour_points))
                            if i == 0:
                                # First segment: add all points
                                combined_points2.extend(pts)
                            else:
                                # Skip first point to avoid duplicating the shared corner
                                combined_points2.extend(pts[1:])
                    else:
                        # Chain is in natural order
                        for i, seg in enumerate(chain_segs2):
                            if i == 0:
                                # First segment: add all points
                                combined_points2.extend(seg.contour_points)
                            else:
                                # Skip first point to avoid duplicating the shared corner
                                combined_points2.extend(seg.contour_points[1:])

                    # Create temporary combined segments for visualization
                    from ..common.data_classes import ContourSegment, Corner
                    combined_seg1 = ContourSegment(
                        segment_id=seg1_ids[0],
                        piece_id=piece1.piece_id,
                        start_corner=chain_segs1[0].start_corner,
                        end_corner=chain_segs1[-1].end_corner,
                        contour_points=combined_points1,
                        piece_centroid=piece1.centroid,
                        is_border_edge=False
                    )
                    combined_seg2 = ContourSegment(
                        segment_id=seg2_ids[0],
                        piece_id=piece2.piece_id,
                        start_corner=chain_segs2[0].start_corner,
                        end_corner=chain_segs2[-1].end_corner,
                        contour_points=combined_points2,
                        piece_centroid=piece2.centroid,
                        is_border_edge=False
                    )

                    # Use the specialized chain overlay visualization with endpoint alignment
                    # Pass all the chain segments so we can show progressive stages
                    overlay_image = SegmentOverlayVisualizer.create_progressive_chain_visualization(
                        piece1, chain_segs1, piece2, chain_segs2,
                        match, all_matches_in_chain,
                        frame_connection_point1, frame_connection_point2
                    )
                else:
                    # Single segment match - use as-is
                    matches_to_visualize = [match]
                    overlay_image = SegmentOverlayVisualizer.create_overlay_visualization(
                        piece1, segments1, piece2, segments2, matches_to_visualize, max_overlays=1
                    )

                overlay_output_path = prepared_data['temp_folder'] / f'edge_connection_overlay_P{p1_id}_P{p2_id}.png'
                cv2.imwrite(str(overlay_output_path), overlay_image)
                print(f"    Saved to: {overlay_output_path}")

                print(f"    Displaying segment overlay visualization...")
                overlay_viewer = InteractiveImageViewer(f"Connection Overlay {i+1}: P{p1_id} <-> P{p2_id}")
                overlay_viewer.show(overlay_image)

        # Step 6: Visualize the connections
        if show_visualizations:
            print("\n" + "-"*70)
            print("Step 6: Creating visualization of connections...")
            print("-"*70)
            EdgeSolverVisualizer.visualize_connections_svg(
                piece_connections,
                piece_frame_segments,
                all_matches,
                original_image,
                project_root
            )
            print("[OK] SVG visualization saved")

            EdgeSolverVisualizer.visualize_connections_dialog(
                piece_connections,
                piece_frame_segments,
                original_image,
                project_root
            )
            print("[OK] Interactive visualization displayed")

        # # Step 7: Create solution SVG (first piece only)
        # print("\n" + "-"*70)
        # print("Step 7: Creating solution SVG...")
        # print("-"*70)
        # SolutionBuilder.create_solution_svg(
        #     piece_frame_segments,
        #     rotation_angles,
        #     original_image,
        #     project_root,
        #     piece_connections,
        #     all_matches
        # )
        # print("[OK] Solution SVG saved")

        SolverUtils.print_section_footer("Edge-Based Solving Complete")

        # Return results
        return {
            'matches': all_matches,
            'extended_matches': all_extended_matches,
            'algorithm_name': 'edge',
            'all_pieces': [p for p, _, _ in piece_frame_segments],
            'all_segments': all_segments,
            'piece_frame_segments': piece_frame_segments,
            'piece_connections': piece_connections,
            'rotation_angles': rotation_angles
        }
