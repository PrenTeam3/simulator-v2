"""Module for managing connections between puzzle pieces."""


class ConnectionManager:
    """Handles determining the best connections between pieces."""

    @staticmethod
    def determine_best_connections(all_matches, piece_frame_segments):
        """Determine the best 2 connections for each piece.

        Each piece should connect to exactly 2 other pieces (one on each side of the frame corner).

        Args:
            all_matches: List of all SegmentMatch objects sorted by score
            piece_frame_segments: List of tuples (piece, segments, frame_adjacent_segments)

        Returns:
            dict: piece_id -> list of (connected_piece_id, match, side) tuples
        """
        # Initialize connections dictionary
        piece_connections = {}
        for piece, _, frame_segs in piece_frame_segments:
            piece_connections[piece.piece_id] = []

        # Track which matches have been used
        used_matches = set()

        # Greedy approach: assign best matches first
        for match in all_matches:
            p1_id = match.piece1_id
            p2_id = match.piece2_id

            # Check if both pieces still need connections
            p1_needs_connection = len(piece_connections[p1_id]) < 2
            p2_needs_connection = len(piece_connections[p2_id]) < 2

            if p1_needs_connection and p2_needs_connection:
                # Determine which side of the piece this connection is on
                # Side A or Side B (based on segment ID ordering)
                side_p1 = "A" if match.seg1_id in [seg.segment_id for piece, _, frame_segs in piece_frame_segments
                                                     if piece.piece_id == p1_id
                                                     for seg in frame_segs[:1]] else "B"
                side_p2 = "A" if match.seg2_id in [seg.segment_id for piece, _, frame_segs in piece_frame_segments
                                                     if piece.piece_id == p2_id
                                                     for seg in frame_segs[:1]] else "B"

                # Add connection for piece 1
                piece_connections[p1_id].append((p2_id, match, side_p1))

                # Add connection for piece 2
                piece_connections[p2_id].append((p1_id, match, side_p2))

                used_matches.add((match.piece1_id, match.seg1_id, match.piece2_id, match.seg2_id))

                # Check if all pieces have their 2 connections
                all_connected = all(len(conns) == 2 for conns in piece_connections.values())
                if all_connected:
                    break

        return piece_connections

    @staticmethod
    def determine_best_connections_extended(all_matches, piece_frame_segments):
        """Determine the best 2 connections for each piece using extended matches.

        This version works with both ExtendedSegmentMatch and regular SegmentMatch objects.
        For extended matches, it prioritizes longer chains (more segments matched) combined with quality.

        Args:
            all_matches: List of ExtendedSegmentMatch or SegmentMatch objects sorted by score
            piece_frame_segments: List of tuples (piece, segments, frame_adjacent_segments)

        Returns:
            dict: piece_id -> list of (connected_piece_id, match, side) tuples
        """
        from ..common.data_classes import ExtendedSegmentMatch

        # Sort matches by chain length first, then by score
        # This ensures longer chains are prioritized for connections
        def match_ranking_key(match):
            if isinstance(match, ExtendedSegmentMatch):
                # Prioritize chain length heavily (80% weight) + quality (20% weight)
                chain_length = match.total_segments_matched
                quality = match.combined_score
                return (chain_length * 0.8) + (quality * 0.2)
            else:
                # Regular matches get treated as length 1
                return match.match_score * 0.2

        sorted_matches = sorted(all_matches, key=match_ranking_key, reverse=True)

        # Initialize connections dictionary
        piece_connections = {}
        for piece, _, frame_segs in piece_frame_segments:
            piece_connections[piece.piece_id] = []

        # Track which matches have been used
        used_matches = set()

        # Greedy approach: assign best matches first
        for match in sorted_matches:
            # Handle both ExtendedSegmentMatch and regular SegmentMatch
            if isinstance(match, ExtendedSegmentMatch):
                p1_id = match.piece1_id
                p2_id = match.piece2_id
                seg1_id = match.initial_match.seg1_id
                seg2_id = match.initial_match.seg2_id
            else:
                p1_id = match.piece1_id
                p2_id = match.piece2_id
                seg1_id = match.seg1_id
                seg2_id = match.seg2_id

            # Check if both pieces still need connections
            p1_needs_connection = len(piece_connections[p1_id]) < 2
            p2_needs_connection = len(piece_connections[p2_id]) < 2

            if p1_needs_connection and p2_needs_connection:
                # Determine which side of the piece this connection is on
                # Side A or Side B (based on segment ID ordering)
                side_p1 = "A" if seg1_id in [seg.segment_id for piece, _, frame_segs in piece_frame_segments
                                                     if piece.piece_id == p1_id
                                                     for seg in frame_segs[:1]] else "B"
                side_p2 = "A" if seg2_id in [seg.segment_id for piece, _, frame_segs in piece_frame_segments
                                                     if piece.piece_id == p2_id
                                                     for seg in frame_segs[:1]] else "B"

                # Add connection for piece 1
                piece_connections[p1_id].append((p2_id, match, side_p1))

                # Add connection for piece 2
                piece_connections[p2_id].append((p1_id, match, side_p2))

                used_matches.add((p1_id, seg1_id, p2_id, seg2_id))

                # Check if all pieces have their 2 connections
                all_connected = all(len(conns) == 2 for conns in piece_connections.values())
                if all_connected:
                    break

        return piece_connections
