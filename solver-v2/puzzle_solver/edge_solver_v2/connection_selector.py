"""Connection selection module for determining best piece connections."""


class ConnectionSelector:
    """Handles selection of best connections for puzzle assembly."""

    @staticmethod
    def determine_best_connections(all_valid_chains, piece_frame_segments):
        """Determine the best 2 connections for each piece using chain matches.

        Similar to edge_solver v1's ConnectionManager, but works with ChainMatch objects.

        Args:
            all_valid_chains: List of valid ChainMatch objects
            piece_frame_segments: List of (piece, segments, frame_adjacent_segments) tuples

        Returns:
            dict: piece_id -> list of (connected_piece_id, chain, side) tuples
        """
        # Sort chains by: chain length (80%) + shape score (20%)
        def chain_ranking_key(chain):
            chain_length = chain.chain_length
            quality = chain.shape_score / 100.0  # Normalize to 0-1
            return (chain_length * 0.8) + (quality * 0.2)

        sorted_chains = sorted(all_valid_chains, key=chain_ranking_key, reverse=True)

        # Initialize connections dictionary
        piece_connections = {}
        for piece, _, frame_segs in piece_frame_segments:
            piece_connections[piece.piece_id] = []

        # Track which piece pairs have been connected (normalized to avoid duplicates)
        used_piece_pairs = set()

        # Track which segments have been used for each piece
        # Format: {piece_id: set(segment_ids)}
        used_segments_per_piece = {}
        for piece, _, frame_segs in piece_frame_segments:
            used_segments_per_piece[piece.piece_id] = set()

        # Greedy approach: assign best matches first
        for chain in sorted_chains:
            p1_id = chain.piece1_id
            p2_id = chain.piece2_id

            # Create normalized pair key (order doesn't matter)
            pair_key = tuple(sorted([p1_id, p2_id]))

            # Skip if this piece pair is already connected
            if pair_key in used_piece_pairs:
                continue

            # Check if any segments in this chain have already been used
            p1_segments = set(chain.segment_ids_p1)
            p2_segments = set(chain.segment_ids_p2)

            p1_segments_overlap = p1_segments & used_segments_per_piece[p1_id]
            p2_segments_overlap = p2_segments & used_segments_per_piece[p2_id]

            # Skip if any segments are already used
            if p1_segments_overlap or p2_segments_overlap:
                continue

            # Check if both pieces still need connections
            p1_needs_connection = len(piece_connections[p1_id]) < 2
            p2_needs_connection = len(piece_connections[p2_id]) < 2

            if p1_needs_connection and p2_needs_connection:
                # Determine which side of the piece this connection is on
                # Get the first segment ID in the chain
                seg1_id = chain.segment_ids_p1[0]
                seg2_id = chain.segment_ids_p2[0]

                # Simple side determination based on segment ID
                # Side A for lower segment IDs, Side B for higher
                piece1_segments = next(s for p, s, _ in piece_frame_segments if p.piece_id == p1_id)
                piece2_segments = next(s for p, s, _ in piece_frame_segments if p.piece_id == p2_id)

                mid_point_p1 = len(piece1_segments) // 2
                mid_point_p2 = len(piece2_segments) // 2

                side_p1 = "A" if seg1_id < mid_point_p1 else "B"
                side_p2 = "A" if seg2_id < mid_point_p2 else "B"

                # Add connection for piece 1
                piece_connections[p1_id].append((p2_id, chain, side_p1))

                # Add connection for piece 2
                piece_connections[p2_id].append((p1_id, chain, side_p2))

                # Mark this piece pair as used
                used_piece_pairs.add(pair_key)

                # Mark segments as used for both pieces
                used_segments_per_piece[p1_id].update(chain.segment_ids_p1)
                used_segments_per_piece[p2_id].update(chain.segment_ids_p2)

                # Check if all pieces have their 2 connections
                all_connected = all(len(conns) == 2 for conns in piece_connections.values())
                if all_connected:
                    break

        return piece_connections
