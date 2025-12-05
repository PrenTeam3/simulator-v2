import cv2
import numpy as np
import os
import sys

from .Distance import real_edge_compute, generated_edge_compute, shape_only_edge_compute
from .Extractor import Extractor
from .Mover import stick_pieces
from .utils import rotate

from .Enums import (
    Directions,
    Strategy,
    TypePiece,
    TypeEdge,
    get_opposite_direction,
    step_direction,
    rotate_direction,
    directions,
)

from .tuple_helper import (
    equals_tuple,
    add_tuple,
    sub_tuple,
    is_neighbor,
    corner_puzzle_alignment,
    display_dim,
)


class Puzzle:
    """
    Class used to store all informations about the puzzle
    """

    def log(self, *args):
        """Helper to log informations to the GUI"""

        print(" ".join(map(str, args)))
        if self.viewer:
            self.viewer.addLog(args)

    def __init__(self, path, viewer=None, green_screen=False, black_only=False):
        """Extract information of pieces in the img at `path` and start computation of the solution"""
        # Initialize viewer and green_ early so log() can use them
        self.viewer = viewer
        self.green_ = green_screen
        self.black_only = black_only

        self.pieces_ = None
        factor = 0.40
        max_factor = 1.0  # Maximum factor to try
        extraction_attempts = 0

        while self.pieces_ is None or len(self.pieces_) < 4 and factor <= max_factor:
            extraction_attempts += 1
            if black_only:
                self.log(f"[BLACK_ONLY] Extraction attempt {extraction_attempts} with factor {factor:.2f}")

            self.extract = Extractor(path, viewer, green_screen, factor, black_only=black_only)
            self.pieces_ = self.extract.extract()

            if self.pieces_ is None:
                factor += 0.01
                if black_only:
                    self.log(f"[BLACK_ONLY] Extraction failed, trying factor {factor:.2f}")
                continue

            # Check if we have enough pieces
            if black_only:
                self.log(f"[BLACK_ONLY] Extracted {len(self.pieces_)} pieces with factor {factor:.2f}")

            if len(self.pieces_) < 4:
                if factor < max_factor:
                    if black_only:
                        self.log(f"[BLACK_ONLY] Only {len(self.pieces_)} pieces found, trying higher factor...")
                    self.pieces_ = None  # Reset to try again
                    factor += 0.05  # Increase factor more aggressively
                    continue
                else:
                    # Max factor reached without finding 4 pieces
                    break
            else:
                # Found at least 4 pieces, stop here
                if black_only:
                    self.log(f"[BLACK_ONLY] Found {len(self.pieces_)} pieces, stopping extraction")
                break

        if self.pieces_ is None or len(self.pieces_) < 4:
            raise RuntimeError(
                f"Failed to extract at least 4 pieces after {extraction_attempts} attempts. "
                f"Found: {len(self.pieces_) if self.pieces_ else 0} pieces"
            )
        if black_only:
            self.log(f"[BLACK_ONLY] Mode activated with {len(self.pieces_)} pieces")

        self.border_pieces = [p for p in self.pieces_ if p.is_border]
        self.non_border_pieces = [p for p in self.pieces_ if not p.is_border]
        if black_only:
            self.log(f"[BLACK_ONLY] Border pieces: {len(self.border_pieces)}, Non-border: {len(self.non_border_pieces)}")

        self.connected_directions = []
        self.diff = {}
        self.edge_to_piece = {e: p for p in self.pieces_ for e in p.edges_}
        self.possible_dim = self.compute_possible_size(
            len(self.pieces_), len(self.border_pieces)
        )
        self.extremum = (-1, -1, 1, 1)

    def solve_puzzle(self):
        self.log(">>> START solving puzzle")

        if self.black_only:
            self.log(">>> Using simplified solver for black-only puzzle (max 6 pieces, all border)")
            self.solve_puzzle_black_only()
            return

        # Separate border pieces from the other
        connected_pieces = []
        border_pieces = self.border_pieces.copy()
        non_border_pieces = self.non_border_pieces.copy()

        # Start by a corner piece
        for piece in border_pieces:
            if piece.number_of_border() > 1:
                connected_pieces.append(piece)
                border_pieces.remove(piece)
                break

        self.log("Number of border pieces: ", len(border_pieces) + 1)

        self.export_pieces(
            os.path.join(os.environ["ZOLVER_TEMP_DIR"], "stick{0:03d}".format(1) + ".png"),
            os.path.join(os.environ["ZOLVER_TEMP_DIR"], "colored{0:03d}".format(1) + ".png"),
            "Border types".format(),
            "Step {0:03d}".format(1),
            display_border=True,
        )

        self.log(">>> START solve border")
        start_piece = connected_pieces[0]
        start_piece.coord = (0, 0)
        self.corner_pos = [((0, 0), start_piece)]  # we start with a corner

        for i in range(4):
            if (
                start_piece.edge_in_direction(Directions.S).connected
                and start_piece.edge_in_direction(Directions.W).connected
            ):
                break
            start_piece.rotate_edges(1)

        self.extremum = (0, 0, 1, 1)

        self.strategy = Strategy.BORDER
        connected_pieces = self.solve(connected_pieces, border_pieces)
        self.log(">>> START solve middle")
        self.strategy = Strategy.FILL
        self.solve(connected_pieces, non_border_pieces)

        self.log(">>> SAVING result...")
        self.translate_puzzle()
        self.export_pieces(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "stick.png"), os.path.join(os.environ["ZOLVER_TEMP_DIR"], "colored.png"), display=False)

    def solve_puzzle_black_only(self):
        """
        Simplified solver for black-only puzzles with max 6 pieces, all touching border.
        Uses only shape matching (no color).
        """
        self.log(f"[BLACK_ONLY] >>> Solving black-only puzzle with {len(self.pieces_)} pieces")

        # All pieces are border pieces in this case
        left_pieces = self.pieces_.copy()
        connected_pieces = []

        # Start with first piece at origin
        start_piece = left_pieces.pop(0)
        self.log(f"[BLACK_ONLY] Starting with piece 0 at origin (0, 0)")
        start_piece.coord = (0, 0)
        connected_pieces.append(start_piece)
        self.connected_directions = [((0, 0), start_piece)]
        self.extremum = (0, 0, 1, 1)

        # Log edge types of start piece
        edge_types = [e.type.name for e in start_piece.edges_]
        self.log(f"[BLACK_ONLY] Start piece edge types: {edge_types}")

        # Orient start piece so border edges are on S and W
        rotations = 0
        for i in range(4):
            if (
                start_piece.edge_in_direction(Directions.S).type == TypeEdge.BORDER
                and start_piece.edge_in_direction(Directions.W).type == TypeEdge.BORDER
            ):
                rotations = i
                break
            start_piece.rotate_edges(1)
        self.log(f"[BLACK_ONLY] Rotated start piece {rotations} times to align border edges")

        self.corner_pos = [((0, 0), start_piece)]

        # Export initial state
        self.export_pieces(
            os.path.join(os.environ["ZOLVER_TEMP_DIR"], "stick{0:03d}".format(1) + ".png"),
            os.path.join(os.environ["ZOLVER_TEMP_DIR"], "colored{0:03d}".format(1) + ".png"),
            "Initial state",
            "Step {0:03d}".format(1),
            display_border=True,
        )

        # Solve using simplified approach
        self.strategy = Strategy.BORDER
        connected_pieces = self.solve_black_only(connected_pieces, left_pieces)

        self.log(">>> SAVING result...")
        self.translate_puzzle()
        self.export_pieces(
            os.path.join(os.environ["ZOLVER_TEMP_DIR"], "stick.png"),
            os.path.join(os.environ["ZOLVER_TEMP_DIR"], "colored.png"),
            display=False
        )

        # Two sets of pieces: Already connected ones and pieces remaining to connect to the others
        # The first piece has an orientation like that:
        #         N          edges:    0
        #      W     E              3     1
        #         S                    2
        #
        # Pieces are placed on a grid like that (X is the first piece at position (0, 0)):
        # +--+--+--+
        # |  |  |  |
        # +--+--+--+
        # |  | X|  |
        # +--+--+--+
        # |  |  |  |
        # +--+--+--+
        #
        # Then if we test the NORTH edge:
        # +--+--+--+
        # |  | X|  |
        # +--+--+--+
        # |  | X|  |
        # +--+--+--+
        # |  |  |  |
        # +--+--+--+
        # Etc until the puzzle is complete i.e. there is no pieces left on left_pieces.

    def get_bbox(self):
        bboxes = [p.get_bbox() for p in self.pieces_]
        return (
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes),
        )

    def rotate_bbox(self, angle, around):
        # Rotate corners only to optimize
        minX, minY, maxX, maxY = self.get_bbox()
        rotated = [
            rotate((x, y), angle, around) for x in [minX, maxX] for y in [minY, maxY]
        ]
        rotatedX = [p[0] for p in rotated]
        rotatedY = [p[1] for p in rotated]
        return (
            int(min(rotatedX)),
            int(min(rotatedY)),
            int(max(rotatedX)),
            int(max(rotatedY)),
        )

    def solve_black_only(self, connected_pieces, left_pieces):
        """
        Simplified solving for black-only puzzles.
        Uses shape-only matching and simpler strategy.
        """
        self.log(f"[BLACK_ONLY] solve_black_only: {len(connected_pieces)} connected, {len(left_pieces)} left")

        # Check if we need to initialize diffs (either connected_directions is empty OR diff is empty)
        if len(self.connected_directions) == 0:
            self.log("[BLACK_ONLY] Initializing connected_directions and computing initial diffs")
            self.connected_directions = [((0, 0), connected_pieces[0])]
            self.diff = self.compute_diffs(
                left_pieces, self.diff, connected_pieces[0]
            )
            self.log(f"[BLACK_ONLY] Initial diff computed: {len(self.diff)} edges in diff dict")
        elif len(self.diff) == 0:
            # connected_directions is set but diff is empty - need to compute initial diffs
            self.log("[BLACK_ONLY] connected_directions set but diff is empty, computing initial diffs")
            # Get the first connected piece from connected_directions
            first_piece = self.connected_directions[0][1]
            self.diff = self.compute_diffs(
                left_pieces, self.diff, first_piece
            )
            self.log(f"[BLACK_ONLY] Initial diff computed: {len(self.diff)} edges in diff dict")
        else:
            self.log("[BLACK_ONLY] Adding to existing diffs")
            self.diff = self.add_to_diffs(left_pieces)
            self.log(f"[BLACK_ONLY] Updated diff: {len(self.diff)} edges in diff dict")

        iteration = 0
        while len(left_pieces) > 0:
            iteration += 1
            self.log(
                f"[BLACK_ONLY] <--- Iteration {iteration} ---> pieces left: ",
                len(left_pieces),
                "extremum:",
                self.extremum,
            )

            # Log current connected pieces positions
            connected_positions = [(c, p) for c, p in self.connected_directions]
            self.log(f"[BLACK_ONLY] Connected pieces at positions: {connected_positions}")

            block_best_e, best_e = self.best_diff_black_only(
                self.diff, self.connected_directions, left_pieces
            )

            if block_best_e is None or best_e is None:
                self.log("[BLACK_ONLY] ERROR: No match found! Cannot continue solving.")
                break

            block_best_p, best_p = (
                self.edge_to_piece[block_best_e],
                self.edge_to_piece[best_e],
            )

            self.log(f"[BLACK_ONLY] Best match found: block edge type={block_best_e.type.name}, "
                    f"best edge type={best_e.type.name}, direction={block_best_e.direction.name}")

            stick_pieces(block_best_e, best_p, best_e, final_stick=True)
            self.log(f"[BLACK_ONLY] Pieces stuck together")

            self.update_direction(block_best_e, best_p, best_e)
            self.connect_piece(
                self.connected_directions, block_best_p, block_best_e.direction, best_p
            )

            connected_pieces.append(best_p)
            piece_index = left_pieces.index(best_p)
            del left_pieces[piece_index]
            self.log(f"[BLACK_ONLY] Piece {piece_index} connected. {len(left_pieces)} pieces remaining")

            self.diff = self.compute_diffs(
                left_pieces, self.diff, best_p, edge_connected=block_best_e
            )
            self.log(f"[BLACK_ONLY] Updated diff after connection: {len(self.diff)} edges")

            self.export_pieces(
                os.path.join(os.environ["ZOLVER_TEMP_DIR"], "stick{0:03d}.png".format(len(self.connected_directions))),
                os.path.join(os.environ["ZOLVER_TEMP_DIR"], "colored{0:03d}.png".format(len(self.connected_directions))),
                name_colored="Step {0:03d}".format(len(self.connected_directions)),
            )

        return connected_pieces

    def solve(self, connected_pieces, left_pieces):
        """
        Solve the puzzle by finding the optimal piece in left_pieces matching the edges
        available in connected_pieces

        :param connected_pieces: pieces already connected to the puzzle
        :param left_pieces: remaining pieces to place in the puzzle
        :param border: Boolean to determine if the strategy is border
        :return: List of connected pieces
        """

        if len(self.connected_directions) == 0:
            self.connected_directions = [
                ((0, 0), connected_pieces[0])
            ]  # ((x, y), p), x & y relative to the first piece, init with 1st piece
            self.diff = self.compute_diffs(
                left_pieces, self.diff, connected_pieces[0]
            )  # edge on the border of the block -> edge on a left piece -> diff between edges
        else:
            self.diff = self.add_to_diffs(left_pieces)

        while len(left_pieces) > 0:
            self.log(
                "<--- New match ---> pieces left: ",
                len(left_pieces),
                "extremum:",
                self.extremum,
                "puzzle dimension:",
                display_dim(self.possible_dim),
            )
            block_best_e, best_e = self.best_diff(
                self.diff, self.connected_directions, left_pieces
            )
            block_best_p, best_p = (
                self.edge_to_piece[block_best_e],
                self.edge_to_piece[best_e],
            )

            stick_pieces(block_best_e, best_p, best_e, final_stick=True)

            self.update_direction(block_best_e, best_p, best_e)
            self.connect_piece(
                self.connected_directions, block_best_p, block_best_e.direction, best_p
            )

            connected_pieces.append(best_p)
            del left_pieces[left_pieces.index(best_p)]

            self.diff = self.compute_diffs(
                left_pieces, self.diff, best_p, edge_connected=block_best_e
            )

            self.export_pieces(
                os.path.join(os.environ["ZOLVER_TEMP_DIR"], "stick{0:03d}.png".format(len(self.connected_directions))),
                os.path.join(os.environ["ZOLVER_TEMP_DIR"], "colored{0:03d}.png".format(len(self.connected_directions))),
                name_colored="Step {0:03d}".format(len(self.connected_directions)),
            )

        return connected_pieces

    def compute_diffs(self, left_pieces, diff, new_connected, edge_connected=None):
        """
        Compute the diff between the left pieces edges and the new_connected piece edges
        by sticking them and compute the distance

        :param left_pieces: remaining pieces to place in the puzzle
        :param diff: pre computed diff between edges to speed up the process
        :param new_connected: Connected pieces to test for a match
        :return: updated diff matrix
        """

        # Remove former edge from the bloc border
        if edge_connected is not None:
            del diff[edge_connected]

        # build the list of edge to test
        edges_to_test = [
            (piece, edge)
            for piece in left_pieces
            for edge in piece.edges_
            if not edge.connected
        ]

        # Remove the edge of the new piece from the bloc border diffs
        for e in new_connected.edges_:
            for _, v in diff.items():
                if e in v:
                    del v[e]

            if e.connected:
                continue

            diff_e = {}
            computed_count = 0
            for piece, edge in edges_to_test:
                if not e.is_compatible(edge):
                    continue
                for e2 in piece.edges_:
                    e2.backup_shape()
                stick_pieces(e, piece, edge)
                if self.black_only:
                    diff_score = shape_only_edge_compute(edge, e)
                    diff_e[edge] = diff_score
                    computed_count += 1
                    if computed_count <= 5:  # Log first few computations
                        self.log(f"[BLACK_ONLY] compute_diffs: edge pair score={diff_score:.4f}, "
                                f"types=({e.type.name}, {edge.type.name})")
                elif self.green_:
                    diff_e[edge] = real_edge_compute(edge, e)
                else:
                    diff_e[edge] = generated_edge_compute(edge, e)
                for e2 in piece.edges_:
                    e2.restore_backup_shape()

            if self.black_only:
                self.log(f"[BLACK_ONLY] compute_diffs: computed {computed_count} distances for edge {e.type.name}")

            diff[e] = diff_e
        return diff

    def fallback(self, diff, connected_direction, left_piece, strat=Strategy.NAIVE):
        """If a strategy does not work fallback to another one"""

        self.log(
            "Fail to solve the puzzle with", self.strategy, "falling back to", strat
        )
        old_strat = self.strategy
        self.strategy = Strategy.NAIVE
        best_bloc_e, best_e = self.best_diff(diff, connected_direction, left_piece)
        self.strategy = old_strat
        return best_bloc_e, best_e

    def best_diff_black_only(self, diff, connected_direction, left_piece):
        """
        Simplified best_diff for black-only puzzles.
        Uses simpler strategy since all pieces are border pieces.
        """
        self.log(f"[BLACK_ONLY] best_diff_black_only: searching for best match among {len(left_piece)} pieces")
        best_bloc_e, best_e, min_diff = None, None, float("inf")
        minX, minY, maxX, maxY = self.extremum
        self.log(f"[BLACK_ONLY] Search area: extremum=({minX}, {minY}, {maxX}, {maxY})")

        # Find positions with exactly one neighbor (border placement)
        best_coord = []
        for x in range(minX, maxX + 1):
            for y in range(minY, maxY + 1):
                neighbor = list(
                    filter(
                        lambda e: is_neighbor((x, y), e[0], connected_direction),
                        connected_direction,
                    )
                )
                if len(neighbor) == 1 or (len(neighbor) == 2 and len(left_piece) == 1):
                    best_coord.append(((x, y), neighbor[0]))

        self.log(f"[BLACK_ONLY] Found {len(best_coord)} candidate positions for placement")

        tested_combinations = 0
        for c, neighbor in best_coord:
            for p in left_piece:
                for rotation in range(4):
                    tested_combinations += 1
                    diff_score = 0
                    p.rotate_edges(1)
                    block_c, block_p = neighbor

                    direction_exposed = Directions(sub_tuple(c, block_c))
                    edge_exposed = block_p.edge_in_direction(direction_exposed)
                    edge = p.edge_in_direction(
                        get_opposite_direction(direction_exposed)
                    )

                    # Check if edges are compatible and not already connected
                    if (
                        edge_exposed.connected
                        or edge.connected
                        or not edge.is_compatible(edge_exposed)
                    ):
                        diff_score = float("inf")
                        if self.black_only and tested_combinations <= 10:  # Log first few failures
                            self.log(f"[BLACK_ONLY] Position {c}, rotation {rotation}: incompatible "
                                    f"(exposed_connected={edge_exposed.connected}, "
                                    f"edge_connected={edge.connected}, "
                                    f"compatible={edge.is_compatible(edge_exposed)}, "
                                    f"exposed_type={edge_exposed.type.name}, edge_type={edge.type.name})")
                    else:
                        if edge_exposed in diff and edge in diff[edge_exposed]:
                            diff_score = diff[edge_exposed][edge]
                            if tested_combinations <= 10:  # Log first few scores
                                self.log(f"[BLACK_ONLY] Position {c}, rotation {rotation}: "
                                        f"score={diff_score:.4f}, "
                                        f"edge_types=({edge_exposed.type.name}, {edge.type.name}), "
                                        f"direction={direction_exposed.name}")
                        else:
                            diff_score = float("inf")
                            if tested_combinations <= 10:
                                has_exposed = edge_exposed in diff
                                has_edge = has_exposed and edge in diff[edge_exposed] if has_exposed else False
                                self.log(f"[BLACK_ONLY] Position {c}, rotation {rotation}: "
                                        f"no diff computed (exposed_in_diff={has_exposed}, "
                                        f"edge_in_diff={has_edge}, "
                                        f"exposed_type={edge_exposed.type.name}, edge_type={edge.type.name})")

                    if diff_score < min_diff:
                        best_bloc_e, best_e, min_diff = (
                            edge_exposed,
                            edge,
                            diff_score,
                        )
                        self.log(f"[BLACK_ONLY] New best match: position {c}, rotation {rotation}, "
                                f"score={diff_score:.4f}")

        self.log(f"[BLACK_ONLY] Tested {tested_combinations} combinations")

        # Fallback to naive strategy if no match found
        if best_e is None:
            self.log("[BLACK_ONLY] No match found in candidate positions, falling back to naive strategy")
            for block_e, block_e_diff in diff.items():
                for e, diff_score in block_e_diff.items():
                    if diff_score < min_diff:
                        best_bloc_e, best_e, min_diff = block_e, e, diff_score
            if best_e is not None:
                self.log(f"[BLACK_ONLY] Fallback found match with score={min_diff:.4f}")
            else:
                self.log("[BLACK_ONLY] ERROR: No match found even with fallback!")

        return best_bloc_e, best_e

    def best_diff(self, diff, connected_direction, left_piece):
        """
        Find the best matching edge for a piece edge

        :param diff: pre computed diff between edges to speed up the process
        :param connected_direction: Direction of the edge to connect
        :param left_piece: Piece to connect
        :return: the best edge found in the bloc
        """

        best_bloc_e, best_e, _best_p, min_diff = None, None, None, float("inf")
        minX, minY, maxX, maxY = self.extremum

        if self.strategy == Strategy.FILL:
            best_coords = []

            # this is ugly
            for i in range(4, -1, -1):  # 4 to 0
                best_coord = []
                for x in range(minX, maxX + 1):
                    for y in range(minY, maxY + 1):
                        neighbor = list(
                            filter(
                                lambda e: is_neighbor(
                                    (x, y), e[0], connected_direction
                                ),
                                connected_direction,
                            )
                        )
                        if len(neighbor) == i:
                            best_coord.append(((x, y), neighbor))
                best_coords.append(best_coord)

            for best_coord in best_coords:
                for c, neighbor in best_coord:
                    for p in left_piece:
                        for rotation in range(4):
                            diff_score = 0
                            p.rotate_edges(1)
                            last_test = None, None
                            for block_c, block_p in neighbor:
                                direction_exposed = Directions(sub_tuple(c, block_c))
                                edge_exposed = block_p.edge_in_direction(
                                    direction_exposed
                                )
                                edge = p.edge_in_direction(
                                    get_opposite_direction(direction_exposed)
                                )
                                if (
                                    edge_exposed.connected
                                    or edge.connected
                                    or not edge.is_compatible(edge_exposed)
                                ):
                                    diff_score = float("inf")
                                    break
                                else:
                                    diff_score += diff[edge_exposed][edge]
                                    last_test = edge_exposed, edge
                            if diff_score < min_diff:
                                best_bloc_e, best_e, min_diff = (
                                    last_test[0],
                                    last_test[1],
                                    diff_score,
                                )
                if best_e is not None:
                    break
                elif len(best_coord):
                    self.log("Fall back to a worst", self.strategy)
            if best_e is None:
                best_bloc_e, best_e = self.fallback(
                    diff, connected_direction, left_piece
                )
            return best_bloc_e, best_e

        elif self.strategy == Strategy.BORDER:
            best_coord = []
            for x in range(minX, maxX + 1):
                for y in range(minY, maxY + 1):
                    neighbor = list(
                        filter(
                            lambda e: is_neighbor((x, y), e[0], connected_direction),
                            connected_direction,
                        )
                    )
                    if len(neighbor) == 1 or (
                        len(neighbor) == 2 and len(left_piece) == 1
                    ):
                        best_coord.append(((x, y), neighbor[0]))

            for c, neighbor in best_coord:
                for p in left_piece:
                    for rotation in range(4):
                        diff_score = 0
                        p.rotate_edges(1)
                        block_c, block_p = neighbor

                        direction_exposed = Directions(sub_tuple(c, block_c))
                        edge_exposed = block_p.edge_in_direction(direction_exposed)
                        edge = p.edge_in_direction(
                            get_opposite_direction(direction_exposed)
                        )

                        if p.type == TypePiece.ANGLE and (
                            not corner_puzzle_alignment(c, self.corner_pos)
                            or not self.corner_place_fit_size(c)
                        ):
                            diff_score = float("inf")
                        if p.type == TypePiece.BORDER and self.is_edge_at_corner_place(
                            c
                        ):
                            diff_score = float("inf")
                        if (
                            diff_score != 0
                            or edge_exposed.connected
                            or edge.connected
                            or not edge.is_compatible(edge_exposed)
                            or not p.is_border_aligned(block_p)
                        ):
                            diff_score = float("inf")
                        else:
                            diff_score = diff[edge_exposed][edge]

                        if diff_score < min_diff:
                            best_bloc_e, best_e, min_diff = (
                                edge_exposed,
                                edge,
                                diff_score,
                            )
            if best_e is None:
                best_bloc_e, best_e = self.fallback(
                    diff, connected_direction, left_piece, strat=Strategy.FILL
                )
            return best_bloc_e, best_e

        elif self.strategy == Strategy.NAIVE:
            for block_e, block_e_diff in diff.items():
                for e, diff_score in block_e_diff.items():
                    if diff_score < min_diff:
                        best_bloc_e, best_e, min_diff = block_e, e, diff_score
            return best_bloc_e, best_e
        return None, None

    def add_to_diffs(self, left_pieces):
        """build the list of edge to test"""

        if self.black_only:
            self.log(f"[BLACK_ONLY] add_to_diffs: updating diffs for {len(left_pieces)} left pieces")

        edges_to_test = [
            (piece, edge)
            for piece in left_pieces
            for edge in piece.edges_
            if not edge.connected
        ]

        if self.black_only:
            self.log(f"[BLACK_ONLY] add_to_diffs: {len(edges_to_test)} edges to test")

        computed_count = 0
        for e, diff_e in self.diff.items():
            for piece, edge in edges_to_test:
                if not e.is_compatible(edge):
                    continue
                for e2 in piece.edges_:
                    e2.backup_shape()
                stick_pieces(e, piece, edge)
                if self.black_only:
                    diff_score = shape_only_edge_compute(edge, e)
                    diff_e[edge] = diff_score
                    computed_count += 1
                    if computed_count <= 5:  # Log first few
                        self.log(f"[BLACK_ONLY] add_to_diffs: edge pair score={diff_score:.4f}")
                elif self.green_:
                    diff_e[edge] = real_edge_compute(edge, e)
                else:
                    diff_e[edge] = generated_edge_compute(edge, e)
                for e2 in piece.edges_:
                    e2.restore_backup_shape()

        if self.black_only:
            self.log(f"[BLACK_ONLY] add_to_diffs: computed {computed_count} new distances")

        return self.diff

    def update_direction(self, e, best_p, best_e):
        """Update the direction of the edge after matching it"""

        opp = get_opposite_direction(e.direction)
        step = step_direction(opp, best_e.direction)
        for edge in best_p.edges_:
            edge.direction = rotate_direction(edge.direction, step)

    def connect_piece(self, connected_directions, curr_p, dir, best_p):
        """
        Then we need to search the other pieces already in the puzzle that are going to be also connected:
        +--+--+--+
        |  | X| O|
        +--+--+--+
        |  | X| X|
        +--+--+--+
        |  |  |  |
        +--+--+--+

        For example if I am going to put a piece at the marker 'O' only one edge will be connected to the piece
        therefore we need to search the adjacent pieces and connect them properly
        """

        old_coord = list(filter(lambda x: x[1] == curr_p, connected_directions))[0][0]
        new_coord = add_tuple(old_coord, dir.value)

        for coord, p in connected_directions:
            for d in directions:
                if equals_tuple(coord, add_tuple(new_coord, d.value)):
                    for edge in best_p.edges_:
                        if edge.direction == d:
                            edge.connected = True
                            break
                    for edge in p.edges_:
                        if edge.direction == get_opposite_direction(d):
                            edge.connected = True
                            break
        connected_directions.append((new_coord, best_p))

        minX, minY, maxX, maxY = self.extremum
        coeff = [1, 1, 1, 1]
        for i, d in enumerate(directions):
            if best_p.edge_in_direction(d).connected:
                coeff[i] = 0
        self.extremum = (
            min(minX, new_coord[0] - coeff[3]),
            min(minY, new_coord[1] - coeff[2]),
            max(maxX, new_coord[0] + coeff[1]),
            max(maxY, new_coord[1] + coeff[0]),
        )

        if best_p.type == TypePiece.ANGLE:
            self.corner_place_fit_size(new_coord, update_dim=True)
            self.corner_pos.append((new_coord, best_p))
        else:
            self.update_dimension()

        best_p.coord = (new_coord[1], new_coord[0])
        if self.black_only:
            self.log(f"[BLACK_ONLY] Placed: {best_p.type.name} at {best_p.coord}, "
                    f"new_coord={new_coord}, extremum={self.extremum}")
        else:
            self.log("Placed:", best_p.type, "at", best_p.coord)

    def translate_puzzle(self):
        """Translate all pieces to the top left corner to be sure the puzzle is in the image"""

        minX = sys.maxsize
        minY = sys.maxsize
        for p in self.pieces_:
            for e in p.edges_:
                for pixel in e.shape:
                    if pixel[0] < minX:
                        minX = pixel[0]
                    if pixel[1] < minY:
                        minY = pixel[1]

        for p in self.pieces_:
            for e in p.edges_:
                for ip, _ in enumerate(e.shape):
                    e.shape[ip] += (-minX, -minY)

        for p in self.pieces_:
            p.translate(minX, minY)

    def export_pieces(
        self,
        path_contour,
        path_colored,
        name_contour=None,
        name_colored=None,
        display=True,
        display_border=False,
    ):
        """
        Export the contours and the colored image

        :param path_contour: Path used to export contours
        :param path_colored: Path used to export the colored image
        :return: the best edge found in the bloc
        """
        # Save images if:
        # 1. No viewer (command-line mode) - always save
        # 2. Viewer exists and display=True (GUI mode)
        if self.viewer and not display:
            return  # GUI mode but display=False, don't save
        # Otherwise, save the images

        minX, minY, maxX, maxY = self.get_bbox()
        colored_img = np.zeros((maxX - minX, maxY - minY, 3))
        border_img = np.zeros((maxX - minX, maxY - minY, 3))

        for piece in self.pieces_:
            # Reframe piece pixels to (0, 0)
            tmp = [
                (x - minX, y - minY, c)
                for (x, y), c in piece.pixels.items()
                if 0 <= x - minX < colored_img.shape[0]
                and 0 <= y - minY < colored_img.shape[1]
            ]
            x, y, c = (
                list(map(lambda e: int(e[0]), tmp)),
                list(map(lambda e: int(e[1]), tmp)),
                list(map(lambda e: e[2], tmp)),
            )
            colored_img[x, y] = c

            if display_border:
                # Contours
                for e in piece.edges_:
                    for y, x in e.shape:
                        y, x = y - minY, x - minX
                        if (
                            0 <= y < border_img.shape[1]
                            and 0 <= x < border_img.shape[0]
                        ):
                            rgb = (0, 0, 0)
                            if e.type == TypeEdge.HOLE:
                                rgb = (102, 178, 255)
                            if e.type == TypeEdge.HEAD:
                                rgb = (255, 255, 102)
                            if e.type == TypeEdge.UNDEFINED:
                                rgb = (255, 0, 0)
                            if e.connected:
                                rgb = (0, 255, 0)
                            border_img[x, y, 0] = rgb[2]
                            border_img[x, y, 1] = rgb[1]
                            border_img[x, y, 2] = rgb[0]
                cv2.imwrite(path_contour, border_img)
                if self.viewer:
                    self.viewer.addImage(name_contour, path_contour, display=False)
                elif self.black_only:
                    self.log(f"[BLACK_ONLY] Saved contour image: {path_contour}")

        cv2.imwrite(path_colored, colored_img)
        if self.viewer:
            self.viewer.addImage(name_colored, path_colored)
        elif self.black_only:
            self.log(f"[BLACK_ONLY] Saved colored image: {path_colored}")

    def compute_possible_size(self, nb_piece, nb_border) -> list[tuple]:
        """
        Compute all possible size of the puzzle based on the number
        of pieces and the number of border pieces
        """
        nb_edge_border = nb_border - 4
        nb_middle = nb_piece - nb_border
        possibilities = []
        for i in range(nb_edge_border // 2 + 1):
            w, h = i, (nb_edge_border // 2) - i
            if w * h == nb_middle:
                possibilities.append((w + 1, h + 1))
        self.log(
            "Possible sizes: (",
            nb_piece,
            "pieces with",
            nb_border,
            "borders among them):",
            display_dim(possibilities),
        )
        return possibilities

    def corner_place_fit_size(self, c, update_dim=False):
        """Update the possible dimensions of the puzzle when a corner is placed"""

        def almost_equals(idx, target, val):
            return val[idx] == target or val[idx] == -target

        # We have already picked a dimension
        if len(self.possible_dim) == 1:
            return (
                c[0] == 0
                or c[0] == self.possible_dim[0][0]
                or c[0] == -self.possible_dim[0][0]
            ) and (
                c[1] == 0
                or c[1] == self.possible_dim[0][1]
                or c[0] == -self.possible_dim[0][1]
            )

        if c[0] == 0:
            filtered = list(
                filter(lambda x: almost_equals(1, c[1], x), self.possible_dim)
            )
            if len(filtered):
                if update_dim and len(filtered) != len(self.possible_dim):
                    self.log(
                        "Update possible dimensions with corner place:",
                        display_dim(filtered),
                    )
                    self.possible_dim = filtered
                return True
            else:
                return False
        elif c[1] == 0:
            filtered = list(
                filter(lambda x: almost_equals(0, c[0], x), self.possible_dim)
            )
            if len(filtered):
                if update_dim and len(filtered) != len(self.possible_dim):
                    self.log(
                        "Update possible dimensions with corner place:",
                        display_dim(filtered),
                    )
                    self.possible_dim = filtered
                return True
            else:
                return False
        return False

    def is_edge_at_corner_place(self, c):
        """Determine of an edge is at a corner place"""

        if len(self.possible_dim) == 1:
            # We have already picked a dimension
            return (
                c[0] == 0
                or c[0] == self.possible_dim[0][0]
                or c[0] == -self.possible_dim[0][0]
            ) and (
                c[1] == 0
                or c[1] == self.possible_dim[0][1]
                or c[0] == -self.possible_dim[0][1]
            )
        return False

    def update_dimension(self):
        if len(self.possible_dim) == 1:
            return
        dims = []
        _, _, maxX, maxY = self.extremum
        for x, y in self.possible_dim:
            if maxX <= x and maxY <= y:
                dims.append((x, y))
        if len(dims) != len(self.possible_dim):
            self.log(
                "Update possible dimensions with extremum",
                self.extremum,
                ":",
                display_dim(dims),
            )
            self.possible_dim = dims
