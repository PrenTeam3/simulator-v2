"""Common utilities and data structures shared across solver modules."""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class PreparedPuzzleData:
    """Container for prepared puzzle data from the preparation phase.

    Attributes:
        analysis_data: Loaded analysis data from PuzzleDataLoader
        original_image: The image to work with (numpy array)
        all_segments: List of segments for each piece
        annotated_image: Initial visualization with all pieces and segments
        temp_folder: Path to the temp folder
        project_root: Path to the project root
    """
    analysis_data: Any
    original_image: Any  # numpy array
    all_segments: List[List[Any]]
    annotated_image: Any  # numpy array
    temp_folder: Any  # Path object
    project_root: Any  # Path object


@dataclass
class SolverResults:
    """Container for results from a solving algorithm.

    Attributes:
        matches: List of segment matches found
        piece1: First piece analyzed
        piece2: Second piece analyzed
        segments1: Segments from first piece
        segments2: Segments from second piece
        algorithm_name: Name of the algorithm used
        additional_data: Dictionary for algorithm-specific results
    """
    matches: List[Any]
    piece1: Any
    piece2: Any
    segments1: List[Any]
    segments2: List[Any]
    algorithm_name: str
    additional_data: Optional[Dict[str, Any]] = None


class SolverUtils:
    """Utility functions shared across solver modules."""

    @staticmethod
    def validate_piece_ids(piece_id_1, piece_id_2, num_pieces):
        """Validate that piece IDs are valid and different.

        Args:
            piece_id_1: First piece ID
            piece_id_2: Second piece ID
            num_pieces: Total number of pieces available

        Returns:
            tuple: (is_valid, error_message) where is_valid is bool and error_message is str or None
        """
        if piece_id_1 < 0 or piece_id_1 >= num_pieces:
            return False, f"Piece ID {piece_id_1} is out of range (0-{num_pieces-1})"

        if piece_id_2 < 0 or piece_id_2 >= num_pieces:
            return False, f"Piece ID {piece_id_2} is out of range (0-{num_pieces-1})"

        if piece_id_1 == piece_id_2:
            return False, "Piece IDs must be different"

        return True, None

    @staticmethod
    def print_section_header(title, width=70):
        """Print a formatted section header.

        Args:
            title: Title text to display
            width: Width of the header line
        """
        print("\n" + "="*width)
        print(title)
        print("="*width + "\n")

    @staticmethod
    def print_section_footer(title, width=70):
        """Print a formatted section footer.

        Args:
            title: Title text to display
            width: Width of the footer line
        """
        print("\n" + "="*width)
        print(title)
        print("="*width + "\n")
