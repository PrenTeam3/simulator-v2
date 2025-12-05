"""Entry point for puzzle_analyzer analysis."""

from puzzle_analyzer.core import analyze_puzzle
from pathlib import Path


def run_analysis(image_path: str = "../images/puzzle.jpg",
                 debug: bool = True,
                 target_frame_corners: int = 4) -> str:
    """
    Run puzzle analysis and return the temp folder name.

    Args:
        image_path: Path to puzzle image
        debug: Enable debug logging
        target_frame_corners: Target number of frame corners

    Returns:
        str: Name of the temp folder (e.g., 'analysis_20251120_100049')
    """
    # Analyze the puzzle image
    # Results will be saved to temp/analysis_TIMESTAMP/ directory
    analyzer = analyze_puzzle(
        image_path=image_path,
        debug=debug,
        target_frame_corners=target_frame_corners
    )

    # Extract and return just the folder name (not full path)
    temp_folder_name = Path(analyzer.temp_dir).name

    print(f"\n{'='*70}")
    print(f"Analysis complete! Results saved to: {temp_folder_name}")
    print(f"{'='*70}\n")

    return temp_folder_name


if __name__ == "__main__":
    # When run directly, just execute analysis
    temp_folder = run_analysis(
        image_path="../images/puzzle.jpg",
        debug=True,
        target_frame_corners=4
    )
    print(f"You can now run solve.py with folder: {temp_folder}")
