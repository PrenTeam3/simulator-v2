"""Combined entry point: Analyze + Solve puzzle in one run."""

from analyze import run_analysis
from solve import run_solver


def run_full_pipeline(image_path: str = "../images/puzzle.jpg",
                     solver_algorithm: str = 'edge_v2',
                     piece_id_1: int = None,
                     piece_id_2: int = None,
                     show_visualizations: bool = True,
                     debug_analysis: bool = False,
                     target_frame_corners: int = 4):
    """
    Complete pipeline: Analyze puzzle image, then solve it.

    Args:
        image_path: Path to puzzle image
        solver_algorithm: Algorithm to use ('matrix', 'edge', 'edge_v2')
        piece_id_1: First piece ID for comparison (optional)
        piece_id_2: Second piece ID for comparison (optional)
        show_visualizations: Whether to display visualizations
        debug_analysis: Enable debug mode for analysis phase
        target_frame_corners: Target number of frame corners to find

    Returns:
        tuple: (temp_folder_name, solver_results)
    """
    print("\n" + "="*70)
    print("FULL PIPELINE: ANALYZE + SOLVE")
    print("="*70 + "\n")

    # Phase 1: Analysis
    print(">>> PHASE 1: ANALYSIS")
    temp_folder_name = run_analysis(
        image_path=image_path,
        debug=debug_analysis,
        target_frame_corners=target_frame_corners
    )

    # Phase 2: Solving
    print("\n>>> PHASE 2: SOLVING")
    results = run_solver(
        temp_folder_name=temp_folder_name,
        piece_id_1=piece_id_1,
        piece_id_2=piece_id_2,
        solver_algorithm=solver_algorithm,
        show_visualizations=show_visualizations
    )

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)

    return temp_folder_name, results


if __name__ == "__main__":
    # Example 1: Full automatic pipeline
    temp_folder, results = run_full_pipeline(
        image_path="../images/puzzle3.jpg",
        solver_algorithm='edge_v2',
        show_visualizations=False
    )

    # Example 2: With specific piece comparison
    # temp_folder, results = run_full_pipeline(
    #     image_path="../images/puzzle.jpg",
    #     solver_algorithm='edge_v2',
    #     piece_id_1=1,
    #     piece_id_2=3,
    #     show_visualizations=False,
    #     debug_analysis=True
    # )

    # Example 3: Different image
    # temp_folder, results = run_full_pipeline(
    #     image_path="../images/puzzle5.png",
    #     solver_algorithm='matrix',
    #     target_frame_corners=4
    # )
