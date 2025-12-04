"""Main puzzle solver function - orchestrates preparation and solving phases."""
from .preparation import PuzzlePreparer
from .matrix_solver import MatrixSolver
from .edge_solver import EdgeSolver
from .edge_solver_v2 import EdgeSolver as EdgeSolverV2
from .assembly_solver import AssemblySolver
from .assembly_solver.assembly_visualizer import AssemblyVisualizer
from .common import SolverUtils
import cv2


def solve_puzzle(temp_folder_name=None, piece_id_1=None, piece_id_2=None,
                 solver_algorithm='matrix', show_visualizations=True):
    """Load analysis data, segment contours, and solve using specified algorithm.

    Args:
        temp_folder_name: Optional folder name for analysis data
        piece_id_1: First puzzle piece ID for comparison (optional, defaults to piece 0)
        piece_id_2: Second puzzle piece ID for comparison (optional, defaults to piece 1)
        solver_algorithm: Algorithm to use for solving ('matrix', 'edge', 'edge_v2')
        show_visualizations: Whether to display visualization windows (default: True)
    """
    SolverUtils.print_section_header("PUZZLE SOLVER - Modular Architecture")

    # Phase 1: Prepare puzzle data
    prepared_data = PuzzlePreparer.prepare_puzzle_data(
        temp_folder_name=temp_folder_name,
        show_visualization=show_visualizations
    )

    if prepared_data is None:
        print("Error: Failed to prepare puzzle data")
        return

    # Phase 2: Solve using specified algorithm
    if solver_algorithm == 'matrix':
        results = MatrixSolver.solve_with_matrices(
            prepared_data=prepared_data,
            piece_id_1=piece_id_1,
            piece_id_2=piece_id_2,
            show_visualizations=show_visualizations
        )
    elif solver_algorithm == 'edge':
        results = EdgeSolver.solve_with_edges(
            prepared_data=prepared_data,
            piece_id_1=piece_id_1,
            piece_id_2=piece_id_2,
            show_visualizations=show_visualizations
        )
    elif solver_algorithm == 'edge_v2':
        results = EdgeSolverV2.solve_with_edges(
            prepared_data=prepared_data,
            show_visualizations=show_visualizations
        )

        # Phase 3: Assembly phase (only for edge_v2)
        if results is not None:
            assembly_results = AssemblySolver.assemble_puzzle(
                edge_solver_results=results,
                prepared_data=prepared_data,
                show_visualizations=show_visualizations
            )

            # Generate visualizations for completed steps
            if assembly_results and assembly_results['assembly_steps']:
                step_images = []

                for step in assembly_results['assembly_steps']:
                    if step.step_number == 1:
                        step_img = AssemblyVisualizer.visualize_step1_centroids(
                            step.visualization_data
                        )
                        step_images.append(step_img)
                    elif step.step_number == 2:
                        step_img = AssemblyVisualizer.visualize_step2_orientation(
                            step.visualization_data
                        )
                        step_images.append(step_img)
                    elif step.step_number == 3:
                        step_img = AssemblyVisualizer.visualize_step3_anchor_selection(
                            step.visualization_data
                        )
                        step_images.append(step_img)
                    elif step.step_number == 4:
                        step_img = AssemblyVisualizer.visualize_step4_anchor_placement(
                            step.visualization_data
                        )
                        step_images.append(step_img)
                    elif step.step_number == 5:
                        step_img = AssemblyVisualizer.visualize_step5_second_piece(
                            step.visualization_data
                        )
                        step_images.append(step_img)
                    elif step.step_number == 6:
                        step_img = AssemblyVisualizer.visualize_step6_third_piece(
                            step.visualization_data
                        )
                        step_images.append(step_img)
                    elif step.step_number == 7:
                        step_img = AssemblyVisualizer.visualize_step7_fourth_piece(
                            step.visualization_data
                        )
                        step_images.append(step_img)
                    elif step.step_number == 8:
                        step_img = AssemblyVisualizer.visualize_step8_final_assembly(
                            step.visualization_data
                        )
                        step_images.append(step_img)

                # Create combined visualization
                if step_images:
                    combined_img = AssemblyVisualizer.create_combined_visualization(step_images)

                    # Save the combined image to analysis temp folder
                    output_path = prepared_data['temp_folder'] / 'assembly_steps_combined.png'
                    cv2.imwrite(str(output_path), combined_img)
                    print(f"\n[OK] Assembly visualization saved to: {output_path}")

                    # Display if requested
                    if show_visualizations:
                        cv2.imshow("Assembly Steps", combined_img)
                        print("\nPress any key to close...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

            # Add assembly results to the main results
            results['assembly_results'] = assembly_results
    else:
        print(f"Error: Unknown solver algorithm '{solver_algorithm}'")
        print("Available algorithms: 'matrix', 'edge', 'edge_v2'")
        return

    if results is None:
        print("Error: Solving phase failed")
        return

    SolverUtils.print_section_footer("Puzzle solving complete")

    return results

