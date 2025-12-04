"""Puzzle solver entry point."""

from puzzle_solver import solve_puzzle


if __name__ == "__main__":
    array_files = ['puzzle_analysis_20251120_102708', 'puzzle_analysis_20251120_102801','puzzle_analysis_20251120_102822']

    # solve_puzzle(array_files[1])

    #solve_puzzle(array_files[0], piece_id_1=0, piece_id_2=1)
    solve_puzzle(array_files[1], piece_id_1=0, piece_id_2=1)
    #solve_puzzle(array_files[2], piece_id_1=2, piece_id_2=3)
