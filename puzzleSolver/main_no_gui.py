import argparse
import atexit
import multiprocessing as mp
import os
import sys
import tempfile
from datetime import datetime

import matplotlib.pyplot as plt

from Puzzle.Puzzle import Puzzle

if __name__ == '__main__':
    # Create timestamped subfolder for this analysis run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_folder = f"analysis_{timestamp}"
    
    # Create temp folder if it doesn't exist
    temp_base = "temp"
    if not os.path.exists(temp_base):
        os.makedirs(temp_base)
    
    # Create timestamped subfolder inside temp
    analysis_path = os.path.join(temp_base, analysis_folder)
    os.makedirs(analysis_path, exist_ok=True)
    
    os.environ["ZOLVER_TEMP_DIR"] = analysis_path
    print(f"Images will be saved to: {os.path.abspath(analysis_path)}")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Solve Puzzles!")
    parser.add_argument(
        "-g", "--green_screen", help="enable green background removing", action="store_true"
    )
    parser.add_argument(
        "-b", "--black_only", help="enable black-only mode (max 6 pieces, all border)", action="store_true"
    )
    parser.add_argument("-p", "--profile", help="enable profiling", action="store_true")
    parser.add_argument("file", type=str, help="input_file")
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = fig_size[0] * 2
    plt.rcParams["figure.figsize"] = fig_size

    args = parser.parse_args()

    # Set multiprocessing start method (fork on Unix, spawn on Windows)
    if sys.platform == "win32":
        # Windows requires 'spawn' method
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass  # Already set, use existing method
    else:
        # Unix-like systems can use 'fork'
        try:
            mp.set_start_method("fork")
        except (RuntimeError, ValueError):
            # Already set or not available, use default
            pass

    if args.profile:
        import cProfile
        import pstats
        import io
        from pstats import SortKey

        with cProfile.Profile() as pr:
            puzzle = Puzzle(args.file, green_screen=args.green_screen, black_only=args.black_only)
            puzzle.solve_puzzle()

            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
            ps.print_stats(50)
            print(s.getvalue())
    else:
        Puzzle(args.file, green_screen=args.green_screen, black_only=args.black_only)
