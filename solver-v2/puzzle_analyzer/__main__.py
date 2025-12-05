"""Entry point for puzzle_analyzer."""

import sys
from puzzle_analyzer.core import analyze_puzzle


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m puzzle_analyzer <image_path> [output_image] [output_svg]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_image = sys.argv[2] if len(sys.argv) > 2 else "output.png"
    output_svg = sys.argv[3] if len(sys.argv) > 3 else "pieces.svg"

    analyze_puzzle(image_path, output_image, output_svg)
