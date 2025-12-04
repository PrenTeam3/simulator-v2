"""
Puzzle analyzer package.

This package provides high-level analysis functionality for puzzle images:
- Automatic background detection (light/dark)
- Puzzle piece segmentation
- Comprehensive piece analysis (corners, edges, border detection)
- Visualization of results
"""

from puzzle_analyzer.core import analyze_puzzle_pieces

__all__ = ['analyze_puzzle_pieces']
