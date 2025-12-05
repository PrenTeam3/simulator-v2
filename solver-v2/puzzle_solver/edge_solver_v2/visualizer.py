"""Visualization module for edge solver v2 - main coordinator."""

from .visualizer_helpers import VisualizerHelpers
from .segment_visualizer import SegmentVisualizer
from .chain_visualizer import ChainVisualizer


class EdgeSolverV2Visualizer:
    """Main visualizer class that coordinates all visualization tasks."""

    # Delegate chain visualization methods to ChainVisualizer
    @staticmethod
    def visualize_chain(piece1, segments1, chain_segs1, piece2, segments2, chain_segs2, original_image):
        """Create a visualization showing chain on pieces, raw chains, and overlay."""
        return ChainVisualizer.visualize_chain(
            piece1, segments1, chain_segs1, piece2, segments2, chain_segs2, original_image
        )

    @staticmethod
    def visualize_progressive_chains(piece1, segments1, piece2, segments2, progressive_chains, original_image):
        """Create a visualization showing all progressive chain extensions vertically."""
        return ChainVisualizer.visualize_progressive_chains(
            piece1, segments1, piece2, segments2, progressive_chains, original_image
        )

    # Delegate segment visualization methods to SegmentVisualizer
    @staticmethod
    def visualize_segment_pairs(piece1, segments1, frame_segs1, piece2, segments2, frame_segs2, original_image):
        """Create a visualization showing all frame-adjacent segment pairs between two pieces."""
        return SegmentVisualizer.visualize_segment_pairs(
            piece1, segments1, frame_segs1, piece2, segments2, frame_segs2, original_image
        )

    # Expose helper methods for backward compatibility
    @staticmethod
    def _find_frame_touching_segment_ids(piece, all_segments):
        """Find segment IDs that directly touch frame corners."""
        return VisualizerHelpers._find_frame_touching_segment_ids(piece, all_segments)

    @staticmethod
    def _calculate_segment_arrow(segment, piece):
        """Calculate arrow direction for a segment."""
        return VisualizerHelpers._calculate_segment_arrow(segment, piece)

    @staticmethod
    def _draw_outward_arrow(image, segment_points, piece, offset_x, offset_y, color=(255, 0, 0)):
        """Draw an arrow pointing outward from the segment."""
        return VisualizerHelpers._draw_outward_arrow(image, segment_points, piece, offset_x, offset_y, color)

    @staticmethod
    def _draw_forbidden_zone(image, segment_points, piece, offset_x, offset_y):
        """Draw the forbidden zone (outside area) for a segment."""
        return VisualizerHelpers._draw_forbidden_zone(image, segment_points, piece, offset_x, offset_y)
