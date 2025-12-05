"""SVG vector graphics visualizer for puzzle piece analysis."""

import numpy as np
from typing import List, Tuple


class SVGVisualizer:
    """Create scalable vector graphics (SVG) visualizations of puzzle pieces."""

    @staticmethod
    def create_svg_from_contours(contours: List[np.ndarray], image_shape: Tuple[int, int, int],
                                filename: str = "contours.svg") -> str:
        """
        Create an SVG from detected contours (simple linear contours, no smoothing).

        Args:
            contours: List of OpenCV contours
            image_shape: Shape of original image (height, width, channels)
            filename: Output SVG filename

        Returns:
            SVG content as string
        """
        height, width = image_shape[0], image_shape[1]

        # Create SVG document
        svg_parts = []
        svg_parts.append('<?xml version="1.0" encoding="UTF-8"?>')
        svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
        svg_parts.append('<defs>')
        svg_parts.append('  <style>')
        svg_parts.append('    .contour { fill: none; stroke: #0088FF; stroke-width: 2; }')
        svg_parts.append('    .contour-filled { fill: #E8F4FF; opacity: 0.3; stroke: #0088FF; stroke-width: 2; }')
        svg_parts.append('    text { font-family: Arial, sans-serif; }')
        svg_parts.append('  </style>')
        svg_parts.append('</defs>')

        # Add white background
        svg_parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')

        # Draw contours
        for idx, contour in enumerate(contours):
            contour_flat = contour[:, 0, :]
            if len(contour_flat) < 3:
                continue

            # Create simple linear path (no smoothing)
            path_data = SVGVisualizer._contour_to_linear_path(contour_flat)
            svg_parts.append(f'<path class="contour-filled" d="{path_data}"/>')
            svg_parts.append(f'<path class="contour" d="{path_data}"/>')

            # Add contour ID label
            centroid = np.mean(contour_flat, axis=0)
            svg_parts.append(f'<text x="{centroid[0]:.1f}" y="{centroid[1]:.1f}" '
                           f'font-size="12" fill="black" text-anchor="middle" dominant-baseline="middle">'
                           f'Piece {idx}</text>')

        svg_parts.append('</svg>')

        svg_content = '\n'.join(svg_parts)

        # Save to file
        with open(filename, 'w') as f:
            f.write(svg_content)

        print(f"SVG saved to: {filename}")
        return svg_content

    @staticmethod
    def _contour_to_linear_path(contour_pts: np.ndarray) -> str:
        """
        Convert contour points to a simple linear SVG path (no smoothing).

        Args:
            contour_pts: Array of contour points

        Returns:
            SVG path data string
        """
        if len(contour_pts) < 2:
            return ""

        path_parts = []
        path_parts.append(f"M {contour_pts[0][0]:.1f} {contour_pts[0][1]:.1f}")

        for pt in contour_pts[1:]:
            path_parts.append(f"L {pt[0]:.1f} {pt[1]:.1f}")

        path_parts.append("Z")
        return " ".join(path_parts)
