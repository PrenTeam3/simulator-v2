"""SVG visualization of detected corners."""

import numpy as np
from typing import List, Tuple, Dict


class CornerVisualizer:
    """Creates SVG visualizations with corner markings."""

    @staticmethod
    def create_svg_with_corners(contours: List[np.ndarray], image_shape: Tuple[int, int, int],
                               corners_list: List[Dict], filename: str = "pieces_with_corners.svg") -> str:
        """
        Create SVG with contours and corner markings.

        Args:
            contours: List of OpenCV contours
            image_shape: Shape of original image (height, width, channels)
            corners_list: List of corner detection results for each contour
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
        svg_parts.append('    .outer-corner { fill: #FF6600; stroke: #FF3300; stroke-width: 2; }')
        svg_parts.append('    .inner-corner { fill: #00CC00; stroke: #00AA00; stroke-width: 2; }')
        svg_parts.append('    text { font-family: Arial, sans-serif; font-size: 10px; }')
        svg_parts.append('  </style>')
        svg_parts.append('</defs>')

        # Add white background
        svg_parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')

        # Draw contours and corners
        for idx, contour in enumerate(contours):
            contour_flat = contour[:, 0, :]
            if len(contour_flat) < 3:
                continue

            # Create simple linear path
            path_data = CornerVisualizer._contour_to_linear_path(contour_flat)
            svg_parts.append(f'<path class="contour-filled" d="{path_data}"/>')
            svg_parts.append(f'<path class="contour" d="{path_data}"/>')

            # Add corners for this piece if available
            if idx < len(corners_list) and corners_list[idx]:
                corner_info = corners_list[idx]
                outer_corners = corner_info.get('outer_corners', [])
                inner_corners = corner_info.get('inner_corners', [])

                # Draw outer corners (orange)
                for i, corner in enumerate(outer_corners):
                    svg_parts.append(
                        f'<circle cx="{corner[0]}" cy="{corner[1]}" r="6" class="outer-corner"/>'
                    )
                    svg_parts.append(
                        f'<text x="{corner[0] + 10}" y="{corner[1] - 10}" fill="#FF6600">O{i}</text>'
                    )

                # Draw inner corners (green)
                for i, corner in enumerate(inner_corners):
                    svg_parts.append(
                        f'<circle cx="{corner[0]}" cy="{corner[1]}" r="6" class="inner-corner"/>'
                    )
                    svg_parts.append(
                        f'<text x="{corner[0] + 10}" y="{corner[1] - 10}" fill="#00CC00">I{i}</text>'
                    )

            # Add piece label
            centroid = np.mean(contour_flat, axis=0)
            svg_parts.append(
                f'<text x="{centroid[0]:.1f}" y="{centroid[1]:.1f}" '
                f'font-size="14" font-weight="bold" fill="black" '
                f'text-anchor="middle" dominant-baseline="middle">Piece {idx}</text>'
            )

        # Add legend
        svg_parts.append('<g transform="translate(10, 20)">')
        svg_parts.append('<circle cx="10" cy="0" r="5" class="outer-corner"/>')
        svg_parts.append('<text x="25" y="5" fill="black">Outer Corners (Convex)</text>')
        svg_parts.append('</g>')

        svg_parts.append('<g transform="translate(10, 45)">')
        svg_parts.append('<circle cx="10" cy="0" r="5" class="inner-corner"/>')
        svg_parts.append('<text x="25" y="5" fill="black">Inner Corners (Concave)</text>')
        svg_parts.append('</g>')

        svg_parts.append('</svg>')

        svg_content = '\n'.join(svg_parts)

        # Save to file
        with open(filename, 'w') as f:
            f.write(svg_content)

        print(f"SVG with corners saved to: {filename}")
        return svg_content

    @staticmethod
    def _contour_to_linear_path(contour_pts: np.ndarray) -> str:
        """
        Convert contour points to a simple linear SVG path.

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
