from puzzle_analyzer import analyze_puzzle_pieces


if __name__ == "__main__":
    image_path = "../images/puzzle.jpg"
    #image_path = "images/puzzle5.png"

    show_debug_zones = False
    verbose_logging = False

    analyze_puzzle_pieces(image_path, debug_visualization=show_debug_zones, verbose_logging=verbose_logging)

