from PyQt5.QtCore import QThread

from Puzzle.Puzzle import Puzzle


class SolveThread(QThread):
    """Main thread used to launch the puzzle solving"""

    def __init__(self, path, viewer, green_screen=False, black_only=False):
        QThread.__init__(self)
        self.path = path
        self.viewer = viewer
        self.green_screen = green_screen
        self.black_only = black_only

    def run(self):
        puzzle = Puzzle(self.path, self.viewer, self.green_screen, self.black_only)
        puzzle.solve_puzzle()
