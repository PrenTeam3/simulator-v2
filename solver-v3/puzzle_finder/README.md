# Puzzle Finder

Install (from repo root):

- `python -m venv .venv`
- `.\.venv\Scripts\activate`
- `python -m pip install -r .\solver-v2\requirements.txt`

Run from repo root:

- `python .\solver-v2\puzzle_finder\a4_finder_static.py ".\solver-v2\puzzle_finder\bilder\test9.jpg"`
- `python .\solver-v2\puzzle_finder\__main__.py ".\solver-v2\puzzle_finder\bilder\test9.jpg"`
- `python .\solver-v2\puzzle_finder\a4_finder_border.py ".\solver-v2\puzzle_finder\bilder\test9.jpg"`
- `python .\solver-v2\puzzle_finder\convert_to_A4_cords.py --image-width 2340 --image-height 1630 --x 1170 --y 815`

Run with `-m` (only if cwd is `solver-v2`):

Static A4 crop test:

`python -m puzzle_finder.a4_finder_static ".\puzzle_finder\bilder\test9.jpg"`

Optional args (static):

- `--output-dir <path>`
- `--x <int>`
- `--y <int>`
- `--width <int>`
- `--height <int>`

Dynamic A4 finder test:

`python -m puzzle_finder ".\puzzle_finder\bilder\test9.jpg"`

Optional args (dynamic):

- `--output-dir <path>`
- `--no-debug`

Green-border A4 finder test:

`python -m puzzle_finder.a4_finder_border ".\puzzle_finder\bilder\test9.jpg"`

Optional args (green-border):

- `--output-dir <path>` (default: `solver-v2/puzzle_finder/output/border`)
- `--no-debug`

A4 Cords converter:

- `python -m puzzle_finder.convert_to_A4_cords --image-width 2340 --image-height 1630 --x 1170 --y 815`

python -m compileall "solver-v2/puzzle_finder"

python -m compileall "solver-v2/puzzle_finder/a4_finder_static.py" "solver-v2/puzzle_finder/__init__.py"

python -m compileall "solver-v2/puzzle_finder/a4_finder_static.py"

python -m compileall "solver-v2/puzzle_finder/convert_to_A4_cords.py" "solver-v2/puzzle_finder/__init__.py"
