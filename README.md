# AI Pathfinder

Interactive pathfinding visualization with multiple search algorithms, plus a script to generate assignment screenshots.

## Requirements

- Python 3.10+ (tested with Python 3.12)
- `pip`
- A GUI environment for interactive mode (`TkAgg` backend via Tkinter)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run The Interactive App

```bash
python pathfinder_matplotlib.py
```

## Run The Console Based App

```bash
python console_pathfinder.py
```

What you get:
- Click algorithm buttons: `BFS`, `DFS`, `UCS`, `DLS`, `IDDFS`, `Bidirectional`
- Load sample maps (`Sample 1`, `Sample 2`)
- Draw and erase walls, reset search, or clear the grid

## Generate Screenshots (Non-GUI)

```bash
python generate_screenshots.py
```

This writes PNG files to `screenshots/`.

## Notes

- If `python` does not point to your virtual environment, use:

```bash
.venv/bin/python generate_screenshots.py
.venv/bin/python pathfinder_matplotlib.py
```

- If interactive mode fails with a Tk backend error, install Tkinter support (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y python3-tk
```

