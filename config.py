from dataclasses import dataclass

@dataclass
class ROI:
    left: int
    top: int
    width: int
    height: int

@dataclass
class Keymap:
    left: str = "a"
    right: str = "d"
    soft_drop: str = "w"
    hard_drop: str = "s"
    rot_ccw: str = "left"
    rot_cw: str = "right"
    rot_180: str = "up"
    hold: str = "shift"

# Debes calibrarlo con calibrate.py
BOARD_ROI = ROI(left=700, top=180, width=300, height=600)

# Tablero Tetris estándar visible
BOARD_COLS = 10
BOARD_ROWS = 20