from __future__ import annotations

import numpy as np

# All tetrominoes with rotation states as (row, col) offsets
PIECES = {
    "I": [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
    ],
    "O": [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    "T": [
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (1, 2), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
    ],
    "S": [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
    "Z": [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],
    "J": [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
    ],
    "L": [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (1, 2), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
}


def column_heights(board: np.ndarray) -> np.ndarray:
    rows, cols = board.shape
    h = np.zeros(cols, dtype=int)
    for c in range(cols):
        ys = np.where(board[:, c] == 1)[0]
        h[c] = rows - ys[0] if len(ys) else 0
    return h


def count_holes(board: np.ndarray) -> int:
    rows, cols = board.shape
    holes = 0
    for c in range(cols):
        filled_seen = False
        for r in range(rows):
            if board[r, c] == 1:
                filled_seen = True
            elif filled_seen:
                holes += 1
    return int(holes)


def bumpiness(heights: np.ndarray) -> int:
    return int(np.sum(np.abs(np.diff(heights))))


def clear_lines(board: np.ndarray):
    full = np.where(np.all(board == 1, axis=1))[0]
    n = len(full)
    if n == 0:
        return board, 0
    new = np.delete(board, full, axis=0)
    zeros = np.zeros((n, board.shape[1]), dtype=np.uint8)
    new = np.vstack((zeros, new))
    return new, int(n)


def score_board(board: np.ndarray, lines: int) -> float:
    h = column_heights(board)
    agg_h = np.sum(h)
    holes = count_holes(board)
    bump = bumpiness(h)

    # Heuristic weights (classic baseline)
    return (
        0.76 * lines
        - 0.51 * agg_h
        - 0.36 * holes
        - 0.18 * bump
    )


def placeable(board: np.ndarray, shape, r: int, c: int) -> bool:
    rows, cols = board.shape
    for dr, dc in shape:
        rr, cc = r + dr, c + dc
        if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
            return False
        if board[rr, cc] == 1:
            return False
    return True


def drop_row(board: np.ndarray, shape, c: int):
    if not placeable(board, shape, 0, c):
        return None
    r = 0
    while placeable(board, shape, r + 1, c):
        r += 1
    return r


def place(board: np.ndarray, shape, r: int, c: int):
    b = board.copy()
    for dr, dc in shape:
        b[r + dr, c + dc] = 1
    b, lines = clear_lines(b)
    return b, lines


def _line_clear_bonus(lines: int, combo: int, b2b: int):
    """
    Lightweight heuristic inspired by common Tetris scoring:
    - Reward back-to-back Tetrises.
    - Keep combo momentum.
    - Penalize losing b2b streaks.
    """
    extra_score = 0
    new_combo = combo
    new_b2b = b2b

    if lines > 0:
        new_combo = combo + 1
        if lines >= 4:
            # Strong bonus for multi-line clears, with extra weight for b2b streaks.
            extra_score += (400 + 100 * (lines - 4)) + (200 if b2b else 0)
            new_b2b = b2b + 1
        else:
            extra_score += 70 * lines
            new_b2b = 0

        # Combo momentum
        if new_combo > 2:
            extra_score += 20
        if new_combo > 6:
            extra_score += 20
        if new_combo > 17:
            extra_score += 30
    else:
        new_combo = 0
        # Only penalize breaking a b2b streak when no lines are cleared.
        if b2b > 0:
            extra_score -= 500

    return extra_score, new_combo, new_b2b


def best_move(board: np.ndarray, piece: str = "T", *, next_piece: str | None = None, combo: int = 0, b2b: int = 0):
    # safe fallback if classifier returns unknown
    if piece not in PIECES:
        piece = "T"

    shapes = PIECES[piece]
    best = None
    best_score = -1e18

    for rot_idx, shape in enumerate(shapes):
        min_dc = min(dc for _, dc in shape)
        max_dc = max(dc for _, dc in shape)

        # columns where shape fits horizontally
        c_start = -min_dc
        c_end = board.shape[1] - 1 - max_dc

        for c in range(c_start, c_end + 1):
            r = drop_row(board, shape, c)
            if r is None:
                continue
            placed_board, lines = place(board, shape, r, c)
            extra_score, new_combo, new_b2b = _line_clear_bonus(lines, combo, b2b)
            move_score = score_board(placed_board, lines) + extra_score

            # Optional one-piece lookahead to favor setups for the queue.
            # This keeps depth at 1 to contain search cost.
            if next_piece is not None:
                future = best_move(
                    placed_board,
                    piece=next_piece,
                    next_piece=None,
                    combo=new_combo,
                    b2b=new_b2b,
                )
                if future is not None:
                    move_score += future[2]

            if move_score > best_score:
                best_score = move_score
                best = (rot_idx, c, move_score)

    return best
