from __future__ import annotations

import numpy as np

LINE_CLEAR_BONUS = [0.0, 1.0, 2.8, 5.0, 8.0]
FUTURE_DISCOUNT_FACTOR = 0.35
INVALID_FUTURE_PENALTY = -50.0

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
    max_h = np.max(h)
    holes = count_holes(board)
    bump = bumpiness(h)

    # Tuned heuristic:
    # - prioritize line clears strongly
    # - heavily penalize holes
    # - keep stack low and smooth
    # Non-linear reward for line clears: singles help, but doubles/triples/tetrises
    # are rewarded increasingly more to encourage setup plays.
    # `lines` is 0..4 for tetromino placement; clamp defensively for robustness.
    line_bonus = LINE_CLEAR_BONUS[max(0, min(int(lines), 4))]
    return (
        2.2 * line_bonus
        - 0.36 * agg_h
        - 1.25 * holes
        - 0.20 * bump
        - 0.45 * max_h
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


def _best_move_single(board: np.ndarray, piece: str = "T"):
    # safe fallback if classifier returns unknown
    if piece not in PIECES:
        piece = "T"

    shapes = PIECES[piece]
    best = None
    best_score = -float("inf")

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
            b2, lines = place(board, shape, r, c)
            s = score_board(b2, lines)
            if s > best_score:
                best_score = s
                best = (rot_idx, c, s)

    return best


def best_move(board: np.ndarray, piece: str = "T", next_piece: str | None = None):
    """
    Return the best move for `piece` as (rotation_index, column, score).

    If `next_piece` is provided and valid, applies a one-step lookahead:
      score = immediate_score + FUTURE_DISCOUNT_FACTOR * best_future_score
    where best_future_score is the best single-move score for `next_piece`
    on the board after the current candidate move.
    """
    # Keep original behavior when no lookahead piece is available
    if next_piece is None:
        return _best_move_single(board, piece)

    if piece not in PIECES:
        piece = "T"
    if next_piece not in PIECES:
        return _best_move_single(board, piece)

    best = None
    best_score = -float("inf")

    for rot_idx, shape in enumerate(PIECES[piece]):
        min_dc = min(dc for _, dc in shape)
        max_dc = max(dc for _, dc in shape)
        c_start = -min_dc
        c_end = board.shape[1] - 1 - max_dc

        for c in range(c_start, c_end + 1):
            r = drop_row(board, shape, c)
            if r is None:
                continue

            b2, lines = place(board, shape, r, c)
            immediate_score = score_board(b2, lines)

            # One-step lookahead with discount:
            # still optimize current move, but prefer boards that give a better next move.
            future = _best_move_single(b2, next_piece)
            future_score = future[2] if future is not None else INVALID_FUTURE_PENALTY
            s = immediate_score + FUTURE_DISCOUNT_FACTOR * future_score

            if s > best_score:
                best_score = s
                best = (rot_idx, c, s)

    return best
