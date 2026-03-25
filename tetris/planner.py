from __future__ import annotations
import numpy as np

PIECES = {
    "I": [
        [(0,0),(0,1),(0,2),(0,3)],
        [(0,0),(1,0),(2,0),(3,0)],
    ],
    "O": [
        [(0,0),(0,1),(1,0),(1,1)],
    ],
    "T": [
        [(0,1),(1,0),(1,1),(1,2)],
        [(0,0),(1,0),(1,1),(2,0)],
        [(0,0),(0,1),(0,2),(1,1)],
        [(0,1),(1,0),(1,1),(2,1)],
    ],
    "S": [
        [(0,1),(0,2),(1,0),(1,1)],
        [(0,0),(1,0),(1,1),(2,1)],
    ],
    "Z": [
        [(0,0),(0,1),(1,1),(1,2)],
        [(0,1),(1,0),(1,1),(2,0)],
    ],
    "J": [
        [(0,0),(1,0),(1,1),(1,2)],
        [(0,0),(0,1),(1,0),(2,0)],
        [(0,0),(0,1),(0,2),(1,2)],
        [(0,1),(1,1),(2,0),(2,1)],
    ],
    "L": [
        [(0,2),(1,0),(1,1),(1,2)],
        [(0,0),(1,0),(2,0),(2,1)],
        [(0,0),(0,1),(0,2),(1,0)],
        [(0,0),(0,1),(1,1),(2,1)],
    ],
}

def normalize(shape):
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return [(r - min_r, c - min_c) for r, c in shape]

def heights(board):
    rows, cols = board.shape
    out = np.zeros(cols, dtype=int)
    for c in range(cols):
        ys = np.where(board[:, c] == 1)[0]
        out[c] = 0 if len(ys) == 0 else rows - ys[0]
    return out

def holes(board):
    rows, cols = board.shape
    h = 0
    for c in range(cols):
        seen = False
        for r in range(rows):
            if board[r, c]:
                seen = True
            elif seen:
                h += 1
    return h

def bumpiness(h):
    return int(np.abs(np.diff(h)).sum())

def clear_lines(board):
    full = np.where(np.all(board == 1, axis=1))[0]
    n = len(full)
    if n == 0:
        return board, 0
    b = np.delete(board, full, axis=0)
    b = np.vstack([np.zeros((n, board.shape[1]), dtype=np.uint8), b])
    return b, n

def can_place(board, shape, r, c):
    rows, cols = board.shape
    for dr, dc in shape:
        rr, cc = r + dr, c + dc
        if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
            return False
        if board[rr, cc]:
            return False
    return True

def drop_row(board, shape, c):
    if not can_place(board, shape, 0, c):
        return None
    r = 0
    while can_place(board, shape, r + 1, c):
        r += 1
    return r

def place(board, shape, r, c):
    b = board.copy()
    for dr, dc in shape:
        b[r + dr, c + dc] = 1
    return clear_lines(b)

def score(board, lines):
    h = heights(board)
    agg_h = h.sum()
    hs = holes(board)
    bump = bumpiness(h)
    # Dellacherie-inspired weights: height penalised most to keep the board low,
    # which is the primary driver of survival and score maximisation.
    return (0.761 * lines) - (0.511 * agg_h) - (0.357 * hs) - (0.184 * bump)

def best_move(board, piece="T"):
    if piece not in PIECES:
        piece = "T"
    best = None
    best_s = -1e18

    for rot_idx, raw in enumerate(PIECES[piece]):
        shape = normalize(raw)
        max_dc = max(dc for _, dc in shape)
        for c in range(0, board.shape[1] - max_dc):
            r = drop_row(board, shape, c)
            if r is None:
                continue
            b2, lines = place(board, shape, r, c)
            s = score(b2, lines)
            if s > best_s:
                best_s = s
                best = (rot_idx, c, s)
    return best