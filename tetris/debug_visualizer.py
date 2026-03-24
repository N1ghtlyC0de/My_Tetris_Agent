from __future__ import annotations

import cv2
import numpy as np

from vision import Vision
from piece_classifier import classify_piece


def draw_board_grid(img: np.ndarray, rows: int = 20, cols: int = 10) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for c in range(cols + 1):
        x = int(c * w / cols)
        cv2.line(out, (x, 0), (x, h), (80, 80, 80), 1, cv2.LINE_AA)
    for r in range(rows + 1):
        y = int(r * h / rows)
        cv2.line(out, (0, y), (w, y), (80, 80, 80), 1, cv2.LINE_AA)
    return out


def draw_occupancy_overlay(board_bgr: np.ndarray, occ: np.ndarray) -> np.ndarray:
    out = board_bgr.copy()
    h, w = out.shape[:2]
    rows, cols = occ.shape

    overlay = out.copy()
    for r in range(rows):
        for c in range(cols):
            if occ[r, c] == 1:
                x0 = int(c * w / cols)
                x1 = int((c + 1) * w / cols)
                y0 = int(r * h / rows)
                y1 = int((r + 1) * h / rows)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 200, 0), -1)

    out = cv2.addWeighted(overlay, 0.30, out, 0.70, 0.0)
    out = draw_board_grid(out, rows=rows, cols=cols)
    return out


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target_h / max(h, 1)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def pad_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_w:
        return img
    pad = np.zeros((h, target_w - w, 3), dtype=img.dtype)
    return np.hstack([img, pad])


def add_header(canvas: np.ndarray, text: str) -> np.ndarray:
    header_h = 36
    header = np.zeros((header_h, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([header, canvas])


def main() -> None:
    vis = Vision()
    print("Debug visualizer running. Press Q to quit.")

    while True:
        board = vis.grab_board()
        next1 = vis.grab_next1()
        next2 = vis.grab_next2()
        hold = vis.grab_hold()

        occ = vis.board_occupancy(board)

        p1 = classify_piece(next1)
        p2 = classify_piece(next2)
        ph = classify_piece(hold)

        board_dbg = put_label(draw_occupancy_overlay(board, occ), "BOARD + occupancy overlay")
        next1_dbg = put_label(next1, f"NEXT1 detected: {p1}")
        next2_dbg = put_label(next2, f"NEXT2 detected: {p2}")
        hold_dbg = put_label(hold, f"HOLD detected: {ph}")

        tile_h = 160
        n1r = resize_to_height(next1_dbg, tile_h)
        n2r = resize_to_height(next2_dbg, tile_h)
        hr = resize_to_height(hold_dbg, tile_h)

        max_w = max(n1r.shape[1], n2r.shape[1], hr.shape[1])
        n1r = pad_to_width(n1r, max_w)
        n2r = pad_to_width(n2r, max_w)
        hr = pad_to_width(hr, max_w)
        right_col = np.vstack([n1r, n2r, hr])

        target_h = right_col.shape[0]
        board_resized = resize_to_height(board_dbg, target_h)

        pad = np.zeros((target_h, 12, 3), dtype=np.uint8)
        canvas = np.hstack([board_resized, pad, right_col])

        cv2.putText(
            canvas,
            f"Occupied cells: {int(occ.sum())}",
            (10, canvas.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Global detection line (easy to read)
        summary = f"Detected -> NEXT1: {p1} | NEXT2: {p2} | HOLD: {ph}"
        canvas = add_header(canvas, summary)

        cv2.imshow("TETR.IO Bot Debug", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()