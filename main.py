from __future__ import annotations

import time
from typing import Optional, Dict, Any

import keyboard
import numpy as np

from config import Keymap
from controller import Controller
from vision import Vision
from planner import best_move
from piece_classifier import classify_piece


# ------------------------------
# Helpers
# ------------------------------

def detect_queue_and_hold(vis: Vision) -> Dict[str, Optional[str]]:
    """
    Detect pieces from ROI captures:
      - next1
      - next2
      - hold
    """
    p_next1 = classify_piece(vis.grab_next1())
    p_next2 = classify_piece(vis.grab_next2())
    p_hold = classify_piece(vis.grab_hold())
    return {"next1": p_next1, "next2": p_next2, "hold": p_hold}


def estimate_spawn_col(piece: str) -> int:
    """
    Approx spawn column for 10-wide board (0-indexed).
    You can tune this per piece/system if needed.
    """
    # Good default for most pieces in modern clients
    return 4


def execute_move(
    ctrl: Controller,
    target_rot: int,
    target_col: int,
    piece: str,
    lock_with_hard_drop: bool = False,
) -> None:
    """
    Execute the selected move:
      1) rotate
      2) horizontal move
      3) optional hard drop
    """
    # Rotate
    if target_rot == 1:
        ctrl.rotate_cw()
    elif target_rot == 2:
        ctrl.rotate_180()
    elif target_rot == 3:
        ctrl.rotate_ccw()

    # Move to target column
    spawn_col = estimate_spawn_col(piece)
    dx = target_col - spawn_col
    if dx < 0:
        ctrl.move_left(-dx)
    elif dx > 0:
        ctrl.move_right(dx)

    if lock_with_hard_drop:
        ctrl.hard_drop()


def choose_piece_for_turn(
    detected: Dict[str, Optional[str]],
    allow_hold: bool = True,
) -> Dict[str, Any]:
    """
    Decide which piece to play this turn.
    Minimal policy:
      - If next1 exists, play next1.
      - If next1 missing and hold exists + allow_hold -> use hold.
      - Otherwise fallback piece 'T'.

    Returns:
      {
        "piece": str,
        "use_hold": bool
      }
    """
    n1 = detected.get("next1")
    h = detected.get("hold")

    if n1 is not None:
        return {"piece": n1, "use_hold": False}

    if allow_hold and h is not None:
        return {"piece": h, "use_hold": True}

    return {"piece": "T", "use_hold": False}


# ------------------------------
# Main loop
# ------------------------------

def main() -> None:
    vis = Vision()
    ctrl = Controller(Keymap())

    print("Bot ready.")
    print("Hotkeys:")
    print("  F8  -> Start/Pause")
    print("  F9  -> Toggle hold usage")
    print("  ESC -> Exit")

    running = False
    allow_hold = True
    last_toggle_time = 0.0

    while True:
        now = time.time()

        if keyboard.is_pressed("esc"):
            print("Exiting...")
            break

        if keyboard.is_pressed("f8") and (now - last_toggle_time > 0.25):
            running = not running
            last_toggle_time = now
            print(f"RUNNING = {running}")

        if keyboard.is_pressed("f9") and (now - last_toggle_time > 0.25):
            allow_hold = not allow_hold
            last_toggle_time = now
            print(f"ALLOW_HOLD = {allow_hold}")

        if not running:
            time.sleep(0.02)
            continue

        # 1) Capture board occupancy
        board_bgr = vis.grab_board()
        board_occ = vis.board_occupancy(board_bgr)  # expected shape (20,10)

        # Safety: ensure binary uint8 matrix
        board_occ = (board_occ > 0).astype(np.uint8)

        # 2) Detect next/hold pieces from ROIs
        detected = detect_queue_and_hold(vis)

        # 3) Decide current playable piece
        decision = choose_piece_for_turn(detected, allow_hold=allow_hold)
        piece = decision["piece"]
        use_hold = decision["use_hold"]

        # 4) Plan best move for selected piece
        mv = best_move(board_occ, piece=piece, next_piece=detected.get("next2"))

        # If no legal move, fallback with T
        if mv is None and piece != "T":
            mv = best_move(board_occ, piece="T")
            piece = "T"
            use_hold = False

        if mv is not None:
            target_rot, target_col, score = mv

            # If policy says use hold, press hold once before movement
            if use_hold and allow_hold:
                ctrl.hold()
                # tiny settle delay so game state updates
                time.sleep(0.012)

            execute_move(
                ctrl=ctrl,
                target_rot=target_rot,
                target_col=target_col,
                piece=piece,
                lock_with_hard_drop=False,
            )

            # Optional debug
            print(
                f"[MOVE] piece={piece} rot={target_rot} col={target_col} "
                f"score={score:.3f} next1={detected.get('next1')} "
                f"next2={detected.get('next2')} hold={detected.get('hold')}"
            )

        # Loop pacing
        time.sleep(0.025)


if __name__ == "__main__":
    main()
