# from __future__ import annotations

# import time
# from typing import Optional, Dict, Any

# import keyboard
# import numpy as np

# from config import Keymap
# from controller import Controller
# from vision import Vision
# from planner import best_move
# from piece_classifier import classify_piece


# # ------------------------------
# # Helpers
# # ------------------------------

# def detect_queue_and_hold(vis: Vision) -> Dict[str, Optional[str]]:
#     """
#     Detect pieces from ROI captures:
#       - next1
#       - next2
#       - hold
#     """
#     p_next1 = classify_piece(vis.grab_next1())
#     p_next2 = classify_piece(vis.grab_next2())
#     p_hold = classify_piece(vis.grab_hold())
#     return {"next1": p_next1, "next2": p_next2, "hold": p_hold}


# def estimate_spawn_col(piece: str) -> int:
#     """
#     Approx spawn column for 10-wide board (0-indexed).
#     You can tune this per piece/system if needed.
#     """
#     # Good default for most pieces in modern clients
#     return 4


# def execute_move(
#     ctrl: Controller,
#     target_rot: int,
#     target_col: int,
#     piece: str,
#     lock_with_hard_drop: bool = True,
# ) -> None:
#     """
#     Execute the selected move:
#       1) rotate
#       2) horizontal move
#       3) hard drop
#     """
#     # Rotate
#     if target_rot == 1:
#         ctrl.rotate_cw()
#     elif target_rot == 2:
#         ctrl.rotate_180()
#     elif target_rot == 3:
#         ctrl.rotate_ccw()

#     # Move to target column
#     spawn_col = estimate_spawn_col(piece)
#     dx = target_col - spawn_col
#     if dx < 0:
#         ctrl.move_left(-dx)
#     elif dx > 0:
#         ctrl.move_right(dx)

#     if lock_with_hard_drop:
#         ctrl.hard_drop()


# def choose_piece_for_turn(
#     detected: Dict[str, Optional[str]],
#     allow_hold: bool = True,
# ) -> Dict[str, Any]:
#     """
#     Decide which piece to play this turn.
#     Minimal policy:
#       - If next1 exists, play next1.
#       - If next1 missing and hold exists + allow_hold -> use hold.
#       - Otherwise fallback piece 'T'.

#     Returns:
#       {
#         "piece": str,
#         "use_hold": bool
#       }
#     """
#     n1 = detected.get("next1")
#     h = detected.get("hold")

#     if n1 is not None:
#         return {"piece": n1, "use_hold": False}

#     if allow_hold and h is not None:
#         return {"piece": h, "use_hold": True}

#     return {"piece": "T", "use_hold": False}


# # ------------------------------
# # Main loop
# # ------------------------------

# def main() -> None:
#     vis = Vision()
#     ctrl = Controller(Keymap())

#     print("Bot ready.")
#     print("Hotkeys:")
#     print("  F8  -> Start/Pause")
#     print("  F9  -> Toggle hold usage")
#     print("  ESC -> Exit")

#     running = False
#     allow_hold = True
#     last_toggle_time = 0.0

#     while True:
#         now = time.time()

#         if keyboard.is_pressed("esc"):
#             print("Exiting...")
#             break

#         if keyboard.is_pressed("f8") and (now - last_toggle_time > 0.25):
#             running = not running
#             last_toggle_time = now
#             print(f"RUNNING = {running}")

#         if keyboard.is_pressed("f9") and (now - last_toggle_time > 0.25):
#             allow_hold = not allow_hold
#             last_toggle_time = now
#             print(f"ALLOW_HOLD = {allow_hold}")

#         if not running:
#             time.sleep(0.02)
#             continue

#         # 1) Capture board occupancy
#         board_bgr = vis.grab_board()
#         board_occ = vis.board_occupancy(board_bgr)  # expected shape (20,10)

#         # Safety: ensure binary uint8 matrix
#         board_occ = (board_occ > 0).astype(np.uint8)

#         # 2) Detect next/hold pieces from ROIs
#         detected = detect_queue_and_hold(vis)

#         # 3) Decide current playable piece
#         decision = choose_piece_for_turn(detected, allow_hold=allow_hold)
#         piece = decision["piece"]
#         use_hold = decision["use_hold"]

#         # 4) Plan best move for selected piece
#         mv = best_move(board_occ, piece=piece)

#         # If no legal move, fallback with T
#         if mv is None and piece != "T":
#             mv = best_move(board_occ, piece="T")
#             piece = "T"
#             use_hold = False

#         if mv is not None:
#             target_rot, target_col, score = mv

#             # If policy says use hold, press hold once before movement
#             if use_hold and allow_hold:
#                 ctrl.hold()
#                 # tiny settle delay so game state updates
#                 time.sleep(0.012)

#             execute_move(
#                 ctrl=ctrl,
#                 target_rot=target_rot,
#                 target_col=target_col,
#                 piece=piece,
#                 lock_with_hard_drop=True,
#             )

#             # Optional debug
#             print(
#                 f"[MOVE] piece={piece} rot={target_rot} col={target_col} "
#                 f"score={score:.3f} next1={detected.get('next1')} "
#                 f"next2={detected.get('next2')} hold={detected.get('hold')}"
#             )

#         # Loop pacing
#         time.sleep(0.025)


# if __name__ == "__main__":
#     main()

# from __future__ import annotations
# import time
# import keyboard
# import numpy as np

# from config import Keymap
# from controller import Controller
# from vision import Vision
# from planner import best_move
# from piece_classifier import classify_piece


# def detect_queue_and_hold(vis: Vision):
#     return {
#         "next1": classify_piece(vis.grab_next1()),
#         "next2": classify_piece(vis.grab_next2()),
#         "hold": classify_piece(vis.grab_hold()),
#     }


# def safe_piece(p):
#     return p if p in {"I","J","L","O","S","T","Z"} else "T"


# def execute_move_safe(ctrl: Controller, target_rot: int, target_col: int):
#     1) Normalize position: slam left so we know exact reference
#     ctrl.move_left(10)
#     time.sleep(0.01)

#     2) Rotate using CW only for consistency
#     for _ in range(target_rot % 4):
#         ctrl.rotate_cw()
#         time.sleep(0.008)

#     3) Move from col 0 to target
#     if target_col > 0:
#         ctrl.move_right(target_col)
#         time.sleep(0.008)

#     4) Drop
#     ctrl.hard_drop()


# def main():
#     vis = Vision()
#     ctrl = Controller(Keymap())

#     print("Bot ready.")
#     print("F8 start/pause | ESC exit")

#     running = False
#     debounce = 0.0

#     while True:
#         now = time.time()

#         if keyboard.is_pressed("esc"):
#             print("Exiting...")
#             break

#         if keyboard.is_pressed("f8") and now - debounce > 0.25:
#             running = not running
#             debounce = now
#             print("RUNNING =", running)

#         if not running:
#             time.sleep(0.02)
#             continue

#         board = vis.grab_board()
#         occ = (vis.board_occupancy(board) > 0).astype(np.uint8)

#         Remove top spawn rows noise
#         plan_board = occ.copy()
#         plan_board[:4, :] = 0

#         pieces = detect_queue_and_hold(vis)
#         piece = safe_piece(pieces["next1"])

#         mv = best_move(plan_board, piece=piece)
#         if mv is None:
#             piece = "T"
#             mv = best_move(plan_board, piece=piece)

#         if mv is not None:
#             rot, col, _score = mv
#             execute_move_safe(ctrl, rot, col)

#         time.sleep(0.03)


# if __name__ == "__main__":
#     main()

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


VALID_PIECES = {"I", "J", "L", "O", "S", "T", "Z"}


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


def safe_piece(p: Optional[str]) -> str:
    return p if p in VALID_PIECES else "T"


def choose_piece_for_turn(
    detected: Dict[str, Optional[str]],
    allow_hold: bool = True,
) -> Dict[str, Any]:
    """
    Conservative selection policy:
      - Prefer NEXT1 if valid.
      - If NEXT1 invalid and HOLD valid and allow_hold=True, use hold.
      - Else fallback T.
    """
    n1 = detected.get("next1")
    h = detected.get("hold")

    if n1 in VALID_PIECES:
        return {"piece": n1, "use_hold": False}

    if allow_hold and h in VALID_PIECES:
        return {"piece": h, "use_hold": True}

    return {"piece": "T", "use_hold": False}


def execute_move_safe(ctrl: Controller, target_rot: int, target_col: int) -> None:
    """
    Stable execution strategy:
      1) rotate using CW only at spawn position (before hitting any wall)
      2) slam left to normalize x-position after rotation
      3) move right to target_col
      4) hard drop

    Rotation must happen before slamming left so that wall-kick offsets
    (applied when rotating against the left wall) do not corrupt the column
    reference.  After rotation the slam guarantees the leftmost cell is at
    column 0, and then moving right target_col steps places it correctly.
    """
    # 1) rotation first (CW-only, at spawn where there is room)
    for _ in range(target_rot % 4):
        ctrl.rotate_cw()
        time.sleep(0.008)

    # 2) normalize horizontal position after rotation
    ctrl.move_left(10)
    time.sleep(0.010)

    # 3) move to desired column from 0
    if target_col > 0:
        ctrl.move_right(target_col)
        time.sleep(0.008)

    # 4) lock piece
    ctrl.hard_drop()


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
        board_occ = vis.board_occupancy(board_bgr)
        board_occ = (board_occ > 0).astype(np.uint8)

        # Ignore top spawn rows to reduce moving-piece noise
        plan_board = board_occ.copy()
        plan_board[:4, :] = 0

        # 2) Detect queue/hold
        detected = detect_queue_and_hold(vis)

        # 3) Choose piece
        decision = choose_piece_for_turn(detected, allow_hold=allow_hold)
        piece = safe_piece(decision["piece"])
        use_hold = bool(decision["use_hold"])

        # 4) Plan
        mv = best_move(plan_board, piece=piece)

        # Fallback if no legal move
        if mv is None and piece != "T":
            piece = "T"
            use_hold = False
            mv = best_move(plan_board, piece=piece)

        if mv is not None:
            rot, col, score = mv

            # Optional hold action before move
            if use_hold and allow_hold:
                ctrl.hold()
                time.sleep(0.012)

            execute_move_safe(ctrl, rot, col)

            print(
                f"[MOVE] piece={piece} rot={rot} col={col} score={score:.3f} "
                f"next1={detected.get('next1')} next2={detected.get('next2')} "
                f"hold={detected.get('hold')} use_hold={use_hold}"
            )

        # Loop pacing
        time.sleep(0.03)


if __name__ == "__main__":
    main()