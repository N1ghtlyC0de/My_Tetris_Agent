from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import mss


RUNTIME_CONFIG_PATH = Path("configs/runtime_config.json")


def save_runtime_config(runtime: Dict) -> None:
    RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUNTIME_CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(runtime, f, indent=2)


def _select_roi(title: str, frame_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    r = cv2.selectROI(title, frame_bgr, showCrosshair=True, fromCenter=False)
    x, y, w, h = [int(v) for v in r]
    if w <= 0 or h <= 0:
        raise RuntimeError(f"No ROI selected for {title}.")
    return x, y, w, h


def _abs_roi(x: int, y: int, w: int, h: int) -> Dict[str, int]:
    return {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}


def _rel_to_board(board_abs: Dict[str, int], roi_abs: Dict[str, int]) -> Dict[str, int]:
    return {
        "dx": int(roi_abs["left"] - board_abs["left"]),
        "dy": int(roi_abs["top"] - board_abs["top"]),
        "width": int(roi_abs["width"]),
        "height": int(roi_abs["height"]),
    }


def main() -> None:
    print("Open TETR.IO in your browser and keep all panels visible (board, next, hold).")
    print("Taking full-screen screenshot in 2 seconds...")
    cv2.waitKey(1)
    import time
    time.sleep(2)

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        shot = np.array(sct.grab(monitor), dtype=np.uint8)  # BGRA
        frame_bgr = shot[:, :, :3].copy()

    # Select ROIs from same fullscreen image
    bx, by, bw, bh = _select_roi("Select BOARD ROI (10x20 playfield)", frame_bgr)
    n1x, n1y, n1w, n1h = _select_roi("Select NEXT1 ROI (first next piece only)", frame_bgr)
    n2x, n2y, n2w, n2h = _select_roi("Select NEXT2 ROI (second next piece only)", frame_bgr)
    hx, hy, hw, hh = _select_roi("Select HOLD ROI (hold piece area)", frame_bgr)
    cv2.destroyAllWindows()

    board_abs = _abs_roi(bx, by, bw, bh)
    next1_abs = _abs_roi(n1x, n1y, n1w, n1h)
    next2_abs = _abs_roi(n2x, n2y, n2w, n2h)
    hold_abs = _abs_roi(hx, hy, hw, hh)

    runtime = {
        "board": {"roi": board_abs},
        "next": {
            "roi1_rel_to_board": _rel_to_board(board_abs, next1_abs),
            "roi2_rel_to_board": _rel_to_board(board_abs, next2_abs),
        },
        "hold": {
            "roi_rel_to_board": _rel_to_board(board_abs, hold_abs),
        },
    }

    save_runtime_config(runtime)
    print(f"Saved runtime config to {RUNTIME_CONFIG_PATH.as_posix()}")
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()