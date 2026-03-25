from __future__ import annotations

from typing import Dict


def rel_to_abs(board_abs: Dict[str, int], rel: Dict[str, int]) -> Dict[str, int]:
    return {
        "left": int(board_abs["left"] + rel["dx"]),
        "top": int(board_abs["top"] + rel["dy"]),
        "width": int(rel["width"]),
        "height": int(rel["height"]),
    }


def get_all_abs_rois(runtime_cfg: Dict) -> Dict[str, Dict[str, int]]:
    board_abs = runtime_cfg["board"]["roi"]

    next1_abs = rel_to_abs(board_abs, runtime_cfg["next"]["roi1_rel_to_board"])
    next2_abs = rel_to_abs(board_abs, runtime_cfg["next"]["roi2_rel_to_board"])
    hold_abs = rel_to_abs(board_abs, runtime_cfg["hold"]["roi_rel_to_board"])

    return {
        "board": board_abs,
        "next1": next1_abs,
        "next2": next2_abs,
        "hold": hold_abs,
    }