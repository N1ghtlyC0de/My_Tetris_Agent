from __future__ import annotations

import numpy as np
import cv2
import mss

from config_runtime import load_runtime_config
from vision_rois import get_all_abs_rois

BOARD_COLS = 10
BOARD_ROWS = 20


class Vision:
    def __init__(self):
        self.sct = mss.mss()
        self.runtime_cfg = load_runtime_config()
        rois = get_all_abs_rois(self.runtime_cfg)

        self.board_roi = rois["board"]
        self.next1_roi = rois["next1"]
        self.next2_roi = rois["next2"]
        self.hold_roi = rois["hold"]

    def _grab_roi(self, roi):
        shot = np.array(self.sct.grab(roi), dtype=np.uint8)  # BGRA
        return shot[:, :, :3].copy()  # BGR

    def grab_board(self):
        return self._grab_roi(self.board_roi)

    def grab_next1(self):
        return self._grab_roi(self.next1_roi)

    def grab_next2(self):
        return self._grab_roi(self.next2_roi)

    def grab_hold(self):
        return self._grab_roi(self.hold_roi)

    def board_occupancy(self, bgr: np.ndarray) -> np.ndarray:
        """
        Convert board ROI image into occupancy grid (20x10), uint8 {0,1}.
        """
        h, w, _ = bgr.shape
        cell_w = w / BOARD_COLS
        cell_h = h / BOARD_ROWS

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        occ = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.uint8)

        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                cx = int((c + 0.5) * cell_w)
                cy = int((r + 0.5) * cell_h)

                y0, y1 = max(0, cy - 2), min(h, cy + 3)
                x0, x1 = max(0, cx - 2), min(w, cx + 3)
                patch = hsv[y0:y1, x0:x1]

                sat = float(patch[:, :, 1].mean())
                val = float(patch[:, :, 2].mean())

                # Tune these thresholds for your theme/brightness
                occ[r, c] = 1 if (sat > 45 and val > 45) else 0

        return occ