from __future__ import annotations

from typing import Dict, Optional, Tuple
import cv2
import numpy as np

PIECES = ["I", "J", "L", "O", "S", "T", "Z"]

HSV_RANGES: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    "I": ((75,  50,  70), (100, 255, 255)),   # cyan
    "J": ((100, 60,  60), (130, 255, 255)),   # blue
    "L": ((8,   70,  70), (24,  255, 255)),   # orange
    "O": ((18,  35, 120), (40,  255, 255)),   # yellow (wider, brighter)
    "S": ((40,  60,  60), (85,  255, 255)),   # green
    "T": ((130, 45,  55), (170, 255, 255)),   # purple
    "Z": ((0,   70,  70), (10,  255, 255)),   # red low
}
Z2_RANGE = ((170, 70, 70), (179, 255, 255))   # red high


def _mask_for_piece(hsv: np.ndarray, piece: str) -> np.ndarray:
    lo, hi = HSV_RANGES[piece]
    mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    if piece == "Z":
        lo2, hi2 = Z2_RANGE
        mask2 = cv2.inRange(hsv, np.array(lo2, dtype=np.uint8), np.array(hi2, dtype=np.uint8))
        mask = cv2.bitwise_or(mask, mask2)
    return mask


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


def _largest_blob_area(mask: np.ndarray) -> int:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    return int(max(cv2.contourArea(c) for c in contours))


def classify_piece_by_color(roi_bgr: np.ndarray, min_norm_score: float = 0.015) -> Optional[str]:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    roi_area = roi_bgr.shape[0] * roi_bgr.shape[1]

    best_piece = None
    best_score = 0.0

    for p in PIECES:
        mask = _clean_mask(_mask_for_piece(hsv, p))
        area = _largest_blob_area(mask)
        score = area / max(roi_area, 1)  # normalize by ROI size
        if score > best_score:
            best_score = score
            best_piece = p

    if best_score < min_norm_score:
        return None
    return best_piece


# --- shape fallback ---
SHAPE_SIGNATURES = {
    "I": {(0,0), (0,1), (0,2), (0,3)},
    "O": {(0,0), (0,1), (1,0), (1,1)},
    "T": {(0,1), (1,0), (1,1), (1,2)},
    "S": {(0,1), (0,2), (1,0), (1,1)},
    "Z": {(0,0), (0,1), (1,1), (1,2)},
    "J": {(0,0), (1,0), (1,1), (1,2)},
    "L": {(0,2), (1,0), (1,1), (1,2)},
}

def _to_4x4_occupancy(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    bw = cv2.medianBlur(bw, 3)

    h, w = bw.shape
    occ = np.zeros((4, 4), dtype=np.uint8)
    for r in range(4):
        for c in range(4):
            y0, y1 = int(r*h/4), int((r+1)*h/4)
            x0, x1 = int(c*w/4), int((c+1)*w/4)
            patch = bw[y0:y1, x0:x1]
            occ[r, c] = 1 if (patch > 0).mean() > 0.20 else 0
    return occ

def _normalize_coords(occ: np.ndarray):
    ys, xs = np.where(occ == 1)
    if len(xs) == 0:
        return set()
    miny, minx = ys.min(), xs.min()
    return {(int(y-miny), int(x-minx)) for y, x in zip(ys, xs)}

def classify_piece_by_shape(roi_bgr: np.ndarray) -> Optional[str]:
    occ = _to_4x4_occupancy(roi_bgr)
    coords = _normalize_coords(occ)
    if len(coords) < 3:
        return None

    best, best_d = None, 10**9
    for p, sig in SHAPE_SIGNATURES.items():
        d = len(coords.symmetric_difference(sig))
        if d < best_d:
            best_d = d
            best = p
    return best if best_d <= 4 else None


def classify_piece(roi_bgr: np.ndarray) -> Optional[str]:
    # Bias to color first (works best for TETR.IO default skin)
    p = classify_piece_by_color(roi_bgr)
    if p is not None:
        return p
    return classify_piece_by_shape(roi_bgr)