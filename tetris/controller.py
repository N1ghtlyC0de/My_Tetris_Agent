import time
import pyautogui
from config import Keymap

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

class Controller:
    def __init__(self, keymap: Keymap):
        self.k = keymap

    def tap(self, key: str, delay=0.012):
        pyautogui.keyDown(key)
        time.sleep(delay)
        pyautogui.keyUp(key)

    def move_left(self, n=1):
        for _ in range(n):
            self.tap(self.k.left)

    def move_right(self, n=1):
        for _ in range(n):
            self.tap(self.k.right)

    def rotate_cw(self, n=1):
        for _ in range(n):
            self.tap(self.k.rot_cw)

    def rotate_ccw(self, n=1):
        for _ in range(n):
            self.tap(self.k.rot_ccw)

    def rotate_180(self, n=1):
        for _ in range(n):
            self.tap(self.k.rot_180)

    def hold(self):
        self.tap(self.k.hold)

    def hard_drop(self):
        self.tap(self.k.hard_drop, delay=0.008)