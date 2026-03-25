import unittest

import numpy as np

from planner import best_move, score_board


class PlannerHeuristicTests(unittest.TestCase):
    def test_score_penalizes_holes(self):
        board_no_hole = np.zeros((20, 10), dtype=np.uint8)
        board_no_hole[17, 0] = 1
        board_no_hole[18, 0] = 1
        board_no_hole[19, 0] = 1

        board_with_hole = np.zeros((20, 10), dtype=np.uint8)
        board_with_hole[16, 0] = 1
        board_with_hole[18, 0] = 1
        board_with_hole[19, 0] = 1

        self.assertLess(score_board(board_with_hole, 0), score_board(board_no_hole, 0))

    def test_score_rewards_more_lines(self):
        empty = np.zeros((20, 10), dtype=np.uint8)
        self.assertGreater(score_board(empty, 4), score_board(empty, 2))
        self.assertGreater(score_board(empty, 2), score_board(empty, 1))

    def test_best_move_prefers_line_clear(self):
        board = np.zeros((20, 10), dtype=np.uint8)
        board[19, :] = 1
        board[19, 3:7] = 0

        mv = best_move(board, piece="I")
        self.assertIsNotNone(mv)
        rot, col, _ = mv

        self.assertEqual(rot, 0)
        self.assertEqual(col, 3)


if __name__ == "__main__":
    unittest.main()
