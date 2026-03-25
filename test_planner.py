import unittest

import numpy as np

from planner import best_move, score_board, PIECES, drop_row, place


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

    def test_best_move_uses_next_piece_lookahead(self):
        board = np.zeros((20, 10), dtype=np.uint8)
        filled = [
            (11, 7), (11, 8), (11, 9),
            (12, 0), (12, 5), (12, 7), (12, 8), (12, 9),
            (13, 0), (13, 5), (13, 7), (13, 9),
            (14, 0), (14, 1), (14, 2), (14, 5), (14, 6), (14, 7), (14, 8), (14, 9),
            (15, 0), (15, 1), (15, 2), (15, 4), (15, 5), (15, 6), (15, 7), (15, 8), (15, 9),
            (16, 0), (16, 1), (16, 2), (16, 3), (16, 4), (16, 5), (16, 6), (16, 7), (16, 8), (16, 9),
            (17, 0), (17, 1), (17, 2), (17, 3), (17, 4), (17, 5), (17, 6), (17, 7), (17, 8), (17, 9),
            (18, 0), (18, 1), (18, 2), (18, 3), (18, 4), (18, 5), (18, 6), (18, 7), (18, 8), (18, 9),
            (19, 0), (19, 1), (19, 2), (19, 3), (19, 4), (19, 5), (19, 6), (19, 7), (19, 8), (19, 9),
        ]
        for r, c in filled:
            board[r, c] = 1

        no_lookahead = best_move(board, piece="I")
        with_lookahead = best_move(board, piece="I", next_piece="T")

        self.assertIsNotNone(no_lookahead)
        self.assertIsNotNone(with_lookahead)
        self.assertNotEqual((no_lookahead[0], no_lookahead[1]), (with_lookahead[0], with_lookahead[1]))

        # Validate the lookahead-selected move creates a better board
        # for the provided next piece under the same planner objective.
        def apply_move(b, piece_name, move):
            rot, col, _ = move
            shape = PIECES[piece_name][rot]
            row = drop_row(b, shape, col)
            self.assertIsNotNone(row)
            b2, _ = place(b, shape, row, col)
            return b2

        b_no = apply_move(board, "I", no_lookahead)
        b_look = apply_move(board, "I", with_lookahead)
        future_from_no = best_move(b_no, piece="T")
        future_from_look = best_move(b_look, piece="T")

        self.assertIsNotNone(future_from_no)
        self.assertIsNotNone(future_from_look)
        self.assertGreaterEqual(future_from_look[2], future_from_no[2])


if __name__ == "__main__":
    unittest.main()
