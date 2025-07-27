
from connect4.board import get_legal_moves, add_move, check_valid_board
import numpy as np
import pytest

X=1
O=-1

def test_get_legal_moves_empty_board(empty_board_arr):
    legal_moves = get_legal_moves(empty_board_arr)

    for col, row in legal_moves.items():
        copy_board = empty_board_arr.copy()
        add_move(copy_board, X, (row, col))
        assert check_valid_board(copy_board), f"Board is invalid after adding move at {(row, col)}"


def test_get_legal_moves_some_full_columns():
    # Fill column 0 and 3 completely, others empty
    starting_board = np.array([
        [X, 0, 0, O, 0, 0, 0],
        [X, 0, 0, O, 0, 0, 0],
        [X, 0, 0, O, 0, 0, 0],
        [X, 0, 0, O, 0, 0, 0],
        [X, 0, 0, O, 0, 0, 0],
        [X, 0, 0, O, 0, 0, 0],
    ])

    legal_moves = get_legal_moves(starting_board)

    for col, row in legal_moves.items():
        copy_board = starting_board.copy()
        add_move(copy_board, X, (row, col))
        assert check_valid_board(copy_board), f"Board is invalid after adding move at {(row, col)}"

