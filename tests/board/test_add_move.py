
from connect4.board import get_legal_moves, add_move, check_valid_board
import numpy as np
import pytest

X=1
O=-1

def test_get_legal_moves_empty_board(empty_board_arr):
    legal_moves = get_legal_moves(empty_board_arr)

    for col, row in legal_moves.items():
        new_board = add_move(empty_board_arr, X, (row, col))
        assert check_valid_board(new_board), f"Board is invalid after adding move at {(row, col)}"


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
        new_board = add_move(starting_board, X, (row, col))
        assert check_valid_board(new_board), f"Board is invalid after adding move at {(row, col)}"


