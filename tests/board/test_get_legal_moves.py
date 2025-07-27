
from connect4.board import get_legal_moves
import numpy as np
import pytest

X=1
O=-1

def test_get_legal_moves_empty_board(empty_board_arr):
    legal_moves = get_legal_moves(empty_board_arr)
    assert legal_moves == {0: 5, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5}, (
        "All columns should be legal on an empty board."
    )


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
    expected = {1: 5, 2: 5, 4: 5, 5: 5, 6: 5} # no legal moves in columns 0 and 3
    assert legal_moves == expected

def test_get_legal_moves_all_full():
    # All columns full, alternating players
    starting_board = np.array([
        [X, O, X, O, X, O, X],
        [X, O, X, O, X, O, X],
        [X, O, X, O, X, O, X],
        [X, O, X, O, X, O, X],
        [X, O, X, O, X, O, X],
        [X, O, X, O, X, O, X],
    ])
    legal_moves = get_legal_moves(starting_board)
    expected = {}
    assert legal_moves == expected
