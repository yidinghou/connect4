import numpy as np
from connect4.board import check_board_state, check_board_state_incremental

X=1
O=-1


def test_check_board_state_draw():
    # Fill column 0 and 3 completely, others empty
    board_arr= np.array([
        [O, O, O, X, O, O, O],
        [X, X, X, O, X, X, X],
        [O, O, O, X, O, O, O],
        [X, X, X, O, X, X, X],
        [O, O, O, X, O, O, O],
        [X, X, X, O, X, X, X]
    ], dtype=int)


    is_terminal, result = check_board_state(board_arr)
    assert is_terminal is True
    assert result == 0  # Draw

def test_player_x_already_won_horizontal():
    board_arr = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [X, X, X, X, 0, 0, 0],
    ], dtype=int)


    is_terminal, result = check_board_state(board_arr)
    assert is_terminal is True
    assert result == 1  # Player X wins
    
    for i in range(4):
        is_terminal, result = check_board_state_incremental(board_arr, row=5, col=i, player=X)
        assert is_terminal is True
        assert result == 1  # Player X wins

def test_player_o_already_won_vertical():
    board_arr = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [O, 0, 0, 0, 0, 0, 0],
        [O, 0, 0, 0, 0, 0, 0],
        [O, 0, 0, 0, 0, 0, 0],
        [O, 0, 0, 0, 0, 0, 0],
    ], dtype=int)
    is_terminal, result = check_board_state(board_arr)
    assert is_terminal is True
    assert result == -1  # Player O wins

    for i in range(4):
        is_terminal,result = check_board_state_incremental(board_arr, row=i, col=0, player=O)
        assert is_terminal is True
        assert result == -1  # Player O wins