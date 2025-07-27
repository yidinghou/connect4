

from connect4.board import check_win, check_incremental_win
import numpy as np
import pytest
X=1
O=-1


#region No Win Tests
@pytest.fixture
def no_win_board_1():
    return np.array([
        [X, X, X, 0, X, X, X],
        [X, X, X, 0, X, X, X],
        [0, 0, 0, 0, 0, 0, 0],
        [X, X, X, 0, X, X, X],
        [X, X, X, 0, X, X, X],
        [X, X, X, 0, X, X, X]
    ], dtype=int)

@pytest.fixture
def no_win_full_board_1():
    return np.array([
        [O, O, O, X, O, O, O],
        [X, X, X, O, X, X, X],
        [O, O, O, X, O, O, O],
        [X, X, X, O, X, X, X],
        [O, O, O, X, O, O, O],
        [X, X, X, O, X, X, X]
    ], dtype=int)

@pytest.fixture
def no_win_full_board_2():
    return np.array([
        [O, X, O, X, O, X, O],
        [X, O, O, O, X, O, X],
        [X, O, X, O, X, O, X],
        [X, O, X, O, X, X, X],
        [O, X, O, X, O, X, O],
        [X, O, X, O, O, O, X]
    ], dtype=int)

@pytest.fixture
def no_win_board_2():
    return np.array([
        [0, X, X, X, 0, X, X],
        [X, X, X, 0, X, X, X],
        [X, X, 0, X, X, X, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [X, X, X, 0, X, X, X],
        [X, X, X, 0, X, X, X]
    ], dtype=int)


@pytest.fixture
def board_fixture(request):
    """Meta-fixture that resolves to the actual board fixture"""
    return request.getfixturevalue(request.param)

@pytest.mark.parametrize("board_fixture", [
    "no_win_board_1",
    "no_win_board_2",
    "no_win_full_board_1",
    "no_win_full_board_2"
], indirect=True)
def test_boards_with_floating_pieces(board_fixture):
    """Test that boards with floating pieces are correctly identified as invalid"""
    result = check_win(board_fixture)
    assert result is False

    result = check_win(board_fixture*-1)
    assert result is False

#endregion

#region Win Tests

DIAG_WINS = [
    # 1st row
    np.array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0]]),

    np.array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0]]),

    np.array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0]]),

    np.array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0]]),

    # 2nd row
    np.array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]]),

    np.array([[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0]]),

    np.array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]]),

    np.array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]]),

    # 3rd row
    np.array([[0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]]),

    np.array([[0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]]),

    np.array([[0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]]),

    np.array([[0, 0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]]),
]

@pytest.mark.parametrize("board", DIAG_WINS)
def test_diagonal_wins(board):
    """Test diagonal winning conditions"""
    result = check_win(board)
    assert result is True, f"Expected win for diagonal board:\n{board}"

    result = check_win(board*-1)  # Check for player -1
    assert result is True, f"Expected win for diagonal board:\n{board}"



@pytest.fixture
def player_x_already_won_diagonal():
    """Board where Player X has a down-right diagonal win."""
    return np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [X, 0, 0, 0, 0, 0, 0],
        [O, X, 0, 0, 0, 0, 0],
        [O, O, X, 0, 0, 0, 0],
        [O, O, O, X, 0, 0, 0],
    ], dtype=int)

def test_player_x_already_won_horizontal():
    board_arr = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [X, X, X, X, 0, 0, 0],
    ], dtype=int)
    """Test that check_incremental_win detects Player X's horizontal win."""
    for i in range(4):
        result = check_incremental_win(board_arr, row=5, col=i, player=X)
        assert result is True, "Expected Player X to have already won horizontally."

        result = check_incremental_win(board_arr, row=5, col=i, player=O)
        assert result is False, "Checking wrong player."

    result = check_incremental_win(board_arr, row=4, col=1, player=X)
    assert result is False, "Checking wrong coordinate."


def test_player_o_already_won_vertical():
    board_arr = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [O, 0, 0, 0, 0, 0, 0],
        [O, 0, 0, 0, 0, 0, 0],
        [O, 0, 0, 0, 0, 0, 0],
        [O, 0, 0, 0, 0, 0, 0],
    ], dtype=int)

    """Test that check_incremental_win detects Player O's vertical win."""
    for i in range(4):
        result = check_incremental_win(board_arr, row=i, col=0, player=O)
        assert result is True, "Expected Player O to have already won vertically."

        result = check_incremental_win(board_arr, row=5, col=1, player=X)
        assert result is False, "Checking wrong player."


    result = check_incremental_win(board_arr, row=5, col=1, player=O)
    assert result is False, "Checking wrong coordinate."

#endregion

