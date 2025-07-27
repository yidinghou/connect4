
from connect4.board import check_floating_pieces
import numpy as np
import pytest

X=1
O=-1

# Fixtures for invalid board states (boards with floating pieces)
@pytest.fixture
def floating_piece_simple():
    """Single floating piece - piece in air with empty space below"""
    return np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, X, 0, 0, 0],  # Floating piece
        [0, 0, 0, 0, 0, 0, 0],  # Empty space below
        [0, 0, 0, 0, 0, 0, 0]
    ], dtype=int)


@pytest.fixture
def floating_piece_with_gap():
    """Piece floating with gap in middle of column"""
    return np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, X, 0, 0, 0],  # Floating piece
        [0, 0, 0, 0, 0, 0, 0],  # Gap
        [0, 0, 0, X, 0, 0, 0],  # Another piece
        [0, 0, 0, O, 0, 0, 0]  # Bottom piece
    ], dtype=int)


@pytest.fixture
def multiple_floating_pieces():
    """Multiple columns with floating pieces"""
    return np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, X, 0, 0, 0, O, 0],  # Floating pieces in cols 1 and 5
        [0, 0, 0, 0, 0, 0, 0],  # Empty spaces below
        [0, 0, 0, X, 0, 0, 0],  # Floating piece in col 3
        [0, 0, 0, 0, 0, 0, 0],  # Empty space below
        [X, O, X, O, 0, 0, 0] # Bottom row with proper pieces
    ], dtype=int)


@pytest.fixture
def floating_stack():
    """Stack of pieces floating together"""
    return np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],  # Top of floating stack
        [0, 0, -1, 0, 0, 0, 0], # Middle of floating stack
        [0, 0, 1, 0, 0, 0, 0],  # Bottom of floating stack
        [0, 0, 0, 0, 0, 0, 0],  # Empty space below stack
        [0, 0, 0, 0, 0, 0, 0]
    ], dtype=int)


@pytest.fixture
def floating_top_heavy():
    """Pieces at top with gaps below - clearly invalid"""
    return np.array([
        [X, O, X, O, 0, 0, 0],  # Pieces at very top
        [0, 0, 0, 0, 0, 0, 0],    # Empty spaces below
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ], dtype=int)


@pytest.fixture
def mixed_valid_invalid():
    """Some columns valid, others have floating pieces"""
    return np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, X, 0, 0, 0, 0],  # Floating in col 2
        [0, 0, 0, 0, 0, 0, 0],  # Gap
        [0, X, 0, X, 0, X, 0],  # Mixed: col 1,3,5 have pieces
        [X, X, 0, X, O, X, 0]  # Bottom: cols 0,1,3,4,5 valid, col 2 has gap above
    ], dtype=int)


# Fixtures for valid board states (no floating pieces)
@pytest.fixture
def valid_board_no_floating():
    """Valid board with no floating pieces - all pieces have fallen due to gravity"""
    return np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, X, 0, 0, 0],
        [0, 0, X, X, 0, 0, 0],
        [X, O, X, O, 0, 0, 0]  # All pieces at bottom
    ], dtype=int)


@pytest.fixture
def empty_board():
    """Empty board - valid state"""
    return np.zeros((6, 7), dtype=int)


# Meta-fixture for parametrized tests
@pytest.fixture
def board_fixture(request):
    """Meta-fixture that resolves to the actual board fixture"""
    return request.getfixturevalue(request.param)


# Parametrized tests for boards with floating pieces (should return False)
@pytest.mark.parametrize("board_fixture", [
    "floating_piece_simple",
    "floating_piece_with_gap", 
    "multiple_floating_pieces",
    "floating_stack",
    "floating_top_heavy",
    "mixed_valid_invalid"
], indirect=True)
def test_boards_with_floating_pieces(board_fixture):
    """Test that boards with floating pieces are correctly identified as invalid"""
    result = check_floating_pieces(board_fixture)
    assert result is False


# Parametrized tests for valid boards (should return True)
@pytest.mark.parametrize("board_fixture", [
    "valid_board_no_floating",
    "empty_board"
], indirect=True)
def test_valid_boards(board_fixture):
    """Test that valid boards (no floating pieces) are correctly identified as valid"""
    result = check_floating_pieces(board_fixture)
    assert result is True
