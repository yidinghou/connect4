
from connect4.board import check_floating_pieces
import numpy as np

#TODO: Create fixtures for each board state and parametrize tests

def floating_piece_simple():
    """Single floating piece - piece in air with empty space below"""
    return np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],  # Floating piece
        [0, 0, 0, 0, 0, 0, 0],  # Empty space below
        [0, 0, 0, 0, 0, 0, 0]
    ], dtype=int)

def floating_piece_with_gap():
    """Piece floating with gap in middle of column"""
    return np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],  # Floating piece
        [0, 0, 0, 0, 0, 0, 0],  # Gap
        [0, 0, 0, 1, 0, 0, 0],  # Another piece
        [0, 0, 0, -1, 0, 0, 0]  # Bottom piece
    ], dtype=int)

def multiple_floating_pieces():
    """Multiple columns with floating pieces"""
    return np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, -1, 0],  # Floating pieces in cols 1 and 5
        [0, 0, 0, 0, 0, 0, 0],  # Empty spaces below
        [0, 0, 0, 1, 0, 0, 0],  # Floating piece in col 3
        [0, 0, 0, 0, 0, 0, 0],  # Empty space below
        [1, -1, 1, -1, 0, 0, 0] # Bottom row with proper pieces
    ], dtype=int)

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

def valid_board_no_floating():
    """Valid board with no floating pieces - all pieces have fallen due to gravity"""
    return np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [1, -1, 1, -1, 0, 0, 0]  # All pieces at bottom
    ], dtype=int)


def floating_top_heavy():
    """Pieces at top with gaps below - clearly invalid"""
    return np.array([
        [1, -1, 1, -1, 0, 0, 0],  # Pieces at very top
        [0, 0, 0, 0, 0, 0, 0],    # Empty spaces below
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ], dtype=int)

def mixed_valid_invalid():
    """Some columns valid, others have floating pieces"""
    return np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],  # Floating in col 2
        [0, 0, 0, 0, 0, 0, 0],  # Gap
        [0, 1, 0, 1, 0, 1, 0],  # Mixed: col 1,3,5 have pieces
        [1, 1, 0, 1, -1, 1, 0]  # Bottom: cols 0,1,3,4,5 valid, col 2 has gap above
    ], dtype=int)

def test_check_floating_pieces():
    """
    This single test will run 7 times, once for each fixture.
    The `board_fixture` argument will hold the resolved object
    from each fixture in turn.
    """
    for board in [
        floating_piece_simple(),
        floating_piece_with_gap(),
        multiple_floating_pieces(),
        floating_stack(),
        floating_top_heavy(),
        mixed_valid_invalid()
    ]:
        result = check_floating_pieces(board)
        assert result is False


def test_valid_boards():
    """
    This single test will run 7 times, once for each fixture.
    The `board_fixture` argument will hold the resolved object
    from each fixture in turn.
    """
    for board in [
        valid_board_no_floating(),
        np.zeros((6, 7), dtype=int)  # Empty board
    ]:
        result = check_floating_pieces(board)
        assert result is True
