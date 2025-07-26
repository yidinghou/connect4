# Board module to represent connect 4 boards
import numpy as np


def check_valid_board(board_arr: np.ndarray) -> bool:
    """
    Checks if the board is a valid Connect 4 board.

    Args:
        board_arr (np.ndarray): NumPy array representing the board.

    Returns:
        bool: True if the board is valid, False otherwise.
    """

    # Check if sum board pieces add up to 0 or 1
    if np.sum(board_arr) not in [0, 1]:
        return False
    
    # Check there are no floating pieces
    if not check_floating_pieces(board_arr):
        return False

def check_floating_pieces(board_arr: np.ndarray) -> bool:
    """
    A floating zero is a 0 that has a non-zero number in the cell
    directly above it. This function works for any array with 2 or more rows.

    Args:
        board (np.ndarray): An 6 x 7 NumPy array.

    Returns:
        bool: True if the board state is valid (no floating zeros), False otherwise.
    """
    # Slice containing all rows EXCEPT the last one (the "above" rows)
    rows_above = board_arr[:-1, :]

    # Slice containing all rows EXCEPT the first one (the "current" rows)
    rows_below = board_arr[1:, :]

    # A floating zero exists where a value in `rows_below` is 0 AND
    # the corresponding value in `rows_above` is not 0.
    has_floating_zero = (rows_above != 0) & (rows_below == 0)

    # The board is valid if no such condition exists anywhere.
    return not np.any(has_floating_zero)
