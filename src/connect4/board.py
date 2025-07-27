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

    if np.sum(board_arr) not in [0, 1]:
        return False
    
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

    rows_above = board_arr[:-1, :]
    rows_below = board_arr[1:, :]
    has_floating_zero = (rows_above != 0) & (rows_below == 0)

    return not np.any(has_floating_zero)


def check_win(board_arr: np.ndarray) -> bool:
    """
    Checks if there is a winning condition on the board.

    Args:
        board_arr (np.ndarray): NumPy array representing the board.

    Returns:
        bool: True if there is a winning condition, False otherwise.
    """
    for player in [1, -1]:
        #check each row for four in a row
        for row in board_arr:
            if np.any(np.convolve(row, np.ones(4, dtype=int), 'valid') == 4 * player):
                return True
        
        #check each column for four in a row
        for col in board_arr.T:
            if np.any(np.convolve(col, np.ones(4, dtype=int), 'valid') == 4 * player):
                return True

        #check diagonal (top-left to bottom-right)
        for i in range(board_arr.shape[0] - 3):
            for j in range(board_arr.shape[1] - 3):
                if (board_arr[i, j] == player and
                    board_arr[i + 1, j + 1] == player and
                    board_arr[i + 2, j + 2] == player and
                    board_arr[i + 3, j + 3] == player):
                    return True

        #check diagonal (bottom-left to top-right)
        for i in range(3, board_arr.shape[0]):
            for j in range(board_arr.shape[1] - 3):
                if (board_arr[i, j] == player and
                    board_arr[i - 1, j + 1] == player and
                    board_arr[i - 2, j + 2] == player and
                    board_arr[i - 3, j + 3] == player):
                    return True

    return False  # No winning condition found

def check_incremental_win(board_arr: np.ndarray, row: int, col: int, player: int) -> bool:
    """
    Checks if the last move by 'player' at (row, col) resulted in a win.
    Only checks the local region around the last move.

    Args:
        board_arr (np.ndarray): The board.
        row (int): Row index of the last move.
        col (int): Column index of the last move.
        player (int): The player value (1 for X, -1 for O).

    Returns:
        bool: True if the last move resulted in a win, False otherwise.
    """

    # Check horizontal
    # get a slice of the entire row
    row_slice = board_arr[row, :]
    # check for four in a row
    if np.any(np.convolve(row_slice, np.ones(4, dtype=int), 'valid') == 4 * player):
        return True
    # Check vertical
    # get a slice of the entire column
    col_slice = board_arr[:, col]
    # check for four in a row
    if np.any(np.convolve(col_slice, np.ones(4, dtype=int), 'valid') == 4 * player):
        return True
    
    #get a slice of the diagonal (top-left to bottom-right)
    diag_slice = np.diagonal(board_arr, offset=col - row)
    if len(diag_slice) >= 4 and np.any(np.convolve(diag_slice, np.ones(4, dtype=int), 'valid') == 4 * player):
        return True
    
    #get a slice of the diagonal (bottom-left to top-right)
    diag_slice = np.diagonal(np.fliplr(board_arr), offset=col + row - (board_arr.shape[1] - 1))
    if len(diag_slice) >= 4 and np.any(np.convolve(diag_slice, np.ones(4, dtype=int), 'valid') == 4 * player):
        return True
    
    return False  # No win found in the local region