# Board module to represent connect 4 boards

from typing import List, Dict, Tuple, Union
import numpy as np
import random

import connect4.configs as configs


def check_valid_board(board_arr: np.ndarray) -> bool:
    """
    Checks if the board is a valid Connect 4 board.
    :param board_arr: np array of the board
    :return: True if valid, False otherwise
    """

    # Check if sum board pieces add up to 0 or 1
    if np.sum(board_arr) not in [0, 1]:
        return False
    
    # Check there are no floating pieces

def check_floating_pieces(board_arr: np.ndarray) -> bool:
    """
    Checks an M x N NumPy array to ensure there are no 'floating zeros'.

    A floating zero is a 0 that has a non-zero number in the cell
    directly above it. This function works for any array with 2 or more rows.

    Args:
        board (np.ndarray): An M x N NumPy array.

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


class Board:
    """
    A data class representing the state of a Connect 4 board.

    Attributes:
        board (np.ndarray): The current state of the board as a NumPy array.
        board_str (str): The string representation of the board.
        player_turn (int): The current player's turn (1 for player 1, -1 for player 2).
        is_terminal (Optional[bool]): Whether the board state is terminal (win or draw). None means not checked
        winner (Optional[int]): The winner of the game (1 for player 1, -1 for player 2, 0 for draw, None if not terminal).
    """

    def __init__(self, board_arr: np.ndarray):
        """
        Initializes the Board instance.

        :param board_input: The input board state, either as a string or a NumPy array.
        """
        if isinstance(board_arr, np.ndarray):
            self.board_arr = board_arr
        else:
            raise ValueError("Invalid input type. Must be a string or a NumPy array.")

        total_pieces = np.count_nonzero(self.board_arr)
        self.player_turn = 1 if total_pieces % 2 == 0 else -1

        self.is_terminal = None
        self.winner = None
        self.check_update_state()

    def check_update_state(self) -> None:
        """
        Updates the BoardState instance to reflect whether the board is terminal and determines the winner.
        1: player 1 win, -1: player 2 win, 0: draw, None: not terminal
        """
        player_win = self.check_win(self.board)
        if player_win != 0:
            self.is_terminal = True
            self.winner = player_win
        elif self.is_full():
            self.is_terminal = True
            self.winner = 0
        else:
            self.is_terminal = False
            self.winner = None
    
    def is_full(self) -> bool:
        """
        Checks if the board is full (i.e., no empty spaces left).

        :return: True if the board is full, False otherwise.
        """
        return is_full(self.board)
    
    def copy(self) -> 'Board':
        """
        Creates a copy of the current Board instance.

        :return: A new Board instance with the same state.
        """
        return Board(self.board.copy())
    
    def add_move(self, player_int: int, loc: Tuple[int, int]) -> None:
        """
        Adds a move to the board at the specified location.

        :param player_int: The player making the move (1 for player 1, -1 for player 2).
        :param loc: A tuple (col, row) representing the column and row of the move.
        """
        col, row = loc
        if row < 0 or col < 0 or col >= self.board.shape[1] or row >= self.board.shape[0]:
            raise ValueError("Invalid move location: Out of bounds")

        if self.board[row, col] != 0:
            raise ValueError("Invalid move: Cell is already occupied")

        self.board[row, col] = player_int
        self.check_valid_state()
        self.check_update_state()

    def get_legal_moves(self) -> Dict[int, int]:
        """
        Class method wrapper for the standalone get_legal_moves function.
        """
        return get_legal_moves(self.board)

    def get_child_board_arr(self) -> Dict[int, np.ndarray]:
        """
        Returns a dict of board arrays for each legal move.
        If no legal moves, then dict is empty; if one column is full then that key:value pair does not exist in dict.
        """
        self.check_update_state()
        if self.is_terminal:
            raise ValueError("Terminal Board State: no childboard")

        legal_moves = self.get_legal_moves()
        child_board_dict = {}
        for col, row in legal_moves.items():
            board_copy = np.copy(self.board)
            board_obj = Board(board_copy)
            board_obj.add_move(self.player_turn, (col, row))
            child_board_dict[col] = board_obj.board
        return child_board_dict

    def select_random_move(self):
        """
        Instance method to select a random legal move from the current board state.
        Returns (col, row) or raises ValueError if no legal moves.
        """
        return select_random_move(self.board)

# Standalone function for random move selection

def select_random_move(board_input: Union[str, np.ndarray]) -> tuple:
    """
    Standalone function to select a random legal move from a board string or array.
    Returns (col, row) or raises ValueError if no legal moves.
    """
    legal_moves_dict = get_legal_moves(board_input)
    if not legal_moves_dict:
        raise ValueError("no legal moves")
    col, row = random.choice(list(legal_moves_dict.items()))
    return (col, row)



def apply_gravity(board_arr: np.ndarray) -> np.ndarray:
    """
    :param board_arr: np array
    :return: np array of the input board with no empty vertical space between pieces
    """
    copy_board = np.copy(board_arr)
    return np.array([sorted(column, key=bool) for column in copy_board.T]).T


def arr_to_str(board_arr: np.ndarray) -> str:
    """
    :param board_arr: np array
    :return: string representation of board, 6*7=42 length for c4
    """
    return ''.join(configs.INT2STR_MAP[i] for i in board_arr.ravel(order='F'))


def str_to_arr(board_str: str) -> np.ndarray:
    """
    :param board_str: string representation of board, 6*7=42 length for c4
    :return: board_arr
    """
    b = np.array([configs.STR2INT_MAP[i] for i in board_str]).reshape([configs.N_COLS, configs.N_ROWS])
    return np.fliplr(np.rot90(b, k=3))


def shuffle_str(string: str) -> str:
    """This function takes input string and shuffles the string and returns a new string"""
    str_var = list(string)
    random.shuffle(str_var)
    return ''.join(str_var)


def gen_rand_boards(n: int, seed: int = 42) -> List[np.ndarray]:
    """
    :param n: number of boards to generate
    :param seed: random seed
    :return: list of board_arr
    """

    random.seed(seed)

    "This block of code generates lists of size n for player 1 moves and player 2 moves"
    p1_n_moves = random.choices(range(1, int(configs.MAX_MOVES / 2)), k=n)  # sample [1,21] for p1 moves
    delta = random.choices(list(range(2)), k=n)  # sample [0,1] for differences in p1 and p2
    p2_n_moves = [a - b for a, b in zip(p1_n_moves, delta)]  # calculate p2 moves based on p1 and delta

    "This block of code converts the lists of size n to list of board_strs"
    p1_str_list = [n * "x" for n in p1_n_moves]
    p2_str_list = [n * "o" for n in p2_n_moves]
    board_str_list = [p1_str_list[i] + p2_str_list[i] for i in range(n)]
    board_str_list = [x.ljust(configs.MAX_MOVES) for x in board_str_list]  # fill string to be of size 42
    board_str_list = [shuffle_str(b) for b in board_str_list]  # shuffle string to randomize

    "This block of code converts list of board_strs to a list of board_arrs"
    board_arr_list = [str_to_arr(b) for b in board_str_list]
    board_arr_list = [apply_gravity(b) for b in board_arr_list]

    return board_arr_list


def convert_board_input(board_input: Union[str, np.ndarray]) -> np.ndarray:
    """
    This function converts input of either string or np array to the right format: np array
    """
    if isinstance(board_input, str):
        board_arr = str_to_arr(board_input)
    elif isinstance(board_input, np.ndarray):
        board_arr = board_input
    else:
        raise ValueError("Invalid input type")

    return board_arr


def is_full(board_arr: np.ndarray) -> bool:
    return np.sum(board_arr == 0) == 0  # no empty spots left

def get_legal_moves(board_input: Union[str, np.ndarray]) -> Dict[int, int]:
    """
    Returns a dictionary of legal moves for a Connect 4 board.
    The key is the column index, and the value is the row index of the next available spot.

    :param board_input: The input board state, either as a string or a NumPy array.
    :return: A dictionary of legal moves. Empty if no moves are available.
    """
    # Convert input to a NumPy array
    board_arr = convert_board_input(board_input)

    # Validate board dimensions
    if board_arr.shape != (configs.N_ROWS, configs.N_COLS):
        raise ValueError("Invalid board dimensions")

    # Find the first empty row in each column
    legal_moves = {}
    for col in range(7):
        empty_rows = np.where(board_arr[:, col] == 0)[0]
        if empty_rows.size > 0:
            legal_moves[col] = int(empty_rows[-1])  # Get the lowest empty row

    return legal_moves

