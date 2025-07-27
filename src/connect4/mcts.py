"""Monte Carlo Tree Search (MCTS) for Connect 4 game."""

from connect4 import board
import numpy as np
import random

def rollout(board_arr: np.ndarray, player: int, debug=False) -> int:
    """
    Perform a random rollout from the current board state.
    
    Args:
        board_arr (np.ndarray): The current board state.
        player (int): The player to make the first move in the rollout.

    Returns:
        int: The result of the rollout (1 for player 1 win, -1 for player 2 win, 0 for draw).
    """
    is_terminal, result = board.check_board_state(board_arr)
    while not is_terminal:
        legal_moves = board.get_legal_moves(board_arr)
        col, row = random.choice(list(legal_moves.items()))

        board.add_move(board_arr, player=player, loc=(row, col))
        if debug:
            print(board_arr)
            print(f"Player {player} added move at ({row}, {col})")
            assert board.check_valid_board(board_arr)

        is_terminal, result = board.check_board_state_incremental(board_arr, row, col, player)
        player *= -1

    return result

