from connect4.mcts import rollout
import numpy as np


def test_rollout_valid(empty_board_arr):
    """Test the rollout function on an empty board."""
    player = 1
    for i in range(10):
        rollout(empty_board_arr, player, debug=True) # Perform 10 rollouts, will fail if any boards are not valid

