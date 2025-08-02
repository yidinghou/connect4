import numpy as np
import connect4.mcts as mcts


def convert_mcts_nodes_data_to_target(nodes_data, player_perspective):
    """
    Convert MCTS nodes data to target format for training.
    Args:
        nodes_data (numpy.ndarray): The MCTS nodes data.

    Returns:
        (value, policy): Tuple containing:
            - value (float): The value of the board state.
            - policy (numpy.ndarray): The policy vector for the actions.
    """

    root_node = nodes_data[0]
    value = (root_node[mcts.WINS_COL] / root_node[mcts.N_VISITS_COL]
             if root_node[mcts.N_VISITS_COL] > 0 else 0)
    # root is always from player perspective, but in training we will keep it from p1 perspective
    # this way the model learns to predict the outcome from the perspective of player 1
    if player_perspective == 1:
        value = 1-value

    policy = np.zeros(7)
    for col in range(7):
        child_key = col
        child_node = nodes_data[nodes_data[:, mcts.ACTION_COL] == child_key]
        if child_node.size > 0:
            policy[col] = child_node[0, mcts.N_VISITS_COL]

    policy /= policy.sum() if policy.sum() > 0 else 1
    return value, policy