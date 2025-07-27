
from connect4.mcts import MCTSTree
from connect4 import board
import numpy as np



def test_select_leaf_initial():
    # when initiated from scratch
    # then leaf_node and lead_board are index 0, and starting board
    # then tree.node_data first index is [-1, -1, 0, 0, 0]

    initial_board = np.zeros((6, 7), dtype=int)  # Empty board
    tree = MCTSTree(initial_board)

    # Select leaf from the initial state
    leaf_node, leaf_board = tree.select_leaf(0, initial_board)
    assert leaf_node == 0
    assert np.array_equal(leaf_board, initial_board)
    assert tree.node_data[0, tree.PARENT_COL] == -1
    assert tree.node_data[0, tree.ACTION_COL] == -1


def test_select_mock_expansion():
    # when there is an expansion
    # then the select_leaf should return the new node index and board state
    
    initial_board = np.zeros((6, 7), dtype=int)  # Empty board
    tree = MCTSTree(initial_board)

    pre_count = tree.node_count
    # Simulate an expansion by creating a new node, real expan
    # Simulate an expansion by creating a new node, real expansion will create multiple
    # child nodes
    sample_col = 1
    tree._create_new_node(0, action_col=sample_col)
    tree.node_data[0, tree.EXPANDED_COL] = 1

    assert tree.node_count == pre_count + 1

    leaf_node, leaf_board = tree.select_leaf(0, initial_board)
    assert leaf_node == 1  # The new node created

    expected_board = board.add_move(initial_board, 1, (5, sample_col))
    assert np.array_equal(leaf_board, expected_board)