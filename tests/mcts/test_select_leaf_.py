
from connect4.mcts import MCTSTree
from connect4 import board
import numpy as np

X=1
O=-1

def test_select_leaf_initial():
    # when initiated from scratch
    # then leaf_node and lead_board are index 0, and starting board
    # then tree.node_data first index is [-1, -1, 0, 0, 0]

    initial_board = np.zeros((6, 7), dtype=int)  # Empty board
    tree = MCTSTree(initial_board)

    # Select leaf from the initial state
    leaf_node, leaf_board, _= tree.select_leaf(0, initial_board)
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

    leaf_node, leaf_board, path = tree.select_leaf(0, initial_board)
    assert leaf_node == 1  # The new node created

    expected_board = board.add_move(initial_board, 1, (5, sample_col))
    assert np.array_equal(leaf_board, expected_board)

    assert path == [0, 1]

def test_expand():
    initial_board = np.zeros((6,7))
    tree = MCTSTree(initial_board)
    tree.expand_node(0, initial_board)

    expected = np.array([[-1, -1,  0,  0,  0,  1],
       [ 0,  0,  0,  0,  0,  0],
       [ 0,  1,  0,  0,  0,  0],
       [ 0,  2,  0,  0,  0,  0],
       [ 0,  3,  0,  0,  0,  0],
       [ 0,  4,  0,  0,  0,  0],
       [ 0,  5,  0,  0,  0,  0],
       [ 0,  6,  0,  0,  0,  0]])

    node_data_head = tree.node_data[:8, :]
    assert np.array_equal(node_data_head, expected)


def test_backpropagation():
    initial_board = np.zeros((6, 7), dtype=int)  # Empty board
    tree = MCTSTree(initial_board)

    sample_col = 1
    tree._create_new_node(0, action_col=sample_col)
    tree.node_data[0, tree.EXPANDED_COL] = 1
    leaf_node, leaf_board, path_with_players = tree.select_leaf(0, initial_board)

    tree.backpropagate(path_with_players, 1)

    nodes_df = tree.to_pandas() 
    assert nodes_df.loc[0, "n_visits"] == 1
    assert nodes_df.loc[0, "wins"] == 0

    tree.backpropagate(path_with_players, -1)
    assert nodes_df.loc[0, "n_visits"] == 2
    assert nodes_df.loc[0, "wins"] == 1


def test_mcts_terminal_node():
    board_arr = np.zeros((6, 7), dtype=int)
    # Set a horizontal win in the bottom row for player 1:
    # Assume bottom row index = 5; winning columns = 0,1,2,3.
    board_arr[5, 0:4] = 1
    tree = MCTSTree(board_arr, player=1, iterations=10)
    pre_count = tree.node_count

    # Run an MCTS step. Since the board is terminal, no expansion should occur.
    for i in range(10):
        tree.mcts_step()

    # Node count remains unchanged since we never expand a terminal node
    assert tree.node_count == pre_count
    
    # The root node should reflect the terminal state
    assert tree.node_data[0, tree.N_VISITS_COL] == 10
    assert tree.node_data[0, tree.WINS_COL] == 10  # All visits are wins for player 1

    # With backpropagation, the root's wins should reflect a win.
    # get_training_sample uses _get_value on root: wins/n_visits,
    # which should equal 1 for a terminal win.
    value, _ = tree.get_training_sample()
    assert value == 1

    # Also, no children should have been added.
    assert tree.children_map == {}

def test_mcts_almost_terminal_node():
    board_arr =  np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, O, 0, 0, 0],
        [0, 0, 0, O, 0, 0, 0],
        [0, X, X, X, O, 0, 0]  # All pieces at bottom
    ], dtype=int)

    # Set a horizontal win in the bottom row for player 1:
    n = 100
    tree = MCTSTree(board_arr, player=1, iterations=n)
    # Run an MCTS step. Since the board is terminal, no expansion should occur.
    for i in range(n):
        tree.mcts_step()

    # child index 1 should be 100% win from parent players perspective


    node_data = tree.node_data
    assert node_data[1, tree.WINS_COL]/node_data[1, tree.N_VISITS_COL] == 1
    assert node_data[1, tree.EXPANDED_COL] == 0