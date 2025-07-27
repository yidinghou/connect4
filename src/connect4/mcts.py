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

        board_arr = board.add_move(board_arr, player=player, loc=(row, col))
        if debug:
            print(board_arr)
            print(f"Player {player} added move at ({row}, {col})")
            assert board.check_valid_board(board_arr)

        is_terminal, result = board.check_board_state_incremental(board_arr, row, col, player)
        player *= -1

    return result

# Initialize nodes data structure
N = 100
NODES_DATA = np.zeros((N, 6), dtype=int)

#adding parent node
NODES_DATA[:, PARENT_COL] = -1  # parent_idx
NODES_DATA[:, ACTION_COL] = -1  # action_idx
NEXT_IDX = 1

PLAYER = 1

def traverse_select(node_idx, node_board):
    """
    Traverse the tree to select a node based on UCT (Upper Confidence Bound for Trees).
    Returns:
        int: The index of the selected node.
        board_arr: np.ndarray: The board state after the selected node's action.
    """

    if NODES_DATA[node_idx, EXPANDED_COL] == 0:
        # We've found a leaf node, so selection is over.
        return node_idx, node_board
    
    children = get_children(node_idx)
    best_action = score_select_uct(children)
    node_board = board.add_move(node_board, PLAYER, best_action)

    child_key = (parent_idx, action)
    if child_key is NOT in children_map: # then add to tree
        NODES_DATA[NEXT_IDX, PARENT_COL] = node_idx
        NODES_DATA[NEXT_IDX, ACTION_COL] = best_action
        node_idx = NEXT_IDX
    else:
        node_idx = children_map[child_key]



class MCTSTree:

    def __init__(self, root_board):
        self.root_board = root_board
        self.player = 1 # Starting player
        self.children_map = {}  # Maps (parent_idx, action) to child_idx
        self.node_data = np.zeros((100, 6), dtype=int)  # Initial size, can be resized later

        self.node_count = 0
        # 6 data elements: parent_idx, action_idx, n_visits, wins, prior, expanded
        self.PARENT_COL = 0
        self.ACTION_COL = 1
        self.N_VISITS_COL = 2
        self.WINS_COL = 3
        self.PRIOR_COL = 4
        self.EXPANDED_COL = 5

    def select_leaf(self, node_idx, node_board):
        current_player = self.player
        current_node = node_idx
        current_board = node_board.copy()

        found_leaf = False
        while self.node_data[current_node, self.EXPANDED_COL] == 1:  # While not a leaf
            legal_moves = board.get_legal_moves(current_board)
            if not legal_moves:  # Board is full
                break
                
            # For now, use random selection (replace with UCT)
            col = random.choice(list(legal_moves.keys()))
            row = legal_moves[col]
            best_action = (row, col)
            
            current_board = board.add_move(node_board, current_player, best_action)
            child_key = (current_node, col)  # Use column as action identifier

            if child_key not in self.children_map:
                # Create new leaf node
                self.children_map[child_key] = self.node_count
                self.node_data[self.node_count, self.PARENT_COL] = current_node
                self.node_data[self.node_count, self.ACTION_COL] = col
                current_node = self.node_count
                self.node_count += 1
                break  # Found leaf, exit loop
            else:
                current_node = self.children_map[child_key]
            
            current_player *= -1
        
        return current_node, current_board

        if self.node_data[node_idx, self.EXPANDED_COL] == 0:
            return node_idx, node_board
        
        is_leaf = False
        while not is_leaf:
            # children = get_children(node_idx)
            # best_action = score_select_uct(children)
            best_action = 1
            node_board = board.add_move(node_board, self.player, best_action)

            child_key = (node_idx, best_action)
            if child_key not in self.children_map:
                is_leaf = True

                # add to children_map
                self.children_map[child_key] = self.node_count

                # add to node_data
                self.node_data[self.node_count, self.PARENT_COL] = node_idx
                self.node_data[self.node_count, self.ACTION_COL] = best_action
                node_idx = self.node_count
                self.node_count += 1
            else:
                node_idx = self.children_map[child_key]

            self.player *= -1

        return node_idx, node_board

node_board = np.zeros((6,7))
tree = MCTSTree(node_board)