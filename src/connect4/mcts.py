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

        is_terminal, result = board.check_board_state_incremental(
            board_arr, row, col, player
        )
        player *= -1

    return result

class MCTSTree:
    def __init__(self, root_board, player, iterations=10):
        self.root_board = root_board
        self.player = player  # player who moves from root
        self.children_map = {}  # Maps (parent_idx, action) to child_idx
        
        data_size = iterations * 7 + 1  # Each iteration can have up to 7 children (one for each column), adding 1 for the root node
        self.node_data = np.zeros((data_size, 6), dtype=float)

        # 6 data elements: parent_idx, action_idx, n_visits, wins, prior, expanded
        self.PARENT_COL = 0
        self.ACTION_COL = 1
        self.N_VISITS_COL = 2
        self.WINS_COL = 3
        self.PRIOR_COL = 4
        self.EXPANDED_COL = 5

        self.node_data[0, self.PARENT_COL]=-1
        self.node_data[0, self.ACTION_COL] = -1 
        self.node_count = 1


    def select_leaf(self, node_idx, node_board):
        """
        Traverse tree to find a leaf node using UCT selection.

        Args:
            node_idx (int): Starting node index
            node_board (np.ndarray): Board state at starting node

        Returns:
            tuple: (leaf_node_idx, leaf_board_state)
        """
        current_player = self.player
        current_node = node_idx
        current_board = node_board.copy()
        path = [current_node]  # (node, player_to_move)

        # Keep traversing until we find a leaf
        while self.node_data[current_node, self.EXPANDED_COL] == 1:
            current_node, current_board, current_player = (
                self._traverse_one_step(current_node, current_board, current_player)
            )
            path.append(current_node)


        return current_node, current_board, path

    def _traverse_one_step(self, node_idx, board_state, player):
        """
        Perform one step of tree traversal using UCT selection among existing children.
        """
        # Check if game is over (no legal moves)
        legal_moves = board.get_legal_moves(board_state)
        if not legal_moves:
            return node_idx, board_state, player, True  # Terminal node - force break
        
        # Get existing children only
        existing_children = []
        for col in legal_moves.keys():
            child_key = (node_idx, col)
            if child_key in self.children_map:
                existing_children.append(col)
        
        if not existing_children:
            # This should never happen in Option 1 since we only call this on expanded nodes
            # But if it does, it means the node claims to be expanded but has no children
            raise ValueError(f"Node {node_idx} claims to be expanded but has no children")
        
        # Select among existing children (replace with UCT)
        col = random.choice(existing_children)
        row = legal_moves[col]
        
        new_board_state = board.add_move(board_state, player, (row, col))
        child_idx = self.children_map[(node_idx, col)]
        
        return child_idx, new_board_state, -player

    def expand_node(self, node_idx, board_state):
        """
        Expand a node by creating all possible child nodes.
        
        Args:
            node_idx (int): Index of node to expand
            board_state (np.ndarray): Board state at this node
        """
        legal_moves = board.get_legal_moves(board_state)
        
        # Create child for each legal move
        for col in legal_moves.keys():
            if (node_idx, col) not in self.children_map:
                self._create_new_node(node_idx, col)
        
        # Mark this node as expanded
        self.node_data[node_idx, self.EXPANDED_COL] = 1
        
    def _create_new_node(self, parent_idx, action_col):
        """
        Create a new node in the tree.

        Args:
            parent_idx (int): Index of parent node
            action_col (int): Column action that leads to this node

        Returns:
            int: Index of newly created node
        """
        # Create new node
        new_node_idx = self.node_count
        self.children_map[(parent_idx, action_col)] = new_node_idx

        # Initialize node data
        self.node_data[new_node_idx, self.PARENT_COL] = parent_idx
        self.node_data[new_node_idx, self.ACTION_COL] = action_col
        self.node_data[new_node_idx, self.N_VISITS_COL] = 0
        self.node_data[new_node_idx, self.WINS_COL] = 0
        self.node_data[new_node_idx, self.PRIOR_COL] = 0
        self.node_data[new_node_idx, self.EXPANDED_COL] = (
            0  # New nodes start unexpanded
        )

        self.node_count += 1
        return new_node_idx
        
    def backpropagate(self, path, value):
        """
        Backpropagate the result through the path using forward iteration.
        
        Args:
            path (list): List of node indices from root to leaf
            result (int): Game result (1 for player 1 win, -1 for player 2 win, 0 for draw)
        """
         
        # path is the first indexed element of the path_with_players list:

        self.node_data[path, self.N_VISITS_COL] += 1
        self.node_data[path[::2], self.WINS_COL] += value  # Update wins for player 1's turns
        self.node_data[path[1::2], self.WINS_COL] += (1-value)  # Update wins for player 2's turns

    def mcts_step(self):
        # 1. Selection - find leaf and track path
        leaf_node, leaf_board, path = self.select_leaf(0, self.root_board)
        
        # 2. Expansion - create children for the leaf
        self.expand_node(leaf_node, leaf_board)
        
        # 3. Simulation - random rollout from leaf
        result = rollout(leaf_board.copy(), self.player)
        
        # 4. Backpropagation - update statistics along path
        self.backpropagate(path, result)
    
    def select_and_expand(self):
        """
        Perform one MCTS step: select a leaf, expand it, simulate a rollout, and backpropagate the result.
        """
        leaf_node, leaf_board, path = self.select_leaf(0, self.root_board)
        
        # 2. Expansion - create children for the leaf
        self.expand_node(leaf_node, leaf_board)

        return leaf_node, leaf_board, path

    def to_pandas(self):
        """
        Convert the node data to a pandas DataFrame for easier analysis.
        
        Returns:
            pd.DataFrame: DataFrame representation of the tree node data
        """
        import pandas as pd
        columns = [
            "parent_idx", "action_col", "n_visits", "wins", "prior", "expanded"
        ]
        return pd.DataFrame(self.node_data, columns=columns)

    def _get_child(self):
        child = self.node_data[self.node_data[:, self.PARENT_COL] == 0]
        child = child[child[:, self.N_VISITS_COL] > 0]
        return child

    def _get_value(self):
        root_stats = self.node_data[0,:]
        return root_stats[self.WINS_COL] / root_stats[self.N_VISITS_COL]
    
    def get_training_sample(self):
        value = self._get_value()
        child = self._get_child()
        visits = child[:, self.N_VISITS_COL] / sum(child[:, self.N_VISITS_COL])
        return value, visits

# TODO: test mcts_step better