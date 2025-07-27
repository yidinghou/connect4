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
    def __init__(self, root_board, iterations=10):
        self.root_board = root_board
        self.player = 1  # player who moves from root
        self.children_map = {}  # Maps (parent_idx, action) to child_idx
        self.node_data = np.zeros(
            (iterations, 6), dtype=int
        )  # Initial size, can be resized later

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
        path_with_players = [(current_node, -1 * current_player)]  # (node, player_to_move)

        # Keep traversing until we find a leaf
        while self.node_data[current_node, self.EXPANDED_COL] == 1:
            current_node, current_board, current_player = (
                self._traverse_one_step(current_node, current_board, current_player)
            )
            path_with_players.append((current_node, -1 *current_player))


        return current_node, current_board, path_with_players

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
        
    def backpropagate(self, path_with_players, result):
        """
        Backpropagate the result through the path using forward iteration.
        
        Args:
            path (list): List of node indices from root to leaf
            result (int): Game result (1 for player 1 win, -1 for player 2 win, 0 for draw)
        """
         
        # path is the first indexed element of the path_with_players list:
        path, players = map(list, zip(*path_with_players))

        if result == 0:
            self.node_data[path, self.WINS_COL] += 0.5
        elif result == self.player:
            # only update even indexed nodes
            self.node_data[path[::2], self.WINS_COL] += 1
        elif result == -self.player:
            self.node_data[path[1::2], self.WINS_COL] += 1
    
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

            
# initial_board = np.zeros((6,7))
# tree = MCTSTree(initial_board)