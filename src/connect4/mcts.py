"""Monte Carlo Tree Search (MCTS) for Connect 4 game."""

from connect4 import board
import numpy as np
import random
import math

global PARENT_COL, ACTION_COL, N_VISITS_COL, WINS_COL, PRIOR_COL, EXPANDED_COL
PARENT_COL = 0
ACTION_COL = 1
N_VISITS_COL = 2
WINS_COL = 3
PRIOR_COL = 4
EXPANDED_COL = 5

class MCTSTree:
    def __init__(self, root_board, player=1, iterations=10, exploration_factor=math.sqrt(2)):
        self.root_board = root_board
        self.player = player  # player who moves from root
        self.children_map = {}  # Maps (parent_idx, action) to child_idx
        
        data_size = iterations * 7 + 1  # Each iteration can have up to 7 children (one for each column), adding 1 for the root node
        self.node_data = np.zeros((data_size, 6), dtype=float)

        self.node_data[0, PARENT_COL] = -1
        self.node_data[0, ACTION_COL] = -1 
        self.node_count = 1
        self.exploration_factor = exploration_factor

    def mcts_step(self):
        """
        Perform one complete MCTS iteration: select, expand, simulate, and backpropagate.
        """
        # 1 & 2. Selection and Expansion (with virtual loss)
        leaf_node, leaf_board, path = self.select_and_expand()
        
        # 3. Simulation - random rollout from leaf
        result = rollout(leaf_board.copy(), self.player)
        value = (result + 1) / 2

        # 4. Backpropagation - update statistics along path
        self.backpropagate(path, value)
    
    def select_and_expand(self):
        """
        Select a leaf node and expand it. Used for batching simulations.
        
        Returns:
            tuple: (leaf_node_idx, leaf_board_state, path)
        """
        leaf_node, leaf_board, path = self.select_leaf(0, self.root_board)
        
        # Apply virtual loss and expand
        self.apply_virtual_loss(path)
        self.expand_node(leaf_node, leaf_board)

        return leaf_node, leaf_board, path

    def select_leaf(self, node_idx, node_board):
        """
        Traverse tree to find a leaf node using UCT selection.

        Args:
            node_idx (int): Starting node index
            node_board (np.ndarray): Board state at starting node

        Returns:
            tuple: (leaf_node_idx, leaf_board_state, path)
        """
        current_player = self.player
        current_node = node_idx
        current_board = node_board.copy()
        path = [current_node]

        # Keep traversing until we find a leaf
        while self.node_data[current_node, EXPANDED_COL] == 1:
            legal_moves = board.get_legal_moves(current_board)
            if not legal_moves:
                break  # Terminal node
            
            # Gather children and corresponding UCT scores
            child_scores = {}
            parent_visits = self.node_data[current_node, N_VISITS_COL]
            for col in legal_moves.keys():
                child_key = (current_node, col)
                if child_key in self.children_map:
                    child_idx = self.children_map[child_key]
                    score = uct_score(self.node_data, child_idx, parent_visits, self.exploration_factor)
                    child_scores[col] = score

            if not child_scores:
                raise ValueError(f"Node {current_node} is marked expanded but has no children")

            # Select column with highest UCT score
            selected_col = weighted_sample(child_scores)
            row = legal_moves[selected_col]
            current_board = board.add_move(current_board, current_player, (row, selected_col))
            current_node = self.children_map[(current_node, selected_col)]
            current_player = -current_player
            path.append(current_node)

        return current_node, current_board, path

    def expand_node(self, node_idx, board_state):
        """
        Expand a node by creating all possible child nodes.
        
        Args:
            node_idx (int): Index of node to expand
            board_state (np.ndarray): Board state at this node
        """
        is_terminal, _ = board.check_board_state(board_state)
        if is_terminal:
            return
        
        legal_moves = board.get_legal_moves(board_state)
        
        # Create child for each legal move
        for col in legal_moves.keys():
            if (node_idx, col) not in self.children_map:
                self._create_new_node(node_idx, col)
        
        # Mark this node as expanded
        self.node_data[node_idx, EXPANDED_COL] = 1
        
    def backpropagate(self, path, value):
        """
        Backpropagate the result through the path using forward iteration.

        Args:
            path (list): List of node indices from root to leaf
            result (int): Game result (1 for player 1 win, -1 for player 2 win,
        """

        # If root player is -1, flip the initial pattern       fix back prop log
        if self.player == -1:
            self.node_data[path[::2], WINS_COL] += value     # P2's turns
            self.node_data[path[1::2], WINS_COL] += (1-value) # P1's turns
        else:
            self.node_data[path[1::2], WINS_COL] += value     # P1's turns
            self.node_data[path[::2], WINS_COL] += (1-value) # P2's turns

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
        self.node_data[new_node_idx, PARENT_COL] = parent_idx
        self.node_data[new_node_idx, ACTION_COL] = action_col
        self.node_data[new_node_idx, N_VISITS_COL] = 0
        self.node_data[new_node_idx, WINS_COL] = 0
        self.node_data[new_node_idx, PRIOR_COL] = 0
        self.node_data[new_node_idx, EXPANDED_COL] = (
            0  # New nodes start unexpanded
        )

        self.node_count += 1
        return new_node_idx

    def apply_virtual_loss(self, path, loss=0.1):
        """
        Apply a virtual loss to each node along the path.
        """
        self.node_data[path, N_VISITS_COL] += 1
        # For example, subtract virtual loss from wins for the player's turn
        self.node_data[path, WINS_COL] -= loss

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

    def select_best_child(self):
        """
        Select the child with the highest visit count from the root node.

        Returns:
            int: Column index of the best child action
        """
        # root children are when parent_idx is 0 and 
        root_children = self.node_data[:10]
        root_children = root_children[root_children[:, PARENT_COL]==0]
        wins = root_children[:, WINS_COL] / root_children[:, N_VISITS_COL]
        best_child = root_children[np.argmax(wins), ACTION_COL]
        return best_child

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


def weighted_sample(child_scores: dict) -> int:
    """
    If multiple actions share the maximum UCT score, choose one uniformly at random.
    Otherwise, return the unique max.
    
    Args:
        child_scores (dict): mapping of action column -> score

    Returns:
        int: a selected column
    """
    max_score = max(child_scores.values())
    # Using a tolerance if scores are floats:
    candidates = [col for col, score in child_scores.items() if abs(score - max_score) < 1e-8]
    if len(candidates) > 1:
        return random.choice(candidates)
    return candidates[0]


def uct_score(node_data, child_idx, parent_visits, exploration_factor):
    """
    Calculate the UCT (Upper Confidence Bound applied to Trees) score for a child node.
    
    Args:
        node_data (np.ndarray): Array containing node statistics
        child_idx (int): Index of the child node
        parent_visits (int): Number of visits to the parent node
        exploration_factor (float): Exploration constant (typically sqrt(2))
        
    Returns:
        float: UCT score for the child node
    """
    wins = node_data[child_idx, WINS_COL]
    visits = node_data[child_idx, N_VISITS_COL]
    if visits == 0:
        # Favor unexplored children
        return 10e6
    exploitation = wins / visits
    exploration = exploration_factor * math.sqrt(math.log(parent_visits) / visits)
    return exploitation + exploration