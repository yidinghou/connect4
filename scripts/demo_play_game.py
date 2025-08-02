from connect4.mcts import MCTSTree
from connect4 import board
from connect4 import data_collector
import numpy as np
import time


O = -1
X = 1

board_arr = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
])


is_terminal = False
N = 2000
player = 1
while not is_terminal:
    start = time.time()
    tree = MCTSTree(board_arr, player=player, iterations=N)
    # Run an MCTS step. Since the board is terminal, no expansion should occur.
    for i in range(N):
        tree.mcts_step()

    best_move_col = int(tree.select_best_child())
    print(f"Best move for player {player}: {best_move_col}")

    legal_moves = board.get_legal_moves(board_arr)
    row = legal_moves[best_move_col]

    board_arr = board.add_move(board_arr, player=player, loc=(row, best_move_col))
    print(tree.to_pandas().head(8))
    board.pretty_print(board_arr)
    is_terminal = board.check_win(board_arr)
    print(data_collector.convert_mcts_nodes_data_to_target(
        tree.node_data, player_perspective=player))
    if is_terminal:
        print(f"is terminal: {is_terminal}")
    
    player = player * -1
    print(f"Time taken for MCTS step: {time.time() - start:.2f} seconds")
    print("--------------------------------------------------")