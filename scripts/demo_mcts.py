from connect4.mcts import MCTSTree
import numpy as np  

board_arr = np.zeros((6, 7), dtype=int)  # Create an empty board
N = 1000
tree = MCTSTree(board_arr, player=1, iterations=N)
# Run an MCTS step. Since the board is terminal, no expansion should occur.
for i in range(N):
    tree.mcts_step()

print(tree.to_pandas().head(10))