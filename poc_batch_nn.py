import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from connect4.mcts import MCTSTree

class MockNN(nn.Module): # SAVE WORKING, yiding hou (44 minutes ago)
    def __init__(self):
        super().__init__()
        self.name = "MockNN"

    def score(self, boards):
        # Mock scoring function
        # time.sleep(0.1) # Simulate some processing time
        return np.random.rand(len(boards)) # Random scores for all boards

class SimpleValueNN(nn.Module):
    """
    A simple neural network that takes a 6x7 board and predicts a single value.
    The input board has values -1, 0, or 1.
    The output value is a prediction of the game outcome (e.g., win probability).
    """

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(6 * 7, 128).to(self.device)
        self.fc2 = nn.Linear(128, 64).to(self.device)
        self.fc3 = nn.Linear(64, 1).to(self.device)
        self.eval() # Set to evaluation mode

    def forward(self, x):
        x = x.view(-1, 6 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = torch.sigmoid(self.fc3(x))
        return value

    def score(self, boards):
        """
        Score a batch of boards
        Args:
            boards: list[np.ndarray] or np.ndarray of shape (N, 6, 7)
        Returns:
            np.ndarray of shape (N,) with values between 0 and 1
        """
        with torch.no_grad():
            # Convert to tensor
            if isinstance(boards, list):
                boards = np.stack(boards)
            x = torch.FloatTensor(boards).to(self.device)
            
            # Forward pass
            values = self.forward(x)

            # Convert to numpy and flatten
            return values.cpu().numpy().flatten()


# Sequential MCTS
root_board = np.zeros((6, 7))
tree = MCTSTree(root_board, iterations=8000)

print("Initializing Neural Network...")
nn_init_start = time.time()
nn_runner = SimpleValueNN()
nn_init_time = time.time() - nn_init_start
print(f"Neural Network initialization took {nn_init_time:.2f} seconds")

start_time = time.time()
N = 1000
paths = []
leaf_boards: list[np.ndarray] = []
total_select_expand_time = 0
total_scoring_time = 0

for i in range(N):
    if i > 0 and i % 100  == 0:
        # Progress update every 100 iterations, skip first
        print(f"Sequential MCTS: {i}/{N} steps completed")
        print("Scoring NN...")
        scoring_start = time.time()
        predictions = nn_runner.score(leaf_boards)
        scoring_time = time.time() - scoring_start
        total_scoring_time += scoring_time
        print(f"Batch scoring took {scoring_time:.2f} seconds")

        # Batch backpropagate the predictions
        for path, prediction in zip(paths, predictions):
            tree.backpropagate(path, prediction)

        # Clear the paths and boards for next batch
        paths = []
        leaf_boards = []

    select_expand_start = time.time()
    leaf_node, leaf_board, path = tree.select_and_expand()
    select_expand_time = time.time() - select_expand_start
    total_select_expand_time += select_expand_time

    paths.append(path)
    leaf_boards.append(leaf_board)

# Process final batch if there are any remaining boards
if leaf_boards:
    print(f"Processing final batch with {len(leaf_boards)} boards...")
    scoring_start = time.time()
    predictions = nn_runner.score(leaf_boards)
    scoring_time = time.time() - scoring_start
    total_scoring_time += scoring_time
    print(f"Final batch scoring took {scoring_time:.2f} seconds")

    for path, prediction in zip(paths, predictions):
        tree.backpropagate(path, prediction)

total_time = time.time() - start_time
print("\nPerformance Summary:")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average select/expand time: {total_select_expand_time/N:.4f} seconds")
print(f"Total scoring time: {total_scoring_time:.2f} seconds")
print(f"Average scoring time per batch: {total_scoring_time/(N/100):.2f} seconds")

print("\nTree Analysis:")
print(tree.to_pandas().head(40))

# Add total script time at the end
script_total_time = time.time() - start_time
print(f"\nTotal script execution time: {script_total_time:.2f} seconds")