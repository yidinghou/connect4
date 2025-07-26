import pytest
import numpy as np

@pytest.fixture
def empty_board_arr():
    return np.zeros((6, 7), dtype=int)

