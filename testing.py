import numpy as np

from fifteen_puzzle import FifteenPuzzle
from dqn_agent import DQNAgent

# Initialize the agent with the correct state and action sizes
state_size = 16  # 4x4 puzzle
action_size = 4  # 4 possible moves: UP, DOWN, LEFT, RIGHT
agent = DQNAgent(state_size, action_size, epsilon=0) #epsilon=0 when exploring states outside of the recommended states is not necessary (it is necessary when training but not when testing)

# Load the saved weights
agent.load("dqn_15_puzzle.weights.h5")

# Given 15-puzzle state
puzzle_state = np.array([[ 1,  2,  3,  4],
                       [ 5,  6,  7,  8],
                       [ 9, 10, 11, 12],
                       [13, 14, 0,  15]])

# Flatten the puzzle state to match the input shape expected by the model
state = puzzle_state.flatten()

# Use the agent to predict the next best move
next_move = agent.act(state)

# Map the predicted action to a readable move
action_map = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
suggested_move = action_map[next_move]



print(f"current configuration: \n {puzzle_state}" )
print(f"Suggested next move: move blank (zero) tile {suggested_move} if possible")
