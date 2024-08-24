# DeepQLearning-15Puzzle

# 15-Puzzle Solver using Deep Q-Learning (DQN)

This project implements a Deep Q-Learning (DQN) agent to solve the classic 15-puzzle game. The agent is trained using reinforcement learning to learn how to solve the puzzle from randomly shuffled states.

## Project Structure

The project is organized into the following files:

- **fifteen_puzzle.py**: Contains the `FifteenPuzzle` class, which defines the puzzle environment, including methods for moving tiles, checking if the puzzle is solved, shuffling, and getting the current state.
- **dqn_model.py**: Defines the `DQN` class, which builds and manages the neural network model used by the agent. The model includes a convolutional layer to process the puzzle's board state.
- **dqn_agent.py**: Contains the `DQNAgent` class, which implements the agent's behavior, including action selection, experience replay, and updating the target model.
- **training.py**: Contains the `train_dqn` function, which runs the training process for the DQN agent. It initializes the environment and agent, and then iteratively trains the agent through multiple episodes.
- **testing.py**: Script to test the trained model on a specific 15-puzzle configuration.

## How It Works

1. **Environment**: The 15-puzzle is represented as a 4x4 grid with numbers 1 through 15 and an empty space (0). The goal is to move the tiles to achieve a solved state where the numbers are in order from 1 to 15 with the empty space in the bottom-right corner.

2. **DQN Agent**: The agent uses a neural network to estimate Q-values for each possible action (up, down, left, right) given the current state of the puzzle. The agent selects actions based on these Q-values to maximize future rewards.

3. **Training**: The agent is trained over a series of episodes. In each episode, the puzzle is shuffled, and the agent attempts to solve it within a limited number of steps. The agent's experience is stored in memory and used to train the neural network through experience replay. The target model is periodically updated to stabilize training.

## Requirements

- Python 3.6+
- NumPy
- TensorFlow
- Matplotlib (optional, for visualization if added)

You can install the required packages using pip:

```
pip install numpy tensorflow matplotlib
```


## Running the Training code

To train the DQN agent, run the `main.py` script:

```
python main.py
```

You can customize the training parameters, such as the number of episodes, learning rate, and epsilon settings, directly in the `train_dqn` function call.


###Example Output
The training process will output information about each episode, including the number of steps taken to solve the puzzle (or a timeout if the puzzle is not solved within the step limit):

```
Episode: 0/10, Score: 9, Epsilon: 0.05
Episode: 1/10, Score: 15, Epsilon: 0.05
...
Episode: 9/10, Score: 8, Epsilon: 0.05
```
## Testing the Model

After training the model, you can test it on a specific puzzle configuration using the testing.py file.

1. Ensure the model weights are saved (e.g., `dqn_15_puzzle.weights.h5`).
2. Run the `testing.py` script:
```
python testing.py
```
This script will load the trained model and predict the best next move for a given puzzle configuration.
### Example:
Given the following 15-puzzle state:

```
1  2  3  4
5  6  7  8
9  10 11 12
13 14 0  15
```
where '0' is the empty tile.

```
Current configuration: 
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14  0  15]]

Suggested next move: move blank (zero) tile RIGHT if possible
```

## Future Improvements
- Visualization: Add a visualization of the puzzle and the agent's actions during training.
- Advanced Training Techniques: Experiment with different neural network architectures, reward functions, and training strategies to improve performance.
- Generalization: Extend the agent to solve puzzles of different sizes (e.g., 3x3 or 5x5).

