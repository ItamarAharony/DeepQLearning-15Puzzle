# DeepQLearning-15Puzzle


# NOTE: Project is still work in progress. Future updates will add more features for tinkering with the model's and training process' parameters. 

# 15-Puzzle Solver using Deep Q-Learning (DQN)

This project implements a Deep Q-Learning (DQN) agent to solve the classic 15-puzzle game. The agent is trained using reinforcement learning to learn how to solve the puzzle from randomly shuffled states.

## Project Structure

The project is organized into the following files:

- **fifteen_puzzle.py**: Contains the `FifteenPuzzle` class, which defines the puzzle environment, including methods for moving tiles, checking if the puzzle is solved, shuffling, and getting the current state.
- **dqn_model.py**: Defines the `DQN` class, which builds and manages the neural network model used by the agent. The model includes a convolutional layer to process the puzzle's board state.
- **dqn_agent.py**: Contains the `DQNAgent` class, which implements the agent's behavior, including action selection, experience replay, and updating the target model.
- **training.py**: Contains the `train_dqn` function, which runs the training process for the DQN agent. It initializes the environment and agent, and then iteratively trains the agent through multiple episodes.

## How It Works

1. **Environment**: The 15-puzzle is represented as a 4x4 grid with numbers 1 through 15 and an empty space (0). The goal is to move the tiles to achieve a solved state where the numbers are in order from 1 to 15 with the empty space in the bottom-right corner.

2. **DQN Agent**: The agent uses a neural network to estimate Q-values for each possible action (up, down, left, right) given the current state of the puzzle. The agent selects actions based on these Q-values to maximize future rewards.

3. **Training**: The agent is trained over a series of episodes. In each episode, the puzzle is shuffled, and the agent attempts to solve it within a limited number of steps. The agent's experience is stored in memory and used to train the neural network through experience replay. The target model is periodically updated to stabilize training.

## Requirements

- Python 3.x
- NumPy
- TensorFlow
- Matplotlib (optional, for visualization if added)

You can install the required packages using pip:

```
pip install numpy tensorflow matplotlib
```


## Running the Code

To train the DQN agent, run the `training.py` script:

```
python training.py
```
