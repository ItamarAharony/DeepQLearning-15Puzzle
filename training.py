from fifteen_puzzle import FifteenPuzzle
from dqn_agent import DQNAgent

def train_dqn(episodes, num_shuffles=1, max_steps_per_episode=10, epsilon=1.0, epsilon_decay=0.999, learning_rate=0.01, resume_training=False, filename="dqn_15_puzzle.weights.h5"):
    """
    Train a DQN agent to solve the 15-puzzle game.
    :param episodes: Number of episodes (games) to train the agent.
    :param num_shuffles: Number of random moves to shuffle the puzzle at the start of each episode.
    :param max_steps_per_episode: Maximum number of steps (moves) allowed per episode.
    :param epsilon: Initial exploration rate (probability of taking a random action).
    :param epsilon_decay: Decay rate for epsilon, controlling how exploration decreases over time.
    :param learning_rate: Learning rate for the DQN model.
    :param resume_training: If True, load existing weights and continue training.
    :param filename: Name of the file to save/load the model weights.
    """
    
    # Initialize the environment (15-puzzle game) and agent
    env = FifteenPuzzle()  # The environment representing the 15-puzzle game
    state_size = env.size * env.size  # The size of the state (e.g., 16 for a 4x4 puzzle)
    action_size = 4  # Number of possible actions (UP, DOWN, LEFT, RIGHT)
    agent = DQNAgent(state_size, action_size, epsilon=epsilon, epsilon_decay=epsilon_decay, learning_rate=learning_rate, load_weights=resume_training, weights_path=filename)
    
    batch_size = 32  # Size of the minibatch for experience replay

    # Loop through each episode (game)
    for e in range(episodes):
        env.reset()  # Reset the puzzle to the solved state
        env.shuffle(num_shuffles)  # Shuffle the puzzle to create a new starting state
        state = env.get_state()  # Get the initial state of the puzzle

        # Loop through each step (move) in the episode
        for time in range(max_steps_per_episode):
            action = agent.act(state)  # Choose an action based on the current state
            if not env.move(action):
                continue  # Skip to the next iteration if the move is invalid

            next_state = env.get_state()  # Get the new state after the move
            reward = -1  # Default penalty for each move to encourage faster solutions
            
            # Check if the puzzle is solved
            if env.is_solved():
                reward = 100  # Reward for solving the puzzle
                done = True  # Mark the episode as done
            else:
                done = False  # Continue the episode

            # Store the experience in memory
            agent.remember(state, action, reward, next_state, done)

            state = next_state  # Update the current state

            # Break the loop if the puzzle is solved
            if done:
                print(f"Episode: {e}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2}")
                break

            # Break the loop if the maximum number of steps is reached
            if time >= max_steps_per_episode - 1:
                print(f"Episode: {e}/{episodes}, Timeout at {time} steps")
                break
        
        # Train the agent with the experiences in memory if enough samples are available
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        # Update the target model every 10 episodes for stability
        if e % 10 == 0:
            agent.update_target_model()

    # Save the trained model weights
    agent.save(filename)

# Train the DQN agent with the specified parameters
train_dqn(episodes=100, num_shuffles=0, max_steps_per_episode=10, epsilon=0.05, epsilon_decay=1, learning_rate=0.1, resume_training=False, filename="dqn_15_puzzle.weights.h5")
