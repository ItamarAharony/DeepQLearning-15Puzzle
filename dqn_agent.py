import random
import numpy as np
import tensorflow as tf
from collections import deque
from dqn_model import DQN

class DQNAgent:
    def __init__(self, state_size, action_size, epsilon=1.0, epsilon_decay=0.999, learning_rate=0.01, load_weights=False, weights_path="dqn_15_puzzle.weights.h5"):
        """
        Initialize the DQN agent.
        :param state_size: Size of the input state (e.g., 16 for a 4x4 puzzle).
        :param action_size: Number of possible actions (e.g., 4 for UP, DOWN, LEFT, RIGHT).
        :param epsilon: Initial exploration rate (probability of taking a random action).
        :param epsilon_decay: Decay rate for epsilon, controlling how exploration decreases over time.
        :param learning_rate: Learning rate for the DQN model.
        :param load_weights: If True, load weights from a file.
        :param weights_path: Path to the weights file.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay memory to store past experiences
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration
        self.learning_rate = learning_rate  # Learning rate for the DQN model
        self.model = DQN(state_size, action_size)  # Main DQN model
        self.target_model = DQN(state_size, action_size)  # Target DQN model for stability in training
        if load_weights:
            self.load(weights_path)  # Load weights if specified
        self.update_target_model()  # Initialize the target model with the same weights as the main model

    def update_target_model(self):
        """
        Update the target model's weights with the weights of the main model.
        This is done to stabilize training, as the target model is used to calculate target Q-values.
        """
        self.target_model.update_weights(self.model.model)

    def act(self, state):
        """
        Choose an action based on the current state using an epsilon-greedy policy.
        :param state: The current state of the environment.
        :return: The chosen action (an integer representing UP, DOWN, LEFT, or RIGHT).
        """
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))  # Random action (exploration)
        state = np.reshape(state, [1, self.state_size])  # Reshape state for the model
        state = tf.convert_to_tensor(state)  # Convert the state to a tensor for TensorFlow
        act_values = self.model.predict(state)  # Predict Q-values for the state
        return np.argmax(act_values[0])  # Choose the action with the highest Q-value (exploitation)

    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience in the replay memory.
        :param state: The current state of the environment.
        :param action: The action taken.
        :param reward: The reward received after taking the action.
        :param next_state: The state of the environment after taking the action.
        :param done: A boolean indicating if the episode has ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """
        Train the DQN model by sampling a batch of experiences from the replay memory.
        :param batch_size: The number of experiences to sample for training.
        """
        minibatch = random.sample(self.memory, batch_size)  # Sample a random batch from memory
        for state, action, reward, next_state, done in minibatch:
            target = reward  # Start with the immediate reward
            if not done:
                # If the episode is not done, add the discounted future reward
                next_state = np.reshape(next_state, [1, self.state_size])
                next_state = tf.convert_to_tensor(next_state)
                target = (reward + self.gamma *
                          np.amax(self.target_model.predict(next_state)[0]))
            state = np.reshape(state, [1, self.state_size])
            state = tf.convert_to_tensor(state)  # Convert state to a tensor for TensorFlow
            target_f = self.model.predict(state)  # Predict the current Q-values
            target_f[0][action] = target  # Update the Q-value for the action taken
            self.model.train(state, target_f)  # Train the model with the updated Q-values
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Decay the exploration rate

    def load(self, name):
        """
        Load model weights from a file.
        :param name: Path to the weights file.
        """
        self.model.model.load_weights(name)

    def save(self, name):
        """
        Save model weights to a file.
        :param name: Path to save the weights file.
        """
        self.model.model.save_weights(name)
