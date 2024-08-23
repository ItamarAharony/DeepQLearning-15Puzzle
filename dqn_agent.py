import random
import numpy as np
import tensorflow as tf
from collections import deque
from dqn_model import DQN

class DQNAgent:
    def __init__(self, state_size, action_size, epsilon=1.0, epsilon_decay=0.999, learning_rate=1e-5, load_weights=False, weights_path="dqn_15_puzzle_model.h5"):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200)
      
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        self.gamma = 0.95  # discount rate
        self.epsilon_min = 0.01
        
        self.model = DQN(state_size, action_size, learning_rate = learning_rate)
        self.target_model = DQN(state_size, action_size, learning_rate = learning_rate)
        
        if load_weights:
            self.model.load(weights_path)
        self.update_target_model()

    def save(self, file_path):
        """
        Save the DQNAgent's model.
        """
        self.model.save(file_path)

    def load(self, file_path):
        """
        Load the DQNAgent's model.
        """
        self.model.load(file_path)

    def update_target_model(self):
        """
        Update the target model's weights with the current model's weights.
        """
        self.target_model.update_weights(self.model)

    def remember(self, state, action, reward, next_state, done):
        """
        Store the experience in the replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose an action based on the current state using an epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        """
        Train the model using experience replay.
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.target_model.model.predict(next_state.reshape(1, -1),verbose=0)[0])
            target_f = self.model.model.predict(state.reshape(1, -1),verbose=0)
            target_f[0][action] = target
            self.model.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
