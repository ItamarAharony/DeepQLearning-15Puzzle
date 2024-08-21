import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

class DQN:
    def __init__(self, state_size, action_size, learning_rate = 1e-5):
        """
        Initialize the DQN (Deep Q-Network).
        :param state_size: The size of the input state (e.g., 16 for a 4x4 puzzle).
        :param action_size: The number of possible actions (e.g., 4 for UP, DOWN, LEFT, RIGHT).
        """
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()  # Build the neural network model
    
    def _build_model(self):
        """
        Build the DQN model using TensorFlow/Keras.
        The model architecture includes:
        - A reshaping layer to convert the input to 4x4x1 (for a 4x4 puzzle)
        - A convolutional layer to extract features on the scale of 2 x 2 puzzle tiles
        - Two dense layers for learning
        - An output layer with linear activation for Q-values.
        :return: The compiled Keras model.
        """
        model = models.Sequential()  # Initialize a sequential model

        # Reshape the input to (4, 4, 1) to represent the 4x4 board with a single channel
        model.add(layers.Reshape((4, 4, 1), input_shape=(self.state_size,)))
        
        # Add a convolutional layer with 32 filters and a 2x2 kernel, using ReLU activation
        model.add(layers.Conv2D(32, (2, 2), activation='relu', padding='same'))
        
        # Flatten the output of the convolutional layer to feed into dense layers
        model.add(layers.Flatten())
        
        # Add the first dense layer with 256 neurons and ReLU activation
        model.add(layers.Dense(256, activation='relu'))
        
        # Add the second dense layer with 256 neurons and ReLU activation
        model.add(layers.Dense(256, activation='relu'))
        
        # Output layer with 'action_size' neurons (one for each possible action) and linear activation
        model.add(layers.Dense(self.action_size, activation='linear'))
        
        # Compile the model with mean squared error loss and Adam optimizer
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=learning_rate))
                
        return model

    def predict(self, state):
        """
        Predict Q-values for a given state.
        :param state: The input state (expected to be preprocessed).
        :return: A numpy array containing Q-values for all possible actions.
        """
        return self.model.predict(state, verbose=0)

    def train(self, state, q_values):
        """
        Train the DQN model on a single state-Q-value pair.
        :param state: The input state (expected to be preprocessed).
        :param q_values: The target Q-values for the given state.
        """
        self.model.fit(state, q_values, epochs=1, verbose=0)

    def update_weights(self, other_model):
        """
        Update the weights of this model with the weights of another model.
        Typically used to synchronize the target model with the current model.
        :param other_model: The model whose weights are to be copied.
        """
        self.model.set_weights(other_model.get_weights())
