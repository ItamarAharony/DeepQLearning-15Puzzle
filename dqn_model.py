from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

class DQN:
    def __init__(self, state_size, action_size, learning_rate=1e-5):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        """
        Build a convolutional neural network model for the 15-puzzle.
        """
        model = Sequential()

        # Reshape the input to (4, 4, 1) to represent the 4x4 board with a single channel
        model.add(Reshape((4, 4, 1), input_shape=(self.state_size,)))

        # Convolutional layer with 32 filters, 2x2 kernel, ReLU activation
        model.add(Conv2D(32, (2, 2), activation='relu', padding='same'))

        # Flatten the output of the convolutional layer
        model.add(Flatten())

        # Two dense layers with 256 units each
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))

        # Output layer with the number of actions as units
        model.add(Dense(self.action_size, activation='linear'))

        # Compile the model with Mean Squared Error loss and Adam optimizer
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def save(self, file_path):
        """
        Save only the model's weights to the specified file path.
        """
        self.model.save_weights(file_path)


    def load(self, file_path):
        """
        Load the model's weights from the specified file path.
        """
        self.model.load_weights(file_path)


    def update_weights(self, other_model):
        """
        Update the model's weights with those from another model.
        :param other_model: Another DQN model object from which to copy weights.
        """
        self.model.set_weights(other_model.model.get_weights())