import numpy as np
import random
import copy

class FifteenPuzzle:
    # Constants representing the four possible move directions
    UP, DOWN, LEFT, RIGHT = range(4)

    def __init__(self, size=4):
        """
        Initialize the Fifteen Puzzle game.
        :param size: The size of the puzzle (default is 4x4).
        """
        self.size = size
        self.reset()

    def reset(self):
        """
        Reset the puzzle to the solved state.
        The solved state has tiles numbered 1 to (size^2 - 1) in order,
        with the last tile being the empty space (represented by 0).
        :return: The board in the solved state as a flattened array.
        """
        self.board = np.arange(1, self.size ** 2 + 1)  # Create a list from 1 to size^2
        self.board[-1] = 0  # Set the last tile to 0 (empty space)
        self.empty_position = self.size ** 2 - 1  # Track the empty space position
        return self.board

    def is_solved(self):
        """
        Check if the puzzle is in the solved state.
        :return: True if solved, otherwise False.
        """
        # The puzzle is solved if all tiles except the last are in order
        return np.all(self.board[:-1] == np.arange(1, self.size ** 2))

    def move(self, direction):
        """
        Attempt to move the empty space in the specified direction.
        :param direction: The direction to move (UP, DOWN, LEFT, RIGHT).
        :return: True if the Move is successful, otherwise False.
        """
        row, col = divmod(self.empty_position, self.size)  # Convert the linear index to 2D coordinates
        
        # Determine the target position based on the direction
        if direction == self.UP: 
            if row == 0:  # Cannot move up if the empty space is in the top row
                return False
            target = self.empty_position - self.size  # Move the empty space up one row
        elif direction == self.DOWN:
            if row == self.size - 1:  # Cannot move down if the empty space is in the bottom row
                return False
            target = self.empty_position + self.size  # Move the empty space down one row
        elif direction == self.LEFT: 
            if col == 0:  # Cannot move left if the empty space is in the leftmost column
                return False
            target = self.empty_position - 1  # Move the empty space left one column
        elif direction == self.RIGHT:  
            if col == self.size - 1:  # Cannot move right if the empty space is in the rightmost column
                return False
            target = self.empty_position + 1  # Move the empty space right one column
        else:
            return False  # Invalid direction

        # Swap the empty space with the target tile
        self.board[self.empty_position], self.board[target] = self.board[target], self.board[self.empty_position]
        self.empty_position = target  # Update the empty space position
        return True

    def shuffle(self, steps):
        """
        Shuffle the puzzle by making a series of random moves.
        :param steps: The number of random moves to make.
        """
        for _ in range(steps):
            self.move(random.choice([self.UP, self.DOWN, self.LEFT, self.RIGHT]))

    def get_state(self):
        """
        Get a copy of the current board state.
        :return: A copy of the board array.
        """
        return copy.deepcopy(self.board)

    def get_possible_actions(self):
        """
        Get a list of possible move directions based on the current position of the empty space.
        :return: A list of valid move directions (UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3).
        """
        row, col = divmod(self.empty_position, self.size)
        actions = []
        # Add each direction to the list if it's a valid move
        if row > 0: actions.append(self.UP)  # Can move up if not in the top row
        if row < self.size - 1: actions.append(self.DOWN)  # Can move down if not in the bottom row
        if col > 0: actions.append(self.LEFT)  # Can move left if not in the leftmost column
        if col < self.size - 1: actions.append(self.RIGHT)  # Can move right if not in the rightmost column
        return actions


    def print_state(self):
        """
        Print out the configuration of the board as a string, with '0' representing the empty square.
        """
        print(self.board.reshape(4,4))