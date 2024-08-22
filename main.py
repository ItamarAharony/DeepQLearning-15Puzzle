from training import train_dqn

if __name__ == "__main__":
    """
    The file to which the Deep-Q Network weights and biases will be saved is filename="dqn_15_puzzle.weights.h5".
    If you are running this code while a file called dqn_15_puzzle.weights.h5 does not exist in this program's directory, keep resume_training=False and
    the h5 file will be generated.
    Otherwise if dqn_15_puzzle.weights.h5 exists, resume_training=False will overwrite the file while resume_training=True will resume the training from the saved model
    in dqn_15_puzzle.weights.h5.
    """
    train_dqn(episodes=10, num_shuffles=0, max_steps_per_episode=10, epsilon=0.05, epsilon_decay=1, learning_rate=0.1, resume_training=False, filename="dqn_15_puzzle.weights.h5")
