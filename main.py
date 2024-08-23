from training import train_dqn

if __name__ == "__main__":
    """
    The file to which the Deep-Q Network weights and biases will be saved is filename="dqn_15_puzzle.weights.h5".
    If you are running this code while a file called dqn_15_puzzle.weights.h5 does not exist in this program's directory, keep resume_training=False and
    the h5 file will be generated.
    Otherwise, if dqn_15_puzzle.weights.h5 exists, resume_training=False will overwrite the file while resume_training=True will resume the training from the saved model
    in dqn_15_puzzle.weights.h5.
    """
    for i in range(1,12):
        print(f"step {i}")
        print("==============================")
        train_dqn(episodes=100, num_shuffles=i, max_steps_per_episode=50, epsilon=0.3/i, epsilon_decay=1, learning_rate=1e-3, resume_training=True, filename="dqn_15_puzzle.weights.h5")
        