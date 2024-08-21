from .training import train_dqn

if __name__ == "__main__":
    train_dqn(episodes=100, num_shuffles=0, max_steps_per_episode=10, epsilon=0.05, epsilon_decay=1, learning_rate=0.1, resume_training=False, filename="dqn_15_puzzle.weights.h5")
