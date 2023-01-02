import numpy as np
import gym
import random
import time

# 1. Environment Setup
env = gym.make("FrozenLake-v1", is_slippery=False) # Deterministic version for easier learning

# 2. Q-Table Initialization
# Q-table has dimensions (states, actions)
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# 3. Hyperparameters
LEARNING_RATE = 0.9       # Alpha (α) - how much new information overrides old
DISCOUNT_FACTOR = 0.8     # Gamma (γ) - importance of future rewards
EPSILON = 1.0             # Epsilon (ε) - exploration-exploitation trade-off
EPSILON_DECAY_RATE = 0.001
RANDOM_SEED = 42

EPISODES = 10000          # Total number of episodes to train the agent
MAX_STEPS_PER_EPISODE = 100 # Max steps per episode to prevent infinite loops

# Set random seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)

# 4. Q-Learning Algorithm
def train_q_learning(q_table, env, episodes, max_steps, lr, gamma, epsilon, epsilon_decay_rate):
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()[0] # Reset environment and get initial state
        done = False
        rewards_current_episode = 0

        for step in range(max_steps):
            # Exploration-exploitation trade-off
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state,:]) # Exploit learned values

            # Take action and observe new state and reward
            new_state, reward, done, _, _ = env.step(action)

            # Update Q-table
            q_table[state, action] = q_table[state, action] * (1 - lr) + \
                                     lr * (reward + gamma * np.max(q_table[new_state, :]))

            state = new_state
            rewards_current_episode += reward

            if done:
                break
        
        # Decay epsilon to reduce exploration over time
        epsilon = max(0.01, epsilon - epsilon_decay_rate)
        rewards_per_episode.append(rewards_current_episode)

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}: Average reward = {sum(rewards_per_episode[-1000:]) / 1000}")

    print("\nTraining finished.\n")
    return q_table, rewards_per_episode

# 5. Test the trained agent
def test_agent(q_table, env, episodes=100):
    total_rewards = 0
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        rewards_current_episode = 0
        for step in range(MAX_STEPS_PER_EPISODE):
            action = np.argmax(q_table[state,:]) # Take greedy action
            new_state, reward, done, _, _ = env.step(action)
            rewards_current_episode += reward
            state = new_state
            if done:
                break
        total_rewards += rewards_current_episode
    print(f"Agent achieved an average reward of {total_rewards / episodes} over {episodes} test episodes.")

if __name__ == "__main__":
    print("Starting Q-Learning training...")
    trained_q_table, episode_rewards = train_q_learning(
        q_table.copy(), env, EPISODES, MAX_STEPS_PER_EPISODE, 
        LEARNING_RATE, DISCOUNT_FACTOR, EPSILON, EPSILON_DECAY_RATE
    )
    print("\nTesting trained agent...")
    test_agent(trained_q_table, env)
    env.close()
