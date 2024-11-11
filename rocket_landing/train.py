# rocket_landing/train.py

import gym
from rocket_landing.envs.rocket_landing_env import RocketLandingEnv
from rocket_landing.models.dqn_agent import DQNAgent
from rocket_landing.utils.utils import set_seed, save_model, plot_learning_curve

def train():
    env = RocketLandingEnv()
    agent = DQNAgent(env)
    set_seed(42)
    
    episodes = 1000
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        rewards.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode {episode} - Total Reward: {total_reward}")

    plot_learning_curve(rewards, filename="learning_curve.png")
    save_model(agent.model, "models/dqn_rocket_landing.pth")

if __name__ == "__main__":
    train()
