# rocket_landing/evaluate.py

import torch
from rocket_landing.envs.rocket_landing_env import RocketLandingEnv
from rocket_landing.models.dqn_agent import DQNAgent
from rocket_landing.utils.utils import load_model

def evaluate():
    env = RocketLandingEnv()
    agent = DQNAgent(env)
    load_model(agent.model, "models/dqn_rocket_landing.pth")
    
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state, epsilon=0)
        state, reward, done, _ = env.step(action)
        env.render()

    env.close()

if __name__ == "__main__":
    evaluate()
