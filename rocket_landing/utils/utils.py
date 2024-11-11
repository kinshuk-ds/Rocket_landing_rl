# rocket_landing/utils/utils.py

import numpy as np
import random
import torch
import os

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_observation(obs, min_values, max_values):
    return 2 * (obs - min_values) / (max_values - min_values) - 1

def save_model(model, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath: str):
    model.load_state_dict(torch.load(filepath))
    print(f"Model loaded from {filepath}")
    return model

def plot_learning_curve(rewards, filename='learning_curve.png'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Learning curve saved as {filename}")
