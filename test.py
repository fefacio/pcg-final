from train import MODEL_NAME, LOG_DIR


import gymnasium as gym
import numpy as np
import pandas as pd
import time
import os 

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

SAVE_DIR = './results/test/'
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_FILE = os.path.join(SAVE_DIR, MODEL_NAME)
env = gym.make("Maze-v0", render_mode="human")
print(f'Model name: f{MODEL_NAME}')
print(LOG_DIR)
model = PPO.load(os.path.join(LOG_DIR, "best_model.zip"))
obs, info = env.reset()
total_term = 0
total_trunc = 0

# Coletar dados de teste
test_data = []
episode_reward = 0
episode_length = 0
episode_id = 0

test_length = 1_000
print(f'Starting testing loop')
while episode_id < test_length:
    env.render()  
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    episode_reward += reward
    episode_length += 1

    if terminated or truncated:
        test_data.append({
            "episode": episode_id,
            "reward_total": episode_reward,
            "length": episode_length,
            "terminated": terminated,
            "truncated": truncated,
            **info  # Se info contiver dados úteis como path_length, etc.
        })

        total_term += int(terminated)
        total_trunc += int(truncated)

        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_id += 1
    if episode_id % 100 == 0:
        print(f' Episode: {episode_id}')
# Salvar os dados
df = pd.DataFrame(test_data)
df.to_csv(os.path.join(SAVE_DIR, MODEL_NAME+"test_metrics.csv"), index=False)

print("Dados de teste salvos em test_metrics.csv")