# Custom 
#from config import save_env
import gymnasium_env

import gymnasium as gym
import numpy as np
import pandas as pd
import time

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
import json
import os

class CustomLoggingCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.metrics = []
        self.last_time = time.time()
        self.df = pd.DataFrame(columns=[
            "timesteps", 
            "episodes", 
            "avg_steps_per_episode", 
            "mean_reward", 
            "best_mean_reward",
            "step_duration"
        ])

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            start_time = time.time()
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                n_episodes = len(y)
                avg_steps_per_episode = x[-1] / n_episodes if n_episodes > 0 else 0
                step_duration = start_time - self.last_time 

                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                new_metrics = {
                    "timesteps": self.num_timesteps,
                    "episodes": n_episodes,
                    "avg_steps_per_episode": avg_steps_per_episode,
                    "mean_reward": mean_reward,
                    "best_mean_reward": self.best_mean_reward,
                    "step_duration": step_duration
                }
                self.df = pd.concat([self.df, pd.DataFrame([new_metrics])], ignore_index=True)
                self.df.to_csv(os.path.join(self.log_dir, "training_metrics.csv"), index=False)

                self.last_time = time.time()

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
                
        return True



MODEL_NAME = "maze-ppo-NRGC2R3"
LOG_DIR = os.path.join('./results/', MODEL_NAME)
os.makedirs(LOG_DIR, exist_ok=True)

def train_model():
    env = gym.make("Maze-v0", render_mode="human")
    env = Monitor(env, LOG_DIR)


    callback = CustomLoggingCallback(
        check_freq=1000,
        log_dir=LOG_DIR,
    )

    model = PPO("MultiInputPolicy", env, device='cpu', verbose=1)
    model.learn(total_timesteps=100_000, callback=callback, log_interval=1000)
    model.save(MODEL_NAME)

if __name__ == "__main__":
    train_model()
