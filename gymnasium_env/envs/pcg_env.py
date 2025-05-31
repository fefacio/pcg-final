# Custom defined modules
from .game import GAMES
from .representation import REPRESENTATION
from gymnasium_env.envs.utils.dtypes import GenerationType, TileType
from gymnasium_env.envs.utils.rewards import *
from gymnasium_env.envs.representation import WideRepresentation
import config

# Third party modules
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict

class PcgrlEnv(gym.Env):
    # Supported render modes
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Enviromnent constructor
    def __init__(self, 
                 game: str,
                 representation: str,
                 reward_strategy: RewardStrategy,
                 action_tiles: list[TileType],
                 env_config: Dict,
                 render_mode = None):
        config.debug_print("[INIT]")
        self._env_config = env_config
        game_config = self._env_config.get("game.config", {})
        self._init_game(game, game_config)

        representation_config = self._env_config.get("representation.config", {})
        self._init_representation(representation, representation_config)

        self._reward = reward_strategy
        self._action_tiles = action_tiles
        self._stats = None
        self._iteration = 0
        self._changes = 0
        self._change_rate = self._env_config.get("change_rate", 0.2)
        self._max_changes = max(int(self._change_rate * self._prob._width * self._prob._height), 1)
        
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        self.render_mode = render_mode
        config.debug_print(f'Possible actions {len(self._action_tiles)}')
        config.debug_print(f'Actions {self._action_tiles}')
        self.action_space = self._representation.get_action_space(
            self._prob.height,
            self._prob.width,
            len(self._action_tiles)
        )
        self.observation_space = self._representation.get_observation_space(
            self._prob.height,
            self._prob.width,
            self._prob.get_num_tiles()
        )

        self._heatmap = np.zeros((self._prob._height, self._prob._width))
        self.observation_space.spaces['heatmap'] = \
            spaces.Box(low=0, high=self._max_changes, dtype=np.uint8, 
                       shape=(self._prob._height, self._prob._width))


    def _init_game(self, game, game_config):
        height = game_config.get("height", 6)
        width = game_config.get("width", 6)
        render_mode = game_config.get("render_mode", "human")
        render_type = game_config.get("render_type", "step")
        render_ws_height = game_config.get("render_ws_height", 768)
        render_ws_width = game_config.get("render_ws_width", 768)
        self._prob = GAMES[game](height, 
                                 width,
                                 render_mode,
                                 render_type,
                                 render_ws_width,
                                 render_ws_height)


    def _init_representation(self, representation, representation_config):
        generation = representation_config.get("generation", GenerationType.RANDOM)
        random_start = representation_config.get("random_start", True)
        representation_kwargs = {
            "random_start": random_start
        }
        self._representation = REPRESENTATION[representation](generation, 
                                                              **representation_kwargs)


    # Resets current enviroment
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed) 

        self._changes = 0
        self._iteration = 0

    
        self._representation.reset(self._prob.height, self._prob.width)
        self._stats = self._reward.compute_stats(self._representation._grid)
        self._prob.reset(self._stats)
        self._heatmap = np.zeros((self._prob._height, self._prob._width), dtype=np.uint8)

        observation = self._representation.get_observation()
        observation["heatmap"] = self._heatmap.copy()

        config.debug_print(f'[RESET]')
        if config.DEBUG_MODE:
            for key, value in self._stats.items():
                config.debug_print(f'Stat {key}: {value}')
        return observation, self._stats

    
    # Change the current state
    def step(self, action):
        config.debug_print(f'[STEP]')
        config.debug_print(f'Action {action}')
        self._iteration += 1
        # Save copy of older grid stats
        old_stats = self._stats
        # Update grid state and get its new stats
        if isinstance(self._representation, WideRepresentation):
            x, y, tile_idx = action
            tile = self._action_tiles[tile_idx]
            action_to_enum = [x, y, tile]
        else:
            action_to_enum = self._action_tiles[action]

        change, x, y, action = self._representation.update(action_to_enum, len(self._action_tiles))
        config.debug_print(f'Action {action}')
        if change > 0:
            self._changes += change
            self._heatmap[y][x] += 1
            self._stats = self._reward.compute_stats(self._representation._grid)
        
        if config.DEBUG_MODE:
            for key, value in self._stats.items():
                config.debug_print(f'Stat {key}: {value}')
                config.debug_print(f'Old stat {key}: {old_stats[key]}')

        # Current grid state
        observation = self._representation.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        # Episode reward based on the change (old and new grid stats)
        reward = self._reward.get_rewards(self._stats, old_stats)
        config.debug_print(f'Final reward {reward}')

        #print(f'stats {self._stats}')
        #print(f'reward {reward}')
        # Episoded ended beucase the goal was reached
        done = self._reward.get_episode_over(self._stats, old_stats)

        # Episoded ended because iterative limitations reached
        truncated = self._changes >= self._max_changes or self._iteration >= self._max_iterations

        # Episode debugging info
        info = self._reward.compute_stats(self._representation._grid)
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        
        return observation, reward, done, truncated, info

    
    def render(self):
        self._prob.render(self._representation._grid)
  

    def close(self):
        self._prob.render(self._representation._grid)
