from gymnasium_env.envs.utils.helper import *
from gymnasium_env.envs.utils.dtypes import TileType
import config
from functools import partial
from typing import Callable
from functools import partial

import inspect

POSSIBLE_MAZE_REWARDS = {
        "num_empty": partial(get_tile_count, TileType.EMPTY),
        "num_wall": partial(get_tile_count, TileType.WALL),
        "num_start": partial(get_tile_count, TileType.START),
        "num_end": partial(get_tile_count, TileType.END),
        "is_grid_solvable": lambda maze: is_maze_solvable(maze)[0],
        "path_length": lambda maze: is_maze_solvable(maze)[1],
        "num_regions": get_num_regions
}


class RewardStrategy():
    def __init__(self):
        self.reward_dict = {}
        self.weight_dict = {}
        self.stats_dict = {}
        self.episode_end_cond = {}

    """
    The first thing to define is the possible stats that 
    are going to be used for the reward calculation
    """
    def set_stats(self, key) -> bool:
        if key not in POSSIBLE_MAZE_REWARDS:
            #raise Exception("key is not valid")
            return False
        if key in self.stats_dict:
            #raise Exception("key already exists")
            return False
        
        self.stats_dict[key] = POSSIBLE_MAZE_REWARDS[key]
        return True
    
    def compute_stats(self, grid) -> dict:
        stats = {}
        for key, func in self.stats_dict.items():
            stats[key] = func(grid)  
            
        return stats
    
    
    
    """
    After defining these stats, we can dynamic create the associate
    reward and weight function for each stats 
    """
    def set_key_reward(self, key, reward) -> bool:
        if key not in self.stats_dict.keys():
            return False
        self.reward_dict[key] = reward
        return True

    def set_key_weight(self, key, weight) -> bool:
        if key not in self.stats_dict.keys():
            return False
        self.weight_dict[key] = weight
        return True
    
    
    """
    After defining the reward and weight functions, we
    can get the rewards given a set of stats
    """
    def set_reward_function(self, key: str, fn: Callable[[dict], float]):
        self.reward_dict[key] = fn

    def get_rewards(self, new_stats: dict, old_stats: dict) -> dict:
        config.debug_print("[REWARDS MODULE] [get_rewards]")
        # if config.DEBUG_MODE:
        #     for key, value in new_stats.items():
        #         config.debug_print(f'Stat {key}: {value}')
        #         config.debug_print(f'Old stat {key}: {old_stats[key]}')
        rewards = 0
        for key in self.reward_dict:
            num_params = len(inspect.signature(self.reward_dict[key]).parameters)
            if num_params == 1:
                reward_value = self.reward_dict[key](new_stats)
            else:
                reward_value = self.reward_dict[key](new_stats, old_stats)
            weight = self.weight_dict.get(key, 1.0)
            

            weighted_reward = reward_value * weight
            rewards += weighted_reward
            # if old_stats == new_stats:
            #     rewards += -10  # Penalize no changes between states
            config.debug_print(f"Reward {reward_value} for key {key}, weighted: {weighted_reward}")

        config.debug_print(f"Total weighted reward: {rewards}")
        return rewards


    """
    Define conditions for end of episode
    """
    def set_episode_end_cond(self, key: str, cond_fn: Callable[[dict], bool]):
        if key not in self.stats_dict.keys():
            return False
        self.episode_end_cond[key] = cond_fn
        return True

    def get_episode_over(self, new_stats: dict, old_stats: dict) -> bool:
        return all(cond(new_stats) for cond in self.episode_end_cond.values())


    def debug_info(self):
        return [
            self.reward_dict,
            self.weight_dict,
            self.stats_dict,
            self.episode_end_cond
        ]
    
   
def get_range_reward(new_value, old_value, low, high):
    config.debug_print("[REWARDS MODULE] [get_range_rewards]")
    config.debug_print(f'new_value: {new_value} old_value: {old_value} low: {low} high: {high}')
    
    # Calculate if the new and old values are inside the range
    new_in = low <= new_value <= high
    old_in = low <= old_value <= high

    # Both in the interval
    if new_in and old_in:
        return 0.0
    
    # Both outside lower interval
    if new_value < low and old_value < low:
        # Penalize no changes
        if new_value == old_value:
            return -1
    
        if new_value > old_value:
            return new_value - old_value  # Better change
        else:
            return new_value - old_value  # Worse change

    # Both outside higher interval
    if new_value > high and old_value > high:
        if new_value == old_value:
            return -1
        if new_value < old_value:
            return old_value - new_value  # Better change
        else:
            return old_value - new_value  # Worse change

    # Values jump outside the interval
    if new_value > high and old_value < low:
        return (high - new_value) + (old_value - low)

    if new_value < low and old_value > high:
        return (high - old_value) + (new_value - low)

    # One value inside and the other outside
    if new_in and not old_in:
        return -1.0  

    # One value inside and the other outside
    if not new_in and old_in:
        return -1.0  
    
    # Default
    return 0.0

def get_range_reward2(new_value, old_value, low, high):
    if new_value >= low and new_value <= high and old_value >= low and old_value <= high:
        return 0
    if old_value <= high and new_value <= high:
        return min(new_value,low) - min(old_value,low)
    if old_value >= low and new_value >= low:
        return max(old_value,high) - max(new_value,high)
    if new_value > high and old_value < low:
        return high - new_value + old_value - low
    if new_value < low and old_value > high:
        return high - old_value + new_value - low
    

"""
    Custom defined reward scenarios
    All functions should return an object of RewardStrategy
"""

"""
Agent needs to maximize a given path
"""
# These functions results in worse values
# def maze_reward_scenario1(path_length=15) -> RewardStrategy:
#     strategy = RewardStrategy()
#     key = "path_length"
#     strategy.set_stats(key)
#     strategy.set_reward_function(key, lambda stats, k=key: -abs(stats[k] - path_length))
#     strategy.set_key_weight(key, 1)
#     strategy.set_episode_end_cond(key, lambda stats, k=key: stats[k] >= path_length)
#     return strategy


# def maze_reward_scenario2(path_length=15) -> RewardStrategy:
#     strategy = RewardStrategy()
#     key = "path_length"
#     strategy.set_stats(key)
#     print(type(key), key)

#     strategy.set_reward_function(key, lambda stats, k=key: -abs(stats[k] - path_length))
#     strategy.set_key_weight(key, 1)
#     strategy.set_episode_end_cond(key, lambda stats, k=key: stats[k] >= path_length)

#     key = "num_start"
#     strategy.set_stats(key)
#     strategy.set_reward_function(key, lambda stats, k=key: -abs(stats[k] - 1))
#     strategy.set_key_weight(key, 3)
#     strategy.set_episode_end_cond(key, lambda stats, k=key: stats[k] == 1)

#     key = "num_end"
#     strategy.set_stats(key)
#     strategy.set_reward_function(key, lambda stats, k=key: -abs(stats[k] - 1))
#     strategy.set_key_weight(key, 3)
#     strategy.set_episode_end_cond(key, lambda stats, k=key: stats[k] == 1)

#     return strategy


def maze_reward_scenario3(target_path_length=11) -> RewardStrategy:
    strategy = RewardStrategy()
    key = "path_length"
    strategy.set_stats(key)
    strategy.set_reward_function(key, lambda new_stats, old_stats, k=key: 
                                 get_range_reward(new_stats[k], 
                                                  old_stats[k], 
                                                  low=target_path_length, 
                                                  high=target_path_length))
    strategy.set_key_weight(key, 1)
    strategy.set_episode_end_cond(key, lambda stats, k=key: stats[k] >= target_path_length)

    key = "num_wall"
    strategy.set_stats(key)
    strategy.set_reward_function(key, lambda new_stats, old_stats, k=key: 
                                 get_range_reward(new_stats[k], 
                                                  old_stats[k], 
                                                  low=np.inf, 
                                                  high=np.inf))
    strategy.set_key_weight(key, 1)
    strategy.set_episode_end_cond(key, lambda stats, k=key: stats[k] >= 5)

    return strategy


def maze_reward_scenario4(target_path_length=15) -> RewardStrategy:
    strategy = RewardStrategy()
    key = "path_length"
    strategy.set_stats(key)
    strategy.set_reward_function(key, lambda new_stats, old_stats, k=key: 
                                 get_range_reward(new_stats[k], 
                                                  old_stats[k], 
                                                  low=np.inf, 
                                                  high=np.inf))
    strategy.set_key_weight(key, 1)
    strategy.set_episode_end_cond(key, lambda stats, k=key: stats[k] >= target_path_length)
    
    key = "num_wall"
    strategy.set_stats(key)
    strategy.set_reward_function(key, lambda new_stats, old_stats, k=key: 
                                 get_range_reward(new_stats[k], 
                                                  old_stats[k], 
                                                  low=np.inf, 
                                                  high=np.inf))
    strategy.set_key_weight(key, 1)
    strategy.set_episode_end_cond(key, lambda stats, k=key: stats[k] >= 10)
    
    key = "num_start"
    strategy.set_stats(key)
    strategy.set_reward_function(key, lambda new_stats, old_stats, k=key: 
                                 get_range_reward(new_stats[k], 
                                                  old_stats[k], 
                                                  low=1, 
                                                  high=1))
    strategy.set_key_weight(key, 1)
    strategy.set_episode_end_cond(key, lambda stats, k=key: stats[k] == 1)

    key = "num_end"
    strategy.set_stats(key)
    strategy.set_reward_function(key, lambda new_stats, old_stats, k=key: 
                                 get_range_reward(new_stats[k], 
                                                  old_stats[k], 
                                                  low=1, 
                                                  high=1))
    strategy.set_key_weight(key, 1)
    strategy.set_episode_end_cond(key, lambda stats, k=key: stats[k] == 1)
    return strategy


def maze_reward_scenario5(target_path_length=15) -> RewardStrategy:
    strategy = RewardStrategy()
    
    key = "num_start"
    strategy.set_stats(key)
    strategy.set_reward_function(key, lambda new_stats, old_stats, k=key: 
                                 get_range_reward(new_stats[k], 
                                                  old_stats[k], 
                                                  low=1, 
                                                  high=1))
    strategy.set_key_weight(key, 1)
    strategy.set_episode_end_cond(key, lambda stats, k=key: stats[k] == 1)

    key = "num_end"
    strategy.set_stats(key)
    strategy.set_reward_function(key, lambda new_stats, old_stats, k=key: 
                                 get_range_reward(new_stats[k], 
                                                  old_stats[k], 
                                                  low=1, 
                                                  high=1))
    strategy.set_key_weight(key, 1)
    strategy.set_episode_end_cond(key, lambda stats, k=key: stats[k] == 1)
    return strategy