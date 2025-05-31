import gymnasium_env.envs.utils.rewards as rewards
import gymnasium_env.envs.utils.generation as generation
from gymnasium_env.envs.utils.dtypes import TileType, GenerationType

import os
import json
import copy
import sys 

DEBUG_MODE = False
if len(sys.argv) > 1 and sys.argv[1] == "debug":
    DEBUG_MODE = True
    print("!DEBUG MODE ACTIVATED")

def debug_print(*args):
    if DEBUG_MODE:
        print("[DEBUG]", *args)


RESULTS_PATH = './results/'

"""
Change environment creation parameters:

    * Set custom reward scenarios in: import gymnasium_env.envs.utils.rewards

"""

ENV_PARAMS = {
    "game": "maze",
    "representation": "narrow",
    "reward_strategy": rewards.maze_reward_scenario3,
    "action_tiles": [
        TileType.EMPTY, 
        TileType.WALL, 
        #TileType.START,
        #TileType.END
        ],   
}

ENV_CONFIG = {
    "game.config": {
        "height": 6,
        "width": 6,
        "render_mode": "human",
        "render_type": "step",
        "render_ws_width": 640,
        "render_ws_height": 480,
    },
    "representation.config": {
        "generation": GenerationType.CUSTOM2,
        "random_start": True
    },
    "change_rate": 0.3
}  

def config_path():
    os.makedirs('./results/', exist_ok=True)

def serialize_env(env_params, env_config):
    serializable_params = copy.deepcopy(env_params)
    serializable_config = copy.deepcopy(env_config)

    # Convert reward strategy to string
    if callable(serializable_params.get("reward_strategy")):
        serializable_params["reward_strategy"] = serializable_params["reward_strategy"].__name__

    # Convert enums to strings
    serializable_params["action_tiles"] = [
        tile.name if hasattr(tile, "name") else tile
        for tile in serializable_params.get("action_tiles", [])
    ]

    rep_config = serializable_config.get("representation.config", {})
    if "generation" in rep_config and hasattr(rep_config["generation"], "name"):
        rep_config["generation"] = rep_config["generation"].name

    return serializable_params, serializable_config

def save_env():
    config_path()
    serializable_params, serializable_config = serialize_env(ENV_PARAMS, ENV_CONFIG)

    with open("env_params.jsonl", "a") as f:
        f.write(json.dumps(serializable_params) + "\n")

    with open("env_config.jsonl", "a") as f:
        f.write(json.dumps(serializable_config) + "\n")

    # with open(os.path.join(RESULTS_PATH, "env_params.json"), "w") as f:
    #     json.dump(serializable_params, f)

    # with open(os.path.join(RESULTS_PATH, "env_config.json"), "w") as f:
    #     json.dump(serializable_config, f)


# def load_env():
#     with open("env_params.json") as f:
#         ENV_PARAMS = json.load(f)

#     with open("env_config.json") as f:
#         ENV_CONFIG = json.load(f)