from gymnasium.envs.registration import register


from config import ENV_PARAMS, ENV_CONFIG


register(
    id="Maze-v0",
    entry_point="gymnasium_env.envs:PcgrlEnv",
    kwargs={
        "game": ENV_PARAMS["game"],
        "representation": ENV_PARAMS["representation"],
        "reward_strategy": ENV_PARAMS["reward_strategy"](),
        "action_tiles": ENV_PARAMS["action_tiles"],
        "env_config": ENV_CONFIG
    }
)