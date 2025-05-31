from gymnasium_env.envs.utils.dtypes import GenerationType, GRID_SIZE_DTYPE
from gymnasium_env.envs.utils.generation import *

from gymnasium.utils import seeding


""""
This modules defines how the agent will see and interact with a given enviromnent
"""
class Representation():
    def __init__(self, 
                 gen_type: GenerationType = GenerationType.RANDOM,
                 **kwargs):
        self._grid: GRID_ARR_DTYPE = None
        self._gen_type = gen_type
        self._gen_kwargs = kwargs
        self.seed()


    def seed(self, seed=None):
        self._random, seed = seeding.np_random(seed)
        return seed
    
    def reset(self, height: GRID_SIZE_DTYPE, width: GRID_SIZE_DTYPE) -> None:
        if self._gen_type == GenerationType.EMPTY:
            self._grid = generation_empty(height, width)
        elif self._gen_type == GenerationType.FULL:
            tile = self._gen_kwargs.get("full_tile", TileType.WALL)
            self._grid = generation_full_tile(height, width, tile)
        elif self._gen_type == GenerationType.RANDOM:
            self._grid = random_gen(
                height, width,
                tile_probs = {
                    TileType.EMPTY: 0.7,
                    TileType.WALL: 0.2,
                    TileType.START: 0.05,
                    TileType.END: 0.05,
                },
                rng=self._random
            )
        elif self._gen_type == GenerationType.CUSTOM1:
            self._grid = maze_custom_gen1(height, width)
        elif self._gen_type == GenerationType.CUSTOM2:
            self._grid = maze_custom_gen2(height, width)
        else:
            raise ValueError(f"Unknown generation type: {self._gen_type}")
    
    

    def get_action_space(self, height, width, tile_values):
        raise NotImplementedError('get_action_space is not implemented')
    
    def get_observation_space(self, height, width, num_tiles):
        raise NotImplementedError('get_observation_space is not implemented')

    def get_observation(self):
        raise NotImplementedError('get_observation is not implemented')

    def update(self, action, num_tiles):
        raise NotImplementedError('update is not implemented')
    
    