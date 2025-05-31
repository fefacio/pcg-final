from gymnasium.utils import seeding
from gymnasium_env.envs.utils.dtypes import GRID_ARR_DTYPE, GRID_SIZE_DTYPE


from enum import Enum
from typing import List, Optional

# Base class for 2D grid-related games
"""
This interface is responsible for game specific attributes and methods,
and to evaluate a given grid, specified in the Representation interface 
"""
class GridWorld:
    
    # Constructor
    def __init__(self,
                 width: GRID_SIZE_DTYPE,
                 height: GRID_SIZE_DTYPE
                 ):
        self.width = width
        self.height = height
        self.grid = None


    @property
    def width(self) -> GRID_SIZE_DTYPE:
        return self._width
    
    @width.setter
    def width(self, value: GRID_SIZE_DTYPE) -> None:
        if value <= 0:
            raise ValueError("Width must be positive")
        self._width = value

    @property
    def height(self) -> GRID_SIZE_DTYPE:
        return self._height
    
    @height.setter
    def height(self, value: GRID_SIZE_DTYPE) -> None:
        if value <= 0:
            raise ValueError("Height must be positive")
        self._height = value



    # Public setters and getters
    def set_width(self, width) -> None:
        self.width = width

    def set_height(self, height) -> None:
        self.height = height

    def get_width(self) -> GRID_SIZE_DTYPE:
        return self.width
    
    def get_height(self) -> GRID_SIZE_DTYPE:
        return self.height
    
    
    

    # Get game tile types
    def get_tile_types(self) -> List[Enum]:
        raise NotImplementedError('get_tile_types is not implemented')

    # Set game tile types
    def set_tile_probs(self) -> List[float]:
        raise NotImplementedError('set_tile_probs is not implemented')

    
    # Get the current stats of the map, e.g, number of walls, path length...
    def get_stats(self, map) -> dict[str, any]:
        raise NotImplementedError('get_stats is not implemented')


    # Get rewards based on the changes made
    def get_reward(self, new_stats, old_stats) -> float:
        raise NotImplementedError('get_reward is not implemented')


    # Uses the stats to check if the problem ended (episode_over) which means reached
    # a satisfying quality based on the stats
    def get_episode_over(self, new_stats, old_stats) -> bool:
        raise NotImplementedError('get_graphics is not implemented')

    


    # Render the game using grid values
    def render(self, grid):
        raise NotImplementedError('render is not implemented')
    
    
    # Close game rendering
    def close(self):
        raise NotImplementedError('close is not implemented')
    
    
