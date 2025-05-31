from gymnasium_env.envs.utils.dtypes import GenerationType
from .representation import Representation
from PIL import Image
from gymnasium import spaces
import numpy as np


class WideRepresentation(Representation):
    def __init__(self,
                 gen_type: GenerationType,
                 **kwargs):
        super().__init__(gen_type, **kwargs)


    def get_action_space(self, width, height, num_tiles):
        return spaces.MultiDiscrete([width, height, num_tiles])

    
    def get_observation_space(self, width, height, num_tiles):
        return spaces.Dict({
            "grid": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height, width))
        })


    def get_observation(self):
        return {
            "grid": self._grid.copy()
        }

    
    def update(self, action, num_tiles):
        x, y, new_tile = action
        old_tile = self._grid[y][x]
        change = int(old_tile != new_tile)
        self._grid[y][x] = new_tile
        return change, x, y, action