from gymnasium_env.envs.utils.dtypes import GRID_DTYPE, GRID_SIZE_DTYPE, TileType
from gymnasium_env.envs.utils.dtypes import GenerationType
from .representation import Representation
from PIL import Image
from gymnasium import spaces
import numpy as np
from collections import OrderedDict
from typing import Dict, Tuple
from numpy.typing import NDArray
import config



class NarrowRepresentation(Representation):
    def __init__(self, 
                 gen_type: GenerationType, 
                 **kwargs):
        self._random_start = kwargs.pop("random_start", True)
        print(f' representation{self._random_start}')
        super().__init__(gen_type, **kwargs)
        self._x = None
        self._y = None

    

    def reset(self, width: GRID_SIZE_DTYPE, height: GRID_SIZE_DTYPE):
        super().reset(width, height)
        
        if self._random_start:
            self._x = self._random.integers(0, width)
            self._y = self._random.integers(0, height)
        else:
            self._x = 0
            self._y = 0

    
    def _set_random_pos(self, num_tiles):
        ...

    def get_action_space(self, 
                         height: GRID_SIZE_DTYPE, 
                         width: GRID_SIZE_DTYPE, 
                         num_tiles: int):
        return spaces.Discrete(num_tiles)

   
    def get_observation_space(self, 
                              height: GRID_SIZE_DTYPE, 
                              width: GRID_SIZE_DTYPE, 
                              num_tiles: int
                              ) -> Dict[str, spaces.Box]:
        return spaces.Dict({
            "pos": spaces.Box(low=np.array([0, 0]), high=np.array([width-1, height-1]), dtype=GRID_DTYPE),
            "grid": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height, width))
        })

    
    def get_observation(self) -> OrderedDict[str, NDArray]:
        return OrderedDict({
            "pos": np.array([self._x, self._y], dtype=np.uint8),
            "grid": self._grid.copy()
        })

    
    
    def update(self, action: int, number_tiles: int) -> Tuple[bool, int, int, int]:
        current_tile = self._grid[self._y][self._x]

        if number_tiles == 2 and (current_tile == TileType.START or current_tile == TileType.END):
            change = 0 
        else:
            change = int(current_tile != action)
            self._grid[self._y][self._x] = action
        
        if self._random_start:
            while True:
                self._x = self._random.integers(0, self._grid.shape[1])
                self._y = self._random.integers(0, self._grid.shape[0])
                tile = self._grid[self._y, self._x]

                if number_tiles != 2:
                    break  # qualquer posição serve

                if tile == TileType.EMPTY or tile == TileType.WALL:
                    break  # posição válida encontrada
        else:
            self._x += 1
            if self._x >= self._grid.shape[1]:
                self._x = 0
                self._y += 1
                if self._y >= self._grid.shape[0]:
                    self._y = 0

        return change, self._x, self._y, action
