from .dtypes import GRID_SIZE_DTYPE, GRID_DTYPE, GRID_ARR_DTYPE, TileType

from typing import List, Dict
import numpy as np
from enum import IntEnum
import random


"""
All Generation functions should: 
    * have grid height and width as parameters
    * return a valid grid of GRID_ARR_DTYPE
"""


"""
GenerationType: EMPTY = 0,
"""
def generation_empty(height: GRID_SIZE_DTYPE, 
                     width: GRID_SIZE_DTYPE
                     ) -> GRID_ARR_DTYPE:
    return np.full((height, width), TileType.EMPTY, dtype=GRID_DTYPE) 

"""
GenerationType: FULL = 1,
"""
def generation_full_tile(height: GRID_SIZE_DTYPE, 
                         width: GRID_SIZE_DTYPE, 
                         tile_type: IntEnum
                         ) -> GRID_ARR_DTYPE:
    return np.full((height, width), tile_type, dtype=GRID_DTYPE)


"""
GenerationType: RANDOM = 2,
"""
def random_gen(height: GRID_SIZE_DTYPE,
               width: GRID_SIZE_DTYPE,
               tile_probs: Dict[IntEnum, float],
               rng: np.random.Generator
               ) -> GRID_ARR_DTYPE:
    tile_types = list(tile_probs.keys())
    probs = list(tile_probs.values())
    assert abs(sum(probs) - 1.0) < 1e-6, "Probabilities must sum to 1.0"

    flat_choices = rng.choice(tile_types, size=(height * width), p=probs)
    grid = flat_choices.reshape((height, width)).astype(GRID_DTYPE)
    return grid


"""
CUSTOM GENERATIONS
"""

"""
Start with an empty grid
The start will be the first position [0][0]
The goal will be the end position [height-1][width-1]
"""

def maze_custom_gen1(height: GRID_SIZE_DTYPE,
                     width: GRID_SIZE_DTYPE) ->\
                     GRID_ARR_DTYPE:
    grid = generation_empty(height, width)
    grid[0][0] = TileType.START
    grid[height - 1][width - 1] = TileType.END

    return grid

"""
Start with an empty grid
The start and end will be random positions
"""
def maze_custom_gen2(height: GRID_SIZE_DTYPE,
                     width: GRID_SIZE_DTYPE
                     ) -> GRID_ARR_DTYPE:
    grid = generation_empty(height, width)

    start_y = random.randint(0, height - 1)
    start_x = random.randint(0, width - 1)
    grid[start_y][start_x] = TileType.START

    # Gera posição aleatória para o END, garantindo que seja diferente do START
    while True:
        end_y = random.randint(0, height - 1)
        end_x = random.randint(0, width - 1)
        if (end_y, end_x) != (start_y, start_x):
            break
    grid[end_y][end_x] = TileType.END

    return grid



