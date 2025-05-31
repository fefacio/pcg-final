import numpy as np
from numpy.typing import NDArray
from enum import IntEnum


GRID_SIZE_DTYPE = np.uint32
GRID_DTYPE = np.uint8
GRID_ARR_DTYPE = NDArray[GRID_DTYPE]
IDX_POSITION_DTYPE = NDArray[np.intp]



class TileType(IntEnum):
    EMPTY = 0
    WALL = 1
    START = 2
    END = 3


class GenerationType(IntEnum):
    EMPTY = 0
    FULL = 1
    RANDOM = 2
    DFS = 3
    CUSTOM1 = 100
    CUSTOM2 = 101
    