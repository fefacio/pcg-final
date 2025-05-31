import numpy as np
from enum import IntEnum
from typing import Tuple
from collections import deque

from .dtypes import TileType


    
def get_tile_count(grid, tile_type: IntEnum):
    return np.sum(grid == tile_type)

"""
Check if a given maze is solvable
Returns:
A bool indicating whether the maze is solvable
An integer>0 representing the path_length if the 
maze is solvable and -1 otherwise 
"""
def is_maze_solvable(grid) -> Tuple[bool, int]:
    if get_tile_count(grid, TileType.START) > 1:
        return (False, -1)
    if get_tile_count(grid, TileType.END) > 1:
        return (False, -1)
    
    rows, cols = len(grid), len(grid[0])
    
    # Encontra o ponto de início
    start = None
    end = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == TileType.START:
                start = (r, c)
            if grid[r][c] == TileType.END:
                end = (r, c)

    if start is None or end is None:
        return False, -1
    
    queue = deque()
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    queue.append((start[0], start[1], 0))  # (row, column, steps)
    visited[start[0]][start[1]] = True

    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    while queue:
        r, c, steps = queue.popleft()

        if (r, c) == end:
            return True, steps  # achou o caminho mais curto

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols:
                if not visited[nr][nc] and grid[nr][nc] != TileType.WALL:
                    visited[nr][nc] = True
                    queue.append((nr, nc, steps + 1))

    return False, -1  # não há caminho
    


    
    
def get_num_regions(grid) -> int:
    ...


def obs_to_key(obs: dict) -> tuple:
    return tuple(obs["pos"].flatten()) + tuple(obs["grid"].flatten())

