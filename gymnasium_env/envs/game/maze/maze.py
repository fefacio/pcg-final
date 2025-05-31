from ..grid_world import GridWorld
from gymnasium_env.envs.utils.dtypes import *

from gymnasium import spaces

from typing import List, Dict, Optional
from enum import IntEnum

import random
import numpy as np
import pygame
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPRITES_PATH = 'sprites'

class Maze(GridWorld):
    def __init__(self,
                 width,
                 height,
                 render_mode,
                 render_type,
                 render_ws_width,
                 render_ws_height,
                 random_prob: bool = False,
                 tile_probs: Optional[Dict[TileType, float]] = None):
        super().__init__(width, height)
        # Valor padrão seguro
        
        self._tiles = [
            TileType.EMPTY,
            TileType.WALL,
            TileType.START,
            TileType.END
        ]
        self._tile_probs = tile_probs
        self.set_tile_probs()

        self._start_stats = None
        self._render_mode = render_mode
        self._render_type = render_type
        self._render_ws_width = render_ws_width
        self._render_ws_height = render_ws_height
        if self._render_mode == "human":
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode(
                (self._render_ws_width, self._render_ws_height))
            self._clock = pygame.time.Clock()
        


    #############################################
    #           Tile related methods            #
    #############################################
    def get_tile_types(self) -> list[TileType]:
        return list(TileType)
    
    def set_tile_probs(self) -> None:
        if self._tile_probs is None:
            self._tile_probs = {
                TileType.EMPTY: 0.7,
                TileType.WALL: 0.2,
                TileType.START: 0.05,
                TileType.END: 0.05,
            }
    
    #     # Check if the probs are in the correct format and already_assigned
    #     if self._tile_probs is not None and isinstance(self._tile_probs, dict):
    #         if all(isinstance(k, MazeTileType) and isinstance(v, float) for k, v in self._tile_probs.items()):
    #             return 

    #     tile_types = self.get_tile_types()
    #     # Random probs for each tile type
    #     if self._random_probs:
    #         raw_probs = [random() for _ in tile_types]
    #         total = sum(raw_probs)
    #         normalized_probs = [p / total for p in raw_probs]
    #         self._tile_probs = dict(zip(tile_types, normalized_probs))
    #     # All tile types have the same probability
    #     else:
    #         uniform_prob = 1.0 / len(tile_types)
    #         self._tile_probs = {tile: uniform_prob for tile in tile_types}


    def get_tiles(self) -> List[IntEnum]:
        return self._tiles
    
    def get_num_tiles(self) -> int:
        return len(self._tiles)


    
    def reset(self, start_stats):
        self._start_stats = start_stats


    
    def render(self, grid):
        canvas = pygame.Surface(
            (self._render_ws_width, self._render_ws_height))
        canvas_width, canvas_height = canvas.get_size()
        canvas.fill((255, 255, 255))

        pix_square_size_w = self._render_ws_width / self.width
        pix_square_size_h = self._render_ws_height / self.height
        pix_square_size = min(pix_square_size_w, pix_square_size_h)
        pix_square_size = int(pix_square_size)

        # Color render
        tile_colors = {
            TileType.EMPTY.value: (255, 255, 255),   
            TileType.START.value: (0, 255, 0),       
            TileType.END.value: (255, 0, 0),         
            TileType.WALL.value: (0, 0, 0),          
        }

        # Image rendering
        def load_sprite(sprite_name):
            img = pygame.image.load(os.path.join(BASE_DIR, SPRITES_PATH, sprite_name)).convert_alpha()
            img_norm = pygame.transform.scale(img, (
                pix_square_size_w, pix_square_size_h))
            return img_norm

        tile_images = {
            TileType.EMPTY.value: load_sprite("empty.png"),
            TileType.START.value: load_sprite("start.png"),
            TileType.END.value: load_sprite("end.png"),
            TileType.WALL.value: load_sprite("wall.png"),
        }

        # Draw tile values on screen
        for y in range(self.height):
            for x in range(self.width):
                tile_type = grid[y][x]  
                sprite = tile_images.get(tile_type)  

                if sprite:
                    canvas.blit(sprite, (x * pix_square_size_w, y * pix_square_size_h))
                else:
                    color = tile_colors.get(tile_type, (128, 128, 128))
                    rect = pygame.Rect(
                        x * pix_square_size_w,
                        y * pix_square_size_h,
                        pix_square_size,
                        pix_square_size
                    )
                    pygame.draw.rect(canvas, color, rect)

        # Horizontal lines
        for y in range(self.height + 1):
            pygame.draw.line(
                canvas,
                0,  # COLOR: BLACK 
                (0, int(pix_square_size_h * y)),
                (self._render_ws_width, int(pix_square_size_h * y)),
                width=3,
            )

        # Vertical lines
        for x in range(self.width + 1):
            pygame.draw.line(
                canvas,
                0,  # COLOR: BLACK
                (int(pix_square_size_w * x), 0),
                (int(pix_square_size_w * x), self._render_ws_height),
                width=3,
    )

        if self._render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self._window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            # self.clock.tick(4)
            if self._render_type == "step":
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            exit()
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                pygame.quit()
                                exit()
                            else:
                                waiting = False

        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        


    
