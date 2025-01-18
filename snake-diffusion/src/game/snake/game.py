import random
from typing import Optional, Union, List
from enum import Enum
from dataclasses import dataclass

import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Define constants
SPEED = 10
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

class Direction(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

@dataclass
class Point:
    x: int
    y: int

    def equal_with_block(self, other: 'Point', block_size: int) -> bool:
        return (other.x >= self.x and other.x <= self.x + block_size / 2 and other.y >= self.y and other.y <= self.y + block_size / 2) or \
        (self.x >= other.x and self.x <= other.x + block_size / 2 and self.y >= other.y and self.y <= other.y + block_size / 2)

    def distance(self, other: 'Point') -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5

class SnakeGame:
    def __init__(self, width=640, height=480, speed=SPEED, block_size=20):
        self.width = width
        self.height = height
        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.speed = speed

        self.block_size = block_size
        
        # init game state
        self.reset()

        # import os
        # import matplotlib.pyplot as plt
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # plt.imsave(os.path.join(dir_path, f'test.jpg'), self.get_snapshot().transpose(1,0,2))
    
    def reset(self):
        # init snake
        self.direction = Direction.RIGHT
        self.head = Point(self.width/2, self.height/2)
        self.snake = [
            self.head,
            Point(self.head.x-self.block_size, self.head.y),
            Point(self.head.x-(2*self.block_size), self.head.y)
        ]
        
        self.score = 0
        self.food = None
        self._place_food()
        self._update_ui()

    def get_snapshot(self) -> np.ndarray:
        return pygame.surfarray.array3d(self.display)
        
    def _place_food(self):
        x = random.randint(0, (self.width-self.block_size)//self.block_size)*self.block_size
        y = random.randint(0, (self.height-self.block_size)//self.block_size)*self.block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _get_direction_from_event(self, event: pygame.event.Event) -> Direction:
        if event.type != pygame.KEYDOWN:
            return self.direction
        if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
            return Direction.LEFT
        elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
            return Direction.RIGHT
        elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
            return Direction.UP
        elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
            return Direction.DOWN
        return self.direction
    
    def _get_direction_from_int_value(self, value: int) -> Direction:
        new_direction = Direction(value)
        if new_direction == Direction.LEFT and self.direction != Direction.RIGHT:
            return Direction.LEFT
        elif new_direction == Direction.RIGHT and self.direction != Direction.LEFT:
            return Direction.RIGHT
        elif new_direction == Direction.UP and self.direction != Direction.DOWN:
            return Direction.UP
        elif new_direction == Direction.DOWN and self.direction != Direction.UP:
            return Direction.DOWN
        return self.direction
        # return self.direction
        # clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        # idx = clock_wise.index(self.direction)

        # if value == 0:
        #     return self.direction
        # elif value == 1:
        #     next_idx = (idx + 1) % 4
        #     return clock_wise[next_idx]
        # else:
        #     next_idx = (idx - 1) % 4
        #     return clock_wise[next_idx]
    
    def _get_direction_from_list_int_value(self, value: np.ndarray) -> Direction:
        value = value.argmax()
        if value > 3:
            return self.direction
        new_direction = Direction(value)
        if new_direction == Direction.LEFT and self.direction != Direction.RIGHT:
            return Direction.LEFT
        elif new_direction == Direction.RIGHT and self.direction != Direction.LEFT:
            return Direction.RIGHT
        elif new_direction == Direction.UP and self.direction != Direction.DOWN:
            return Direction.UP
        elif new_direction == Direction.DOWN and self.direction != Direction.UP:
            return Direction.DOWN
        return self.direction
        # clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        # idx = clock_wise.index(self.direction)

        # if np.array_equal(value, [1, 0, 0]):
        #     new_dir = clock_wise[idx] # no change
        # elif np.array_equal(value, [0, 1, 0]):
        #     next_idx = (idx + 1) % 4
        #     new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        # else: # [0, 0, 1]
        #     next_idx = (idx - 1) % 4
        #     new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        # return new_dir
    
    def play_step(self, value: Optional[Union[int, List[int]]] = None):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            self.direction = self._get_direction_from_event(event)
        if value is not None:
            if isinstance(value, int) or isinstance(value, np.int64):
                self.direction = self._get_direction_from_int_value(value)
            else:
                self.direction = self._get_direction_from_list_int_value(value)
        # 2. move
        self._move(self.direction)
        self.snake.insert(0, self.head)

        # 4. place new food or just move
        if self.head.equal_with_block(self.food, self.block_size):
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        
        # 3. check if game over
        game_over = False
        if self.is_collision():
            print(f"fps {self.clock.get_fps()}")
            game_over = True
            self.draw_game_over()
            return game_over, self.score
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(self.speed)
        
        # 6. return game over and score
        return game_over, self.score
    
    def draw_game_over(self):
        font = pygame.font.Font(None, self.block_size)
        text = font.render('Game Over - Press Enter to Play Again', True, WHITE)
        text_rect = text.get_rect(center=(self.width/2, self.height/2))
        self.display.blit(text, text_rect)
        pygame.display.flip()
    
    def is_collision(self, point: Optional[Point] = None):
        if point is None:
            point = self.head
        # hits boundary
        if (point.x > self.width - self.block_size or point.x < 0 or \
            point.y > self.height - self.block_size or point.y < 0):
            return True
        # hits itself
        for body in self.snake[1:]:
            if body.equal_with_block(point, self.block_size):
                return True
        return False
    
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+(self.block_size // 5), pt.y+(self.block_size // 5), self.block_size * 3 / 5, self.block_size * 3 / 5))
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))
        pygame.display.flip()
    
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += self.block_size
        elif direction == Direction.LEFT:
            x -= self.block_size
        elif direction == Direction.DOWN:
            y += self.block_size
        elif direction == Direction.UP:
            y -= self.block_size
        self.head = Point(x, y)

if __name__ == '__main__':
    game = SnakeGame(w=60, h=60, speed=10, block_size=4)
    
    # game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over:            
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                        game.reset()
                        waiting = False
    
    print(f'Final Score: {score}')
    pygame.quit()
