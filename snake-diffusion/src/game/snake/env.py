from typing import Tuple, Union, List

import numpy as np

from ..env import Environment, ActionResult
from .game import SnakeGame, Direction, Point

class GameEnvironment(Environment):
    def __init__(self, game: SnakeGame):
        super().__init__()
        self.steps_taken = 0
        self.game = game
        game.reset()

    def actions_length(self) -> int:
        return 4

    def reset(self):
        self.game.reset()
        self.steps_taken = 0

    def get_snapshot(self) -> np.ndarray:
        return self.game.get_snapshot().transpose(1,0,2)

    def do_action(self, action: np.ndarray) -> ActionResult:
        self.steps_taken += 1
        
        # Execute action in game
        reward, terminated = self._take_action(action)
        return ActionResult(self.get_state(), reward, terminated, self.game.score)
    
    def get_state(self) -> np.ndarray:
        head = self.game.head
        point_l = Point(head.x - self.game.block_size, head.y)
        point_r = Point(head.x + self.game.block_size, head.y)
        point_u = Point(head.x, head.y - self.game.block_size)
        point_d = Point(head.x, head.y + self.game.block_size)
        
        dir_l = self.game.direction == Direction.LEFT
        dir_r = self.game.direction == Direction.RIGHT
        dir_u = self.game.direction == Direction.UP
        dir_d = self.game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.game.is_collision(point_r)) or 
            (dir_l and self.game.is_collision(point_l)) or 
            (dir_u and self.game.is_collision(point_u)) or 
            (dir_d and self.game.is_collision(point_d)),

            # Danger right
            (dir_u and self.game.is_collision(point_r)) or 
            (dir_d and self.game.is_collision(point_l)) or 
            (dir_l and self.game.is_collision(point_u)) or 
            (dir_r and self.game.is_collision(point_d)),

            # Danger left
            (dir_d and self.game.is_collision(point_r)) or 
            (dir_u and self.game.is_collision(point_l)) or 
            (dir_r and self.game.is_collision(point_u)) or 
            (dir_l and self.game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            self.game.food.x < head.x,  # food left
            self.game.food.x > head.x,  # food right
            self.game.food.y < head.y,  # food up
            self.game.food.y > head.y  # food down
        ]

        return np.array(state, dtype=int)

    def _take_action(self, action: np.ndarray) -> Tuple[int, bool]:
        prev_score = self.game.score
        game_over, score = self.game.play_step(action)
        reward = 0
        if game_over:
            reward = -10
        elif score > prev_score:
            reward = 10
        if self.steps_taken >= 100 * len(self.game.snake):
            game_over = True
            self.game.draw_game_over()
            reward = -10
        return reward, game_over

if __name__ == '__main__':
    game = SnakeGame()
    env = GameEnvironment(game)
    state = env._get_state()
    print(state.shape)
    print(env.observation_space)
    print(env.action_space)