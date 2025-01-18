from collections import deque
from typing import Union, Tuple, List, Optional
import random
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import shutil

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from models.agent.blocks import Linear_QNet, QTrainer
from game.env import Environment, ActionResult

def plot(scores, mean_scores):
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

class ValueForEndGame(Enum):
    last_action = "last_action"
    not_exist = "not_exist"

@dataclass
class QAgentConfig:
    max_memory: int
    batch_size: int
    lr: float
    hidden_state: int
    value_for_end_game: ValueForEndGame
    iterations: int
    min_deaths_to_record: int
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    gamma: float = 0.9
    train_every_iteration: int = 10
    save_every_iteration: Optional[int] = None

class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.memory, min(len(self.memory), batch_size))
        return zip(*batch)
    
    def __len__(self):
        return len(self.memory)

class QAgent:
    def __init__(
        self,
        env: Environment,
        config: QAgentConfig,
        model_path: str,
        dataset_path: str,
        last_checkpoint: Optional[str]
    ):
        self.config = config
        self.model_path = model_path
        self.memory = ReplayMemory(config.max_memory)
        self.model = Linear_QNet(len(env.get_state()), self.config.hidden_state, env.actions_length())
        self.trainer = QTrainer(self.model, gamma=config.gamma)
        self.env = env
        self.steps = 0
        self.dataset_path = dataset_path
        self.count_games = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.recorded_actions = []
        self.epsilon = config.epsilon_start
        self.begin_iteration = 0
        if last_checkpoint:
            parameters = torch.load(last_checkpoint)
            self.model.load_state_dict(parameters["model"])
            self.optimizer.load_state_dict(parameters["optimizer"])
            self.count_games = parameters.get("count_games", 0)
            self.begin_iteration = parameters.get("begin_iteration", 0)
    
    def _remember(
        self,
        state: np.ndarray,
        action: Union[np.ndarray, List[int]],
        reward: Union[np.ndarray, float],
        next_state: np.ndarray,
        done: Union[np.ndarray, bool]
    ):
        self.memory.append((state, action, reward, next_state, done))

    def _train_long_memory(self):
        if len(self.memory) > self.config.batch_size:
            mini_sample = random.sample(self.memory, self.config.batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self._train_step(states, actions, rewards, next_states, dones)

    def _train_step(
        self,
        state: np.ndarray,
        action: Union[np.ndarray, List[int]],
        reward: Union[np.ndarray, float],
        next_state: np.ndarray,
        done: Union[np.ndarray, bool]
    ):
        self.optimizer.zero_grad()
        loss = self.trainer.train_step(state, action, reward, next_state, done)
        loss.backward()
        self.optimizer.step()

    @property
    def snapshots_path(self):
        return os.path.join(self.dataset_path, "snapshots")

    @property
    def actions_path(self):
        return os.path.join(self.dataset_path, "actions")

    def _get_action(self, state: np.ndarray) -> Tuple[np.ndarray, int]:
        if random.random() < self.epsilon:
            max_index = random.randint(0, self.env.actions_length() - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float)
                q_values = self.model(state_tensor)
                max_index = torch.argmax(q_values).item()
        final_move = [0] * self.env.actions_length()
        final_move[max_index] = 1
        return np.array(final_move), max_index
    
    def _save_snapshot(self, step: int):
        plt.imsave(os.path.join(self.snapshots_path, f'{step}.jpg'), self.env.get_snapshot())
    
    def _save_actions(self):
        with open(self.actions_path, mode="w") as file:
            file.write("\n".join([str(action) for action in self.recorded_actions]))
    
    def play_step(
        self,
        record: bool = False,
        step: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, ActionResult]:
        old_state = self.env.get_state()
        action, max_index = self._get_action(old_state)
        self.steps += 1
        if step is None:
            step = self.steps
        result = self.env.do_action(action)
        if record:
            self._save_snapshot(step)
            self.recorded_actions.append(max_index)
            self._save_actions()
        return old_state, action, result

    def train(self, show_plot: bool = False, record: bool = False, clear_old: bool = False):
        self._setup_training(clear_old)
        
        plot_scores = []
        plot_mean_scores = []
        top_result = 0
        total_score = 0
        print(f"Begin iteration is {self.begin_iteration}")
        print(f"All iteration is {self.config.iterations}")
        if self.begin_iteration >= self.config.iterations:
            return
        for iteration in range(self.begin_iteration, self.config.iterations):
            old_state, action, result = self.play_step(
                record=record and self.count_games >= self.config.min_deaths_to_record
            )
            reward, new_state, done = result.reward, result.new_state, result.terminated
            self.memory.push(old_state, action, result.reward, result.new_state, result.terminated)

            def do_training():
                batch = self.memory.sample(self.config.batch_size)
                self._train_step(*batch)

            if len(self.memory) > self.config.batch_size and iteration % self.config.train_every_iteration == 0:
                do_training()
            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

            if done:
                self.count_games += 1
                score = result.score
                self.env.reset()
                do_training()
                if record and self.count_games > self.config.min_deaths_to_record:
                    if self.config.value_for_end_game.value == ValueForEndGame.last_action.value:
                        self.steps += 1
                        self.recorded_actions.append(self.env.actions_length())
                        self._save_snapshot(self.steps)
                    elif self.config.value_for_end_game.value == ValueForEndGame.not_exist.value:
                        pass
                self._save_actions()

                if score > top_result:
                    top_result = score
                    self.save_agent(iteration)

                print('Game', self.count_games, 'Score', score, 'Record:', top_result, "Iteration:", iteration)
                if show_plot:
                    plot_scores.append(score)
                    total_score += score
                    mean_score = total_score / self.count_games
                    plot_mean_scores.append(mean_score)
                    plot(plot_scores, plot_mean_scores)
            if self.config.save_every_iteration is not None and iteration % self.config.save_every_iteration == 0:
                self.save_agent(iteration)
        self._save_actions()
        self.save_agent(iteration+1)
        print(f"finish iteration is {iteration}")

    def _setup_training(self, clear_old: bool):
        if clear_old:
            self._clear_training_data()
        else:
            self._load_training_data()
        os.makedirs(self.snapshots_path, exist_ok=True)
        if os.path.dirname(self.model_path) != "":
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def _clear_training_data(self):
        self.steps = 0
        self.recorded_actions = []
        shutil.rmtree(self.dataset_path)

    def _load_training_data(self):
        try:
            self.steps = len([f for f in os.listdir(self.snapshots_path) if f.endswith('.jpg')])
            with open(self.actions_path) as f:
                self.recorded_actions = [int(line) for line in f]
        except:
            self.steps = 0
            self.recorded_actions = []
        print(self.steps, len(self.recorded_actions))
        assert self.steps == len(self.recorded_actions)

    def save_agent(self, iteration: int):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "count_games": self.count_games,
            "begin_iteration": iteration
        }, self.model_path)