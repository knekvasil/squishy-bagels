
import os
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class QTrainer:
    def __init__(self, model: torch.nn.Module, gamma: float):
        self.gamma = gamma
        self.model = model
        self.criterion = nn.MSELoss()

    def train_step(
        self,
        state: np.ndarray,
        action: Union[np.ndarray, int],
        reward: Union[np.ndarray, float],
        next_state: np.ndarray,
        done: Union[np.ndarray, bool]
    ) -> torch.Tensor:
        state = torch.tensor(np.array(state, dtype=np.float32), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state, dtype=np.float32), dtype=torch.float)
        action = torch.tensor(np.array(action, dtype=np.int64), dtype=torch.long)
        reward = torch.tensor(np.array(reward, dtype=np.float32), dtype=torch.float)
        
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        return self.criterion.forward(target, pred)