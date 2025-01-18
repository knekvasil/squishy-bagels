import abc
from abc import ABC
from typing import List, Any, Optional
from dataclasses import dataclass, field
import numpy as np

@dataclass
class ActionResult:
    new_state: np.ndarray
    reward: float
    terminated: bool
    score: float

class Environment(ABC):
    @abc.abstractmethod
    def get_state(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_snapshot(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def do_action(self, action: np.ndarray) -> ActionResult:
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def actions_length(self) -> int:
        pass
