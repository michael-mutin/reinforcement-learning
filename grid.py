import random
from enum import Enum
from typing import Tuple

import numpy as np


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class GridTile(Enum):
    HOLE = 0
    FIELD = 1
    GOAL = 2
    AGENT = 3
    AGENT_IN_GOAL = 4
    AGENT_IN_HOLE = 5

class Environment:
    def __init__(self, n: int, goal_pos: Tuple[int, int]):
        assert n > 1, "n should be > 1"
        assert goal_pos[0] in range(0, n) and goal_pos[1] in range(0, n), "The position of the goal should be within the grid"
        self._n = n
        self._grid: np.ndarray = np.ones((n, n), dtype=np.uint8) * GridTile.FIELD.value
        self._grid[goal_pos] = GridTile.GOAL.value

    def step(self, state: Tuple[int, int], action: Action) -> Tuple[Tuple[int, int], float, bool]:
        assert state[0] in range(0, self._n) and state[1] in range(0, self._n), "The state should be within the grid"

        if self._grid[state] == GridTile.GOAL.value or self._grid[state] == GridTile.HOLE.value:
            return state, 0, True

        new_state = None
        reward = 0
        done = False
        if action == Action.UP:
            new_state = (state[0]-1, state[1])
        elif action == Action.DOWN:
            new_state = (state[0]+1, state[1])
        elif action == Action.LEFT:
            new_state = (state[0], state[1]-1)
        else: # Action.RIGHT
            new_state = (state[0], state[1]+1)
        
        if (not new_state[0] in range(0, self._n)) or (not new_state[1] in range(0, self._n)):
            new_state = state
            reward = -1
        else:
            if self._grid[new_state] == GridTile.HOLE.value:
                done = True
                reward = -1
            elif self._grid[new_state] == GridTile.GOAL.value:
                done = True
                reward = 1

        return new_state, reward, done
    
    def reset():
        raise NotImplementedError # Should the reset methode exist, if the state is managed from outside?


    def createHole(self, hole_pos: Tuple[int, int]):
        assert hole_pos[0] in range(0, self._n) and hole_pos[1] in range(0, self._n), "The position of the hole should be within the grid"
        assert self._grid[hole_pos] == GridTile.FIELD.value, "The hole can only be placed on a FIELD"
        
        self._grid[hole_pos] = GridTile.HOLE.value

    
    def view_grid(self, state: Tuple[int, int], allow_emojis: bool = True):
        assert state[0] in range(0, self._n) and state[1] in range(0, self._n), "The state should be within the grid"

        grid_copy = self._grid.copy()
        if grid_copy[state] == GridTile.GOAL.value:
            grid_copy[state] = GridTile.AGENT_IN_GOAL.value
        elif grid_copy[state] == GridTile.HOLE.value:
            grid_copy[state] = GridTile.AGENT_IN_HOLE.value
        else:
            grid_copy[state] = GridTile.AGENT.value
        
        if allow_emojis:
            emoji_map = {
                GridTile.HOLE.value: '⭕',
                GridTile.FIELD.value: '🟩',
                GridTile.GOAL.value: '🏁',
                GridTile.AGENT.value: '😺',
                GridTile.AGENT_IN_GOAL.value: '😸',
                GridTile.AGENT_IN_HOLE.value: '🙀'
            }
            emoji_grid = [list(map(emoji_map.get, row)) for row in grid_copy]     
            for row in emoji_grid:
                print(" ".join(row))
        else:
            print(grid_copy)