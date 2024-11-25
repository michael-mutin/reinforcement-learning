import random
from enum import Enum
from typing import Optional, Tuple

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
    def __init__(self, n: int, agent_pos: Tuple[int, int], goal_pos: Tuple[int, int]):
        assert n > 1, "n should be > 1"
        assert agent_pos is None or (agent_pos[0] in range(0, n) and agent_pos[1] in range(0, n)), "The position of the agent should be within the grid"
        assert goal_pos is None or (goal_pos[0] in range(0, n) and goal_pos[1] in range(0, n)), "The position of the goal should be within the grid"
        assert agent_pos != goal_pos, "At the start, the agent should not be in the goal"

        self._n: int = n
        self._grid: np.ndarray = np.ones((n, n), dtype=np.uint8) * GridTile.FIELD.value
        self._initial_agent_pos: Tuple[int, int] = agent_pos
        self._agent_pos: Tuple[int, int] = agent_pos
        self._grid[goal_pos] = GridTile.GOAL.value
        self._return: int = 0
        self._done: bool = False
    
    def get_return(self) -> int:
        return self._return
    
    def get_done(self) -> int:
        return self._done

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool]:
        assert self._agent_pos[0] in range(0, self._n) and self._agent_pos[1] in range(0, self._n), "The state should be within the grid."
        assert not self._done, "Cannot make a step when done."

        if self._grid[self._agent_pos] == GridTile.GOAL.value or self._grid[self._agent_pos] == GridTile.HOLE.value:
            return self._agent_pos, 0, True

        new_state = None
        reward = 0
        if action == Action.UP:
            new_state = (self._agent_pos[0]-1, self._agent_pos[1])
        elif action == Action.DOWN:
            new_state = (self._agent_pos[0]+1, self._agent_pos[1])
        elif action == Action.LEFT:
            new_state = (self._agent_pos[0], self._agent_pos[1]-1)
        else: # Action.RIGHT
            new_state = (self._agent_pos[0], self._agent_pos[1]+1)
        
        if (not new_state[0] in range(0, self._n)) or (not new_state[1] in range(0, self._n)):
            reward = -1
        else:
            self._agent_pos = new_state
            if self._grid[self._agent_pos] == GridTile.HOLE.value:
                self._done = True
                reward = -1
            elif self._grid[self._agent_pos] == GridTile.GOAL.value:
                self._done = True
                reward = 1
        
        self._return += reward

        return self._agent_pos, reward, self._done
    
    def reset(self) -> Tuple[int, int]:
        """Resets Environment and return initial state.

        :return: The initial state
        :rtype: Tuple[int, int]
        """
        self._agent_pos = self._initial_agent_pos
        self._return = 0
        self._done = False

        return self._initial_agent_pos

    def create_hole(self, hole_pos: Tuple[int, int]):
        assert (hole_pos[0] in range(0, self._n) and hole_pos[1] in range(0, self._n)), "The position of the hole should be within the grid"
        assert self._grid[hole_pos] == GridTile.FIELD.value, "The hole can only be placed on a FIELD"

        self._grid[hole_pos] = GridTile.HOLE.value

    
    def print_grid(self, allow_emojis: bool = True):
        assert self._agent_pos[0] in range(0, self._n) and self._agent_pos[1] in range(0, self._n), "The state should be within the grid"

        grid_copy = self._grid.copy()
        if grid_copy[self._agent_pos] == GridTile.GOAL.value:
            grid_copy[self._agent_pos] = GridTile.AGENT_IN_GOAL.value
        elif grid_copy[self._agent_pos] == GridTile.HOLE.value:
            grid_copy[self._agent_pos] = GridTile.AGENT_IN_HOLE.value
        else:
            grid_copy[self._agent_pos] = GridTile.AGENT.value
        
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

if __name__ == "__main__":
    import keyboard

    env = Environment(7, (6,6), (0,0))
    env.create_hole((1,1))
    env.create_hole((1,2))
    env.create_hole((5,3))
    env.create_hole((4,6))
    env.create_hole((0,3))

    def print_next_state(reward, done):
        print('\033[2J\033[H', end='')
        env.print_grid()
        print(f"Reward: {reward}, Done: {done}")
        if done:
            print(f"Return: {env.get_return()}")
            print(f"Press any key except ESC to reset the environment. Press ESC to exit.")

    def on_key_event(event):
        global next_state
        if event.event_type == 'down' and event.name != 'esc':
            if env.get_done():
                env.reset()
                print('\033[2J\033[H', end='')
                env.print_grid()
            else:
                match event.scan_code:
                    case 72: # up                
                        next_state, reward, done = env.step(Action.UP)
                        print_next_state(reward, done)
                    case 80: # down
                        next_state, reward, done = env.step(Action.DOWN)
                        print_next_state(reward, done)
                    case 75: # left
                        next_state, reward, done = env.step(Action.LEFT)
                        print_next_state(reward, done)
                    case 77: # right
                        next_state, reward, done = env.step(Action.RIGHT)
                        print_next_state(reward, done)

    keyboard.on_press(on_key_event)
    print("Programm läuft. Drücke ESC zum Beenden...")
    env.print_grid()
    keyboard.wait('esc')