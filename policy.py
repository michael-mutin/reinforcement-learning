from typing import Tuple

import numpy as np
from grid import Action, Environment


class Policy():
    def __init__(self, n: int):
        self._n = n
        self._possible_actions = [action for action in Action]
        self._action_policy = np.random.choice([action.value for action in self._possible_actions], size=(n, n))

    def __call__(self, state: Tuple[int, int]) -> Action:
        assert state[0] in range(0, self._n) and state[1] in range(0, self._n), "The state should be within the grid"
        return Action(self._action_policy[state])
    
    def evaluation(self, value_func: np.ndarray, env: Environment, discount=0.9, threshold=0.1) -> np.ndarray:
        value_func = value_func.copy()
        while True:
            delta = 0
            for i, j in np.ndindex(value_func.shape):
                v = value_func[(i, j)]
                action = Action(self._action_policy[(i, j)])
                next_state, reward, _ = env.step((i,j), action)
                new_v = reward + discount*value_func[next_state]
                value_func[(i, j)] = new_v
                delta = max(delta, abs(v - new_v))
            if delta < threshold:
                return value_func
            
    def improvement(self, value_func: np.ndarray, env: Environment, discount=0.9) -> bool:
        policy_stable = True
        for i, j in np.ndindex(value_func.shape):
            old_action = self._action_policy[(i, j)]
            action_value = {}
            for action in self._possible_actions:
                next_state, reward, _ = env.step((i,j), action)
                action_value[action] = reward + discount*value_func[next_state]
            max_action = max(action_value, key=action_value.get)
            self._action_policy[(i,j)] = max_action.value
            if old_action != max_action.value:
                policy_stable = False
        return policy_stable
    
    def iteration(self, value_func: np.ndarray, env:Environment, discount=0.9, threshold=0.1) -> None:
        while True:
            value_func = policy.evaluation(value_func, env, discount, threshold)
            if policy.improvement(value_func, env, discount):
                break

def print_next_state(next_state, reward, done):
    print('\033[2J\033[H', end='')
    env.view_grid(next_state)
    print(f"Reward: {reward}, Done: {done}")
    
if __name__ == "__main__":
    import random
    import time

    n = 7
    goal_pos = (2,2)
    holes_pos = [(1,1), (1,2), (5,3), (4,6), (0,3)]
    env = Environment(n, goal_pos)
    for hole_pos in holes_pos:
        env.createHole(hole_pos)
    policy = Policy(n)
    value_func = np.zeros((n,n))

    policy.iteration(value_func, env)

    occupied = holes_pos + [goal_pos]
    while True:
        next_state = random.choice([(i,j) for i in range(n) for j in range(n) if (i,j) not in occupied])
        print('\033[2J\033[H', end='')
        env.view_grid(next_state)
        input("Press ENTER to start the agent.")   
        while True:            
            time.sleep(1)
            action = policy(next_state)
            next_state, reward, done = env.step(next_state, action)
            print_next_state(next_state, reward, done)
            if done:
                input("Press ENTER to restart the agent.") 
                break
