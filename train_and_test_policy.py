import numpy as np
import matplotlib.pyplot as plt
import random
import time
from typing import Tuple

from grid import Environment
from policy import Policy, vis_matrix


def print_next_state(next_state: Tuple[int, int], reward: float, done: bool):
    print('\033[2J\033[H', end='')
    env.print_grid(next_state)
    print(f"Reward: {reward}, Done: {done}")

n = 7
goal_pos = (2,2)
holes_pos = [(1,1), (1,2), (5,3), (4,6), (0,3)]
env = Environment(n, goal_pos)
for hole_pos in holes_pos:
    env.create_hole(hole_pos)
policy = Policy(n)
value_func = np.zeros((n,n))

value_func = policy.policy_iteration(value_func, env)
# value_func = policy.value_iteration(value_func, env)

vis_matrix(value_func)

occupied = holes_pos + [goal_pos]
while True:
    next_state = random.choice([(i,j) for i in range(n) for j in range(n) if (i,j) not in occupied])
    print('\033[2J\033[H', end='')
    env.print_grid(next_state)
    input("Press ENTER to start the agent.")   
    while True:            
        time.sleep(1)
        action = policy(next_state)
        next_state, reward, done = env.step(next_state, action)
        print_next_state(next_state, reward, done)
        if done:
            input("Press ENTER to restart the agent.") 
            break