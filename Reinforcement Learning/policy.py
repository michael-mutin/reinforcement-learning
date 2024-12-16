import time
from typing import Tuple
import matplotlib.pyplot as plt

import numpy as np
from grid import Action, Environment


class Policy():
    def __init__(self, n: int):
        """
        Creates a new deterministic Policy by randomly picking actions for each state.
        The Policy is being created for a nxn grid.

        :param n: The number of rows and the number of columns of the grid, the policy will be used with.
        :type n: int
        """
        assert n > 1, "n should be greater than 1."

        self._n = n
        self._possible_actions = [action for action in Action]
        # self._action_policy = np.random.choice([action.value for action in self._possible_actions], size=(n, n))
        q_function_size = (self._n, self._n) + (len(self._possible_actions),)
        self._q_function = np.zeros(q_function_size)

    def __call__(self, state: Tuple[int, int]) -> Action:
        """
        Returns the best action for a particular state.

        :param state: The state for which you want the action.
        :type state: Tuple[int, int]
        :return: The action, the policy would take in the provided state.
        :rtype: Action
        """
        assert state[0] in range(0, self._n) and state[1] in range(0, self._n), "The state should be within the grid."

        a = np.argmax(self._q_function[state])
        return Action(a)
    
    def epsilon_greedy(self, epsilon, state: Tuple[int, int]) -> Action:
        return np.random.choice(self._possible_actions) if np.random.rand() < epsilon else Action(np.argmax(self._q_function[state]))
    
    def sarsa(self, env: Environment, episodes=10000, learning_rate=0.3, discount=0.9, epsilon=0.3):     
        for i in range(episodes):
            state = env.reset()
            print(f"Episode {i}")
            reached_terminal_state = False
            while not reached_terminal_state:
                action: Action = self.epsilon_greedy(epsilon, state)

                next_state, reward, done = env.step(action)

                next_action: Action = self.epsilon_greedy(epsilon, next_state)

                self._q_function[state+(action.value,)] = self._q_function[state+(action.value,)] + learning_rate * (reward + discount * self._q_function[next_state+(next_action.value,)] - self._q_function[state+(action.value,)])
                
                state = next_state
                reached_terminal_state = done
            env.reset()

    def q_learning(self, env: Environment, episodes=10000, learning_rate=0.3, discount=0.9, epsilon=0.7):
        
        for i in range(episodes):
            state = env.reset()
            print(f"Episode {i}")
            reached_terminal_state = False
            while not reached_terminal_state:
                action: Action = self.epsilon_greedy(epsilon, state)

                next_state, reward, done = env.step(action)

                self._q_function[state+(action.value,)] = self._q_function[state+(action.value,)] + learning_rate * (reward + discount * max([self._q_function[next_state + (possible_action.value,)] for possible_action in self._possible_actions]) - self._q_function[state+(action.value,)])
                
                state = next_state
                reached_terminal_state = done
            env.reset()

    
    def vis_matrix(self, cmap=plt.cm.Blues):
        """Visualizes a matrix.

        :param cmap: cmap., defaults to plt.cm.Blues
        """
        greedy_arr = np.max(self._q_function, axis=-1)
        fig, ax = plt.subplots()
        ax.matshow(greedy_arr, cmap=cmap)
        for i in range(greedy_arr.shape[0]):
            for j in range(greedy_arr.shape[1]):
                c = greedy_arr[j, i]
                ax.text(i, j, "%.2f" % c, va="center", ha="center")
        fig.show()


import keyboard
def print_next_state(next_state: Tuple[int, int], reward: float, done: bool):
    print('\033[2J\033[H', end='')
    env.print_grid(next_state)
    print(f"Reward: {reward}, Done: {done}")

n = 7
goal_pos = (2,2)
holes_pos = [(1,1), (1,2), (5,3), (4,6), (0,3)]
env = Environment(n, goal_pos, (6,6))
for hole_pos in holes_pos:
    env.create_hole(hole_pos)
policy = Policy(n)
policy.q_learning(env)
policy.vis_matrix()
input("Press ENTER to continue.")  
initial_state = env.reset()
env.print_grid()

occupied = holes_pos + [goal_pos]
while True:
    next_state = initial_state
    print('\033[2J\033[H', end='')
    env.print_grid(next_state)
    input("Press ENTER to start the agent.")   
    while True:            
        time.sleep(0.5)
        action = policy(next_state)
        next_state, reward, done = env.step(action)
        print_next_state(next_state, reward, done)
        if done:
            input("Press ENTER to restart the agent.") 
            env.reset()
            break