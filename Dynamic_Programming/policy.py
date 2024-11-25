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
        self._action_policy = np.random.choice([action.value for action in self._possible_actions], size=(n, n))

    def __call__(self, state: Tuple[int, int]) -> Action:
        """
        Returns the action for a particular state.

        :param state: The state for which you want the action.
        :type state: Tuple[int, int]
        :return: The action, the policy would take in the provided state.
        :rtype: Action
        """
        assert state[0] in range(0, self._n) and state[1] in range(0, self._n), "The state should be within the grid."

        return Action(self._action_policy[state])
    
    def policy_evalution(self, value_func: np.ndarray, env: Environment, discount: float = 0.9, threshold: float = 0.1) -> np.ndarray:
        """
        Policy Evaluation algorithm.

        :param value_func: The value function.
        :type value_func: np.ndarray
        :param env: The environment.
        :type env: Environment
        :param discount: The discount factor., defaults to 0.9
        :type discount: float, optional
        :param threshold: The threshold value., defaults to 0.1
        :type threshold: float, optional
        :return: The new value function.
        :rtype: np.ndarray
        """
        assert value_func.shape == (self._n, self._n), "The value_func has to be of shape nxn where n is the value provided in __init__ when creating the policy."
        assert 0 <= discount and discount <= 1, "The discount has to be between 0 and 1."
        assert threshold > 0, "The threshold has to be a small positive number (bigger than 0)."

        value_func = value_func.copy()
        while True:
            delta = 0
            for i, j in np.ndindex(value_func.shape):
                v = value_func[(i, j)]
                action = Action(self._action_policy[(i, j)])
                next_state, reward, _ = env.step((i,j), action)
                value_func[(i, j)] = reward + discount*value_func[next_state]
                delta = max(delta, abs(v - value_func[(i, j)]))
            if delta < threshold:
                return value_func
            
    def policy_improvement(self, value_func: np.ndarray, env: Environment, discount: float = 0.9) -> bool:
        """
        Policy Improvement algorithm.

        :param value_func: The value function.
        :type value_func: np.ndarray
        :param env: The environment
        :type env: Environment
        :param discount: The discount factor., defaults to 0.9
        :type discount: float, optional
        :return: If the policy is stable.
        :rtype: bool
        """
        assert value_func.shape == (self._n, self._n), "The value_func has to be of shape nxn where n is the value provided in __init__ when creating the policy."
        assert 0 <= discount and discount <= 1, "The discount has to be between 0 and 1."

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
    
    def policy_iteration(self, value_func: np.ndarray, env: Environment, discount: float = 0.9, threshold:float = 0.1) -> np.ndarray:
        """
        Policy Iteration Algorithm.
        Combines Policy Evaluation and Policy Improvement.

        :param value_func: The value function.
        :type value_func: np.ndarray
        :param env: The environment.
        :type env: Environment
        :param discount: The discount factor., defaults to 0.9
        :type discount: float, optional
        :param threshold: The threshold value., defaults to 0.1
        :type threshold: float, optional
        :return: The resulting value function.
        :rtype: np.ndarray
        """
        assert value_func.shape == (self._n, self._n), "The value_func has to be of shape nxn where n is the value provided in __init__ when creating the policy."
        assert 0 <= discount and discount <= 1, "The discount has to be between 0 and 1."
        assert threshold > 0, "The threshold has to be a small positive number (bigger than 0)."

        while True:
            value_func = self.policy_evalution(value_func, env, discount, threshold)
            if self.policy_improvement(value_func, env, discount):
                return value_func
            
    def value_iteration(self, value_func: np.ndarray, env: Environment, discount=0.9, threshold=0.1) -> np.ndarray:
        """Value Iteration algorithm.

        .. important::
            value_func has to be zero for terminal states.
        
        :param value_func: The value function.
        :type value_func: np.ndarray
        :param env: The environment.
        :type env: Environment
        :param discount: The discount factor., defaults to 0.9
        :type discount: float, optional
        :param threshold: The threshold value., defaults to 0.1
        :type threshold: float, optional
        :return: The resulting value function.
        :rtype: np.ndarray
        """
        assert value_func.shape == (self._n, self._n), "The value_func has to be of shape nxn where n is the value provided in __init__ when creating the policy."
        assert 0 <= discount and discount <= 1, "The discount has to be between 0 and 1."
        assert threshold > 0, "The threshold has to be a small positive number (bigger than 0)."

        value_func = value_func.copy()
        while True:
            delta = 0
            for i, j in np.ndindex(value_func.shape):
                v = value_func[(i, j)]
                action_value = []
                for action in self._possible_actions:
                    next_state, reward, _ = env.step((i,j), action)
                    action_value.append(reward + discount*value_func[next_state])
                value_func[(i, j)] = max(action_value)
                delta = max(delta, abs(v - value_func[(i, j)]))
            if delta < threshold:
                break
        for i, j in np.ndindex(value_func.shape):
            action_value = {}
            for action in self._possible_actions:
                next_state, reward, _ = env.step((i,j), action)
                action_value[action] = reward + discount*value_func[next_state]
            max_action = max(action_value, key=action_value.get)
            self._action_policy[(i,j)] = max_action.value
        return value_func
        

def vis_matrix(M: np.ndarray, cmap=plt.cm.Blues):
    """Visualizes a matrix.

    :param M: The matrix to visualize.
    :type M: np.ndarray
    :param cmap: cmap., defaults to plt.cm.Blues
    """
    fig, ax = plt.subplots()
    ax.matshow(M, cmap=cmap)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            c = M[j, i]
            ax.text(i, j, "%.2f" % c, va="center", ha="center")
    fig.show()
