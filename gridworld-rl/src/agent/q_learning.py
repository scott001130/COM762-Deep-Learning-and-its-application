import random
from typing import Dict

import numpy as np

from environment.gridworld import GridWorld, State


class QLearningAgent:
    """
    Tabular Q-learning agent using epsilon-greedy exploration.
    """

    def __init__(
        self,
        env: GridWorld,
        learning_rate: float = 1.0,
        discount_factor: float = 0.9,
        epsilon: float = 0.2,
    ) -> None:
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.q_table: Dict[State, np.ndarray] = {
            state: np.zeros(len(self.env.actions))
            for state in self.env.get_all_states()
        }

    def choose_action(self, state: State) -> int:
        """
        Epsilon-greedy action selection.
        Explore randomly with probability epsilon.
        Exploit best known Q-value otherwise.
        """

        if random.random() < self.epsilon:
            return random.choice(list(self.env.actions.keys()))

        return int(np.argmax(self.q_table[state]))

    def update_q_value(
        self,
        state: State,
        action: int,
        reward: int,
        next_state: State,
    ) -> None:
        """
        Q-learning update rule:
        Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s')) - Q(s,a)]
        """

        current_q = self.q_table[state][action]
        best_next_q = np.max(self.q_table[next_state])

        updated_q = current_q + self.learning_rate * (
            reward + self.discount_factor * best_next_q - current_q
        )

        self.q_table[state][action] = updated_q

    def get_state_values(self) -> dict[State, float]:
        """
        Convert Q-table into state values for visualisation.
        V(s) = max_a Q(s,a)
        """

        return {
            state: float(np.max(action_values))
            for state, action_values in self.q_table.items()
        }

    def get_best_action(self, state: State) -> int:
        """Return best learned action for a state."""
        return int(np.argmax(self.q_table[state]))

    def get_optimal_path(self, max_steps: int = 20) -> list[State]:
        """
        Follow the best learned action from the start state until the goal is reached.
        """

        path = []
        state = self.env.reset()
        path.append(state)

        for _ in range(max_steps):
            if state == self.env.goal_state:
                break

            action = self.get_best_action(state)
            next_state, _, done = self.env.step(action)

            path.append(next_state)
            state = next_state

            if done:
                break

        return path
