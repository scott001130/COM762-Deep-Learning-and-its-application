from typing import Tuple


State = Tuple[int, int]


class GridWorld:
    """
    5x5 Grid World environment for Q-learning.

    Coursework grid uses 1-indexed positions:
    Start: [2,1]
    Jump:  [2,4] -> [4,4]
    Goal:  [5,5]

    Code uses 0-indexed positions internally:
    Start: (1,0)
    Jump:  (1,3) -> (3,3)
    Goal:  (4,4)
    """

    def __init__(self) -> None:
        self.size = 5

        self.start_state: State = (1, 0)
        self.goal_state: State = (4, 4)

        self.jump_from: State = (1, 3)
        self.jump_to: State = (3, 3)


        self.obstacles = {
            (2, 2),  # [3,3]
            (2, 3),  # [3,4]
            (2, 4),  # [3,5]
            (3, 2),  # [4,3]
        }

        self.actions = {
            0: "North",
            1: "South",
            2: "East",
            3: "West",
        }

        self.state: State = self.start_state

    def reset(self) -> State:
        """Reset the agent to the fixed start state."""
        self.state = self.start_state
        return self.state

    def step(self, action: int) -> tuple[State, int, bool]:
        """
        Apply an action and return:
        next_state, reward, done
        """

        if action not in self.actions:
            raise ValueError(f"Invalid action {action}. Use 0, 1, 2, or 3.")

        row, col = self.state

        # Special jump: [2,4] -> [4,4] when moving South
        if self.state == self.jump_from and action == 1:
            self.state = self.jump_to
            return self.state, 5, False

        if action == 0:      # North
            proposed_state = (row - 1, col)
        elif action == 1:    # South
            proposed_state = (row + 1, col)
        elif action == 2:    # East
            proposed_state = (row, col + 1)
        else:                # West
            proposed_state = (row, col - 1)

        if not self._is_valid_state(proposed_state):
            return self.state, -1, False

        self.state = proposed_state

        if self.state == self.goal_state:
            return self.state, 10, True

        return self.state, -1, False

    def _is_valid_state(self, state: State) -> bool:
        row, col = state

        inside_grid = 0 <= row < self.size and 0 <= col < self.size
        not_obstacle = state not in self.obstacles

        return inside_grid and not_obstacle

    def get_all_states(self) -> list[State]:
        """Return all non-obstacle states."""
        return [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if (row, col) not in self.obstacles
        ]

    def state_to_label(self, state: State) -> str:
        """Convert 0-indexed internal state to coursework-style [row,column]."""
        row, col = state
        return f"[{row + 1},{col + 1}]"
