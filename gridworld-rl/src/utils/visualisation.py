import os

import matplotlib.pyplot as plt
import numpy as np

from agent.q_learning import QLearningAgent


def create_state_value_grid(agent: QLearningAgent) -> np.ndarray:
    """
    Create a 5x5 grid of state values.
    Obstacles are shown as NaN.
    """

    values = agent.get_state_values()
    grid = np.full((agent.env.size, agent.env.size), np.nan)

    for state, value in values.items():
        row, col = state
        grid[row, col] = value

    return grid


def plot_state_values(agent: QLearningAgent, filename: str = "state_values.png") -> None:
    """
    Visualise state values in the grid-world layout.
    """

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_dir = os.path.join(base_dir, "results", "graphs")
    os.makedirs(output_dir, exist_ok=True)

    grid = create_state_value_grid(agent)

    plt.figure(figsize=(7, 6))
    plt.imshow(grid)

    for row in range(agent.env.size):
        for col in range(agent.env.size):
            state = (row, col)

            if state in agent.env.obstacles:
                text = "X"
            elif state == agent.env.start_state:
                text = "S"
            elif state == agent.env.goal_state:
                text = "G"
            elif state == agent.env.jump_from:
                text = "J"
            elif state == agent.env.jump_to:
                text = "L"
            else:
                value = grid[row, col]
                text = "" if np.isnan(value) else f"{value:.1f}"

            plt.text(col, row, text, ha="center", va="center")

    plt.title("State-Value Visualisation of Learned Q-Learning Policy")
    plt.xticks(range(agent.env.size), range(1, agent.env.size + 1))
    plt.yticks(range(agent.env.size), range(1, agent.env.size + 1))
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.colorbar(label="State Value")
    plt.tight_layout()

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300)
    plt.show(block=True)
    plt.close()

    print(f"State-value visualisation saved to: {output_path}")