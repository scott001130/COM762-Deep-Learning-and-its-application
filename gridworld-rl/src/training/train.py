import random
from statistics import mean
from typing import Any

import numpy as np

from agent.q_learning import QLearningAgent
from environment.gridworld import GridWorld

random.seed(42)
np.random.seed(42)

def train_agent(
    learning_rate: float,
    episodes: int = 100,
    discount_factor: float = 0.9,
    epsilon: float = 0.2,
    max_steps_per_episode: int = 100,
    stop_average_reward: float = 10.0,
    stop_window: int = 30,
) -> dict[str, Any]:
    """
    Train one Q-learning agent for a given learning rate.
    """

    env = GridWorld()

    agent = QLearningAgent(
        env=env,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
    )

    episode_rewards: list[int] = []
    episode_steps: list[int] = []

    for episode in range(episodes):
        state = env.reset()
        cumulative_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps_per_episode:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.update_q_value(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
            )

            state = next_state
            cumulative_reward += reward
            steps += 1

        episode_rewards.append(cumulative_reward)
        episode_steps.append(steps)

        if len(episode_rewards) >= stop_window:
            recent_average = mean(episode_rewards[-stop_window:])

            if recent_average > stop_average_reward:
                break

    return {
        "agent": agent,
        "learning_rate": learning_rate,
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "episodes_completed": len(episode_rewards),
        "final_average_reward": mean(episode_rewards[-min(stop_window, len(episode_rewards)):]),
        "average_steps": mean(episode_steps),
    }


def run_learning_rate_experiments() -> list[dict[str, Any]]:
    """
    Run Q-learning using alpha = 1 and several values between 0 and 1.
    """

    learning_rates = [1.0, 0.8, 0.5, 0.3, 0.1]
    results = []

    for learning_rate in learning_rates:
        result = train_agent(learning_rate=learning_rate)
        results.append(result)

    return results
