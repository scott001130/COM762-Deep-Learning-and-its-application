import os

import matplotlib.pyplot as plt

def plot_rewards(results):
    os.makedirs("gridworld-rl/results/graphs", exist_ok=True)

    plt.figure(figsize=(10, 6))

    for result in results:
        rewards = result["episode_rewards"]
        alpha = result["learning_rate"]

        plt.plot(rewards, label=f"α = {alpha}")

    plt.title("Cumulative Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid()

    plt.savefig("gridworld-rl/results/graphs/reward_curve.png")
    plt.show()

def plot_steps(results):
    os.makedirs("gridworld-rl/results/graphs", exist_ok=True)

    plt.figure(figsize=(10, 6))

    for result in results:
        steps = result["episode_steps"]
        alpha = result["learning_rate"]

        plt.plot(steps, label=f"α = {alpha}")

    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps to Goal")
    plt.legend()
    plt.grid()

    plt.savefig("gridworld-rl/results/graphs/steps_curve.png")
    plt.show()
