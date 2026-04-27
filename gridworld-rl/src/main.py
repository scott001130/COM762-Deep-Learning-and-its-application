import csv
import os

from training.train import run_learning_rate_experiments
from utils.plots import plot_rewards, plot_steps
from utils.visualisation import plot_state_values


def save_results_table(results: list[dict]) -> None:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "learning_rate_results.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "Learning Rate",
                "Episodes Completed",
                "Final Average Reward",
                "Average Steps",
                "Learned Path",
            ]
        )

        for result in results:
            agent = result["agent"]
            path = agent.get_optimal_path()
            labelled_path = " -> ".join(
                agent.env.state_to_label(state) for state in path
            )

            writer.writerow(
                [
                    result["learning_rate"],
                    result["episodes_completed"],
                    round(result["final_average_reward"], 2),
                    round(result["average_steps"], 2),
                    labelled_path,
                ]
            )

    print(f"Results table saved to: {output_path}")


def main() -> None:
    results = run_learning_rate_experiments()

    print("Learning rate experiments completed.\n")

    best_result = max(results, key=lambda result: result["final_average_reward"])

    for result in results:
        agent = result["agent"]
        path = agent.get_optimal_path()
        labelled_path = [agent.env.state_to_label(state) for state in path]

        print(f"Learning rate: {result['learning_rate']}")
        print(f"Episodes completed: {result['episodes_completed']}")
        print(f"Final average reward: {result['final_average_reward']:.2f}")
        print(f"Average steps: {result['average_steps']:.2f}")
        print(f"Learned path: {' -> '.join(labelled_path)}")
        print("-" * 40)

    print(
        f"\nBest learning rate based on final average reward: "
        f"{best_result['learning_rate']}"
    )

    save_results_table(results)

    plot_state_values(
        best_result["agent"],
        filename=f"state_values_alpha_{best_result['learning_rate']}.png",
    )

    plot_rewards(results)
    plot_steps(results)


if __name__ == "__main__":
    main()
