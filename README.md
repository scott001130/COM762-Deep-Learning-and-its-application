# Q-Learning Grid World (COM762 Coursework)

## Overview

This project implements and evaluates a Q-Learning agent in a 5x5 Grid World environment as part of COM762 coursework.

The aim is to analyse how different learning rates affect convergence behaviour, policy optimality, and overall performance in a deterministic reinforcement learning setting.

The agent learns to navigate from a fixed start position to a goal state while avoiding obstacles and exploiting a high-reward jump transition.

---

## Features

- 5x5 Grid World environment  
- Tabular Q-Learning implementation  
- Epsilon-greedy exploration strategy  
- Multiple learning rate experiments  
- Performance evaluation (rewards & steps)  
- State-value heatmap visualisation  
- CSV export of results  

---

## Project Structure
```
project_root/  
│  
├── main.py  
├── training/  
│   └── train.py  
├── utils/  
│   ├── plots.py  
│   └── visualisation.py  
├── results/  
│   ├── tables/  
│   └── plots/  
└── README.md  
```
---

## Requirements

- Python 3.10+  
- Required libraries:
  - numpy  
  - matplotlib  

Install dependencies:

pip install numpy matplotlib

---

## How to Run

From the project root directory:

python main.py

---

## What Happens When You Run It

Running `main.py` will:

1. Train Q-Learning agents using multiple learning rates:
   - α = 1.0, 0.8, 0.5, 0.3, 0.1  

2. Output results to the console:
   - Learning rate  
   - Episodes completed  
   - Final average reward  
   - Average steps  
   - Learned optimal path  

3. Save a results table:

results/tables/learning_rate_results.csv  

4. Generate visualisations:
   - Reward curves  
   - Step count graphs  
   - State-value heatmap (best-performing agent)  

Saved in:

results/plots/  

---

## Environment Description

- Grid size: 5x5  
- Start state: [2,1]  
- Goal state: [5,5] (+10 reward)  
- Jump transition: [2,4] → [4,4] (+5 reward)  
- Obstacles:
  - [3,3], [3,4], [3,5], [4,3]  
- All other moves: -1 reward  

---

## Visualisations

The following plots are generated:

- Cumulative Reward vs Episodes  
- Steps per Episode  
- State-Value Heatmap  

The heatmap visualisation corresponds to Task F from the coursework and is generated during execution.

---

## Reproducibility

- A fixed random seed is used  
- Ensures consistent and fair comparison across different learning rates  

---

## Notes

- This implementation directly supports coursework tasks (a–f)  
- All experiments are executed from `main.py`  
- No manual configuration is required  
- Results and plots are automatically saved  

---

## Example Output

- Learning rate: 1.0  
- Episodes completed: 100  
- Final average reward: 9.60  
- Average steps: 8.96  
- Learned path:  
  [2,1] → [2,2] → [2,3] → [2,4] → [4,4] → [5,4] → [5,5]  

---

## Author

Scott Harrison  
COM762 – Deep Learning and Its Application