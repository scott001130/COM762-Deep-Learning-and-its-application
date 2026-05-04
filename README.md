# Q-Learning Grid World (COM762 Coursework)

This project implements a Q-Learning agent to solve a 5x5 Grid World environment using Reinforcement Learning in Python.

The agent learns to navigate from a fixed start position to a goal state while avoiding obstacles and utilising a special jump transition.

---

## Features

- 5x5 Grid World environment
- Q-Learning (tabular implementation)
- Epsilon-greedy exploration strategy
- Multiple learning rate experiments
- Performance evaluation (rewards & steps)
- State-value heatmap visualisation
- CSV export of results

---

## Project Structure

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

---

## How to Run

### 1. Install Python

Ensure Python 3.10+ is installed.

Check version:

python --version

---

### 2. Install Dependencies

pip install matplotlib numpy

---

### 3. Run the Program

From the project root directory:

python main.py

---

## What Happens When You Run It

Running `main.py` will:

1. Train Q-Learning agents using multiple learning rates:
   - α = 1.0, 0.8, 0.5, 0.3, 0.1

2. Output results to console:
   - Learning rate
   - Episodes completed
   - Final average reward
   - Average steps
   - Learned optimal path

3. Save results table to CSV:

results/tables/learning_rate_results.csv

4. Generate visualisations:
   - Reward curves
   - Step count graphs
   - State-value heatmap (best agent)

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

## Output Example

- Learning rate: 1.0  
- Episodes completed: 100
- Final average reward: 9.60  
- Average steps: 8.96  
- Learned path: [2,1] -> [2,2] -> [2,3] -> [2,4] -> [4,4] -> [5,4] -> [5,5]  
----------------------------------------

---

## Visualisations

The following plots are generated:

- Cumulative Reward vs Episodes  
- Steps per Episode  
- State-Value Heatmap  

---

## Reproducibility

- A fixed random seed is used  
- Ensures consistent results across runs  

---

## Notes

- All experiments run from `main.py`  
- No manual configuration required  
- Results are automatically saved  

---

## Author

Scott Harrison  
COM762 – Deep Learning and Its Application