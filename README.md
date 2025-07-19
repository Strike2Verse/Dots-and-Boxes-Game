# 🧠 Dots and Boxes - Reinforcement Learning Game (Q-Learning Agent)

This project is a Python-based implementation of the classic **Dots and Boxes** game using **Reinforcement Learning**. It features a playable environment with **Pygame** where a human player can compete against a trained **Q-learning agent**.

---

## 🎮 Game Overview

Dots and Boxes is a two-player strategy game where players take turns adding a single horizontal or vertical line between two unjoined adjacent dots. A player who completes the fourth side of a 1×1 box earns one point and takes another turn. The player with the most points wins.

---

## 🤖 Project Features

- ✅ Custom game environment built with `Pygame`
- ✅ Q-Learning agent with training and testing modes
- ✅ Support for human vs. agent gameplay
- ✅ Save/load trained Q-tables
- ✅ Reward system based on box completion
- ✅ Epsilon decay for exploration-exploitation balance

---

## 🗂 Project Structure

```
├── environment.py # Game logic and environment class
├── q_learner.py # Q-learning agent logic
├── train.py # Agent training script
├── test.py # Human vs. Agent gameplay
├── agents/
│ ├── agent0_q_table.pkl # Trained Q-table for player 0 (optional)
│ └── agent1_q_table.pkl # Trained Q-table for player 1
├── _pycache_
└── README.md # Project documentation
```

---

## 📦 Installation

Make sure you have Python 3.7+ installed.

