import numpy as np
import random
from collections import defaultdict
from environment import DotsAndBoxesEnv
import pickle

class QLearningAgent:
    def __init__(self, player: int, epsilon: float = 0.1, alpha: float = 0.5, gamma: float = 0.9) -> None:
        self.q_table = defaultdict(lambda: 0)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.player = player

    def get_state(self, state: tuple, current_player: int) -> tuple:
        edges_h, edges_v, player_scores, _ = state
        state_key = (
            tuple(map(tuple, edges_h)),
            tuple(map(tuple, edges_v)),
            tuple(player_scores),
            current_player
        )
        return state_key

    def choose_action(self, state: tuple, available_actions: list, env: 'DotsAndBoxesEnv') -> tuple:
        """
        Chooses an action based on epsilon-greedy policy.
        First, checks for box-completion actions. If none, falls back to Q-learning policy.
        """
        # First, check for any moves that complete boxes
        for action in available_actions:
            edge_type, i, j = action
            temp_env = DotsAndBoxesEnv(grid_size=env.grid_size)
            temp_env.edges_h = env.edges_h.copy()
            temp_env.edges_v = env.edges_v.copy()
            temp_env.boxes = env.boxes.copy()
            temp_env.scores = env.scores.copy()
            temp_env.current_player = self.player

            # Try to simulate the action
            _, reward = temp_env.take_action(*action)

            # If the move completes a box, prioritize it
            if reward > 0:
                return action  # Prioritize box-completion move

        # If no immediate box-completion move, fallback to regular Q-learning action choice
        q_values = [self.q_table[(self.get_state(state, self.player), a)] for a in available_actions]
        max_q = max(q_values)
        return random.choice([a for a, q in zip(available_actions, q_values) if q == max_q])

    def update_q_table(self, state: tuple, action: tuple, reward: int, next_state: tuple, next_actions: list) -> None:
        current_q = self.q_table[(state, action)]
        max_next_q = max([self.q_table[(next_state, a)] for a in next_actions], default=0)
        self.q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

    def save_model(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_model(self, filename: str) -> None:
        with open(filename, 'rb') as f:
            self.q_table = defaultdict(lambda: 0, pickle.load(f))
