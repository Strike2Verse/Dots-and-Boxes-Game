from environment import DotsAndBoxesEnv
from q_learner import QLearningAgent

def train(episodes: int = 20000, epsilon: float = 0.25, alpha: float = 0.07, gamma: float = 0.8) -> None:
    env = DotsAndBoxesEnv(grid_size=4)
    agent0 = QLearningAgent(player=0, epsilon=epsilon, alpha=alpha, gamma=gamma)
    agent1 = QLearningAgent(player=1, epsilon=epsilon, alpha=alpha, gamma=gamma)

    wins = {0: 0, 1: 0, 'draws': 0}
    epsilon_decay = 0.9995
    min_epsilon = 0.01

    try:
        for episode in range(episodes):
            # Alternate who starts first
            starting_player = episode % 2
            env.reset()
            env.current_player = starting_player

            # Update agents' epsilon
            agent0.epsilon = epsilon
            agent1.epsilon = epsilon

            state = env.get_state()
            done = False

            while not done:
                player = env.current_player
                agent = agent0 if player == 0 else agent1
                opponent = agent1 if player == 0 else agent0

                state_tuple = agent.get_state(state, player)
                available_actions = env.get_available_actions()
                action = agent.choose_action(state, available_actions, env)

                reward, _ = env.take_action(*action)
                next_state = env.get_state()
                next_actions = env.get_available_actions()
                next_state_tuple = agent.get_state(next_state, player)

                agent.update_q_table(state_tuple, action, reward, next_state_tuple, next_actions)

                if env.game_over():
                    done = True
                    # Add negative reward to opponent if they lose
                    if env.scores[player] > env.scores[1 - player]:
                        opponent_state = opponent.get_state(state, 1 - player)
                        opponent.update_q_table(opponent_state, action, -reward, next_state_tuple, next_actions)

            # Record results
            if env.scores[0] > env.scores[1]:
                wins[0] += 1
            elif env.scores[1] > env.scores[0]:
                wins[1] += 1
            else:
                wins['draws'] += 1

            # Epsilon decay
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

            # Print progress
            if (episode + 1) % 1000 == 0:
                print(f"Episode {episode + 1}/{episodes}")
                print(f"  Agent 0 Wins: {wins[0]}")
                print(f"  Agent 1 Wins: {wins[1]}")
                print(f"  Draws      : {wins['draws']}")
                print(f"  Current Epsilon: {epsilon:.4f}")
                print("-" * 40)

    finally:
        env.reset()
        print("Training completed!")

    # Save trained models
    agent0.save_model('agents/agent0_q_table.pkl')
    agent1.save_model('agents/agent1_q_table.pkl')


if __name__ == "__main__":
    train()  # You can pass parameters if needed
