import pygame
from environment import DotsAndBoxesEnv
from q_learner import QLearningAgent

# --- Settings ---
GRID_SIZE = 4
WIDTH, HEIGHT = 600, 600
MARGIN = 60
LINE_WIDTH = 4

# Init pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dots and Boxes: You vs Agent")
font = pygame.font.SysFont(None, 30)
clock = pygame.time.Clock()

# Init game
env = DotsAndBoxesEnv(grid_size=GRID_SIZE)
agent = QLearningAgent(player=1)
agent.load_model('agents/agent1_q_table.pkl')

human_player = 0
agent_player = 1
env.current_player = human_player  # You start
spacing = (WIDTH - 2 * MARGIN) / GRID_SIZE

def get_clicked_edge(pos):
    x, y = pos
    for i in range(GRID_SIZE + 1):
        for j in range(GRID_SIZE):
            x1 = MARGIN + j * spacing
            y1 = MARGIN + i * spacing
            if x1 <= x <= x1 + spacing and abs(y - y1) <= 10:
                return ('h', i, j)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE + 1):
            x1 = MARGIN + j * spacing
            y1 = MARGIN + i * spacing
            if y1 <= y <= y1 + spacing and abs(x - x1) <= 10:
                return ('v', i, j)
    return None

# --- Game Loop ---
running = True
env.render(screen, font)
game_over = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if not game_over and env.current_player == human_player and event.type == pygame.MOUSEBUTTONDOWN:
            action = get_clicked_edge(pygame.mouse.get_pos())
            if action and action in env.get_available_actions():
                env.take_action(*action)
                env.render(screen, font)

    if not game_over and env.current_player == agent_player:
        pygame.time.delay(300)
        state = env.get_state()
        available = env.get_available_actions()
        action = agent.choose_action(state, available, env)
        env.take_action(*action)
        env.render(screen, font)

    if env.game_over() and not game_over:
        game_over = True
        winner = "Draw!"
        if env.scores[human_player] > env.scores[agent_player]:
            winner = "You Win!"
        elif env.scores[agent_player] > env.scores[human_player]:
            winner = "Agent Wins!"
        text = font.render(winner, True, (0, 128, 0))
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT - 40))
        pygame.display.flip()

    clock.tick(30)

pygame.quit()
