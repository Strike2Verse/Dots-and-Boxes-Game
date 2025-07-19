import numpy as np
import pygame

# Constants
GRID_SIZE = 4
WIDTH, HEIGHT = 600, 600
MARGIN = 60
DOT_RADIUS = 5
LINE_WIDTH = 4
BOX_COLORS = [(173, 216, 230), (255, 182, 193)]  # Player 0, Player 1

class DotsAndBoxesEnv:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.edges_h = np.zeros((self.grid_size + 1, self.grid_size))
        self.edges_v = np.zeros((self.grid_size, self.grid_size + 1))
        self.boxes = np.full((self.grid_size, self.grid_size), -1)
        self.scores = [0, 0]
        self.current_player = 0
        self.last_scorer = None

    def get_state(self):
        return (
            self.edges_h.copy(),
            self.edges_v.copy(),
            self.scores.copy(),
            self.current_player
        )

    def get_available_actions(self):
        actions = []
        for i in range(self.grid_size + 1):
            for j in range(self.grid_size):
                if self.edges_h[i, j] == 0:
                    actions.append(('h', i, j))
        for i in range(self.grid_size):
            for j in range(self.grid_size + 1):
                if self.edges_v[i, j] == 0:
                    actions.append(('v', i, j))
        return actions

    def take_action(self, edge_type, i, j):
        if edge_type == 'h' and self.edges_h[i, j] == 0:
            self.edges_h[i, j] = 1
        elif edge_type == 'v' and self.edges_v[i, j] == 0:
            self.edges_v[i, j] = 1
        else:
            return 0, 0  # Invalid move

        player_that_moved = self.current_player
        completed = self._check_completed_boxes(player_that_moved)

        if completed == 0:
            self.current_player = 1 - self.current_player
        else:
            self.last_scorer = player_that_moved

        return completed, completed  # reward is number of boxes completed

    def _check_completed_boxes(self, player_that_moved):
        completed = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.boxes[i, j] == -1:
                    if self.edges_h[i, j] and self.edges_h[i + 1, j] and \
                       self.edges_v[i, j] and self.edges_v[i, j + 1]:
                        self.boxes[i, j] = player_that_moved
                        self.scores[player_that_moved] += 1
                        completed += 1
        return completed

    def game_over(self):
        return np.all(self.boxes != -1)

    def render(self, screen, font, human_player=0, agent_player=1):
        spacing = (WIDTH - 2 * MARGIN) / self.grid_size
        screen.fill((255, 255, 255))

        # Draw boxes
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                owner = self.boxes[i, j]
                if owner != -1:
                    x = MARGIN + j * spacing
                    y = MARGIN + i * spacing
                    rect = pygame.Rect(x + 2, y + 2, spacing - 4, spacing - 4)
                    pygame.draw.rect(screen, BOX_COLORS[owner], rect)

        # Draw horizontal edges
        for i in range(self.grid_size + 1):
            for j in range(self.grid_size):
                if self.edges_h[i, j]:
                    x1 = MARGIN + j * spacing
                    y1 = MARGIN + i * spacing
                    pygame.draw.line(screen, (0, 0, 255), (x1, y1), (x1 + spacing, y1), LINE_WIDTH)

        # Draw vertical edges
        for i in range(self.grid_size):
            for j in range(self.grid_size + 1):
                if self.edges_v[i, j]:
                    x1 = MARGIN + j * spacing
                    y1 = MARGIN + i * spacing
                    pygame.draw.line(screen, (255, 0, 0), (x1, y1), (x1, y1 + spacing), LINE_WIDTH)

        # Draw dots
        for i in range(self.grid_size + 1):
            for j in range(self.grid_size + 1):
                x = MARGIN + j * spacing
                y = MARGIN + i * spacing
                pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), DOT_RADIUS)

        # Draw scores
        text = font.render(f"You: {self.scores[human_player]} | Agent: {self.scores[agent_player]}", True, (0, 0, 0))
        screen.blit(text, (20, 10))

        pygame.display.flip()
