import pygame
import heapq
import time
import random
from collections import deque

# Constants
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 20
ROWS, COLS = HEIGHT // CELL_SIZE, WIDTH // CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Solver")
clock = pygame.time.Clock()

class Maze:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.start = (0, 0)
        self.end = (rows-1, cols-1)
        
    def generate_maze(self, obstacle_density=0.2):
        """Generate a random maze with obstacles"""
        for i in range(self.rows):
            for j in range(self.cols):
                if random.random() < obstacle_density and (i, j) not in (self.start, self.end):
                    self.grid[i][j] = 1  # 1 represents an obstacle
                else:
                    self.grid[i][j] = 0  # 0 represents a free path
    
    def draw(self, screen):
        """Draw the maze on the screen"""
        for i in range(self.rows):
            for j in range(self.cols):
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.grid[i][j] == 1:  # Obstacle
                    pygame.draw.rect(screen, BLACK, rect)
                elif (i, j) == self.start:  # Start position
                    pygame.draw.rect(screen, GREEN, rect)
                elif (i, j) == self.end:  # End position
                    pygame.draw.rect(screen, RED, rect)
                else:  # Free path
                    pygame.draw.rect(screen, WHITE, rect)
                pygame.draw.rect(screen, GRAY, rect, 1)  # Grid lines
    
    def is_valid(self, row, col):
        """Check if a cell is within bounds and not an obstacle"""
        return 0 <= row < self.rows and 0 <= col < self.cols and self.grid[row][col] == 0
    
    def get_neighbors(self, row, col):
        """Get valid neighboring cells"""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if self.is_valid(r, c):
                neighbors.append((r, c))
                
        return neighbors

def heuristic(a, b):
    """Manhattan distance heuristic"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, current, draw):
    """Reconstruct and draw the path from start to end"""
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    
    for node in path:
        row, col = node
        rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, PURPLE, rect)
        pygame.display.update()
        clock.tick(60)
    
    return path

def bfs(maze, draw):
    """Breadth-First Search algorithm"""
    start = maze.start
    end = maze.end
    queue = deque([start])
    visited = set([start])
    came_from = {}
    nodes_explored = 0
    
    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
        
        current = queue.popleft()
        nodes_explored += 1
        
        # Visualize the explored nodes
        if current != start and current != end:
            row, col = current
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLUE, rect)
            pygame.display.update()
            clock.tick(60)
        
        if current == end:
            path = reconstruct_path(came_from, end, draw)
            return path, nodes_explored
        
        for neighbor in maze.get_neighbors(*current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
    
    return None, nodes_explored  # No path found

def dfs(maze, draw):
    """Depth-First Search algorithm"""
    start = maze.start
    end = maze.end
    stack = [start]
    visited = set([start])
    came_from = {}
    nodes_explored = 0
    
    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
        
        current = stack.pop()
        nodes_explored += 1
        
        # Visualize the explored nodes
        if current != start and current != end:
            row, col = current
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, YELLOW, rect)
            pygame.display.update()
            clock.tick(60)
        
        if current == end:
            path = reconstruct_path(came_from, end, draw)
            return path, nodes_explored
        
        for neighbor in maze.get_neighbors(*current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)
    
    return None, nodes_explored  # No path found

def greedy_best_first(maze, draw):
    """Greedy Best-First Search algorithm"""
    start = maze.start
    end = maze.end
    open_set = []
    heapq.heappush(open_set, (heuristic(start, end), start))
    came_from = {}
    visited = set([start])
    nodes_explored = 0
    
    while open_set:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
        
        _, current = heapq.heappop(open_set)
        nodes_explored += 1
        
        # Visualize the explored nodes
        if current != start and current != end:
            row, col = current
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, ORANGE, rect)
            pygame.display.update()
            clock.tick(60)
        
        if current == end:
            path = reconstruct_path(came_from, end, draw)
            return path, nodes_explored
        
        for neighbor in maze.get_neighbors(*current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                heapq.heappush(open_set, (heuristic(neighbor, end), neighbor))
    
    return None, nodes_explored  # No path found

def a_star(maze, draw):
    """A* Search algorithm"""
    start = maze.start
    end = maze.end
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    visited = set()
    nodes_explored = 0
    
    while open_set:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
        
        _, current = heapq.heappop(open_set)
        nodes_explored += 1
        
        # Visualize the explored nodes
        if current != start and current != end:
            row, col = current
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLUE, rect)
            pygame.display.update()
            clock.tick(60)
        
        if current == end:
            path = reconstruct_path(came_from, end, draw)
            return path, nodes_explored
        
        visited.add(current)
        
        for neighbor in maze.get_neighbors(*current):
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                if neighbor not in visited:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None, nodes_explored  # No path found

def main():
    maze = Maze(ROWS, COLS)
    maze.generate_maze(obstacle_density=0.3)
    
    algorithms = {
        "BFS": bfs,
        "DFS": dfs,
        "Greedy": greedy_best_first,
        "A*": a_star
    }
    
    results = {}
    
    running = True
    current_algorithm = None
    
    while running:
        screen.fill(WHITE)
        maze.draw(screen)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset maze
                    maze.generate_maze(obstacle_density=0.3)
                    results = {}
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    # Choose algorithm
                    if event.key == pygame.K_1:
                        current_algorithm = "BFS"
                    elif event.key == pygame.K_2:
                        current_algorithm = "DFS"
                    elif event.key == pygame.K_3:
                        current_algorithm = "Greedy"
                    elif event.key == pygame.K_4:
                        current_algorithm = "A*"
                    
                    if current_algorithm not in results:
                        # Run the algorithm
                        start_time = time.time()
                        path, nodes_explored = algorithms[current_algorithm](maze, lambda: None)
                        end_time = time.time()
                        
                        if path:
                            path_length = len(path)
                        else:
                            path_length = 0
                        
                        results[current_algorithm] = {
                            "time": end_time - start_time,
                            "nodes_explored": nodes_explored,
                            "path_length": path_length,
                            "has_path": path is not None
                        }
        
        # Display instructions
        font = pygame.font.SysFont(None, 24)
        instructions = [
            "Press 1: BFS",
            "Press 2: DFS",
            "Press 3: Greedy Best-First",
            "Press 4: A*",
            "Press R: Regenerate maze"
        ]
        
        for i, text in enumerate(instructions):
            text_surface = font.render(text, True, RED)
            screen.blit(text_surface, (10, 10 + i * 25))
        
        # Display results
        if current_algorithm and current_algorithm in results:
            result = results[current_algorithm]
            y_offset = 150
            
            if not result["has_path"]:
                no_path_text = font.render("No path found!", True, RED)
                screen.blit(no_path_text, (10, y_offset))
                y_offset += 30
            else:
                path_text = font.render(f"Path length: {result['path_length']}", True, RED)
                screen.blit(path_text, (10, y_offset))
                y_offset += 30
            
            time_text = font.render(f"Time: {result['time']:.4f}s", True, RED)
            screen.blit(time_text, (10, y_offset))
            y_offset += 30
            
            nodes_text = font.render(f"Nodes explored: {result['nodes_explored']}", True, RED)
            screen.blit(nodes_text, (10, y_offset))
        
        pygame.display.update()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()