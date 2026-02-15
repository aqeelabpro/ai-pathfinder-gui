"""
Screenshot Generator - Creates all screenshots needed for assignment report
Runs all algorithms and saves screenshots automatically
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import deque
import heapq
import os

# Cell Types
EMPTY = 0
WALL = -1
START = 2
TARGET = 3
FRONTIER = 4
EXPLORED = 5
PATH = 6


class GridEnvironment:
    def __init__(self, size=10):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.start = None
        self.target = None
        
    def set_start(self, pos):
        if self.start:
            self.grid[self.start[0]][self.start[1]] = EMPTY
        self.start = pos
        self.grid[pos[0]][pos[1]] = START
    
    def set_target(self, pos):
        if self.target:
            self.grid[self.target[0]][self.target[1]] = EMPTY
        self.target = pos
        self.grid[pos[0]][pos[1]] = TARGET
    
    def add_wall(self, pos):
        if pos != self.start and pos != self.target:
            self.grid[pos[0]][pos[1]] = WALL
    
    def is_valid(self, pos):
        x, y = pos
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        if self.grid[x][y] == WALL:
            return False
        return True
    
    def get_neighbors(self, pos):
        x, y = pos
        directions = [(-1, 0), (0, 1), (1, 0), (1, 1), (0, -1), (-1, -1), (-1, 1), (1, -1)]
        return [(x + dx, y + dy) for dx, dy in directions if self.is_valid((x + dx, y + dy))]
    
    def reset_search_markers(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] in [FRONTIER, EXPLORED, PATH]:
                    self.grid[i][j] = EMPTY
        if self.start:
            self.grid[self.start[0]][self.start[1]] = START
        if self.target:
            self.grid[self.target[0]][self.target[1]] = TARGET


def draw_grid_to_figure(grid_env, title="GOOD PERFORMANCE TIME APP"):
    """Draw grid and return figure"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlim(0, grid_env.size)
    ax.set_ylim(0, grid_env.size)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    colors = {
        EMPTY: 'white', WALL: 'red', START: 'blue', TARGET: 'green',
        FRONTIER: 'yellow', EXPLORED: 'lightblue', PATH: 'purple'
    }
    
    for i in range(grid_env.size):
        for j in range(grid_env.size):
            cell_value = grid_env.grid[i][j]
            rect = patches.Rectangle((j, grid_env.size - i - 1), 1, 1,
                                    linewidth=1, edgecolor='black',
                                    facecolor=colors.get(cell_value, 'white'))
            ax.add_patch(rect)
            
            if cell_value in [EMPTY, FRONTIER, EXPLORED, PATH]:
                text_val = '0'
            elif cell_value == WALL:
                text_val = '-1'
            else:
                text_val = ''
            
            if text_val:
                ax.text(j + 0.5, grid_env.size - i - 0.5, text_val,
                       ha='center', va='center', fontsize=10, color='black')
    
    return fig


def bfs(grid_env):
    """Run BFS and mark explored/frontier cells"""
    queue = deque([grid_env.start])
    came_from = {grid_env.start: None}
    explored = set()
    
    while queue:
        current = queue.popleft()
        if current == grid_env.target:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            for pos in path:
                if pos != grid_env.target and pos != grid_env.start:
                    grid_env.grid[pos[0]][pos[1]] = PATH
            return True
        
        if current in explored:
            continue
        explored.add(current)
        
        if current != grid_env.start:
            grid_env.grid[current[0]][current[1]] = EXPLORED
        
        for neighbor in grid_env.get_neighbors(current):
            if neighbor not in explored and neighbor not in came_from:
                came_from[neighbor] = current
                queue.append(neighbor)
                if neighbor != grid_env.target:
                    grid_env.grid[neighbor[0]][neighbor[1]] = FRONTIER
    return False


def dfs(grid_env):
    """Run DFS"""
    stack = [grid_env.start]
    came_from = {grid_env.start: None}
    explored = set()
    
    while stack:
        current = stack.pop()
        if current == grid_env.target:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            for pos in path:
                if pos != grid_env.target and pos != grid_env.start:
                    grid_env.grid[pos[0]][pos[1]] = PATH
            return True
        
        if current in explored:
            continue
        explored.add(current)
        
        if current != grid_env.start:
            grid_env.grid[current[0]][current[1]] = EXPLORED
        
        for neighbor in reversed(grid_env.get_neighbors(current)):
            if neighbor not in explored and neighbor not in came_from:
                came_from[neighbor] = current
                stack.append(neighbor)
                if neighbor != grid_env.target:
                    grid_env.grid[neighbor[0]][neighbor[1]] = FRONTIER
    return False


def ucs(grid_env):
    """Run UCS"""
    pq = [(0, grid_env.start)]
    came_from = {grid_env.start: None}
    cost_so_far = {grid_env.start: 0}
    explored = set()
    
    while pq:
        current_cost, current = heapq.heappop(pq)
        
        if current == grid_env.target:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            for pos in path:
                if pos != grid_env.target and pos != grid_env.start:
                    grid_env.grid[pos[0]][pos[1]] = PATH
            return True
        
        if current in explored:
            continue
        explored.add(current)
        
        if current != grid_env.start:
            grid_env.grid[current[0]][current[1]] = EXPLORED
        
        for neighbor in grid_env.get_neighbors(current):
            new_cost = current_cost + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))
                if neighbor != grid_env.target and neighbor not in explored:
                    grid_env.grid[neighbor[0]][neighbor[1]] = FRONTIER
    return False


def generate_screenshots():
    """Generate all screenshots"""
    output_dir = "screenshots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("="*60)
    print("GENERATING SCREENSHOTS FOR ASSIGNMENT")
    print("="*60)
    
    # BFS Best Case
    print("\n1. BFS Best Case...")
    grid = GridEnvironment(10)
    grid.set_start((6, 1))
    grid.set_target((5, 6))
    for i in range(1, 7):
        grid.add_wall((i, 5))
    
    fig = draw_grid_to_figure(grid, "BFS - Best Case (Initial)")
    fig.savefig(f"{output_dir}/BFS_best_initial.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    bfs(grid)
    fig = draw_grid_to_figure(grid, "BFS - Best Case (Final)")
    fig.savefig(f"{output_dir}/BFS_best_final.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓ BFS best case saved")
    
    # BFS Worst Case
    print("2. BFS Worst Case...")
    grid = GridEnvironment(10)
    grid.set_start((1, 1))
    grid.set_target((8, 8))
    for i in [2, 4, 6]:
        for j in range(1, 9):
            if j not in [4, 5]:
                grid.add_wall((i, j))
    
    fig = draw_grid_to_figure(grid, "BFS - Worst Case (Initial)")
    fig.savefig(f"{output_dir}/BFS_worst_initial.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    bfs(grid)
    fig = draw_grid_to_figure(grid, "BFS - Worst Case (Final)")
    fig.savefig(f"{output_dir}/BFS_worst_final.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓ BFS worst case saved")
    
    # DFS Best Case
    print("3. DFS Best Case...")
    grid = GridEnvironment(10)
    grid.set_start((1, 1))
    grid.set_target((5, 5))
    
    fig = draw_grid_to_figure(grid, "DFS - Best Case (Initial)")
    fig.savefig(f"{output_dir}/DFS_best_initial.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    dfs(grid)
    fig = draw_grid_to_figure(grid, "DFS - Best Case (Final)")
    fig.savefig(f"{output_dir}/DFS_best_final.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓ DFS best case saved")
    
    # DFS Worst Case
    print("4. DFS Worst Case...")
    grid = GridEnvironment(10)
    grid.set_start((5, 5))
    grid.set_target((5, 6))
    grid.add_wall((4, 5))
    grid.add_wall((5, 4))
    
    fig = draw_grid_to_figure(grid, "DFS - Worst Case (Initial)")
    fig.savefig(f"{output_dir}/DFS_worst_initial.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    dfs(grid)
    fig = draw_grid_to_figure(grid, "DFS - Worst Case (Final)")
    fig.savefig(f"{output_dir}/DFS_worst_final.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓ DFS worst case saved")
    
    # UCS Best Case
    print("5. UCS Best Case...")
    grid = GridEnvironment(10)
    grid.set_start((2, 2))
    grid.set_target((7, 7))
    for i in range(3, 8):
        grid.add_wall((i, 4))
    
    fig = draw_grid_to_figure(grid, "UCS - Best Case (Initial)")
    fig.savefig(f"{output_dir}/UCS_best_initial.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    ucs(grid)
    fig = draw_grid_to_figure(grid, "UCS - Best Case (Final)")
    fig.savefig(f"{output_dir}/UCS_best_final.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓ UCS best case saved")
    
    # UCS Worst Case
    print("6. UCS Worst Case...")
    grid = GridEnvironment(10)
    grid.set_start((1, 1))
    grid.set_target((8, 8))
    for i in [3, 5, 7]:
        for j in range(1, 9):
            if j not in [2, 7]:
                grid.add_wall((i, j))
    
    fig = draw_grid_to_figure(grid, "UCS - Worst Case (Initial)")
    fig.savefig(f"{output_dir}/UCS_worst_initial.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    ucs(grid)
    fig = draw_grid_to_figure(grid, "UCS - Worst Case (Final)")
    fig.savefig(f"{output_dir}/UCS_worst_final.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓ UCS worst case saved")
    
    print("\n" + "="*60)
    print(f"✓ All screenshots saved to '{output_dir}/' folder!")
    print("="*60)
    print("\nScreenshots generated:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")
    print("\nYou can now use these images in your report!")


if __name__ == "__main__":
    generate_screenshots()
