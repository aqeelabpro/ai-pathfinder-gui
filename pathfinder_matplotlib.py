"""
AI Pathfinder - Interactive Matplotlib Version with GUI Buttons
Click buttons to run different algorithms instead of pressing Enter
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import numpy as np
from collections import deque
import heapq
import random
import sys

# Cell Types
EMPTY = 0
WALL = -1
START = 2
TARGET = 3
FRONTIER = 4
EXPLORED = 5
PATH = 6
DYNAMIC_OBSTACLE = -2


class GridEnvironment:
    def __init__(self, size=10):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.start = None
        self.target = None
        self.dynamic_obstacles = set()
        
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
    
    def remove_wall(self, pos):
        if pos != self.start and pos != self.target:
            self.grid[pos[0]][pos[1]] = EMPTY
    
    def add_dynamic_obstacle(self, pos):
        if (pos != self.start and pos != self.target and 
            self.grid[pos[0]][pos[1]] == EMPTY):
            self.grid[pos[0]][pos[1]] = DYNAMIC_OBSTACLE
            self.dynamic_obstacles.add(pos)
            return True
        return False
    
    def is_valid(self, pos):
        x, y = pos
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        if self.grid[x][y] in [WALL, DYNAMIC_OBSTACLE]:
            return False
        return True
    
    def get_neighbors(self, pos):
        """Get neighbors in specific order"""
        x, y = pos
        directions = [
            (-1, 0),   # Up
            (0, 1),    # Right
            (1, 0),    # Bottom
            (1, 1),    # Bottom-Right (diagonal)
            (0, -1),   # Left
            (-1, -1),  # Top-Left (diagonal)
            (-1, 1),   # Top-Right (diagonal)
            (1, -1)    # Bottom-Left (diagonal)
        ]
        
        neighbors = []
        for dx, dy in directions:
            new_pos = (x + dx, y + dy)
            if self.is_valid(new_pos):
                neighbors.append(new_pos)
        return neighbors
    
    def reset_search_markers(self):
        """Reset frontier, explored, and path markers"""
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] in [FRONTIER, EXPLORED, PATH]:
                    self.grid[i][j] = EMPTY
        if self.start:
            self.grid[self.start[0]][self.start[1]] = START
        if self.target:
            self.grid[self.target[0]][self.target[1]] = TARGET
    
    def clear_all(self):
        """Clear entire grid"""
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.start = None
        self.target = None
        self.dynamic_obstacles.clear()


class InteractiveVisualizer:
    def __init__(self, grid_env, delay=0.05):
        self.grid_env = grid_env
        self.delay = delay
        
        # Create figure with extra space for buttons
        self.fig = plt.figure(figsize=(12, 10))
        
        # Main grid axes
        self.ax = self.fig.add_axes([0.1, 0.3, 0.8, 0.65])
        
        # Algorithm buttons
        button_width = 0.12
        button_height = 0.04
        button_y_top = 0.20
        button_y_bottom = 0.13
        
        # Top row buttons (algorithms)
        ax_bfs = self.fig.add_axes([0.05, button_y_top, button_width, button_height])
        ax_dfs = self.fig.add_axes([0.18, button_y_top, button_width, button_height])
        ax_ucs = self.fig.add_axes([0.31, button_y_top, button_width, button_height])
        ax_dls = self.fig.add_axes([0.44, button_y_top, button_width, button_height])
        ax_iddfs = self.fig.add_axes([0.57, button_y_top, button_width, button_height])
        ax_bidir = self.fig.add_axes([0.70, button_y_top, button_width, button_height])
        
        # Bottom row buttons (controls)
        ax_reset = self.fig.add_axes([0.05, button_y_bottom, button_width, button_height])
        ax_clear = self.fig.add_axes([0.18, button_y_bottom, button_width, button_height])
        ax_wall = self.fig.add_axes([0.31, button_y_bottom, button_width, button_height])
        ax_erase = self.fig.add_axes([0.44, button_y_bottom, button_width, button_height])
        ax_sample1 = self.fig.add_axes([0.57, button_y_bottom, button_width, button_height])
        ax_sample2 = self.fig.add_axes([0.70, button_y_bottom, button_width, button_height])
        
        # Create buttons
        self.btn_bfs = Button(ax_bfs, 'BFS', color='lightblue', hovercolor='skyblue')
        self.btn_dfs = Button(ax_dfs, 'DFS', color='lightblue', hovercolor='skyblue')
        self.btn_ucs = Button(ax_ucs, 'UCS', color='lightblue', hovercolor='skyblue')
        self.btn_dls = Button(ax_dls, 'DLS', color='lightblue', hovercolor='skyblue')
        self.btn_iddfs = Button(ax_iddfs, 'IDDFS', color='lightblue', hovercolor='skyblue')
        self.btn_bidir = Button(ax_bidir, 'Bidirectional', color='lightblue', hovercolor='skyblue')
        
        self.btn_reset = Button(ax_reset, 'Reset Search', color='lightgreen', hovercolor='green')
        self.btn_clear = Button(ax_clear, 'Clear Grid', color='lightyellow', hovercolor='yellow')
        self.btn_wall = Button(ax_wall, 'Draw Wall', color='lightcoral', hovercolor='red')
        self.btn_erase = Button(ax_erase, 'Erase', color='lightgray', hovercolor='gray')
        self.btn_sample1 = Button(ax_sample1, 'Sample 1', color='wheat', hovercolor='orange')
        self.btn_sample2 = Button(ax_sample2, 'Sample 2', color='wheat', hovercolor='orange')
        
        # Status text
        self.status_text = self.fig.text(0.5, 0.06, 'Click buttons to run algorithms or draw on grid', 
                                         ha='center', fontsize=11, style='italic')
        
        # Mode tracking
        self.mode = "VIEW"  # Modes: VIEW, DRAW_WALL, ERASE, SET_START, SET_TARGET
        self.drawing = False
        
        # PathFinder instance
        self.pathfinder = None
    
    def draw_grid(self, algorithm_name="GOOD PERFORMANCE TIME APP"):
        self.ax.clear()
        self.ax.set_title(algorithm_name, fontsize=16, fontweight='bold')
        
        # Set up the grid
        self.ax.set_xlim(0, self.grid_env.size)
        self.ax.set_ylim(0, self.grid_env.size)
        self.ax.set_aspect('equal')
        
        # Remove ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Draw grid lines and cells
        for i in range(self.grid_env.size):
            for j in range(self.grid_env.size):
                cell_value = self.grid_env.grid[i][j]
                
                # Draw rectangle based on cell type
                rect = patches.Rectangle((j, self.grid_env.size - i - 1), 1, 1,
                                        linewidth=1, edgecolor='black', 
                                        facecolor=self._get_color(cell_value))
                self.ax.add_patch(rect)
                
                # Add text (0 or -1 for walls)
                if cell_value in [EMPTY, FRONTIER, EXPLORED, PATH]:
                    text_val = '0'
                elif cell_value in [WALL, DYNAMIC_OBSTACLE]:
                    text_val = '-1'
                else:
                    text_val = ''
                
                if text_val:
                    self.ax.text(j + 0.5, self.grid_env.size - i - 0.5, text_val,
                               ha='center', va='center', fontsize=10, color='black')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _get_color(self, cell_type):
        """Get color for cell type"""
        colors = {
            EMPTY: 'white',
            WALL: 'red',
            START: 'blue',
            TARGET: 'green',
            FRONTIER: 'yellow',
            EXPLORED: 'lightblue',
            PATH: 'purple',
            DYNAMIC_OBSTACLE: 'orange'
        }
        return colors.get(cell_type, 'white')
    
    def update_cell(self, pos, cell_type):
        """Update a single cell and refresh display"""
        if pos != self.grid_env.start and pos != self.grid_env.target:
            self.grid_env.grid[pos[0]][pos[1]] = cell_type
        self.draw_grid()
        # Add small delay during algorithm execution
        if self.delay > 0:
            import time
            time.sleep(self.delay)
    
    def update_status(self, message):
        """Update status message"""
        self.status_text.set_text(message)
        self.fig.canvas.draw_idle()
    
    def on_grid_click(self, event):
        """Handle clicks on the grid"""
        if event.inaxes != self.ax:
            return
        
        # Convert click coordinates to grid position
        j = int(event.xdata)
        i = self.grid_env.size - int(event.ydata) - 1
        
        if i < 0 or i >= self.grid_env.size or j < 0 or j >= self.grid_env.size:
            return
        
        pos = (i, j)
        
        if self.mode == "DRAW_WALL":
            self.grid_env.add_wall(pos)
            self.draw_grid()
        elif self.mode == "ERASE":
            self.grid_env.remove_wall(pos)
            self.draw_grid()
        elif self.mode == "SET_START":
            self.grid_env.set_start(pos)
            self.mode = "VIEW"
            self.update_status("Start position set! Now set target or run an algorithm.")
            self.draw_grid()
        elif self.mode == "SET_TARGET":
            self.grid_env.set_target(pos)
            self.mode = "VIEW"
            self.update_status("Target position set! Ready to run algorithms.")
            self.draw_grid()
    
    def setup_callbacks(self, pathfinder):
        """Setup button callbacks"""
        self.pathfinder = pathfinder
        
        # Algorithm buttons
        self.btn_bfs.on_clicked(lambda event: self.run_algorithm('BFS'))
        self.btn_dfs.on_clicked(lambda event: self.run_algorithm('DFS'))
        self.btn_ucs.on_clicked(lambda event: self.run_algorithm('UCS'))
        self.btn_dls.on_clicked(lambda event: self.run_algorithm('DLS'))
        self.btn_iddfs.on_clicked(lambda event: self.run_algorithm('IDDFS'))
        self.btn_bidir.on_clicked(lambda event: self.run_algorithm('Bidirectional'))
        
        # Control buttons
        self.btn_reset.on_clicked(lambda event: self.reset_search())
        self.btn_clear.on_clicked(lambda event: self.clear_grid())
        self.btn_wall.on_clicked(lambda event: self.toggle_draw_wall())
        self.btn_erase.on_clicked(lambda event: self.toggle_erase())
        self.btn_sample1.on_clicked(lambda event: self.load_sample1())
        self.btn_sample2.on_clicked(lambda event: self.load_sample2())
        
        # Grid click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_grid_click)
        self.fig.canvas.mpl_connect('button_release_event', lambda event: setattr(self, 'drawing', False))
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('close_event', self.on_close)

    def on_close(self, event):
        """Handle window close event"""
        print("\nClosing window... Goodbye!")
        sys.exit(0)
        
    def on_mouse_move(self, event):
        """Handle mouse movement for drawing"""
        if event.button == 1 and event.inaxes == self.ax:  # Left button pressed
            if self.mode in ["DRAW_WALL", "ERASE"]:
                try:
                    j = int(event.xdata)
                    i = self.grid_env.size - int(event.ydata) - 1
                    
                    if 0 <= i < self.grid_env.size and 0 <= j < self.grid_env.size:
                        pos = (i, j)
                        if self.mode == "DRAW_WALL":
                            self.grid_env.add_wall(pos)
                        else:
                            self.grid_env.remove_wall(pos)
                        self.draw_grid()
                except:
                    pass
    
    def run_algorithm(self, algo_name):
        """Run selected algorithm"""
        if not self.grid_env.start or not self.grid_env.target:
            self.update_status("⚠️ Please set both Start (click on grid) and Target positions first!")
            return
        
        self.grid_env.reset_search_markers()
        self.update_status(f"Running {algo_name}...")
        
        result = None
        if algo_name == 'BFS':
            result = self.pathfinder.bfs()
        elif algo_name == 'DFS':
            result = self.pathfinder.dfs()
        elif algo_name == 'UCS':
            result = self.pathfinder.ucs()
        elif algo_name == 'DLS':
            result = self.pathfinder.dls(15)
        elif algo_name == 'IDDFS':
            result = self.pathfinder.iddfs(15)
        elif algo_name == 'Bidirectional':
            result = self.pathfinder.bidirectional_search()
        
        if result:
            self.update_status(f"✓ {algo_name} found path of length {len(result)}! Click 'Reset Search' to try another algorithm.")
        else:
            self.update_status(f"✗ {algo_name} did not find a path. Try another algorithm or modify the grid.")
    
    def reset_search(self):
        """Reset search results but keep grid"""
        self.grid_env.reset_search_markers()
        self.draw_grid()
        self.update_status("Search results cleared. Ready to run another algorithm.")
    
    def clear_grid(self):
        """Clear entire grid"""
        self.grid_env.clear_all()
        self.draw_grid()
        self.update_status("Grid cleared. Click to set Start position, then Target position, then draw walls.")
        self.mode = "SET_START"
    
    def toggle_draw_wall(self):
        """Toggle wall drawing mode"""
        if self.mode == "DRAW_WALL":
            self.mode = "VIEW"
            self.update_status("Wall drawing mode OFF. Click algorithm buttons to run search.")
        else:
            self.mode = "DRAW_WALL"
            self.update_status("Wall drawing mode ON. Click and drag on grid to draw walls.")
    
    def toggle_erase(self):
        """Toggle erase mode"""
        if self.mode == "ERASE":
            self.mode = "VIEW"
            self.update_status("Erase mode OFF. Click algorithm buttons to run search.")
        else:
            self.mode = "ERASE"
            self.update_status("Erase mode ON. Click and drag on grid to erase walls.")
    
    def load_sample1(self):
        """Load sample grid 1"""
        self.grid_env.clear_all()
        self.grid_env.set_start((6, 1))
        self.grid_env.set_target((5, 6))
        for i in range(1, 7):
            self.grid_env.add_wall((i, 5))
        self.draw_grid()
        self.update_status("Sample 1 loaded: Simple vertical wall. Ready to run algorithms!")
    
    def load_sample2(self):
        """Load sample grid 2"""
        self.grid_env.clear_all()
        self.grid_env.set_start((1, 1))
        self.grid_env.set_target((8, 8))
        for i in [2, 4, 6]:
            for j in range(1, 9):
                if j not in [4, 5]:
                    self.grid_env.add_wall((i, j))
        self.draw_grid()
        self.update_status("Sample 2 loaded: Maze pattern. Ready to run algorithms!")


# Import PathFinder class from previous implementation
class PathFinder:
    def __init__(self, grid_env, visualizer):
        self.grid = grid_env
        self.viz = visualizer
        self.dynamic_obstacle_prob = 0.00
        
    def spawn_dynamic_obstacle(self):
        if random.random() < self.dynamic_obstacle_prob:
            empty_cells = [(i, j) for i in range(self.grid.size) 
                          for j in range(self.grid.size) 
                          if self.grid.grid[i][j] == EMPTY]
            if empty_cells:
                pos = random.choice(empty_cells)
                self.grid.add_dynamic_obstacle(pos)
                self.viz.update_cell(pos, DYNAMIC_OBSTACLE)
                return pos
        return None
    
    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        for pos in path:
            if pos != self.grid.target:
                self.viz.update_cell(pos, PATH)
        return path
    
    def bfs(self):
        self.viz.draw_grid("BFS - GOOD PERFORMANCE TIME APP")
        queue = deque([self.grid.start])
        came_from = {}
        explored = set()
        
        while queue:
            current = queue.popleft()
            if current == self.grid.target:
                return self.reconstruct_path(came_from, current)
            if current in explored:
                continue
            explored.add(current)
            if current != self.grid.start:
                self.viz.update_cell(current, EXPLORED)
            self.spawn_dynamic_obstacle()
            for neighbor in self.grid.get_neighbors(current):
                if neighbor not in explored and neighbor not in queue:
                    came_from[neighbor] = current
                    queue.append(neighbor)
                    if neighbor != self.grid.target:
                        self.viz.update_cell(neighbor, FRONTIER)
        return None
    
    def dfs(self):
        self.viz.draw_grid("DFS - GOOD PERFORMANCE TIME APP")
        stack = [self.grid.start]
        came_from = {}
        explored = set()
        
        while stack:
            current = stack.pop()
            if current == self.grid.target:
                return self.reconstruct_path(came_from, current)
            if current in explored:
                continue
            explored.add(current)
            if current != self.grid.start:
                self.viz.update_cell(current, EXPLORED)
            self.spawn_dynamic_obstacle()
            neighbors = self.grid.get_neighbors(current)
            for neighbor in reversed(neighbors):
                if neighbor not in explored and neighbor not in came_from:
                    came_from[neighbor] = current
                    stack.append(neighbor)
                    if neighbor != self.grid.target:
                        self.viz.update_cell(neighbor, FRONTIER)
        return None
    
    def ucs(self):
        self.viz.draw_grid("UCS - GOOD PERFORMANCE TIME APP")
        pq = [(0, self.grid.start)]
        came_from = {}
        cost_so_far = {self.grid.start: 0}
        explored = set()
        
        while pq:
            current_cost, current = heapq.heappop(pq)
            if current == self.grid.target:
                return self.reconstruct_path(came_from, current)
            if current in explored:
                continue
            explored.add(current)
            if current != self.grid.start:
                self.viz.update_cell(current, EXPLORED)
            self.spawn_dynamic_obstacle()
            for neighbor in self.grid.get_neighbors(current):
                new_cost = current_cost + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current
                    heapq.heappush(pq, (new_cost, neighbor))
                    if neighbor != self.grid.target and neighbor not in explored:
                        self.viz.update_cell(neighbor, FRONTIER)
        return None
    
    def dls(self, limit=10):
        self.viz.draw_grid(f"DLS (limit={limit}) - GOOD PERFORMANCE TIME APP")
        
        def dls_recursive(node, depth, came_from, explored):
            if node == self.grid.target:
                return came_from, True
            if depth == 0:
                return None, False
            explored.add(node)
            if node != self.grid.start:
                self.viz.update_cell(node, EXPLORED)
            self.spawn_dynamic_obstacle()
            for neighbor in self.grid.get_neighbors(node):
                if neighbor not in explored:
                    came_from[neighbor] = node
                    if neighbor != self.grid.target:
                        self.viz.update_cell(neighbor, FRONTIER)
                    result, found = dls_recursive(neighbor, depth - 1, came_from, explored)
                    if found:
                        return result, True
            return None, False
        
        came_from = {}
        explored = set()
        result, found = dls_recursive(self.grid.start, limit, came_from, explored)
        if found:
            return self.reconstruct_path(result, self.grid.target)
        return None
    
    def iddfs(self, max_depth=20):
        for depth in range(max_depth):
            self.grid.reset_search_markers()
            self.viz.draw_grid(f"IDDFS (depth={depth}) - GOOD PERFORMANCE TIME APP")
            result = self.dls(depth)
            if result is not None:
                return result
        return None
    
    def bidirectional_search(self):
        self.viz.draw_grid("Bidirectional - GOOD PERFORMANCE TIME APP")
        forward_queue = deque([self.grid.start])
        forward_came_from = {self.grid.start: None}
        forward_explored = set()
        backward_queue = deque([self.grid.target])
        backward_came_from = {self.grid.target: None}
        backward_explored = set()
        
        while forward_queue and backward_queue:
            if forward_queue:
                current = forward_queue.popleft()
                forward_explored.add(current)
                if current != self.grid.start:
                    self.viz.update_cell(current, EXPLORED)
                if current in backward_explored:
                    path = []
                    node = current
                    while node is not None:
                        path.append(node)
                        node = forward_came_from[node]
                    path.reverse()
                    node = backward_came_from[current]
                    while node is not None:
                        path.append(node)
                        node = backward_came_from[node]
                    for pos in path:
                        if pos != self.grid.start and pos != self.grid.target:
                            self.viz.update_cell(pos, PATH)
                    return path
                self.spawn_dynamic_obstacle()
                for neighbor in self.grid.get_neighbors(current):
                    if neighbor not in forward_explored and neighbor not in forward_came_from:
                        forward_came_from[neighbor] = current
                        forward_queue.append(neighbor)
                        if neighbor != self.grid.target:
                            self.viz.update_cell(neighbor, FRONTIER)
            if backward_queue:
                current = backward_queue.popleft()
                backward_explored.add(current)
                if current != self.grid.target:
                    self.viz.update_cell(current, EXPLORED)
                if current in forward_explored:
                    path = []
                    node = current
                    while node is not None:
                        path.append(node)
                        node = forward_came_from[node]
                    path.reverse()
                    node = backward_came_from[current]
                    while node is not None:
                        path.append(node)
                        node = backward_came_from[node]
                    for pos in path:
                        if pos != self.grid.start and pos != self.grid.target:
                            self.viz.update_cell(pos, PATH)
                    return path
                for neighbor in self.grid.get_neighbors(current):
                    if neighbor not in backward_explored and neighbor not in backward_came_from:
                        backward_came_from[neighbor] = current
                        backward_queue.append(neighbor)
                        if neighbor != self.grid.start:
                            self.viz.update_cell(neighbor, FRONTIER)
        return None


def main():
    print("="*60)
    print("AI PATHFINDER - Interactive GUI with Buttons")
    print("="*60)
    print("\nInstructions:")
    print("1. Click 'Sample 1' or 'Sample 2' to load a pre-made grid")
    print("2. Or click on grid to set Start (blue), then Target (green)")
    print("3. Click 'Draw Wall' and drag on grid to draw walls")
    print("4. Click algorithm buttons (BFS, DFS, etc.) to run search")
    print("5. Click 'Reset Search' to clear results and try another algorithm")
    print("6. Click 'Clear Grid' to start fresh")
    print("\nClose the window (X button) to exit.")
    print("="*60)
    
    # Create environment
    grid_env = GridEnvironment(size=10)
    
    # Load sample grid
    grid_env.set_start((6, 1))
    grid_env.set_target((5, 6))
    for i in range(1, 7):
        grid_env.add_wall((i, 5))
    
    # Create visualizer
    viz = InteractiveVisualizer(grid_env, delay=0.05)
    
    # Create pathfinder
    pathfinder = PathFinder(grid_env, viz)
    
    # Setup button callbacks
    viz.setup_callbacks(pathfinder)
    
    # Initial draw
    viz.draw_grid("GOOD PERFORMANCE TIME APP - Interactive Mode")
    viz.update_status("Sample 1 loaded. Click algorithm buttons to run, or load Sample 2, or draw your own grid!")
    
    # Show window
    print("\nWindow is open. Close the window (X button) to exit.\n")
    plt.show()


if __name__ == "__main__":
    main()
