"""
AI Pathfinder - Console Version
Algorithms: BFS, DFS, UCS, DLS, IDDFS, Bidirectional
"""

from collections import deque
import heapq

# Cell types
EMPTY  = 0
WALL   = -1
START  = 2
TARGET = 3
PATH   = 6

SYMBOLS = {
    EMPTY:  '_',
    WALL:   '#',
    START:  'S',
    TARGET: 'T',
    PATH:   '*',
}

# ─────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────
class Grid:
    def __init__(self, size=10):
        self.size  = size
        self.cells = [[EMPTY] * size for _ in range(size)]
        self.start  = None
        self.target = None

    def set_start(self, pos):
        if self.start:
            self.cells[self.start[0]][self.start[1]] = EMPTY
        self.start = pos
        self.cells[pos[0]][pos[1]] = START

    def set_target(self, pos):
        if self.target:
            self.cells[self.target[0]][self.target[1]] = EMPTY
        self.target = pos
        self.cells[pos[0]][pos[1]] = TARGET

    def add_wall(self, pos):
        if pos not in (self.start, self.target):
            self.cells[pos[0]][pos[1]] = WALL

    def is_valid(self, pos):
        r, c = pos
        return (0 <= r < self.size and
                0 <= c < self.size and
                self.cells[r][c] != WALL)

    def neighbors(self, pos):
        r, c = pos
        dirs = [(-1,0),(1,0),(0,-1),(0,1),
                (-1,-1),(-1,1),(1,-1),(1,1)]
        return [(r+dr, c+dc) for dr, dc in dirs
                if self.is_valid((r+dr, c+dc))]

    def reset_path(self):
        for r in range(self.size):
            for c in range(self.size):
                if self.cells[r][c] == PATH:
                    self.cells[r][c] = EMPTY
        if self.start:
            self.cells[self.start[0]][self.start[1]]   = START
        if self.target:
            self.cells[self.target[0]][self.target[1]] = TARGET

    def mark_path(self, path):
        for pos in path:
            if pos not in (self.start, self.target):
                self.cells[pos[0]][pos[1]] = PATH

    def print(self):
        print("\n   " + " ".join(str(c) for c in range(self.size)))
        for r in range(self.size):
            row = " ".join(SYMBOLS.get(self.cells[r][c], '?') for c in range(self.size))
            print(f"{r:2} {row}")
        print(f"\n  {SYMBOLS[START]}=Start  {SYMBOLS[TARGET]}=Target  {SYMBOLS[WALL]}=Wall  {SYMBOLS[PATH]}=Path  {SYMBOLS[EMPTY]}=Empty")

# ─────────────────────────────────────────────
# Path reconstruction
# ─────────────────────────────────────────────
def reconstruct(came_from, end):
    path, node = [], end
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path

# ─────────────────────────────────────────────
# Algorithms
# ─────────────────────────────────────────────
def bfs(grid):
    queue, came_from, visited = deque([grid.start]), {}, {grid.start}
    while queue:
        cur = queue.popleft()
        if cur == grid.target:
            return reconstruct(came_from, cur)
        for nb in grid.neighbors(cur):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = cur
                queue.append(nb)
    return None

def dfs(grid):
    stack, came_from, visited = [grid.start], {}, set()
    while stack:
        cur = stack.pop()
        if cur == grid.target:
            return reconstruct(came_from, cur)
        if cur in visited:
            continue
        visited.add(cur)
        for nb in grid.neighbors(cur):
            if nb not in visited:
                came_from[nb] = cur
                stack.append(nb)
    return None

def ucs(grid):
    pq, came_from, cost, visited = [(0, grid.start)], {}, {grid.start: 0}, set()
    while pq:
        g, cur = heapq.heappop(pq)
        if cur == grid.target:
            return reconstruct(came_from, cur)
        if cur in visited:
            continue
        visited.add(cur)
        for nb in grid.neighbors(cur):
            new_cost = g + 1
            if nb not in cost or new_cost < cost[nb]:
                cost[nb], came_from[nb] = new_cost, cur
                heapq.heappush(pq, (new_cost, nb))
    return None

def dls(grid, limit):
    stack, came_from, visited = [(grid.start, 0)], {}, set()
    while stack:
        cur, depth = stack.pop()
        if cur == grid.target:
            return reconstruct(came_from, cur)
        if depth >= limit or cur in visited:
            continue
        visited.add(cur)
        for nb in grid.neighbors(cur):
            if nb not in visited:
                came_from[nb] = cur
                stack.append((nb, depth+1))
    return None

def iddfs(grid, max_depth=30):
    for depth in range(max_depth+1):
        result = dls(grid, depth)
        if result is not None:
            return result
    return None

def bidirectional(grid):
    fwd_queue, fwd_visited = deque([grid.start]), {grid.start: None}
    bwd_queue, bwd_visited = deque([grid.target]), {grid.target: None}
    while fwd_queue or bwd_queue:
        if fwd_queue:
            cur = fwd_queue.popleft()
            for nb in grid.neighbors(cur):
                if nb not in fwd_visited:
                    fwd_visited[nb] = cur
                    fwd_queue.append(nb)
                if nb in bwd_visited:
                    return _merge(fwd_visited, bwd_visited, nb)
        if bwd_queue:
            cur = bwd_queue.popleft()
            for nb in grid.neighbors(cur):
                if nb not in bwd_visited:
                    bwd_visited[nb] = cur
                    bwd_queue.append(nb)
                if nb in fwd_visited:
                    return _merge(fwd_visited, bwd_visited, nb)
    return None

def _merge(fwd, bwd, meet):
    path_fwd, node = [], meet
    while node is not None:
        path_fwd.append(node)
        node = fwd[node]
    path_fwd.reverse()
    node = bwd[meet]
    while node is not None:
        path_fwd.append(node)
        node = bwd[node]
    return path_fwd

# ─────────────────────────────────────────────
# Sample grids
# ─────────────────────────────────────────────
def load_sample1(grid):
    grid.__init__(grid.size)
    grid.set_start((6,1))
    grid.set_target((5,6))
    for r in range(1,7):
        grid.add_wall((r,5))

def load_sample2(grid):
    grid.__init__(grid.size)
    grid.set_start((1,1))
    grid.set_target((8,8))
    for r in [2,4,6]:
        for c in range(1,9):
            if c not in (4,5):
                grid.add_wall((r,c))

# ─────────────────────────────────────────────
# Run algorithm
# ─────────────────────────────────────────────
ALGOS = ["BFS", "DFS", "UCS", "DLS", "IDDFS", "Bidirectional"]

def run_algorithm(name, grid):
    grid.reset_path()
    print(f"\nRunning {name}...")
    if name=="BFS": path=bfs(grid)
    elif name=="DFS": path=dfs(grid)
    elif name=="UCS": path=ucs(grid)
    elif name=="DLS": path=dls(grid, limit=15)
    elif name=="IDDFS": path=iddfs(grid, max_depth=30)
    elif name=="Bidirectional": path=bidirectional(grid)
    else: path=None
    if path:
        grid.mark_path(path)
        grid.print()
        print(f"Path found! Length = {len(path)} steps")
        full_path = [grid.start] + path + [grid.target]
        trace = " → ".join(f"({r},{c})" for r,c in full_path)
        print("\n  Route: "+trace)
    else:
        grid.print()
        print("No path found.")

def main():
    grid = Grid(size=10)
    load_sample1(grid)
    while True:
        print("\n"+"="*42)
        print("   AI PATHFINDER - Console")
        print("="*42)
        print(" Algorithms:")
        print("   1. BFS          2. DFS")
        print("   3. UCS          4. DLS (limit=15)")
        print("   5. IDDFS        6. Bidirectional")
        print(" Grids:")
        print("   7. Sample 1 (vertical wall)")
        print("   8. Sample 2 (maze pattern)")
        print("   9. Show grid")
        print("   0. Exit")
        print("="*42)
        choice=input("Choose: ").strip()
        if choice in ("1","2","3","4","5","6"):
            if not grid.start or not grid.target:
                print("No start/target set. Load a sample first (7 or 8).")
            else:
                run_algorithm(ALGOS[int(choice)-1],grid)
        elif choice=="7": load_sample1(grid); grid.print(); print("Sample 1 loaded.")
        elif choice=="8": load_sample2(grid); grid.print(); print("Sample 2 loaded.")
        elif choice=="9": grid.print()
        elif choice=="0": print("Bye!"); break
        else: print("Invalid choice.")

if __name__=="__main__":
    main()