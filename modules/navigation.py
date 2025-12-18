"""
NOPAINNOGAIN - Navigation and Pathfinding
"""

from __future__ import annotations
import heapq
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.environment import Environment


class Pathfinder:
    """A* pathfinding implementation."""
    
    def __init__(self, environment: Environment):
        self.env = environment
    
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Find shortest path using A* algorithm."""
        if start == goal:
            return [start]
        
        open_set = [(self._heuristic(start, goal), 0, start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            for neighbor in self._get_neighbors(current):
                x, y = neighbor
                move_cost = 1 + self.env.grid[y][x].get("hazards", 0) * 0.5
                tentative_g = current_g + move_cost
                
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        return None
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells."""
        x, y = pos
        neighbors = []
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                neighbors.append((nx, ny))
        
        return neighbors
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
    
    def find_nearest_resource(
        self,
        start: Tuple[int, int],
        resource_type: str,
        max_distance: int = 10
    ) -> Optional[Tuple[int, int]]:
        """Find nearest cell with specified resource."""
        visited = set()
        queue = [(0, start)]
        
        while queue:
            dist, pos = heapq.heappop(queue)
            
            if pos in visited or dist > max_distance:
                continue
            visited.add(pos)
            
            x, y = pos
            cell = self.env.grid[y][x]
            if resource_type in cell.get("resources", {}):
                if cell["resources"][resource_type].quantity > 0:
                    return pos
            
            for neighbor in self._get_neighbors(pos):
                if neighbor not in visited:
                    heapq.heappush(queue, (dist + 1, neighbor))
        
        return None
