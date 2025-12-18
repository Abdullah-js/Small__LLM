"""
NOPAINNOGAIN - Environment Module
"""

from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple, Any

from config import settings
from modules.resources import Plant, Water, Mineral, create_resource


class Cell:
    """Single grid cell containing resources and hazards."""
    
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.resources: Dict[str, Any] = {}
        self.hazards: float = 0.0
        self.pollution: float = 0.0
        self.agents: List = []
    
    def add_resource(self, resource_type: str) -> None:
        """Add a resource to this cell."""
        self.resources[resource_type] = create_resource(resource_type, (self.x, self.y))
    
    def has_resource(self, resource_type: str) -> bool:
        """Check if cell has specified resource with quantity > 0."""
        res = self.resources.get(resource_type)
        return res is not None and res.quantity > 0
    
    def get_total_resources(self) -> float:
        """Get sum of all resource quantities."""
        return sum(r.quantity for r in self.resources.values())
    
    def update(self) -> None:
        """Update cell state (regeneration, decay)."""
        for resource in self.resources.values():
            resource.regenerate()
        
        self.hazards = min(10, self.hazards + self.pollution * 0.1)
        self.pollution = max(0, self.pollution - settings.ENV.POLLUTION_DECAY_RATE)


class Environment:
    """Grid-based world with resources, hazards, and weather."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.weather = "normal"
        self.step_count = 0
        
        self.grid: List[List[Dict]] = [
            [
                {"resources": {}, "hazards": 0.0, "pollution": 0.0, "agents": []}
                for _ in range(width)
            ]
            for _ in range(height)
        ]
        
        self._generate_terrain()
    
    def _generate_terrain(self) -> None:
        """Initialize the world with resources and hazards."""
        all_positions = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
        ]
        
        num_resources = min(settings.INITIAL_RESOURCE_COUNT, len(all_positions))
        resource_positions = random.sample(all_positions, k=num_resources)
        
        for x, y in resource_positions:
            resource_type = random.choices(
                ["plant", "water", "mineral"],
                weights=[0.5, 0.3, 0.2]
            )[0]
            
            if resource_type == "plant":
                self.grid[y][x]["resources"]["plant"] = Plant(position=(x, y))
            elif resource_type == "water":
                self.grid[y][x]["resources"]["water"] = Water(position=(x, y))
            elif resource_type == "mineral":
                self.grid[y][x]["resources"]["mineral"] = Mineral(position=(x, y))
        
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x]["hazards"] = random.uniform(0, settings.HAZARD_BASE_LEVEL)
    
    def update_state(self) -> None:
        """Update all cells and world state."""
        self.step_count += 1
        
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                
                for resource in cell["resources"].values():
                    resource.regenerate()
                
                cell["hazards"] = min(
                    settings.MAX_HAZARD_VALUE,
                    cell["hazards"] + cell["pollution"] * 0.1
                )
                cell["pollution"] = max(0, cell["pollution"] - 0.5)
        
        if self.step_count % 100 == 0:
            self._spawn_new_resources()
    
    def _spawn_new_resources(self) -> None:
        """Occasionally spawn new resources."""
        spawn_count = random.randint(5, 15)
        for _ in range(spawn_count):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            resource_type = random.choice(["plant", "water", "mineral"])
            
            if resource_type not in self.grid[y][x]["resources"]:
                if resource_type == "plant":
                    self.grid[y][x]["resources"]["plant"] = Plant(position=(x, y))
                elif resource_type == "water":
                    self.grid[y][x]["resources"]["water"] = Water(position=(x, y))
                elif resource_type == "mineral":
                    self.grid[y][x]["resources"]["mineral"] = Mineral(position=(x, y))
    
    def get_cell(self, x: int, y: int) -> Dict:
        """Get cell at coordinates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return {"resources": {}, "hazards": 0, "pollution": 0, "agents": []}
    
    def deplete_resource(self, x: int, y: int, resource_type: str, amount: float) -> float:
        """Deplete resource and return actual amount consumed."""
        cell = self.grid[y][x]
        if resource_type in cell["resources"]:
            resource = cell["resources"][resource_type]
            actual = resource.consume(amount)
            cell["pollution"] += 0.5
            return actual
        return 0.0
    
    def get_resource(self, x: int, y: int, resource_type: str) -> Optional[float]:
        """Get quantity of specific resource at location."""
        cell = self.grid[y][x]
        if resource_type in cell["resources"]:
            return cell["resources"][resource_type].quantity
        return None
    
    def add_pollution(self, x: int, y: int, amount: float) -> None:
        """Add pollution to a cell."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x]["pollution"] += amount
    
    def get_neighbors(self, x: int, y: int, radius: int = 1) -> List[Tuple[int, int]]:
        """Get neighboring cell coordinates."""
        neighbors = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append((nx, ny))
        return neighbors
    
    def find_nearest_resource(
        self, x: int, y: int, resource_type: str, max_radius: int = 5
    ) -> Optional[Tuple[int, int]]:
        """Find nearest cell with specified resource."""
        for radius in range(1, max_radius + 1):
            for nx, ny in self.get_neighbors(x, y, radius):
                cell = self.grid[ny][nx]
                if resource_type in cell["resources"]:
                    if cell["resources"][resource_type].quantity > 0:
                        return (nx, ny)
        return None
