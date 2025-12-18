"""
NOPAINNOGAIN - Agent Module
Enhanced with Predator and Prey behaviors
"""

from __future__ import annotations
import random
import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from config import settings, constants

if TYPE_CHECKING:
    from modules.environment import Environment


@dataclass
class Genome:
    """Genetic information container with 10 genes."""
    genes: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.genes:
            self.genes = [random.uniform(0, 10) for _ in range(settings.GENOME_LENGTH)]
    
    def __getitem__(self, idx: int) -> float:
        return self.genes[idx] if idx < len(self.genes) else 5.0
    
    def __len__(self) -> int:
        return len(self.genes)
    
    def copy(self) -> Genome:
        return Genome(genes=self.genes.copy())


@dataclass
class Traits:
    """Phenotype expressed from genome."""
    speed: float = 0.5
    intelligence: float = 0.5
    aggression: float = 0.5
    cooperation: float = 0.5
    pollution_tolerance: float = 0.5
    reproduction_rate: float = 0.1
    lifespan: int = 100
    metabolism: float = 1.0
    vision: float = 0.5
    stealth: float = 0.5
    
    @classmethod
    def from_genome(cls, genome: Genome, is_predator: bool = False) -> Traits:
        """Express phenotype from genetic code."""
        base_traits = cls(
            speed=genome[0] / 10.0,
            intelligence=genome[1] / 10.0,
            aggression=genome[2] / 10.0,
            cooperation=genome[3] / 10.0,
            pollution_tolerance=genome[4] / 10.0,
            reproduction_rate=max(0.02, min(0.5, genome[5] / 50.0)),
            lifespan=int(80 + genome[6] * 15),
            metabolism=max(0.5, min(2.0, genome[7] / 10.0)),
            vision=genome[8] / 10.0 if len(genome.genes) > 8 else 0.5,
            stealth=genome[9] / 10.0 if len(genome.genes) > 9 else 0.5,
        )
        
        if is_predator:
            base_traits.aggression = min(1.0, base_traits.aggression + 0.3)
            base_traits.speed = min(1.0, base_traits.speed + 0.1)
            base_traits.metabolism *= settings.PREDATOR.METABOLISM_MULTIPLIER
        
        return base_traits
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "speed": self.speed,
            "intelligence": self.intelligence,
            "aggression": self.aggression,
            "cooperation": self.cooperation,
            "pollution_tolerance": self.pollution_tolerance,
            "reproduction_rate": self.reproduction_rate,
            "lifespan": self.lifespan,
            "metabolism": self.metabolism,
            "vision": self.vision,
            "stealth": self.stealth,
        }


class Agent:
    """Autonomous agent with genome, behavior, and lifecycle."""
    
    def __init__(
        self,
        name: str,
        species_id: str = "prey",
        position: Optional[Tuple[int, int]] = None,
        genome: Optional[Genome] = None,
        generation: int = 0,
        health: Optional[float] = None,
        energy: Optional[float] = None,
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.species_id = species_id
        self.generation = generation
        self.is_predator = species_id == "predator"
        
        self.position = position or (
            random.randint(0, settings.GRID_WIDTH - 1),
            random.randint(0, settings.GRID_HEIGHT - 1)
        )
        
        self.genome = genome or Genome()
        self.traits = Traits.from_genome(self.genome, self.is_predator)
        
        self.health = health if health is not None else float(settings.MAX_HEALTH)
        self.energy = energy if energy is not None else float(settings.MAX_ENERGY)
        
        self.alive = True
        self.age = 0
        self.state = constants.AgentState.FORAGING
        
        self.color = self._generate_color()
        
        self.total_food_eaten = 0.0
        self.total_distance_traveled = 0.0
        self.children_produced = 0
        self.kills = 0
        self.times_fled = 0
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        
        self.target: Optional[Agent] = None
        self.last_action = "none"
        self.last_action_success = False
        self.was_attacked = False
        self.nearby_threat = False
        self.nearby_prey_count = 0
    
    def _generate_color(self) -> Tuple[int, int, int]:
        """Generate consistent color based on species."""
        if self.species_id == "prey":
            return settings.COLORS.get("prey", (255, 215, 0))
        if self.species_id == "predator":
            return settings.COLORS.get("predator", (220, 20, 60))
        
        random.seed(hash(self.species_id))
        color = (
            random.randint(60, 255),
            random.randint(60, 255),
            random.randint(60, 255)
        )
        random.seed()
        return color
    
    def get_vision_range(self) -> int:
        """Get effective vision range based on traits and species."""
        base = settings.PREDATOR_VISION_RANGE if self.is_predator else settings.PREY_VISION_RANGE
        return int(base * (0.7 + 0.6 * self.traits.vision))
    
    def can_see(self, other: Agent) -> bool:
        """Check if this agent can see another agent."""
        if not other.alive:
            return False
        dist = self.distance_to(other.position)
        stealth_modifier = 1.0 - (other.traits.stealth * 0.5)
        return dist <= self.get_vision_range() * stealth_modifier
    
    def distance_to(self, pos: Tuple[int, int]) -> float:
        """Calculate Manhattan distance to a position."""
        return abs(self.position[0] - pos[0]) + abs(self.position[1] - pos[1])
    
    def find_nearest_threat(self, all_agents: List[Agent]) -> Optional[Agent]:
        """Find the nearest predator (for prey)."""
        if self.is_predator:
            return None
        
        threats = []
        for agent in all_agents:
            if agent.is_predator and agent.alive and self.can_see(agent):
                threats.append((self.distance_to(agent.position), agent))
        
        if threats:
            threats.sort(key=lambda x: x[0])
            return threats[0][1]
        return None
    
    def find_nearest_prey(self, all_agents: List[Agent]) -> Optional[Agent]:
        """Find the nearest prey (for predators)."""
        if not self.is_predator:
            return None
        
        prey_list = []
        for agent in all_agents:
            if not agent.is_predator and agent.alive and self.can_see(agent):
                prey_list.append((self.distance_to(agent.position), agent))
        
        if prey_list:
            prey_list.sort(key=lambda x: x[0])
            return prey_list[0][1]
        return None
    
    def count_nearby_agents(self, all_agents: List[Agent]) -> Tuple[int, int]:
        """Count nearby predators and prey within vision."""
        predators = 0
        prey = 0
        for agent in all_agents:
            if agent.id == self.id or not agent.alive:
                continue
            if self.can_see(agent):
                if agent.is_predator:
                    predators += 1
                else:
                    prey += 1
        return predators, prey
    
    def perform_action(self, action: str, environment: Environment, all_agents: List[Agent] = None) -> bool:
        """Execute an action and return success status."""
        if not self.alive:
            return False
        
        self.was_attacked = False
        base_cost = settings.AGENT.ENERGY_LOSS_PER_STEP * self.traits.metabolism
        self.energy -= base_cost
        
        success = False
        
        if action == "move":
            success = self._move(environment)
        elif action == "eat":
            success = self._eat(environment)
        elif action == "drink":
            success = self._drink(environment)
        elif action == "rest":
            success = self._rest()
        elif action == "flee":
            success = self._flee(environment, all_agents or [])
        elif action == "hunt":
            success = self._hunt(environment, all_agents or [])
        elif action == "attack":
            success = self._attack(all_agents or [])
        
        self.last_action = action
        self.last_action_success = success
        self._check_status()
        return success
    
    def _move(self, environment: Environment) -> bool:
        """Move in a random direction."""
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = random.choice(directions)
        
        x, y = self.position
        new_x = max(0, min(environment.width - 1, x + dx))
        new_y = max(0, min(environment.height - 1, y + dy))
        
        if (new_x, new_y) != self.position:
            self.position = (new_x, new_y)
            self.energy -= settings.ENERGY_LOSS_PER_MOVE * self.traits.speed
            self.total_distance_traveled += 1
            return True
        return False
    
    def _move_toward(self, target_pos: Tuple[int, int], environment: Environment) -> bool:
        """Move toward a target position."""
        x, y = self.position
        tx, ty = target_pos
        
        dx = 0 if tx == x else (1 if tx > x else -1)
        dy = 0 if ty == y else (1 if ty > y else -1)
        
        if random.random() < 0.5:
            dx, dy = dy, dx
        
        new_x = max(0, min(environment.width - 1, x + dx))
        new_y = max(0, min(environment.height - 1, y + dy))
        
        if (new_x, new_y) != self.position:
            self.position = (new_x, new_y)
            self.total_distance_traveled += 1
            return True
        return False
    
    def _move_away_from(self, threat_pos: Tuple[int, int], environment: Environment) -> bool:
        """Move away from a threat."""
        x, y = self.position
        tx, ty = threat_pos
        
        dx = 0 if tx == x else (-1 if tx > x else 1)
        dy = 0 if ty == y else (-1 if ty > y else 1)
        
        if random.random() < 0.5 and dx != 0 and dy != 0:
            dx, dy = dy, dx
        
        new_x = max(0, min(environment.width - 1, x + dx))
        new_y = max(0, min(environment.height - 1, y + dy))
        
        if (new_x, new_y) != self.position:
            self.position = (new_x, new_y)
            self.total_distance_traveled += 1
            return True
        return False
    
    def _eat(self, environment: Environment) -> bool:
        """Consume plant resource at current location."""
        x, y = self.position
        cell = environment.grid[y][x]
        
        plant = cell["resources"].get("plant")
        if plant and plant.quantity > 0:
            amount = min(self.traits.intelligence * 5 + 5, plant.quantity)
            consumed = plant.consume(amount)
            self.energy = min(settings.MAX_ENERGY, self.energy + consumed)
            self.total_food_eaten += consumed
            return True
        return False
    
    def _drink(self, environment: Environment) -> bool:
        """Consume water resource at current location."""
        x, y = self.position
        cell = environment.grid[y][x]
        
        water = cell["resources"].get("water")
        if water and water.quantity > 0:
            consumed = water.consume(5)
            self.energy = min(settings.MAX_ENERGY, self.energy + consumed)
            return True
        return False
    
    def _rest(self) -> bool:
        """Rest to recover energy."""
        recovery = 2.0 * (1 - self.traits.metabolism / 2)
        self.energy = min(settings.MAX_ENERGY, self.energy + recovery)
        self.state = constants.AgentState.RESTING
        return True
    
    def _flee(self, environment: Environment, all_agents: List[Agent]) -> bool:
        """Flee from the nearest predator."""
        if self.is_predator:
            return False
        
        threat = self.find_nearest_threat(all_agents)
        if not threat:
            return self._move(environment)
        
        self.state = constants.AgentState.FLEEING
        self.energy -= settings.FLEE_ENERGY_COST
        
        flee_chance = settings.PREY.FLEE_SUCCESS_BASE + (self.traits.speed * 0.3)
        if random.random() < flee_chance:
            self._move_away_from(threat.position, environment)
            if self.traits.speed > 0.5:
                self._move_away_from(threat.position, environment)
            self.times_fled += 1
            return True
        return self._move_away_from(threat.position, environment)
    
    def _hunt(self, environment: Environment, all_agents: List[Agent]) -> bool:
        """Hunt the nearest prey (predator only)."""
        if not self.is_predator:
            return False
        
        prey = self.find_nearest_prey(all_agents)
        if not prey:
            return self._move(environment)
        
        self.state = constants.AgentState.HUNTING
        self.target = prey
        self.energy -= settings.HUNT_ENERGY_COST
        
        dist = self.distance_to(prey.position)
        if dist <= settings.PREDATOR.ATTACK_RANGE:
            return self._attack(all_agents)
        
        moved = self._move_toward(prey.position, environment)
        if self.traits.speed > 0.6:
            self._move_toward(prey.position, environment)
        
        return moved
    
    def _attack(self, all_agents: List[Agent]) -> bool:
        """Attack nearby prey (predator only)."""
        if not self.is_predator:
            return False
        
        self.energy -= settings.ATTACK_ENERGY_COST
        self.state = constants.AgentState.ATTACKING
        
        for agent in all_agents:
            if agent.id == self.id or agent.is_predator or not agent.alive:
                continue
            
            if self.distance_to(agent.position) <= settings.PREDATOR.ATTACK_RANGE:
                attack_power = settings.ATTACK_DAMAGE * (0.7 + 0.6 * self.traits.aggression)
                agent.take_damage(attack_power, self)
                self.damage_dealt += attack_power
                
                if not agent.alive:
                    self.energy = min(settings.MAX_ENERGY, self.energy + settings.MEAT_ENERGY_GAIN)
                    self.kills += 1
                    self.total_food_eaten += settings.MEAT_ENERGY_GAIN
                    return True
                return True
        
        return False
    
    def take_damage(self, amount: float, attacker: Agent = None) -> None:
        """Receive damage from an attack."""
        self.health -= amount
        self.damage_taken += amount
        self.was_attacked = True
        self.state = constants.AgentState.FLEEING
        
        if self.health <= 0:
            self.health = 0
            self.alive = False
            self.state = constants.AgentState.DEAD
    
    def _check_status(self) -> None:
        """Update alive status based on health, energy, and age."""
        if self.energy <= settings.MIN_ENERGY:
            self.alive = False
            self.state = constants.AgentState.DEAD
        elif self.health <= 0:
            self.alive = False
            self.state = constants.AgentState.DEAD
        elif self.age >= self.traits.lifespan:
            self.alive = False
            self.state = constants.AgentState.DEAD
    
    def can_reproduce(self) -> bool:
        """Check if agent has enough energy to reproduce."""
        return (
            self.alive and
            self.energy >= settings.REPRODUCTION_ENERGY_THRESHOLD
        )
    
    def step(self) -> None:
        """Age the agent by one step."""
        self.age += 1
        self._check_status()
    
    def get_state_features(self, all_agents: List[Agent] = None) -> Dict:
        """Get state features for decision making."""
        predator_count, prey_count = 0, 0
        if all_agents:
            predator_count, prey_count = self.count_nearby_agents(all_agents)
        
        self.nearby_threat = predator_count > 0
        self.nearby_prey_count = prey_count
        
        return {
            "health_bin": min(3, int(self.health / 25)),
            "energy_bin": min(3, int(self.energy / 25)),
            "nearby_predators": min(3, predator_count),
            "nearby_prey": min(3, prey_count),
            "is_predator": 1 if self.is_predator else 0,
            "age_bin": min(3, int(self.age / (self.traits.lifespan / 4))),
        }
    
    def __repr__(self) -> str:
        status = "alive" if self.alive else "dead"
        species = "🦁" if self.is_predator else "🐰"
        return f"Agent({species} {self.name}, {status}, H={self.health:.0f}, E={self.energy:.0f})"
