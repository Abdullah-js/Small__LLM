"""
NOPAINNOGAIN - Creature System
Advanced creatures with physics, neural brains, and sensory systems
"""

from __future__ import annotations
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from enum import Enum, auto

from modules.physics import Vector2, PhysicsBody, flock, separation
from modules.brain import NeuralNetwork, CreatureBrain

if TYPE_CHECKING:
    from modules.world import World, FoodSource


class CreatureState(Enum):
    IDLE = auto()
    WANDERING = auto()
    FORAGING = auto()
    EATING = auto()
    DRINKING = auto()
    FLEEING = auto()
    HUNTING = auto()
    ATTACKING = auto()
    RESTING = auto()
    DEAD = auto()


@dataclass
class CreatureStats:
    """Runtime statistics for a creature."""
    food_eaten: float = 0.0
    water_drunk: float = 0.0
    distance_traveled: float = 0.0
    kills: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    children: int = 0
    times_fled: int = 0
    time_alive: float = 0.0


@dataclass
class Genome:
    """Genetic information that determines creature traits."""
    genes: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.genes:
            self.genes = [random.uniform(0, 1) for _ in range(16)]
    
    def __getitem__(self, idx: int) -> float:
        return self.genes[idx] if idx < len(self.genes) else 0.5
    
    def copy(self) -> Genome:
        return Genome(genes=self.genes.copy())
    
    def mutate(self, rate: float = 0.1, strength: float = 0.15) -> None:
        for i in range(len(self.genes)):
            if random.random() < rate:
                self.genes[i] += random.gauss(0, strength)
                self.genes[i] = max(0, min(1, self.genes[i]))
    
    @staticmethod
    def crossover(parent1: Genome, parent2: Genome) -> Genome:
        child_genes = []
        for i in range(len(parent1.genes)):
            if random.random() < 0.5:
                child_genes.append(parent1.genes[i])
            else:
                child_genes.append(parent2.genes[i])
        return Genome(genes=child_genes)


@dataclass
class Traits:
    """Phenotype expressed from genome."""
    max_speed: float = 4.0
    max_force: float = 0.3
    vision_range: float = 150.0
    vision_angle: float = math.pi * 0.8
    hearing_range: float = 100.0
    smell_range: float = 80.0
    
    attack_power: float = 20.0
    attack_range: float = 20.0
    defense: float = 0.1
    
    metabolism: float = 1.0
    max_stamina: float = 100.0
    stamina_regen: float = 0.5
    
    size: float = 10.0
    mass: float = 1.0
    
    reproduction_threshold: float = 70.0
    reproduction_cost: float = 40.0
    
    lifespan: float = 3000.0
    
    base_color: Tuple[int, int, int] = (255, 200, 100)
    
    @classmethod
    def from_genome(cls, genome: Genome, is_predator: bool = False) -> Traits:
        """Generate traits from genome."""
        base_speed = 3.0 + genome[0] * 3.0
        base_vision = 100.0 + genome[1] * 150.0
        base_attack = 10.0 + genome[2] * 30.0
        base_defense = genome[3] * 0.3
        base_metabolism = 0.7 + genome[4] * 0.8
        base_stamina = 70 + genome[5] * 60
        base_size = 8 + genome[6] * 8
        
        if is_predator:
            base_speed *= 1.2
            base_attack *= 1.5
            base_vision *= 1.3
            base_metabolism *= 1.2
            base_color = (
                180 + int(genome[7] * 75),
                20 + int(genome[8] * 40),
                20 + int(genome[9] * 40),
            )
        else:
            base_defense *= 1.3
            base_stamina *= 1.2
            base_color = (
                200 + int(genome[7] * 55),
                180 + int(genome[8] * 55),
                50 + int(genome[9] * 80),
            )
        
        return cls(
            max_speed=base_speed,
            max_force=0.2 + genome[10] * 0.3,
            vision_range=base_vision,
            vision_angle=math.pi * (0.5 + genome[11] * 0.5),
            hearing_range=60 + genome[12] * 80,
            smell_range=40 + genome[13] * 80,
            attack_power=base_attack,
            attack_range=15 + genome[6] * 10,
            defense=base_defense,
            metabolism=base_metabolism,
            max_stamina=base_stamina,
            stamina_regen=0.3 + genome[14] * 0.5,
            size=base_size,
            mass=0.5 + base_size * 0.1,
            reproduction_threshold=60 + genome[15] * 30,
            reproduction_cost=30 + genome[15] * 20,
            lifespan=2000 + genome[6] * 2000,
            base_color=base_color,
        )


class Creature:
    """A living creature in the simulation."""
    
    def __init__(
        self,
        position: Vector2,
        is_predator: bool = False,
        genome: Optional[Genome] = None,
        brain: Optional[NeuralNetwork] = None,
        generation: int = 0,
        name: str = None,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.name = name or f"{'Pred' if is_predator else 'Prey'}_{self.id}"
        self.is_predator = is_predator
        self.generation = generation
        
        self.genome = genome or Genome()
        self.traits = Traits.from_genome(self.genome, is_predator)
        
        input_size = 16
        hidden1 = 24
        hidden2 = 16
        output_size = 6
        self.brain = brain or NeuralNetwork(
            layer_sizes=[input_size, hidden1, hidden2, output_size]
        )
        
        self.body = PhysicsBody(
            position=position.copy(),
            max_speed=self.traits.max_speed,
            max_force=self.traits.max_force,
            mass=self.traits.mass,
            friction=0.95,
        )
        
        self.health = 100.0
        self.max_health = 100.0
        self.energy = 80.0
        self.max_energy = 100.0
        self.stamina = self.traits.max_stamina
        self.hydration = 80.0
        self.max_hydration = 100.0
        
        self.age = 0.0
        self.alive = True
        self.state = CreatureState.WANDERING
        
        self.stats = CreatureStats()
        
        self.visible_creatures: List[Creature] = []
        self.nearby_food: Optional[FoodSource] = None
        self.nearest_threat: Optional[Creature] = None
        self.nearest_prey: Optional[Creature] = None
        self.target: Optional[Creature] = None
        
        self.last_scent_time = 0.0
        self.attack_cooldown = 0.0
        
        self.color = self.traits.base_color
        self.trail: List[Vector2] = []
        self.max_trail_length = 20
    
    @property
    def position(self) -> Vector2:
        return self.body.position
    
    @position.setter
    def position(self, value: Vector2) -> None:
        self.body.position = value
    
    @property
    def velocity(self) -> Vector2:
        return self.body.velocity
    
    @property
    def heading(self) -> float:
        return self.body.heading
    
    @property
    def speed(self) -> float:
        return self.body.velocity.magnitude
    
    def can_see(self, other: Creature, world: World) -> bool:
        """Check if this creature can see another."""
        if not other.alive:
            return False
        
        dist = self.position.distance_to(other.position)
        effective_range = self.traits.vision_range * world.get_vision_modifier()
        
        if dist > effective_range:
            return False
        
        angle_to_other = self.position.angle_to(other.position)
        angle_diff = abs(angle_to_other - self.heading)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
        
        if angle_diff > self.traits.vision_angle / 2:
            return False
        
        return True
    
    def can_hear(self, other: Creature, world: World) -> bool:
        """Check if this creature can hear another."""
        if not other.alive:
            return False
        
        dist = self.position.distance_to(other.position)
        effective_range = self.traits.hearing_range * world.get_hearing_modifier()
        
        noise_level = other.speed / other.traits.max_speed
        effective_range *= noise_level
        
        return dist < effective_range
    
    def sense_environment(self, world: World, all_creatures: List[Creature]) -> None:
        """Update sensory information."""
        self.visible_creatures = []
        self.nearest_threat = None
        self.nearest_prey = None
        
        nearest_threat_dist = float('inf')
        nearest_prey_dist = float('inf')
        
        for other in all_creatures:
            if other.id == self.id or not other.alive:
                continue
            
            can_detect = self.can_see(other, world) or self.can_hear(other, world)
            
            if can_detect:
                self.visible_creatures.append(other)
                dist = self.position.distance_to(other.position)
                
                if self.is_predator:
                    if not other.is_predator and dist < nearest_prey_dist:
                        nearest_prey_dist = dist
                        self.nearest_prey = other
                else:
                    if other.is_predator and dist < nearest_threat_dist:
                        nearest_threat_dist = dist
                        self.nearest_threat = other
        
        effective_smell = self.traits.smell_range
        if world.wind_strength > 0:
            effective_smell *= (1 + world.wind_strength * 0.5)
        
        self.nearby_food = world.find_nearest_food(self.position, effective_smell * 2)
    
    def think(self, world: World) -> None:
        """Use brain to decide actions."""
        food_dist = 1.0
        food_angle = 0.0
        if self.nearby_food:
            food_dist = min(1.0, self.position.distance_to(self.nearby_food.position) / 200.0)
            food_angle = self.position.angle_to(self.nearby_food.position) - self.heading
        
        threat_dist = 1.0
        threat_angle = 0.0
        if self.nearest_threat:
            threat_dist = min(1.0, self.position.distance_to(self.nearest_threat.position) / 200.0)
            threat_angle = self.position.angle_to(self.nearest_threat.position) - self.heading
        
        prey_dist = 1.0
        prey_angle = 0.0
        if self.nearest_prey:
            prey_dist = min(1.0, self.position.distance_to(self.nearest_prey.position) / 200.0)
            prey_angle = self.position.angle_to(self.nearest_prey.position) - self.heading
        
        ally_dist = 1.0
        ally_angle = 0.0
        allies = [c for c in self.visible_creatures if c.is_predator == self.is_predator]
        if allies:
            nearest_ally = min(allies, key=lambda c: self.position.distance_to(c.position))
            ally_dist = min(1.0, self.position.distance_to(nearest_ally.position) / 150.0)
            ally_angle = self.position.angle_to(nearest_ally.position) - self.heading
        
        inputs = [
            self.health / self.max_health,
            self.energy / self.max_energy,
            self.stamina / self.traits.max_stamina,
            self.hydration / self.max_hydration,
            food_dist,
            food_angle / math.pi,
            threat_dist,
            threat_angle / math.pi,
            prey_dist,
            prey_angle / math.pi,
            ally_dist,
            ally_angle / math.pi,
            1.0 if world.is_day else 0.0,
            world.ambient_light,
            min(1.0, len([c for c in self.visible_creatures if c.is_predator]) / 3.0),
            min(1.0, len(allies) / 5.0),
        ]
        
        outputs = self.brain.forward(inputs)
        
        move_desire = outputs[0]
        turn_desire = (outputs[1] - 0.5) * 2
        
        if self.is_predator:
            hunt_desire = outputs[2]
            attack_desire = outputs[3]
            rest_desire = outputs[4]
            
            if self.nearest_prey and hunt_desire > 0.5:
                self.state = CreatureState.HUNTING
                self.target = self.nearest_prey
            elif attack_desire > 0.6 and self.target and self.attack_cooldown <= 0:
                self.state = CreatureState.ATTACKING
            elif rest_desire > 0.7 and self.stamina < 30:
                self.state = CreatureState.RESTING
            elif move_desire > 0.3:
                self.state = CreatureState.WANDERING
        else:
            eat_desire = outputs[2]
            flee_desire = outputs[3]
            rest_desire = outputs[4]
            
            if self.nearest_threat and flee_desire > 0.4:
                self.state = CreatureState.FLEEING
                self.stats.times_fled += 1
            elif self.nearby_food and eat_desire > 0.5 and self.energy < 70:
                self.state = CreatureState.FORAGING
            elif rest_desire > 0.7 and self.stamina < 30:
                self.state = CreatureState.RESTING
            elif move_desire > 0.3:
                self.state = CreatureState.WANDERING
        
        self._apply_turn(turn_desire)
    
    def _apply_turn(self, turn_amount: float) -> None:
        """Apply turning force."""
        turn_force = turn_amount * 0.1
        perpendicular = Vector2.from_angle(self.heading + math.pi / 2)
        self.body.apply_force(perpendicular * turn_force)
    
    def act(self, world: World, all_creatures: List[Creature]) -> None:
        """Execute current state behavior."""
        if not self.alive:
            return
        
        if self.state == CreatureState.WANDERING:
            self._wander(world)
        elif self.state == CreatureState.FORAGING:
            self._forage(world)
        elif self.state == CreatureState.EATING:
            self._eat(world)
        elif self.state == CreatureState.FLEEING:
            self._flee(world, all_creatures)
        elif self.state == CreatureState.HUNTING:
            self._hunt(world)
        elif self.state == CreatureState.ATTACKING:
            self._attack(all_creatures)
        elif self.state == CreatureState.RESTING:
            self._rest()
        
        if not self.is_predator:
            allies = [c.body for c in self.visible_creatures 
                     if not c.is_predator and c.id != self.id]
            if allies:
                flock_force = flock(self.body, allies, sep_weight=2.0, align_weight=0.8, coh_weight=0.5)
                self.body.apply_force(flock_force * 0.3)
    
    def _wander(self, world: World) -> None:
        """Random wandering behavior."""
        wander_force = self.body.wander(wander_radius=30, wander_distance=60, wander_jitter=0.5)
        self.body.apply_force(wander_force)
        self._consume_stamina(0.3)
    
    def _forage(self, world: World) -> None:
        """Move toward food."""
        if self.nearby_food:
            dist = self.position.distance_to(self.nearby_food.position)
            if dist < self.traits.size + self.nearby_food.radius:
                self.state = CreatureState.EATING
            else:
                seek_force = self.body.seek(self.nearby_food.position)
                self.body.apply_force(seek_force)
                self._consume_stamina(0.5)
        else:
            self.state = CreatureState.WANDERING
    
    def _eat(self, world: World) -> None:
        """Consume food."""
        if self.nearby_food and not self.nearby_food.is_depleted:
            eaten = self.nearby_food.consume(2.0)
            self.energy = min(self.max_energy, self.energy + eaten)
            self.stats.food_eaten += eaten
            
            if self.energy >= self.max_energy * 0.9 or self.nearby_food.is_depleted:
                self.state = CreatureState.WANDERING
        else:
            self.state = CreatureState.WANDERING
    
    def _flee(self, world: World, all_creatures: List[Creature]) -> None:
        """Flee from threats."""
        if self.nearest_threat:
            evade_force = self.body.evade(self.nearest_threat.body, prediction_time=1.5)
            self.body.apply_force(evade_force * 1.5)
            
            speed_boost = 1.3 if self.stamina > 20 else 1.0
            self.body.max_speed = self.traits.max_speed * speed_boost
            
            self._consume_stamina(1.5)
            
            world.add_scent(self.position, "prey", 0.8)
        else:
            self.state = CreatureState.WANDERING
            self.body.max_speed = self.traits.max_speed
    
    def _hunt(self, world: World) -> None:
        """Hunt prey."""
        if self.target and self.target.alive:
            dist = self.position.distance_to(self.target.position)
            
            if dist < self.traits.attack_range:
                self.state = CreatureState.ATTACKING
            else:
                pursue_force = self.body.pursue(self.target.body, prediction_time=1.0)
                self.body.apply_force(pursue_force * 1.2)
                self._consume_stamina(1.0)
                
                world.add_scent(self.position, "predator", 1.0)
        else:
            self.target = None
            self.state = CreatureState.WANDERING
    
    def _attack(self, all_creatures: List[Creature]) -> None:
        """Attack target."""
        if self.target and self.target.alive and self.attack_cooldown <= 0:
            dist = self.position.distance_to(self.target.position)
            
            if dist < self.traits.attack_range:
                damage = self.traits.attack_power * (0.8 + random.random() * 0.4)
                actual_damage = self.target.take_damage(damage, self)
                
                self.stats.damage_dealt += actual_damage
                self._consume_stamina(5.0)
                self.attack_cooldown = 30
                
                if not self.target.alive:
                    self.energy = min(self.max_energy, self.energy + 40)
                    self.stats.kills += 1
                    self.target = None
                    self.state = CreatureState.WANDERING
        else:
            self.state = CreatureState.HUNTING if self.target else CreatureState.WANDERING
    
    def _rest(self) -> None:
        """Rest to recover stamina."""
        self.stamina = min(self.traits.max_stamina, self.stamina + self.traits.stamina_regen * 3)
        self.body.velocity = self.body.velocity * 0.9
        
        if self.stamina > self.traits.max_stamina * 0.8:
            self.state = CreatureState.WANDERING
    
    def _consume_stamina(self, amount: float) -> None:
        """Consume stamina for actions."""
        self.stamina = max(0, self.stamina - amount * self.traits.metabolism)
    
    def take_damage(self, amount: float, attacker: Creature = None) -> float:
        """Take damage and return actual damage dealt."""
        actual_damage = amount * (1 - self.traits.defense)
        self.health -= actual_damage
        self.stats.damage_taken += actual_damage
        
        if self.health <= 0:
            self.die()
        
        return actual_damage
    
    def die(self) -> None:
        """Handle creature death."""
        self.alive = False
        self.state = CreatureState.DEAD
        self.health = 0
    
    def update(self, world: World, dt: float = 1.0) -> None:
        """Update creature state."""
        if not self.alive:
            return
        
        self.body.update(dt)
        self.body.bounce(world.width, world.height, damping=0.5)
        
        self.trail.append(self.position.copy())
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)
        
        self.stats.distance_traveled += self.speed * dt
        
        energy_cost = 0.02 * self.traits.metabolism
        energy_cost += self.speed / self.traits.max_speed * 0.05
        self.energy = max(0, self.energy - energy_cost * dt)
        
        if self.stamina < self.traits.max_stamina:
            self.stamina = min(self.traits.max_stamina, 
                             self.stamina + self.traits.stamina_regen * dt)
        
        if self.attack_cooldown > 0:
            self.attack_cooldown -= dt
        
        self.age += dt
        self.stats.time_alive = self.age
        
        if self.energy <= 0:
            self.health -= 0.5 * dt
        
        if self.age > self.traits.lifespan:
            self.health -= 0.2 * dt
        
        if self.health <= 0:
            self.die()
    
    def can_reproduce(self) -> bool:
        """Check if creature can reproduce."""
        return (
            self.alive and
            self.energy >= self.traits.reproduction_threshold and
            self.health > 50 and
            self.age > 200
        )
    
    def reproduce(self, partner: Creature) -> Optional[Creature]:
        """Reproduce with partner."""
        if not self.can_reproduce() or not partner.can_reproduce():
            return None
        
        child_genome = Genome.crossover(self.genome, partner.genome)
        child_genome.mutate(rate=0.15, strength=0.2)
        
        child_brain = NeuralNetwork.crossover(self.brain, partner.brain)
        child_brain.mutate(mutation_rate=0.1, mutation_strength=0.2)
        
        offset = Vector2.random_unit() * 30
        child_pos = self.position + offset
        
        child = Creature(
            position=child_pos,
            is_predator=self.is_predator,
            genome=child_genome,
            brain=child_brain,
            generation=max(self.generation, partner.generation) + 1,
        )
        
        self.energy -= self.traits.reproduction_cost
        partner.energy -= partner.traits.reproduction_cost * 0.5
        self.stats.children += 1
        partner.stats.children += 1
        
        return child
    
    def get_fitness(self) -> float:
        """Calculate fitness score."""
        fitness = 0.0
        fitness += self.stats.time_alive * 0.01
        fitness += self.stats.food_eaten * 0.5
        fitness += self.stats.children * 50
        fitness += self.stats.kills * 30
        fitness -= self.stats.damage_taken * 0.1
        return max(0, fitness)
