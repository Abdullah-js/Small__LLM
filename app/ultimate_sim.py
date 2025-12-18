"""
🧬 NOPAINNOGAIN - ULTIMATE EVOLUTIONARY ECOSYSTEM v2.0 🧬
======================================================
The most advanced AI ecosystem simulation featuring:

🧠 NEURAL EVOLUTION
- Each creature has a unique neural network brain
- Brains evolve through crossover and mutation
- Fitness-based natural selection

🎯 PHYSICS ENGINE  
- Continuous movement (no grid!)
- Velocity, acceleration, forces
- Realistic collision and bouncing

👀 SENSORY SYSTEM
- Vision cones with distance/angle
- Hearing based on movement noise
- Smell trails for tracking

🦁 PREDATOR-PREY DYNAMICS
- Predators hunt in coordinated packs
- Prey flock together for safety
- Dynamic population balance

🌍 LIVING WORLD
- Day/night cycle affects behavior
- Weather systems (rain, fog, storms)
- Resource regeneration and depletion

📊 ADVANCED DATA LOGGING
- Agent-level CSV exports for ML analysis
- Species-level aggregate statistics
- Real-time performance metrics
- Compatible with analyze_simulation.py v3.0

🎮 NEW IN v2.0
- Pack hunting AI for predators
- Flocking behavior for prey
- Spatial memory system
- Dynamic difficulty adjustment
- Enhanced visual effects
- Real-time statistics dashboard
"""

from __future__ import annotations
import sys
import os
import random
import math
import json
import csv
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from datetime import datetime
from pathlib import Path

import pygame
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.physics import Vector2
from modules.brain import NeuralNetwork


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SimConfig:
    """All simulation parameters in one place."""
    # World
    width: int = 1400
    height: int = 900
    
    # Population
    initial_prey: int = 80
    initial_predators: int = 12
    min_prey: int = 15
    min_predators: int = 4
    max_prey: int = 250
    max_predators: int = 40
    
    # Prey settings
    prey_max_speed: float = 3.5
    prey_vision: float = 120.0
    prey_reproduction_rate: float = 0.08
    prey_reproduction_energy: float = 65.0
    prey_energy_drain: float = 0.015
    prey_flee_boost: float = 1.4
    prey_flock_weight: float = 0.8  # NEW: How strongly prey flock together
    
    # Predator settings  
    predator_max_speed: float = 4.2
    predator_vision: float = 160.0
    predator_attack_damage: float = 35.0
    predator_attack_range: float = 25.0
    predator_kill_energy: float = 70.0
    predator_energy_drain: float = 0.025
    predator_reproduction_rate: float = 0.04
    predator_reproduction_energy: float = 75.0
    predator_can_eat_food: bool = True
    predator_pack_range: float = 150.0  # NEW: Pack coordination range
    predator_pack_bonus: float = 0.2    # NEW: Damage bonus per ally
    
    # World cycles
    day_length: int = 1000
    weather_change_rate: float = 0.002
    
    # Resources
    food_count: int = 150
    food_regen_rate: float = 0.003
    water_count: int = 25
    
    # Evolution
    mutation_rate: float = 0.15
    mutation_strength: float = 0.25
    crossover_rate: float = 0.7
    
    # Memory system
    memory_capacity: int = 20  # NEW: Remembered locations
    memory_decay: float = 0.99  # NEW: Memory strength decay
    
    # Dynamic difficulty
    ecosystem_target_ratio: float = 8.0  # NEW: Target prey:predator ratio
    difficulty_adjustment_rate: float = 0.01  # NEW: How fast to adjust
    
    # Display
    fps: int = 60
    show_trails: bool = False
    show_vision: bool = False
    show_stats: bool = True
    show_heatmap: bool = False  # NEW: Activity heatmap
    
    # Logging
    log_interval: int = 1  # NEW: Log every N steps
    log_dir: str = "data/logs"  # NEW: Log directory


class Weather(Enum):
    CLEAR = auto()
    CLOUDY = auto()
    RAIN = auto()
    STORM = auto()
    FOG = auto()


# =============================================================================
# DATA LOGGING SYSTEM
# =============================================================================

class SimulationLogger:
    """Comprehensive data logging for ML analysis."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Simulation logs (agent-level)
        self.sim_log_path = self.log_dir / 'simulation_logs.csv'
        self.species_log_path = self.log_dir / 'species_logs.csv'
        
        # Initialize CSV files
        self._init_csv_files()
        
        # Buffers for batch writing
        self.sim_buffer: List[Dict] = []
        self.species_buffer: List[Dict] = []
        self.buffer_size = 100
        
    def _init_csv_files(self) -> None:
        """Initialize CSV files with headers."""
        # Agent-level logs
        sim_headers = [
            'step', 'agent_id', 'agent_name', 'species_id', 'is_predator',
            'health', 'energy', 'position_x', 'position_y', 'generation',
            'age', 'speed', 'intelligence', 'aggression', 'vision',
            'action', 'reward', 'kills', 'times_fled'
        ]
        
        # Species-level logs
        species_headers = [
            'step', 'prey_population', 'predator_population', 'total_population',
            'total_kills', 'total_births', 'avg_prey_energy',
            'avg_predator_energy', 'avg_prey_fitness', 'avg_predator_fitness',
            'max_prey_generation', 'max_predator_generation',
            'food_available', 'is_day', 'weather'
        ]
        
        with open(self.sim_log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sim_headers)
            writer.writeheader()
            
        with open(self.species_log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=species_headers)
            writer.writeheader()
            
        print(f"📝 Logging to: {self.log_dir}")
    
    def log_creature(self, step: int, creature: 'Creature', action: str, reward: float = 0) -> None:
        """Log individual creature state."""
        # Derive "traits" from creature properties
        intelligence = getattr(creature, 'intelligence', creature.brain.network.layers[1].mean() if hasattr(creature.brain.network, 'layers') else 0.5)
        aggression = getattr(creature, 'aggression', 0.7 if creature.is_predator else 0.3)
        
        record = {
            'step': step,
            'agent_id': creature.id,
            'agent_name': f"{'Predator' if creature.is_predator else 'Prey'}_{creature.id}",
            'species_id': 'predator' if creature.is_predator else 'prey',
            'is_predator': creature.is_predator,
            'health': round(creature.health, 2),
            'energy': round(creature.energy, 2),
            'position_x': round(creature.position.x, 2),
            'position_y': round(creature.position.y, 2),
            'generation': creature.generation,
            'age': creature.age,
            'speed': round(creature.max_speed, 3),
            'intelligence': round(float(intelligence), 3),
            'aggression': round(float(aggression), 3),
            'vision': round(creature.vision_range, 2),
            'action': action,
            'reward': round(reward, 4),
            'kills': creature.kills,
            'times_fled': getattr(creature, 'times_fled', 0)
        }
        self.sim_buffer.append(record)
        
        if len(self.sim_buffer) >= self.buffer_size:
            self._flush_sim_buffer()
    
    def log_species_state(self, step: int, creatures: List['Creature'], 
                          total_kills: int, total_births: int, world: 'World') -> None:
        """Log species-level aggregate statistics."""
        prey = [c for c in creatures if not c.is_predator and c.alive]
        predators = [c for c in creatures if c.is_predator and c.alive]
        
        food_available = sum(f.amount for f in world.foods)
        
        record = {
            'step': step,
            'prey_population': len(prey),
            'predator_population': len(predators),
            'total_population': len(prey) + len(predators),
            'total_kills': total_kills,
            'total_births': total_births,
            'avg_prey_energy': round(sum(c.energy for c in prey) / max(len(prey), 1), 2),
            'avg_predator_energy': round(sum(c.energy for c in predators) / max(len(predators), 1), 2),
            'avg_prey_fitness': round(sum(c.get_fitness() for c in prey) / max(len(prey), 1), 2),
            'avg_predator_fitness': round(sum(c.get_fitness() for c in predators) / max(len(predators), 1), 2),
            'max_prey_generation': max((c.generation for c in prey), default=0),
            'max_predator_generation': max((c.generation for c in predators), default=0),
            'food_available': round(food_available, 1),
            'is_day': world.is_day,
            'weather': world.weather.name
        }
        self.species_buffer.append(record)
        
        if len(self.species_buffer) >= self.buffer_size:
            self._flush_species_buffer()
    
    def _flush_sim_buffer(self) -> None:
        """Write buffered simulation logs to disk."""
        if not self.sim_buffer:
            return
        with open(self.sim_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.sim_buffer[0].keys())
            writer.writerows(self.sim_buffer)
        self.sim_buffer.clear()
    
    def _flush_species_buffer(self) -> None:
        """Write buffered species logs to disk."""
        if not self.species_buffer:
            return
        with open(self.species_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.species_buffer[0].keys())
            writer.writerows(self.species_buffer)
        self.species_buffer.clear()
    
    def flush_all(self) -> None:
        """Flush all buffers."""
        self._flush_sim_buffer()
        self._flush_species_buffer()
    
    def get_stats(self) -> Dict:
        """Get logging statistics."""
        return {
            'sim_records': self.sim_log_path.stat().st_size // 100 if self.sim_log_path.exists() else 0,
            'species_records': self.species_log_path.stat().st_size // 50 if self.species_log_path.exists() else 0,
        }


# =============================================================================
# SPATIAL MEMORY SYSTEM
# =============================================================================

@dataclass
class MemoryLocation:
    """A remembered location with associated value."""
    position: Vector2
    memory_type: str  # 'food', 'danger', 'safe', 'ally'
    strength: float = 1.0
    timestamp: int = 0


class SpatialMemory:
    """Memory system for creatures to remember important locations."""
    
    def __init__(self, capacity: int = 20, decay: float = 0.995):
        self.memories: List[MemoryLocation] = []
        self.capacity = capacity
        self.decay = decay
    
    def remember(self, position: Vector2, memory_type: str, timestamp: int) -> None:
        """Add or update a memory."""
        # Check if similar memory exists (within 30 units)
        for mem in self.memories:
            if mem.memory_type == memory_type and position.distance_to(mem.position) < 30:
                mem.strength = min(1.0, mem.strength + 0.3)
                mem.timestamp = timestamp
                return
        
        # Add new memory
        self.memories.append(MemoryLocation(
            position=position.copy(),
            memory_type=memory_type,
            strength=1.0,
            timestamp=timestamp
        ))
        
        # Trim to capacity (remove weakest)
        if len(self.memories) > self.capacity:
            self.memories.sort(key=lambda m: m.strength, reverse=True)
            self.memories = self.memories[:self.capacity]
    
    def decay_memories(self) -> None:
        """Decay all memories over time."""
        for mem in self.memories:
            mem.strength *= self.decay
        
        # Remove very weak memories
        self.memories = [m for m in self.memories if m.strength > 0.1]
    
    def get_nearest(self, position: Vector2, memory_type: str, 
                    max_distance: float = 200) -> Optional[MemoryLocation]:
        """Get nearest memory of a type."""
        best = None
        best_score = -1
        
        for mem in self.memories:
            if mem.memory_type != memory_type:
                continue
            dist = position.distance_to(mem.position)
            if dist > max_distance:
                continue
            
            # Score by strength and proximity
            score = mem.strength * (1 - dist / max_distance)
            if score > best_score:
                best_score = score
                best = mem
        
        return best
    
    def get_danger_level(self, position: Vector2, radius: float = 100) -> float:
        """Get danger level at a position based on memories."""
        danger = 0.0
        for mem in self.memories:
            if mem.memory_type == 'danger':
                dist = position.distance_to(mem.position)
                if dist < radius:
                    danger += mem.strength * (1 - dist / radius)
        return min(1.0, danger)
    
    def get_danger_zones(self) -> List[Dict]:
        """Get list of dangerous areas for genetic memory inheritance."""
        zones = []
        for mem in self.memories:
            if mem.memory_type == 'danger' and mem.strength > 0.3:
                zones.append({
                    'position': mem.position.copy(),
                    'threat_level': mem.strength
                })
        # Sort by threat level
        zones.sort(key=lambda z: z['threat_level'], reverse=True)
        return zones
    
    def get_good_resource_spots(self) -> List[Dict]:
        """Get list of good resource locations for genetic memory inheritance."""
        spots = []
        for mem in self.memories:
            if mem.memory_type == 'food' and mem.strength > 0.4:
                spots.append({
                    'position': mem.position.copy(),
                    'value': mem.strength
                })
        spots.sort(key=lambda s: s['value'], reverse=True)
        return spots
    
    def add_predator_sighting(self, position: Vector2, threat_level: float = 1.0) -> None:
        """Add a predator sighting as a danger memory."""
        # Check if similar memory exists
        for mem in self.memories:
            if mem.memory_type == 'danger' and position.distance_to(mem.position) < 40:
                mem.strength = min(1.0, mem.strength + threat_level * 0.5)
                return
        
        self.memories.append(MemoryLocation(
            position=position.copy(),
            memory_type='danger',
            strength=threat_level,
            timestamp=0
        ))
    
    def add_food_memory(self, position: Vector2, value: float = 0.5) -> None:
        """Add a food location memory."""
        for mem in self.memories:
            if mem.memory_type == 'food' and position.distance_to(mem.position) < 30:
                mem.strength = min(1.0, mem.strength + value * 0.3)
                return
        
        self.memories.append(MemoryLocation(
            position=position.copy(),
            memory_type='food',
            strength=value,
            timestamp=0
        ))


# =============================================================================
# RESOURCE SYSTEM
# =============================================================================

@dataclass
class FoodSource:
    position: Vector2
    amount: float = 100.0
    max_amount: float = 100.0
    radius: float = 12.0
    regen_rate: float = 0.2
    
    @property
    def is_depleted(self) -> bool:
        return self.amount <= 0
    
    def consume(self, amount: float) -> float:
        eaten = min(amount, self.amount)
        self.amount -= eaten
        return eaten
    
    def regenerate(self, dt: float = 1.0) -> None:
        if self.amount < self.max_amount:
            self.amount = min(self.max_amount, self.amount + self.regen_rate * dt)


@dataclass  
class WaterSource:
    position: Vector2
    radius: float = 30.0


# =============================================================================
# CREATURE BRAIN
# =============================================================================

class CreatureBrain:
    """Neural network brain for creature decision making."""
    
    def __init__(self, is_predator: bool = False):
        self.is_predator = is_predator
        # Inputs: 20 sensory values
        # Outputs: 8 action values
        self.network = NeuralNetwork(layer_sizes=[20, 32, 24, 8])
        self.fitness = 0.0
    
    def decide(self, inputs: List[float]) -> List[float]:
        """Get action outputs from inputs."""
        return self.network.forward(inputs)
    
    def mutate(self, rate: float = 0.15, strength: float = 0.25) -> None:
        """Mutate the neural network."""
        self.network.mutate(rate, strength)
    
    @staticmethod
    def crossover(parent1: 'CreatureBrain', parent2: 'CreatureBrain') -> 'CreatureBrain':
        """Create child brain from two parents."""
        child = CreatureBrain(is_predator=parent1.is_predator)
        child.network = NeuralNetwork.crossover(parent1.network, parent2.network)
        return child
    
    def copy(self) -> 'CreatureBrain':
        """Create a copy of this brain."""
        new_brain = CreatureBrain(is_predator=self.is_predator)
        new_brain.network = self.network.copy()
        return new_brain


# =============================================================================
# PERSISTENT LEARNING SYSTEM
# =============================================================================

class PersistentLearner:
    """
    Persistent Q-Learning system that saves/loads across simulation runs.
    Creatures learn optimal behaviors over multiple generations and sessions.
    """
    
    # Shared knowledge bases (class-level, persist across instances)
    _predator_q_table: Dict[str, Dict[str, float]] = {}
    _prey_q_table: Dict[str, Dict[str, float]] = {}
    _experience_buffer: List[Dict] = []
    _is_loaded: bool = False
    
    # Learning parameters
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.95
    EXPLORATION_RATE = 0.15
    EXPERIENCE_BUFFER_SIZE = 10000
    
    # File paths
    SAVE_DIR = Path("data/models")
    PREDATOR_Q_FILE = "predator_q_table.json"
    PREY_Q_FILE = "prey_q_table.json"
    EXPERIENCE_FILE = "experience_buffer.json"
    
    # State discretization
    STATE_BINS = {
        'health': [0, 25, 50, 75, 100],
        'energy': [0, 25, 50, 75, 100],
        'nearest_threat_dist': [0, 50, 100, 200, 500],
        'nearest_food_dist': [0, 30, 60, 120, 300],
        'nearest_ally_dist': [0, 50, 100, 200, 500],
        'num_threats_nearby': [0, 1, 2, 3, 5],
        'num_allies_nearby': [0, 1, 3, 5, 10],
        'is_day': [0, 1],
    }
    
    # Actions
    ACTIONS = ['flee', 'hunt', 'forage', 'wander', 'rest', 'flock', 'pack_hunt', 'hide']
    
    @classmethod
    def load_knowledge(cls) -> None:
        """Load Q-tables and experience from disk."""
        if cls._is_loaded:
            return
            
        cls.SAVE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load predator Q-table
        pred_path = cls.SAVE_DIR / cls.PREDATOR_Q_FILE
        if pred_path.exists():
            try:
                with open(pred_path, 'r') as f:
                    cls._predator_q_table = json.load(f)
                print(f"  🧠 Loaded predator knowledge: {len(cls._predator_q_table)} states")
            except Exception as e:
                print(f"  ⚠️ Could not load predator Q-table: {e}")
        
        # Load prey Q-table
        prey_path = cls.SAVE_DIR / cls.PREY_Q_FILE
        if prey_path.exists():
            try:
                with open(prey_path, 'r') as f:
                    cls._prey_q_table = json.load(f)
                print(f"  🧠 Loaded prey knowledge: {len(cls._prey_q_table)} states")
            except Exception as e:
                print(f"  ⚠️ Could not load prey Q-table: {e}")
        
        # Load experience buffer
        exp_path = cls.SAVE_DIR / cls.EXPERIENCE_FILE
        if exp_path.exists():
            try:
                with open(exp_path, 'r') as f:
                    cls._experience_buffer = json.load(f)
                print(f"  🧠 Loaded experience buffer: {len(cls._experience_buffer)} experiences")
            except Exception as e:
                print(f"  ⚠️ Could not load experience buffer: {e}")
        
        cls._is_loaded = True
    
    @classmethod
    def save_knowledge(cls) -> None:
        """Save Q-tables and experience to disk."""
        cls.SAVE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save predator Q-table
        pred_path = cls.SAVE_DIR / cls.PREDATOR_Q_FILE
        with open(pred_path, 'w') as f:
            json.dump(cls._predator_q_table, f, indent=2)
        
        # Save prey Q-table
        prey_path = cls.SAVE_DIR / cls.PREY_Q_FILE
        with open(prey_path, 'w') as f:
            json.dump(cls._prey_q_table, f, indent=2)
        
        # Save experience buffer (keep last N)
        exp_path = cls.SAVE_DIR / cls.EXPERIENCE_FILE
        trimmed = cls._experience_buffer[-cls.EXPERIENCE_BUFFER_SIZE:]
        with open(exp_path, 'w') as f:
            json.dump(trimmed, f)
        
        print(f"  💾 Saved knowledge: {len(cls._predator_q_table)} predator states, "
              f"{len(cls._prey_q_table)} prey states, {len(trimmed)} experiences")
    
    @classmethod
    def _discretize(cls, value: float, bins: List[float]) -> int:
        """Convert continuous value to discrete bin index."""
        for i, threshold in enumerate(bins):
            if value <= threshold:
                return i
        return len(bins) - 1
    
    @classmethod
    def get_state_key(cls, creature: 'Creature', nearby_threats: int, 
                      nearby_allies: int, nearest_threat_dist: float,
                      nearest_food_dist: float, nearest_ally_dist: float,
                      is_day: bool) -> str:
        """Convert creature state to discrete string key for Q-table."""
        health_bin = cls._discretize(creature.health, cls.STATE_BINS['health'])
        energy_bin = cls._discretize(creature.energy, cls.STATE_BINS['energy'])
        threat_dist_bin = cls._discretize(nearest_threat_dist, cls.STATE_BINS['nearest_threat_dist'])
        food_dist_bin = cls._discretize(nearest_food_dist, cls.STATE_BINS['nearest_food_dist'])
        ally_dist_bin = cls._discretize(nearest_ally_dist, cls.STATE_BINS['nearest_ally_dist'])
        threats_bin = cls._discretize(nearby_threats, cls.STATE_BINS['num_threats_nearby'])
        allies_bin = cls._discretize(nearby_allies, cls.STATE_BINS['num_allies_nearby'])
        day_bin = 1 if is_day else 0
        
        return f"{health_bin}_{energy_bin}_{threat_dist_bin}_{food_dist_bin}_{ally_dist_bin}_{threats_bin}_{allies_bin}_{day_bin}"
    
    @classmethod
    def get_q_values(cls, state_key: str, is_predator: bool) -> Dict[str, float]:
        """Get Q-values for a state, initializing if needed."""
        q_table = cls._predator_q_table if is_predator else cls._prey_q_table
        
        if state_key not in q_table:
            # Initialize with small random values
            q_table[state_key] = {action: random.uniform(-0.1, 0.1) for action in cls.ACTIONS}
        
        return q_table[state_key]
    
    @classmethod
    def choose_action(cls, state_key: str, is_predator: bool, 
                      valid_actions: Optional[List[str]] = None) -> str:
        """Choose action using epsilon-greedy policy."""
        if valid_actions is None:
            valid_actions = cls.ACTIONS
        
        # Exploration
        if random.random() < cls.EXPLORATION_RATE:
            return random.choice(valid_actions)
        
        # Exploitation
        q_values = cls.get_q_values(state_key, is_predator)
        valid_q = {a: q_values.get(a, 0) for a in valid_actions}
        return max(valid_q.keys(), key=lambda a: valid_q[a])
    
    @classmethod
    def update_q_value(cls, state_key: str, action: str, reward: float,
                       next_state_key: str, is_predator: bool) -> None:
        """Update Q-value using Q-learning formula."""
        q_table = cls._predator_q_table if is_predator else cls._prey_q_table
        
        # Get current Q-value
        current_q = cls.get_q_values(state_key, is_predator).get(action, 0)
        
        # Get max Q-value for next state
        next_q_values = cls.get_q_values(next_state_key, is_predator)
        max_next_q = max(next_q_values.values()) if next_q_values else 0
        
        # Q-learning update
        new_q = current_q + cls.LEARNING_RATE * (
            reward + cls.DISCOUNT_FACTOR * max_next_q - current_q
        )
        
        q_table[state_key][action] = new_q
    
    @classmethod
    def record_experience(cls, state_key: str, action: str, reward: float,
                          next_state_key: str, is_predator: bool, done: bool) -> None:
        """Record experience for replay learning."""
        experience = {
            'state': state_key,
            'action': action,
            'reward': reward,
            'next_state': next_state_key,
            'is_predator': is_predator,
            'done': done
        }
        cls._experience_buffer.append(experience)
        
        # Trim if too large
        if len(cls._experience_buffer) > cls.EXPERIENCE_BUFFER_SIZE * 1.5:
            cls._experience_buffer = cls._experience_buffer[-cls.EXPERIENCE_BUFFER_SIZE:]
    
    @classmethod
    def replay_batch(cls, batch_size: int = 32) -> None:
        """Learn from random batch of experiences."""
        if len(cls._experience_buffer) < batch_size:
            return
        
        batch = random.sample(cls._experience_buffer, batch_size)
        
        for exp in batch:
            if not exp['done']:
                cls.update_q_value(
                    exp['state'], exp['action'], exp['reward'],
                    exp['next_state'], exp['is_predator']
                )
    
    @classmethod
    def get_stats(cls) -> Dict:
        """Get learning statistics."""
        return {
            'predator_states': len(cls._predator_q_table),
            'prey_states': len(cls._prey_q_table),
            'experiences': len(cls._experience_buffer),
            'is_loaded': cls._is_loaded
        }


# =============================================================================
# EMOTIONS & PERSONALITY SYSTEM
# =============================================================================

@dataclass
class Emotions:
    """Emotional state that affects creature behavior."""
    fear: float = 0.0           # 0-1: Increases flee tendency, decreases risk-taking
    hunger: float = 0.5         # 0-1: Increases food-seeking, desperation hunting
    aggression: float = 0.0     # 0-1: Increases attack likelihood, territorial behavior
    curiosity: float = 0.5      # 0-1: Increases exploration, decreases caution
    exhaustion: float = 0.0     # 0-1: Decreases speed, increases rest need
    
    def update(self, dt: float, creature: 'Creature') -> None:
        """Update emotions based on state and environment."""
        # Fear decay/growth
        if creature.nearest_threat:
            dist = creature.position.distance_to(creature.nearest_threat.position)
            if dist < creature.vision_range * 0.5:
                self.fear = min(1.0, self.fear + 0.02 * dt)
            else:
                self.fear = max(0.0, self.fear - 0.005 * dt)
        else:
            self.fear = max(0.0, self.fear - 0.01 * dt)
        
        # Hunger based on energy
        self.hunger = 1.0 - (creature.energy / creature.max_energy)
        
        # Aggression for predators increases with hunger
        if creature.is_predator:
            self.aggression = min(1.0, self.hunger * 0.8 + creature.personality.aggression * 0.4)
        else:
            # Prey get aggressive when cornered (low fear = low aggression normally)
            self.aggression = max(0.0, 0.1 - self.fear * 0.2) if self.fear < 0.3 else 0.0
        
        # Curiosity decreases with fear
        self.curiosity = max(0.1, 0.7 - self.fear * 0.5 - self.exhaustion * 0.3)
        
        # Exhaustion from stamina
        self.exhaustion = 1.0 - (creature.stamina / creature.max_stamina)
    
    def get_behavior_modifier(self) -> Dict[str, float]:
        """Get modifiers for different behaviors based on emotions."""
        return {
            'flee_urgency': 1.0 + self.fear * 1.5,
            'hunt_drive': self.hunger * (1.0 + self.aggression),
            'explore_tendency': self.curiosity * (1.0 - self.fear),
            'rest_need': self.exhaustion * 2.0,
            'caution': self.fear * (1.0 - self.aggression),
            'desperation': max(0, self.hunger - 0.7) * 2.0,  # Desperate when very hungry
        }


@dataclass
class Personality:
    """Innate personality traits - don't change much over lifetime."""
    boldness: float = 0.5       # 0-1: Willingness to take risks
    aggression: float = 0.5     # 0-1: Base aggression level
    sociability: float = 0.5    # 0-1: Tendency to flock/pack
    patience: float = 0.5       # 0-1: Ability to wait/ambush
    learning_rate: float = 0.5  # 0-1: How fast they learn from experience
    
    @classmethod
    def random(cls, is_predator: bool = False) -> 'Personality':
        """Generate random personality with species-appropriate biases."""
        if is_predator:
            return cls(
                boldness=random.uniform(0.4, 0.9),
                aggression=random.uniform(0.5, 0.9),
                sociability=random.uniform(0.3, 0.7),
                patience=random.uniform(0.3, 0.8),
                learning_rate=random.uniform(0.3, 0.7),
            )
        else:
            return cls(
                boldness=random.uniform(0.2, 0.6),
                aggression=random.uniform(0.1, 0.4),
                sociability=random.uniform(0.5, 0.9),
                patience=random.uniform(0.2, 0.5),
                learning_rate=random.uniform(0.4, 0.8),
            )
    
    @classmethod
    def inherit(cls, parent1: 'Personality', parent2: 'Personality', 
                mutation_rate: float = 0.1) -> 'Personality':
        """Inherit personality from parents with mutation."""
        def blend(v1: float, v2: float) -> float:
            base = (v1 + v2) / 2 + random.uniform(-0.1, 0.1)
            if random.random() < mutation_rate:
                base += random.uniform(-0.2, 0.2)
            return max(0.0, min(1.0, base))
        
        return cls(
            boldness=blend(parent1.boldness, parent2.boldness),
            aggression=blend(parent1.aggression, parent2.aggression),
            sociability=blend(parent1.sociability, parent2.sociability),
            patience=blend(parent1.patience, parent2.patience),
            learning_rate=blend(parent1.learning_rate, parent2.learning_rate),
        )


@dataclass
class Skills:
    """Learned skills that improve with practice."""
    hunting: float = 0.0        # Improves attack success, tracking
    evasion: float = 0.0        # Improves escape success
    foraging: float = 0.0       # Improves food finding efficiency
    stealth: float = 0.0        # Reduces detection, improves ambush
    combat: float = 0.0         # Increases damage dealt, reduces taken
    
    def improve(self, skill: str, amount: float = 0.01) -> None:
        """Improve a skill through practice."""
        current = getattr(self, skill, 0.0)
        # Diminishing returns - harder to improve at higher levels
        improvement = amount * (1.0 - current * 0.5)
        setattr(self, skill, min(1.0, current + improvement))
    
    def get_bonus(self, skill: str) -> float:
        """Get the bonus multiplier for a skill."""
        value = getattr(self, skill, 0.0)
        return 1.0 + value * 0.5  # Up to 50% bonus at max skill


# =============================================================================
# LIFE STAGES & REPUTATION (v3.1)
# =============================================================================

class LifeStage(Enum):
    """Life stages with different abilities."""
    BABY = auto()       # 0-200 age: Small, weak, fast learner
    JUVENILE = auto()   # 200-500: Growing, learning
    ADULT = auto()      # 500-2500: Peak performance
    ELDER = auto()      # 2500+: Wise but slower


@dataclass
class Reputation:
    """Creature's reputation affects how others react to them."""
    fear_rating: float = 0.0      # How scary this creature is (predators)
    respect_rating: float = 0.0   # How much others follow this one
    notoriety: float = 0.0        # How well-known (affects detection)
    
    def increase_fear(self, amount: float = 0.1) -> None:
        """Increase fear rating (successful kills)."""
        self.fear_rating = min(1.0, self.fear_rating + amount)
        self.notoriety = min(1.0, self.notoriety + amount * 0.5)
    
    def increase_respect(self, amount: float = 0.1) -> None:
        """Increase respect (successful escapes, helping others)."""
        self.respect_rating = min(1.0, self.respect_rating + amount)
        self.notoriety = min(1.0, self.notoriety + amount * 0.3)
    
    def decay(self, dt: float = 1.0) -> None:
        """Reputation slowly decays over time."""
        decay_rate = 0.0001 * dt
        self.fear_rating = max(0.0, self.fear_rating - decay_rate)
        self.respect_rating = max(0.0, self.respect_rating - decay_rate)
        self.notoriety = max(0.0, self.notoriety - decay_rate * 0.5)


@dataclass
class Territory:
    """A claimed territory for predators."""
    center: Vector2
    radius: float = 150.0
    owner_id: int = -1
    strength: float = 1.0  # How strongly defended
    
    def contains(self, pos: Vector2) -> bool:
        """Check if position is in territory."""
        return self.center.distance_to(pos) < self.radius
    
    def decay(self, dt: float = 1.0) -> None:
        """Territory claim weakens over time if not defended."""
        self.strength = max(0.0, self.strength - 0.001 * dt)


class EnvironmentalHazard:
    """Dynamic environmental hazards."""
    
    def __init__(self, hazard_type: str, position: Vector2, radius: float = 80.0):
        self.hazard_type = hazard_type  # 'fire', 'flood', 'disease'
        self.position = position
        self.radius = radius
        self.intensity = 1.0
        self.duration = 500  # Steps until hazard dissipates
        self.spread_rate = 0.02  # How fast it spreads
        
    def update(self, dt: float = 1.0) -> bool:
        """Update hazard. Returns False if hazard should be removed."""
        self.duration -= dt
        
        # Hazard behavior
        if self.hazard_type == 'fire':
            # Fire spreads then dies
            if self.duration > 300:
                self.radius = min(150, self.radius + self.spread_rate * dt)
            else:
                self.intensity *= 0.995
        elif self.hazard_type == 'flood':
            # Flood expands then recedes
            if self.duration > 250:
                self.radius = min(200, self.radius + 0.5 * dt)
            else:
                self.radius = max(0, self.radius - 0.3 * dt)
        elif self.hazard_type == 'disease':
            # Disease is invisible but deadly
            self.intensity *= 0.999
        
        return self.duration > 0 and self.intensity > 0.1
    
    def affects(self, pos: Vector2) -> float:
        """Check if position is affected. Returns damage multiplier."""
        dist = self.position.distance_to(pos)
        if dist > self.radius:
            return 0.0
        
        # Damage based on proximity to center
        proximity = 1.0 - (dist / self.radius)
        return proximity * self.intensity
    
    def get_color(self) -> Tuple[int, int, int]:
        """Get hazard color for rendering."""
        if self.hazard_type == 'fire':
            return (255, 100, 0)
        elif self.hazard_type == 'flood':
            return (0, 100, 200)
        elif self.hazard_type == 'disease':
            return (100, 200, 0)
        return (128, 128, 128)


# =============================================================================
# NEW v3.3: PACK HIERARCHY, FRENZY, BURROWS, MUTATIONS
# =============================================================================

class PackRole(Enum):
    """Pack hierarchy roles for predators."""
    ALPHA = auto()    # Leader: +30% damage, +20% vision, others follow
    BETA = auto()     # Second: +15% damage, can become alpha
    OMEGA = auto()    # Lowest: -10% stats, but +20% stealth (survival)
    LONE = auto()     # No pack: Normal stats

class ChronotypeTrait(Enum):
    """Day/night activity preference."""
    DIURNAL = auto()     # Active during day (+20% day stats, -30% night)
    NOCTURNAL = auto()   # Active at night (+20% night stats, -30% day)
    CREPUSCULAR = auto() # Active at dawn/dusk (+10% always)

# =============================================================================
# NEW v3.4: EVOLUTION BRANCHES, ACHIEVEMENTS, DISASTERS, BOSS CREATURES
# =============================================================================

class EvolutionBranch(Enum):
    """Specialized evolution paths creatures can take."""
    NONE = auto()
    # Predator branches
    AMBUSHER = auto()      # +50% ambush damage, +30% stealth
    PACK_LEADER = auto()   # +40% pack coordination, +20% ally buff
    BERSERKER = auto()     # +40% damage when low health, -20% defense
    STALKER = auto()       # +60% stealth, +30% patience
    # Prey branches
    SPEEDSTER = auto()     # +50% speed, +30% stamina
    TANK = auto()          # +50% health, +30% defense, -20% speed
    SCOUT = auto()         # +60% vision, +40% danger sense
    SWARM_MIND = auto()    # +50% group buffs, herd immunity

@dataclass
class Achievement:
    """Track creature accomplishments."""
    name: str
    description: str
    icon: str
    requirement: int
    current: int = 0
    unlocked: bool = False
    
    def check(self, value: int) -> bool:
        self.current = value
        if value >= self.requirement and not self.unlocked:
            self.unlocked = True
            return True
        return False

@dataclass
class NaturalDisaster:
    """Major world events that affect all creatures."""
    disaster_type: str  # 'earthquake', 'meteor', 'volcanic', 'blizzard'
    position: Vector2
    radius: float
    intensity: float
    duration: int
    max_duration: int
    
    def __post_init__(self):
        self.max_duration = self.duration
    
    def update(self) -> bool:
        """Returns False when disaster ends."""
        self.duration -= 1
        # Intensity fades over time
        self.intensity *= 0.995
        return self.duration > 0
    
    def get_effect_at(self, pos: Vector2) -> Dict[str, float]:
        """Get disaster effects at position."""
        dist = self.position.distance_to(pos)
        if dist > self.radius:
            return {}
        
        proximity = 1.0 - (dist / self.radius)
        effects = {}
        
        if self.disaster_type == 'earthquake':
            effects['stun_chance'] = proximity * 0.3
            effects['damage'] = proximity * self.intensity * 5
            effects['speed_penalty'] = proximity * 0.5
        elif self.disaster_type == 'meteor':
            effects['damage'] = proximity * self.intensity * 50  # High damage at center
            effects['fire_chance'] = proximity * 0.8
        elif self.disaster_type == 'volcanic':
            effects['damage'] = proximity * self.intensity * 10
            effects['vision_penalty'] = proximity * 0.6  # Ash clouds
        elif self.disaster_type == 'blizzard':
            effects['speed_penalty'] = proximity * 0.7
            effects['energy_drain'] = proximity * self.intensity * 2
            effects['vision_penalty'] = proximity * 0.5
        
        return effects

@dataclass
class BossCreature:
    """Rare, powerful creature that spawns occasionally."""
    boss_type: str  # 'apex_predator', 'ancient_prey', 'mutant'
    creature_id: int
    title: str
    power_level: float  # Multiplier for all stats
    special_moves: List[str]
    health_bar_visible: bool = True
    defeated: bool = False

@dataclass
class ThoughtBubble:
    """Visual representation of creature thoughts."""
    creature_id: int
    thought: str
    emoji: str
    duration: int = 60
    position: Vector2 = None

@dataclass
class GeneticMutation:
    """Special genetic mutations that give unique abilities."""
    name: str
    rarity: float          # 0.0-1.0, lower = rarer
    color_modifier: Tuple[int, int, int]  # RGB color shift
    stat_bonuses: Dict[str, float]  # Stat name -> multiplier
    special_ability: Optional[str] = None  # Special ability name
    
    @staticmethod
    def roll_mutation() -> Optional['GeneticMutation']:
        """Randomly roll for a mutation."""
        mutations = [
            GeneticMutation("Giant", 0.02, (50, 50, 50), {"size": 1.4, "speed": 0.85, "damage": 1.3}),
            GeneticMutation("Swift", 0.03, (100, 200, 255), {"speed": 1.35, "stamina": 1.2, "size": 0.9}),
            GeneticMutation("Tough", 0.03, (150, 100, 50), {"health": 1.4, "defense": 1.3, "speed": 0.9}),
            GeneticMutation("Genius", 0.02, (255, 215, 0), {"learning": 1.5, "vision": 1.2}, "fast_learner"),
            GeneticMutation("Berserker", 0.015, (200, 0, 50), {"damage": 1.5, "health": 0.8}, "rage_mode"),
            GeneticMutation("Ghost", 0.01, (200, 200, 255), {"stealth": 1.8, "speed": 1.1}, "invisibility"),
            GeneticMutation("Regenerator", 0.02, (50, 255, 100), {"health_regen": 2.0, "energy_regen": 1.3}),
            GeneticMutation("Apex", 0.005, (255, 50, 200), {"damage": 1.4, "speed": 1.2, "vision": 1.3}, "apex_aura"),
        ]
        
        for mut in mutations:
            if random.random() < mut.rarity:
                return mut
        return None

@dataclass
class Burrow:
    """A hiding spot for prey."""
    position: Vector2
    capacity: int = 3           # Max creatures that can hide
    occupants: List[int] = None  # IDs of creatures inside
    safety_rating: float = 0.8   # How safe (chance to avoid detection)
    cooldown: int = 0           # Ticks until can be used again
    
    def __post_init__(self):
        if self.occupants is None:
            self.occupants = []
    
    def can_enter(self) -> bool:
        return len(self.occupants) < self.capacity and self.cooldown <= 0
    
    def enter(self, creature_id: int) -> bool:
        if self.can_enter():
            self.occupants.append(creature_id)
            return True
        return False
    
    def exit(self, creature_id: int) -> bool:
        if creature_id in self.occupants:
            self.occupants.remove(creature_id)
            self.cooldown = 50  # Brief cooldown after exit
            return True
        return False
    
    def update(self) -> None:
        if self.cooldown > 0:
            self.cooldown -= 1

@dataclass  
class CommunicationWave:
    """Visual representation of creature communication."""
    position: Vector2
    wave_type: str        # 'roar', 'alert', 'call', 'howl'
    radius: float = 0.0
    max_radius: float = 150.0
    speed: float = 5.0
    color: Tuple[int, int, int] = (255, 255, 255)
    alpha: int = 200
    
    def update(self) -> bool:
        """Expand wave. Returns False when done."""
        self.radius += self.speed
        self.alpha = int(200 * (1 - self.radius / self.max_radius))
        return self.radius < self.max_radius

@dataclass
class ScentMark:
    """Territorial scent marks left by predators."""
    position: Vector2
    owner_id: int
    strength: float = 1.0
    decay_rate: float = 0.002
    radius: float = 40.0
    color: Tuple[int, int, int] = (139, 69, 19)  # Brown/amber
    
    def update(self) -> bool:
        """Decay scent. Returns False when fully decayed."""
        self.strength -= self.decay_rate
        return self.strength > 0

@dataclass
class HuntingFormation:
    """Pack hunting formation for coordinated attacks."""
    leader_id: int
    member_ids: List[int] = field(default_factory=list)
    target_id: Optional[int] = None
    formation_type: str = 'surround'  # 'surround', 'chase', 'ambush'
    active: bool = True
    positions: Dict[int, Vector2] = field(default_factory=dict)
    
    def get_formation_position(self, member_id: int, target_pos: Vector2, leader_pos: Vector2) -> Vector2:
        """Calculate ideal position for pack member in formation."""
        if member_id == self.leader_id:
            return target_pos  # Leader goes for target
        
        # Calculate positions around target
        idx = self.member_ids.index(member_id) if member_id in self.member_ids else 0
        total = len(self.member_ids) + 1
        angle = (2 * math.pi * idx) / total
        
        if self.formation_type == 'surround':
            # Circle around target
            offset = Vector2(math.cos(angle) * 80, math.sin(angle) * 80)
            return target_pos + offset
        elif self.formation_type == 'chase':
            # Line behind leader
            direction = (target_pos - leader_pos).normalize() if (target_pos - leader_pos).length() > 0 else Vector2(1, 0)
            return leader_pos - direction * (50 * (idx + 1))
        else:  # ambush
            # Fan out ahead of target
            return target_pos + Vector2(math.cos(angle) * 120, math.sin(angle) * 120)


# =============================================================================
# NEW v3.3: SPECIAL ABILITIES
# =============================================================================

class SpecialAbility(Enum):
    """Special abilities creatures can have or evolve."""
    NONE = auto()
    # Predator abilities
    NIGHT_VISION = auto()       # No vision penalty at night
    AMBUSH_MASTER = auto()      # +50% damage from ambush
    PACK_CALLER = auto()        # Can summon nearby pack members
    BLOOD_FRENZY = auto()       # Enter frenzy after kill
    INTIMIDATING_ROAR = auto()  # Scare prey in radius
    # Prey abilities  
    CAMOUFLAGE = auto()         # Harder to detect when still
    BURROW_EXPERT = auto()      # Faster burrow entry, longer stay
    DANGER_SENSE = auto()       # Detect predators at 2x range
    SPEED_BURST = auto()        # Short sprint when fleeing
    HERD_MIND = auto()          # Bonus stats near other prey

@dataclass
class AbilityEffect:
    """Active effect from ability use."""
    ability: SpecialAbility
    duration: int
    strength: float = 1.0
    
    def update(self) -> bool:
        """Returns False when effect expires."""
        self.duration -= 1
        return self.duration > 0


# =============================================================================
# NEW v3.5: SEASONS, FAMILIES, LEGENDARY CREATURES, WORLD EVENTS
# =============================================================================

class Season(Enum):
    """World seasons that affect gameplay."""
    SPRING = auto()    # +30% food spawn, +20% reproduction
    SUMMER = auto()    # Normal, +10% speed
    AUTUMN = auto()    # -20% food, creatures store energy
    WINTER = auto()    # -50% food, -30% speed, survival mode

@dataclass
class CreatureRelationship:
    """Track relationships between creatures."""
    creature_a_id: int
    creature_b_id: int
    relationship_type: str  # 'parent', 'child', 'sibling', 'mate', 'rival', 'friend'
    bond_strength: float = 0.5  # 0.0 to 1.0
    interactions: int = 0
    
    def strengthen(self, amount: float = 0.05):
        self.bond_strength = min(1.0, self.bond_strength + amount)
        self.interactions += 1
    
    def weaken(self, amount: float = 0.02):
        self.bond_strength = max(0.0, self.bond_strength - amount)

@dataclass
class CreatureLineage:
    """Track family tree of a creature."""
    creature_id: int
    parent_ids: List[int] = field(default_factory=list)
    children_ids: List[int] = field(default_factory=list)
    mate_id: Optional[int] = None
    generation: int = 0
    family_name: str = ""
    dynasty_kills: int = 0
    dynasty_survivals: int = 0

@dataclass
class LegendaryCreature:
    """Extremely rare legendary creatures with unique powers."""
    creature_id: int
    legendary_type: str  # 'phoenix', 'shadow_wolf', 'ancient_one', 'storm_bringer'
    title: str
    aura_color: Tuple[int, int, int]
    special_powers: List[str]
    kills_required_to_spawn: int = 100
    legend_level: int = 1  # 1-5

@dataclass
class WorldEvent:
    """Special world events that change gameplay temporarily."""
    event_type: str  # 'blood_moon', 'migration', 'abundance', 'famine', 'predator_invasion'
    name: str
    description: str
    duration: int
    effects: Dict[str, float]
    active: bool = True
    
    def update(self) -> bool:
        self.duration -= 1
        return self.duration > 0

@dataclass 
class CreatureMemorial:
    """Memorial for legendary creatures that died."""
    name: str
    species: str
    kills: int
    survived_days: float
    cause_of_death: str
    legendary_title: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

@dataclass
class Artifact:
    """Special items that boost nearby creatures."""
    position: Vector2
    artifact_type: str  # 'ancient_bone', 'mystic_stone', 'life_spring', 'death_mark'
    radius: float = 60.0
    power: float = 1.0
    effects: Dict[str, float] = field(default_factory=dict)
    duration: int = -1  # -1 = permanent
    glow_color: Tuple[int, int, int] = (255, 215, 0)

@dataclass
class WeatherForecast:
    """Predict upcoming weather."""
    current: str
    next_weather: str
    change_in_steps: int
    season: Season = Season.SPRING

LEGENDARY_TYPES = {
    'phoenix': {
        'title': 'The Undying',
        'aura': (255, 100, 0),
        'powers': ['resurrection', 'flame_aura', 'inspiring_presence'],
        'prey_only': True
    },
    'shadow_wolf': {
        'title': 'Nightmare Hunter',
        'aura': (80, 0, 120),
        'powers': ['shadow_step', 'fear_aura', 'pack_summon'],
        'prey_only': False
    },
    'ancient_one': {
        'title': 'The Eternal',
        'aura': (0, 200, 150),
        'powers': ['time_slow', 'wisdom_aura', 'immortal_will'],
        'prey_only': True
    },
    'storm_bringer': {
        'title': 'Lord of Thunder',
        'aura': (100, 150, 255),
        'powers': ['lightning_strike', 'storm_call', 'wind_rider'],
        'prey_only': False
    }
}

WORLD_EVENTS = {
    'blood_moon': {
        'name': '🌑 Blood Moon',
        'description': 'Predators become frenzied!',
        'duration': 500,
        'effects': {'predator_damage': 1.5, 'predator_speed': 1.2, 'prey_fear': 1.5}
    },
    'migration': {
        'name': '🦋 Great Migration',
        'description': 'Prey population surge!',
        'duration': 400,
        'effects': {'prey_spawn': 2.0, 'food_spawn': 1.5}
    },
    'abundance': {
        'name': '🌸 Season of Plenty',
        'description': 'Resources everywhere!',
        'duration': 600,
        'effects': {'food_spawn': 2.0, 'water_spawn': 1.5, 'reproduction': 1.3}
    },
    'famine': {
        'name': '💀 Great Famine',
        'description': 'Resources are scarce...',
        'duration': 400,
        'effects': {'food_spawn': 0.3, 'energy_drain': 1.5}
    },
    'predator_invasion': {
        'name': '🐺 Predator Invasion',
        'description': 'Apex predators arrive!',
        'duration': 300,
        'effects': {'predator_spawn': 2.0, 'predator_damage': 1.3}
    }
}


# =============================================================================
# NEW v3.6: NAMES, COMBAT SYSTEM, STRUCTURES, PORTALS, LEADERBOARD
# =============================================================================

# Name generation pools
PREDATOR_NAMES = {
    'prefixes': ['Shadow', 'Blood', 'Dark', 'Iron', 'Storm', 'Death', 'Rage', 'Fang', 'Night', 'Grim'],
    'suffixes': ['fang', 'claw', 'bite', 'strike', 'hunter', 'stalker', 'killer', 'ripper', 'slayer', 'reaper']
}

PREY_NAMES = {
    'prefixes': ['Swift', 'Bright', 'Lucky', 'Brave', 'Quick', 'Wise', 'Fleet', 'Nimble', 'Golden', 'Silver'],
    'suffixes': ['foot', 'heart', 'spirit', 'runner', 'dash', 'leap', 'spring', 'whisker', 'tail', 'ears']
}

TITLES = {
    'predator': {
        5: 'Hunter',
        15: 'Slayer', 
        30: 'Destroyer',
        50: 'Apex Predator',
        100: 'Death Incarnate'
    },
    'prey': {
        500: 'Survivor',
        1000: 'Escape Artist',
        2000: 'Living Legend',
        5000: 'Immortal',
        10000: 'The Unkillable'
    }
}

@dataclass
class CombatResult:
    """Result of a combat encounter."""
    attacker_id: int
    defender_id: int
    damage_dealt: float
    was_critical: bool
    was_dodged: bool
    combo_count: int = 0
    special_effect: str = ""

@dataclass
class MegaStructure:
    """Large structures that creatures build/use."""
    position: Vector2
    structure_type: str  # 'nest', 'den', 'hive', 'fortress'
    owner_species: str  # 'predator', 'prey'
    health: float = 100.0
    max_health: float = 100.0
    radius: float = 50.0
    occupants: List[int] = field(default_factory=list)
    buffs: Dict[str, float] = field(default_factory=dict)
    level: int = 1
    build_progress: float = 100.0

@dataclass
class Portal:
    """Teleportation points in the world."""
    position: Vector2
    linked_portal_id: int = -1
    portal_type: str = 'wormhole'  # 'wormhole', 'gateway', 'rift'
    cooldown: int = 0
    color: Tuple[int, int, int] = (150, 0, 255)
    radius: float = 25.0
    active: bool = True

@dataclass
class EventLogEntry:
    """Entry in the scrolling event log."""
    timestamp: int
    message: str
    icon: str
    importance: int = 1  # 1=minor, 2=notable, 3=major, 4=epic, 5=legendary
    color: Tuple[int, int, int] = (255, 255, 255)

@dataclass
class LeaderboardEntry:
    """Entry for creature leaderboards."""
    creature_id: int
    name: str
    species: str
    score: float
    category: str  # 'kills', 'survival', 'children', 'distance'
    is_alive: bool = True

@dataclass
class CombatCombo:
    """Track combat combos for creatures."""
    creature_id: int
    combo_count: int = 0
    last_hit_time: int = 0
    max_combo: int = 0
    combo_damage_bonus: float = 0.0

@dataclass
class CreatureStats:
    """Detailed statistics for a creature."""
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    critical_hits: int = 0
    dodges: int = 0
    combos_started: int = 0
    max_combo: int = 0
    structures_built: int = 0
    portals_used: int = 0
    time_as_leader: int = 0
    allies_helped: int = 0

# Combat modifiers
COMBAT_MODIFIERS = {
    'critical_chance': 0.1,
    'critical_multiplier': 2.0,
    'dodge_chance': 0.15,
    'combo_window': 30,  # Steps to continue combo
    'combo_damage_bonus': 0.1,  # Per combo hit
    'max_combo_bonus': 1.0,  # Max 100% bonus damage
}

STRUCTURE_TYPES = {
    'nest': {
        'species': 'prey',
        'health': 80,
        'radius': 40,
        'buffs': {'health_regen': 1.5, 'safety': 0.3},
        'max_occupants': 5
    },
    'den': {
        'species': 'predator', 
        'health': 120,
        'radius': 50,
        'buffs': {'damage': 1.2, 'rest_speed': 2.0},
        'max_occupants': 4
    },
    'hive': {
        'species': 'prey',
        'health': 150,
        'radius': 60,
        'buffs': {'group_bonus': 1.5, 'alert_range': 2.0},
        'max_occupants': 10
    },
    'fortress': {
        'species': 'predator',
        'health': 200,
        'radius': 70,
        'buffs': {'damage': 1.4, 'defense': 1.3, 'intimidation': 2.0},
        'max_occupants': 6
    }
}


# =============================================================================
# CREATURE STATES (ENHANCED)
# =============================================================================

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
    MATING = auto()
    FLOCKING = auto()  # Prey flocking
    PACK_HUNTING = auto()  # Coordinated predator hunt
    AMBUSHING = auto()  # Predator lying in wait
    STALKING = auto()   # Predator sneaking up
    ALERTING = auto()   # Prey warning others
    INVESTIGATING = auto()  # Checking out interesting spot
    STAMPEDING = auto()  # v3.1: Mass panic flee
    DEFENDING_TERRITORY = auto()  # v3.1: Protecting claimed area
    CHALLENGING = auto()  # v3.1: Dominance challenge
    FLEEING_HAZARD = auto()  # v3.1: Escaping environmental danger
    # NEW v3.3 States
    HIDING = auto()      # Hiding in burrow
    FRENZY = auto()      # Hunger frenzy mode (predators)
    LEADING_PACK = auto()  # Alpha leading the hunt
    HOWLING = auto()     # Communication howl
    SLEEPING = auto()    # Nocturnal creatures during day / vice versa
    DEAD = auto()


class Creature:
    """A living creature with brain, physics, memory, and behavior."""
    
    _id_counter = 0
    
    def __init__(
        self,
        position: Vector2,
        is_predator: bool = False,
        brain: CreatureBrain = None,
        generation: int = 0,
        config: SimConfig = None,
    ):
        Creature._id_counter += 1
        self.id = Creature._id_counter
        self.is_predator = is_predator
        self.generation = generation
        self.config = config or SimConfig()
        
        # Brain
        self.brain = brain or CreatureBrain(is_predator)
        
        # Memory System (NEW)
        self.memory = SpatialMemory(
            capacity=self.config.memory_capacity,
            decay=self.config.memory_decay
        )
        
        # Learning state (NEW v2.1)
        self.current_state_key: str = ""
        self.previous_state_key: str = ""
        self.previous_action: str = "idle"
        self.cumulative_reward: float = 0.0
        
        # Physics
        self.position = position.copy()
        self.velocity = Vector2.random_unit() * random.uniform(0.5, 1.5)
        self.acceleration = Vector2(0, 0)
        
        # Genetic traits (evolved properties)
        if is_predator:
            self.max_speed = self.config.predator_max_speed * random.uniform(0.9, 1.1)
            self.vision_range = self.config.predator_vision * random.uniform(0.9, 1.1)
            self.color = (
                random.randint(180, 255),
                random.randint(20, 60),
                random.randint(20, 60),
            )
        else:
            self.max_speed = self.config.prey_max_speed * random.uniform(0.9, 1.1)
            self.vision_range = self.config.prey_vision * random.uniform(0.9, 1.1)
            self.color = (
                random.randint(180, 255),
                random.randint(180, 255),
                random.randint(50, 100),
            )
        
        self.max_force = 0.3
        self.size = 10.0 if is_predator else 8.0
        
        # NEW v2.3: Personality, Emotions, and Skills
        self.personality = Personality.random(is_predator)
        self.emotions = Emotions()
        self.skills = Skills()
        
        # NEW v3.1: Reputation and Life Stage
        self.reputation = Reputation()
        self.territory: Optional[Territory] = None  # For predators
        self.is_stampeding: bool = False
        self.stampede_direction: Optional[Vector2] = None
        self.rival: Optional['Creature'] = None  # Current rival for dominance
        self.hazard_position: Optional[Vector2] = None  # For fleeing hazards
        
        # Ambush state (for predators)
        self.ambush_position: Optional[Vector2] = None
        self.ambush_timer: float = 0.0
        self.stalk_target: Optional[Creature] = None
        
        # Communication (for prey)
        self.alert_cooldown: float = 0.0
        self.received_alert: bool = False
        self.alert_source: Optional[Vector2] = None
        
        # Stats
        self.health = 100.0
        self.max_health = 100.0
        self.energy = 75.0
        self.max_energy = 100.0
        self.stamina = 100.0
        self.max_stamina = 100.0
        
        # State
        self.alive = True
        self.state = CreatureState.WANDERING
        self.age = 0
        self.last_action = 'idle'  # NEW: For logging
        self.last_reward = 0.0  # NEW: For logging
        
        # Tracking
        self.kills = 0
        self.food_eaten = 0.0
        self.children = 0
        self.distance_traveled = 0.0
        self.times_fled = 0  # NEW: Track fleeing behavior
        self.successful_hunts = 0  # NEW
        
        # Targets
        self.target: Optional[Creature] = None
        self.target_food: Optional[FoodSource] = None
        self.nearest_threat: Optional[Creature] = None
        
        # Pack/Flock awareness (NEW)
        self.nearby_allies: List[Creature] = []
        self.pack_target: Optional[Creature] = None  # Shared target for pack hunting
        
        # Cooldowns
        self.attack_cooldown = 0.0
        self.reproduction_cooldown = 0.0
        
        # Trail for visualization
        self.trail: deque = deque(maxlen=30)
        
        # Wander state
        self.wander_angle = random.uniform(0, 2 * math.pi)
        
        # NEW v3.3: Pack Hierarchy, Mutations, Chronotype
        self.pack_role: PackRole = PackRole.LONE
        self.pack_leader: Optional['Creature'] = None
        self.pack_members: List['Creature'] = []
        self.mutation: Optional[GeneticMutation] = GeneticMutation.roll_mutation()
        self.chronotype: ChronotypeTrait = random.choice([
            ChronotypeTrait.DIURNAL, ChronotypeTrait.DIURNAL,  # Most are diurnal
            ChronotypeTrait.NOCTURNAL,
            ChronotypeTrait.CREPUSCULAR,
        ])
        
        # Frenzy mode (predators)
        self.is_frenzied: bool = False
        self.frenzy_timer: float = 0.0
        
        # Burrow (prey)
        self.hiding_in_burrow: Optional['Burrow'] = None
        self.burrow_cooldown: float = 0.0
        
        # NEW v3.3: Special abilities
        self.special_ability: SpecialAbility = self._roll_special_ability()
        self.ability_cooldown: float = 0.0
        self.active_effects: List[AbilityEffect] = []
        
        # NEW v3.4: Evolution branch & achievements
        self.evolution_branch: EvolutionBranch = EvolutionBranch.NONE
        self.evolution_points: float = 0.0  # Earn through survival/kills
        self.achievements: List[Achievement] = self._init_achievements()
        self.is_boss: bool = False
        self.boss_data: Optional[BossCreature] = None
        self.current_thought: Optional[str] = None
        self.thought_timer: int = 0
        
        # Apply mutation bonuses
        if self.mutation:
            self._apply_mutation()
    
    def _init_achievements(self) -> List[Achievement]:
        """Initialize creature achievement tracking."""
        if self.is_predator:
            return [
                Achievement("First Blood", "Get your first kill", "🩸", 1),
                Achievement("Hunter", "Kill 10 creatures", "🎯", 10),
                Achievement("Apex Predator", "Kill 50 creatures", "👑", 50),
                Achievement("Survivor", "Live for 1000 ticks", "⏱️", 1000),
                Achievement("Elder", "Reach elder age", "👴", 2500),
                Achievement("Pack Master", "Lead 5+ pack hunts", "🐺", 5),
            ]
        else:
            return [
                Achievement("Escape Artist", "Escape 10 attacks", "🏃", 10),
                Achievement("Survivor", "Live for 1000 ticks", "⏱️", 1000),
                Achievement("Elder", "Reach elder age", "👴", 2500),
                Achievement("Parent", "Have 5 children", "👨‍👩‍👧‍👦", 5),
                Achievement("Forager", "Eat 100 food", "🌿", 100),
                Achievement("Social", "Flock with 20+ creatures", "🐑", 20),
            ]
    
    def _roll_special_ability(self) -> SpecialAbility:
        """Randomly assign a special ability with weighted chances."""
        if random.random() > 0.15:  # 85% chance of no ability
            return SpecialAbility.NONE
        
        if self.is_predator:
            abilities = [
                SpecialAbility.NIGHT_VISION,
                SpecialAbility.AMBUSH_MASTER,
                SpecialAbility.PACK_CALLER,
                SpecialAbility.BLOOD_FRENZY,
                SpecialAbility.INTIMIDATING_ROAR,
            ]
        else:
            abilities = [
                SpecialAbility.CAMOUFLAGE,
                SpecialAbility.BURROW_EXPERT,
                SpecialAbility.DANGER_SENSE,
                SpecialAbility.SPEED_BURST,
                SpecialAbility.HERD_MIND,
            ]
        return random.choice(abilities)
    
    def _apply_mutation(self) -> None:
        """Apply genetic mutation bonuses to stats."""
        if not self.mutation:
            return
        
        bonuses = self.mutation.stat_bonuses
        if 'size' in bonuses:
            self.size *= bonuses['size']
        if 'speed' in bonuses:
            self.max_speed *= bonuses['speed']
        if 'health' in bonuses:
            self.max_health *= bonuses['health']
            self.health = self.max_health
        if 'stamina' in bonuses:
            self.max_stamina *= bonuses['stamina']
            self.stamina = self.max_stamina
        if 'vision' in bonuses:
            self.vision_range *= bonuses['vision']
        
        # Apply color modifier
        r = min(255, max(0, self.color[0] + self.mutation.color_modifier[0]))
        g = min(255, max(0, self.color[1] + self.mutation.color_modifier[1]))
        b = min(255, max(0, self.color[2] + self.mutation.color_modifier[2]))
        self.color = (r, g, b)
    
    def get_chronotype_modifier(self, is_day: bool) -> float:
        """Get activity modifier based on chronotype and time of day."""
        if self.chronotype == ChronotypeTrait.DIURNAL:
            return 1.2 if is_day else 0.7
        elif self.chronotype == ChronotypeTrait.NOCTURNAL:
            return 0.7 if is_day else 1.2
        else:  # CREPUSCULAR
            return 1.1  # Always slightly boosted
    
    def get_pack_role_modifier(self) -> Dict[str, float]:
        """Get stat modifiers based on pack role."""
        if self.pack_role == PackRole.ALPHA:
            return {'damage': 1.3, 'vision': 1.2, 'intimidation': 1.5}
        elif self.pack_role == PackRole.BETA:
            return {'damage': 1.15, 'vision': 1.1, 'intimidation': 1.2}
        elif self.pack_role == PackRole.OMEGA:
            return {'damage': 0.9, 'stealth': 1.2, 'intimidation': 0.7}
        return {'damage': 1.0, 'vision': 1.0, 'intimidation': 1.0}
    
    def enter_frenzy(self) -> None:
        """Enter hunger frenzy mode (predators only)."""
        if not self.is_predator or self.is_frenzied:
            return
        self.is_frenzied = True
        self.frenzy_timer = 300  # Lasts 300 ticks
        self.max_speed *= 1.4
        self.emotions.aggression = 1.0
        self.emotions.fear = 0.0
    
    def exit_frenzy(self) -> None:
        """Exit frenzy mode."""
        if not self.is_frenzied:
            return
        self.is_frenzied = False
        self.max_speed /= 1.4
        self.energy -= 30  # Exhausted after frenzy
        self.stamina -= 50
    
    def try_hide_in_burrow(self, burrow: 'Burrow') -> bool:
        """Attempt to hide in a burrow (prey only)."""
        if self.is_predator or self.burrow_cooldown > 0:
            return False
        if burrow.enter(self.id):
            self.hiding_in_burrow = burrow
            self.state = CreatureState.HIDING
            return True
        return False
    
    def exit_burrow(self) -> None:
        """Exit current burrow."""
        if self.hiding_in_burrow:
            self.hiding_in_burrow.exit(self.id)
            self.hiding_in_burrow = None
            self.burrow_cooldown = 100  # Can't re-enter immediately
            self.state = CreatureState.WANDERING
    
    def should_mark_territory(self) -> bool:
        """Check if predator should leave scent mark."""
        if not self.is_predator:
            return False
        # Alphas mark more frequently
        base_chance = 0.01
        if self.pack_role == PackRole.ALPHA:
            base_chance = 0.03
        elif self.pack_role == PackRole.LONE:
            base_chance = 0.02
        return random.random() < base_chance
    
    def can_join_hunt(self) -> bool:
        """Check if predator can join a hunting formation."""
        if not self.is_predator:
            return False
        return (self.energy > 30 and 
                self.stamina > 40 and 
                self.state not in [CreatureState.RESTING, CreatureState.EATING, CreatureState.SLEEPING])
    
    def use_ability(self) -> bool:
        """Attempt to use special ability. Returns True if activated."""
        if self.special_ability == SpecialAbility.NONE:
            return False
        if self.ability_cooldown > 0:
            return False
        
        # Different cooldowns and effects per ability
        if self.special_ability == SpecialAbility.SPEED_BURST:
            self.active_effects.append(AbilityEffect(
                ability=SpecialAbility.SPEED_BURST,
                duration=60,
                strength=1.8,  # 80% speed boost
            ))
            self.ability_cooldown = 300
            return True
        
        elif self.special_ability == SpecialAbility.INTIMIDATING_ROAR:
            self.ability_cooldown = 400
            return True  # Effect handled externally
        
        elif self.special_ability == SpecialAbility.CAMOUFLAGE:
            self.active_effects.append(AbilityEffect(
                ability=SpecialAbility.CAMOUFLAGE,
                duration=200,
                strength=0.5,  # 50% harder to detect
            ))
            self.ability_cooldown = 500
            return True
        
        return False
    
    def has_active_effect(self, ability: SpecialAbility) -> bool:
        """Check if creature has an active ability effect."""
        return any(e.ability == ability for e in self.active_effects)
    
    def get_effect_strength(self, ability: SpecialAbility) -> float:
        """Get strength of active effect, or 0 if not active."""
        for effect in self.active_effects:
            if effect.ability == ability:
                return effect.strength
        return 0.0
    
    def update_effects(self) -> None:
        """Update and remove expired effects."""
        self.active_effects = [e for e in self.active_effects if e.update()]
        if self.ability_cooldown > 0:
            self.ability_cooldown -= 1
    
    def earn_evolution_points(self, amount: float) -> None:
        """Earn evolution points from actions."""
        self.evolution_points += amount
        # Check for evolution at 100 points
        if self.evolution_points >= 100 and self.evolution_branch == EvolutionBranch.NONE:
            self._evolve()
    
    def _evolve(self) -> None:
        """Choose an evolution branch based on creature behavior."""
        if self.is_predator:
            # Choose based on play style
            if self.skills.stealth > 0.6:
                self.evolution_branch = EvolutionBranch.STALKER
            elif self.pack_role == PackRole.ALPHA:
                self.evolution_branch = EvolutionBranch.PACK_LEADER
            elif self.skills.hunting > 0.6:
                self.evolution_branch = EvolutionBranch.AMBUSHER
            else:
                self.evolution_branch = EvolutionBranch.BERSERKER
        else:
            # Prey evolution
            if self.skills.evasion > 0.6:
                self.evolution_branch = EvolutionBranch.SPEEDSTER
            elif self.personality.sociability > 0.6:
                self.evolution_branch = EvolutionBranch.SWARM_MIND
            elif self.max_health > 100:
                self.evolution_branch = EvolutionBranch.TANK
            else:
                self.evolution_branch = EvolutionBranch.SCOUT
        
        # Apply evolution bonuses
        self._apply_evolution_bonuses()
    
    def _apply_evolution_bonuses(self) -> None:
        """Apply stat bonuses from evolution branch."""
        branch = self.evolution_branch
        if branch == EvolutionBranch.STALKER:
            self.skills.stealth = min(1.0, self.skills.stealth + 0.3)
            self.personality.patience = min(1.0, self.personality.patience + 0.3)
        elif branch == EvolutionBranch.PACK_LEADER:
            self.vision_range *= 1.2
        elif branch == EvolutionBranch.AMBUSHER:
            self.skills.hunting = min(1.0, self.skills.hunting + 0.3)
        elif branch == EvolutionBranch.BERSERKER:
            self.max_health *= 0.9
        elif branch == EvolutionBranch.SPEEDSTER:
            self.max_speed *= 1.3
            self.max_stamina *= 1.2
        elif branch == EvolutionBranch.TANK:
            self.max_health *= 1.4
            self.max_speed *= 0.85
        elif branch == EvolutionBranch.SCOUT:
            self.vision_range *= 1.4
        elif branch == EvolutionBranch.SWARM_MIND:
            pass  # Buff applied when near allies
    
    def think(self, inputs: np.ndarray) -> None:
        """Process inputs and update thoughts."""
        # Parent think method
        outputs = self.brain.forward(inputs)
        self._process_outputs(outputs)
        
        # Generate thought bubble occasionally
        if self.thought_timer <= 0 and random.random() < 0.01:
            self._generate_thought()
            self.thought_timer = 100
        elif self.thought_timer > 0:
            self.thought_timer -= 1
    
    def _generate_thought(self) -> None:
        """Generate a thought based on current state."""
        if self.emotions.hunger > 0.8:
            self.current_thought = ("So hungry...", "🍖")
        elif self.emotions.fear > 0.7:
            self.current_thought = ("Must escape!", "😱")
        elif self.state == CreatureState.HUNTING:
            self.current_thought = ("Target acquired", "🎯")
        elif self.state == CreatureState.RESTING:
            self.current_thought = ("zzZ", "😴")
        elif self.emotions.aggression > 0.7:
            self.current_thought = ("RAGE!", "😠")
        elif self.energy > 80 and self.health > 80:
            self.current_thought = ("Feeling good!", "😊")
        elif self.state == CreatureState.FLOCKING:
            self.current_thought = ("Safety in numbers", "🐑")
        elif self.state == CreatureState.STALKING:
            self.current_thought = ("Patience...", "🤫")
        else:
            self.current_thought = None
    
    def check_achievements(self) -> List[str]:
        """Check and unlock achievements. Returns list of newly unlocked."""
        unlocked = []
        for ach in self.achievements:
            if ach.unlocked:
                continue
            
            # Check various achievement conditions
            if ach.name == "First Blood" and self.kills >= 1:
                if ach.check(self.kills):
                    unlocked.append(ach.name)
            elif ach.name == "Hunter" and self.kills >= 10:
                if ach.check(self.kills):
                    unlocked.append(ach.name)
            elif ach.name == "Apex Predator" and self.kills >= 50:
                if ach.check(self.kills):
                    unlocked.append(ach.name)
            elif ach.name == "Survivor" and self.age >= 1000:
                if ach.check(self.age):
                    unlocked.append(ach.name)
            elif ach.name == "Elder" and self.age >= 2500:
                if ach.check(self.age):
                    unlocked.append(ach.name)
            elif ach.name == "Parent" and self.children >= 5:
                if ach.check(self.children):
                    unlocked.append(ach.name)
            elif ach.name == "Forager" and self.food_eaten >= 100:
                if ach.check(int(self.food_eaten)):
                    unlocked.append(ach.name)
        
        return unlocked
    
    @property
    def heading(self) -> float:
        if self.velocity.magnitude > 0.01:
            return math.atan2(self.velocity.y, self.velocity.x)
        return 0.0
    
    @property
    def speed(self) -> float:
        return self.velocity.magnitude
    
    @property
    def life_stage(self) -> LifeStage:
        """Get current life stage based on age."""
        if self.age < 200:
            return LifeStage.BABY
        elif self.age < 500:
            return LifeStage.JUVENILE
        elif self.age < 2500:
            return LifeStage.ADULT
        else:
            return LifeStage.ELDER
    
    @property
    def life_stage_modifier(self) -> Dict[str, float]:
        """Get stat modifiers based on life stage."""
        stage = self.life_stage
        if stage == LifeStage.BABY:
            return {
                'speed': 0.7,
                'damage': 0.3,
                'learning': 2.0,  # Learn fast!
                'vision': 0.6,
                'size': 0.5,
            }
        elif stage == LifeStage.JUVENILE:
            return {
                'speed': 0.9,
                'damage': 0.7,
                'learning': 1.5,
                'vision': 0.85,
                'size': 0.75,
            }
        elif stage == LifeStage.ADULT:
            return {
                'speed': 1.0,
                'damage': 1.0,
                'learning': 1.0,
                'vision': 1.0,
                'size': 1.0,
            }
        else:  # ELDER
            return {
                'speed': 0.75,
                'damage': 0.9,
                'learning': 0.5,
                'vision': 0.8,
                'size': 1.1,  # Slightly larger
            }
    
    @property
    def effective_speed(self) -> float:
        """Get speed modified by life stage."""
        return self.max_speed * self.life_stage_modifier['speed']
    
    @property
    def effective_size(self) -> float:
        """Get size modified by life stage."""
        return self.size * self.life_stage_modifier['size']
    
    def apply_force(self, force: Vector2) -> None:
        """Apply force with mass consideration."""
        self.acceleration = self.acceleration + force
    
    def seek(self, target: Vector2) -> Vector2:
        """Steering force toward target."""
        desired = target - self.position
        dist = desired.magnitude
        if dist > 0:
            desired = desired.normalized() * self.max_speed
            steer = desired - self.velocity
            if steer.magnitude > self.max_force:
                steer = steer.normalized() * self.max_force
            return steer
        return Vector2(0, 0)
    
    def flee(self, target: Vector2) -> Vector2:
        """Steering force away from target."""
        desired = self.position - target
        dist = desired.magnitude
        if dist > 0 and dist < self.vision_range:
            desired = desired.normalized() * self.max_speed
            steer = desired - self.velocity
            if steer.magnitude > self.max_force:
                steer = steer.normalized() * self.max_force
            return steer
        return Vector2(0, 0)
    
    def pursue(self, target: 'Creature', prediction: float = 1.0) -> Vector2:
        """Pursue a moving target."""
        future_pos = target.position + target.velocity * prediction
        return self.seek(future_pos)
    
    def evade(self, threat: 'Creature', prediction: float = 1.5) -> Vector2:
        """Evade a moving threat."""
        future_pos = threat.position + threat.velocity * prediction
        return self.flee(future_pos)
    
    def wander(self) -> Vector2:
        """Random wandering behavior with danger avoidance."""
        wander_radius = 30.0
        wander_distance = 60.0
        jitter = 0.3
        
        self.wander_angle += random.uniform(-jitter, jitter)
        
        circle_center = self.velocity.normalized() * wander_distance if self.velocity.magnitude > 0 else Vector2(1, 0) * wander_distance
        
        displacement = Vector2(
            math.cos(self.wander_angle) * wander_radius,
            math.sin(self.wander_angle) * wander_radius
        )
        
        wander_force = circle_center + displacement
        
        # Check for danger zones in memory and avoid them (prey only)
        if not self.is_predator:
            danger_level = self.memory.get_danger_level(self.position, 100)
            if danger_level > 0.2:
                # Get direction away from danger memories
                danger_mem = self.memory.get_nearest(self.position, 'danger', 100)
                if danger_mem:
                    avoid_force = self.flee(danger_mem.position) * danger_level * 2.0
                    wander_force = wander_force + avoid_force
        
        if wander_force.magnitude > self.max_force:
            wander_force = wander_force.normalized() * self.max_force
        
        return wander_force
    
    # =========================================================================
    # NEW: FLOCKING BEHAVIORS
    # =========================================================================
    
    def flock(self, neighbors: List['Creature']) -> Vector2:
        """Compute flocking behavior (separation, alignment, cohesion)."""
        if not neighbors:
            return Vector2(0, 0)
        
        separation = self._separation(neighbors) * 1.5
        alignment = self._alignment(neighbors) * 1.0
        cohesion = self._cohesion(neighbors) * 1.0
        
        return separation + alignment + cohesion
    
    def _separation(self, neighbors: List['Creature']) -> Vector2:
        """Steer away from nearby creatures."""
        steer = Vector2(0, 0)
        count = 0
        
        for other in neighbors:
            dist = self.position.distance_to(other.position)
            if 0 < dist < 30:  # Too close threshold
                diff = self.position - other.position
                diff = diff.normalized() / max(dist, 0.1)  # Weight by distance
                steer = steer + diff
                count += 1
        
        if count > 0:
            steer = steer / count
            if steer.magnitude > 0:
                steer = steer.normalized() * self.max_speed - self.velocity
                if steer.magnitude > self.max_force:
                    steer = steer.normalized() * self.max_force
        
        return steer
    
    def _alignment(self, neighbors: List['Creature']) -> Vector2:
        """Align velocity with neighbors."""
        avg_vel = Vector2(0, 0)
        count = 0
        
        for other in neighbors:
            dist = self.position.distance_to(other.position)
            if 0 < dist < 60:
                avg_vel = avg_vel + other.velocity
                count += 1
        
        if count > 0:
            avg_vel = avg_vel / count
            avg_vel = avg_vel.normalized() * self.max_speed
            steer = avg_vel - self.velocity
            if steer.magnitude > self.max_force:
                steer = steer.normalized() * self.max_force
            return steer
        
        return Vector2(0, 0)
    
    def _cohesion(self, neighbors: List['Creature']) -> Vector2:
        """Steer toward center of neighbors."""
        center = Vector2(0, 0)
        count = 0
        
        for other in neighbors:
            dist = self.position.distance_to(other.position)
            if 0 < dist < 80:
                center = center + other.position
                count += 1
        
        if count > 0:
            center = center / count
            return self.seek(center)
        
        return Vector2(0, 0)
    
    # =========================================================================
    # NEW: PACK HUNTING BEHAVIORS
    # =========================================================================
    
    def pack_hunt(self, allies: List['Creature'], target: 'Creature') -> Vector2:
        """Coordinated pack hunting behavior."""
        if not target or not target.alive:
            return self.wander()
        
        # Calculate encirclement positions
        num_allies = len(allies) + 1
        my_index = 0
        for i, ally in enumerate(allies):
            if ally.id < self.id:
                my_index += 1
        
        # Spread out around target
        angle_offset = (2 * math.pi / num_allies) * my_index
        target_future = target.position + target.velocity * 2.0  # Predict movement
        
        # Position to surround
        surround_dist = 50.0
        surround_pos = Vector2(
            target_future.x + math.cos(angle_offset) * surround_dist,
            target_future.y + math.sin(angle_offset) * surround_dist
        )
        
        # Move toward surround position, but pursue if close
        dist_to_target = self.position.distance_to(target.position)
        if dist_to_target < self.config.predator_attack_range * 1.5:
            return self.pursue(target) * 1.5
        else:
            return self.seek(surround_pos)
    
    def find_pack_allies(self, creatures: List['Creature']) -> List['Creature']:
        """Find nearby predator allies for pack hunting."""
        allies = []
        for other in creatures:
            if other.id == self.id or not other.alive:
                continue
            if other.is_predator:
                dist = self.position.distance_to(other.position)
                if dist < self.config.predator_pack_range:
                    allies.append(other)
        return allies

    def sense(self, creatures: List['Creature'], foods: List[FoodSource], is_day: bool) -> List[float]:
        """Generate sensory inputs for brain."""
        # Find nearest entities
        nearest_food_dist = self.vision_range
        nearest_food_angle = 0.0
        nearest_threat_dist = self.vision_range
        nearest_threat_angle = 0.0
        nearest_prey_dist = self.vision_range
        nearest_prey_angle = 0.0
        nearest_ally_dist = self.vision_range
        nearest_ally_angle = 0.0
        
        ally_count = 0
        enemy_count = 0
        
        vision = self.vision_range * (1.0 if is_day else 0.6)
        
        for other in creatures:
            if other.id == self.id or not other.alive:
                continue
            
            dist = self.position.distance_to(other.position)
            if dist > vision:
                continue
            
            angle = self.position.angle_to(other.position) - self.heading
            
            if self.is_predator:
                if not other.is_predator:
                    enemy_count += 1  # prey count for predator
                    if dist < nearest_prey_dist:
                        nearest_prey_dist = dist
                        nearest_prey_angle = angle
                        self.target = other
                else:
                    ally_count += 1
                    if dist < nearest_ally_dist:
                        nearest_ally_dist = dist
                        nearest_ally_angle = angle
            else:
                if other.is_predator:
                    enemy_count += 1  # predator is threat
                    if dist < nearest_threat_dist:
                        nearest_threat_dist = dist
                        nearest_threat_angle = angle
                        self.nearest_threat = other
                else:
                    ally_count += 1
                    if dist < nearest_ally_dist:
                        nearest_ally_dist = dist
                        nearest_ally_angle = angle
        
        # Find nearest food
        for food in foods:
            if food.is_depleted:
                continue
            dist = self.position.distance_to(food.position)
            if dist < nearest_food_dist:
                nearest_food_dist = dist
                nearest_food_angle = self.position.angle_to(food.position) - self.heading
                self.target_food = food
        
        # Normalize inputs to [0, 1]
        inputs = [
            self.health / self.max_health,
            self.energy / self.max_energy,
            self.stamina / self.max_stamina,
            1.0 - nearest_food_dist / self.vision_range,
            nearest_food_angle / math.pi,
            1.0 - nearest_threat_dist / self.vision_range,
            nearest_threat_angle / math.pi,
            1.0 - nearest_prey_dist / self.vision_range,
            nearest_prey_angle / math.pi,
            1.0 - nearest_ally_dist / self.vision_range,
            nearest_ally_angle / math.pi,
            min(1.0, ally_count / 5.0),
            min(1.0, enemy_count / 3.0),
            self.speed / self.max_speed,
            1.0 if is_day else 0.0,
            self.age / 3000.0,
            1.0 if self.attack_cooldown > 0 else 0.0,
            1.0 if self.reproduction_cooldown > 0 else 0.0,
            random.random() * 0.1,  # noise
            0.5,  # bias
        ]
        
        return inputs
    
    def think(self, inputs: List[float]) -> None:
        """Use brain to decide action - now with emotions and personality!"""
        outputs = self.brain.decide(inputs)
        
        # Get emotional modifiers
        mood = self.emotions.get_behavior_modifier()
        
        # Output interpretation
        move_forward = outputs[0]
        turn = (outputs[1] - 0.5) * 2
        
        if self.is_predator:
            hunt_desire = outputs[2] * mood['hunt_drive']
            attack_desire = outputs[3]
            rest_desire = outputs[4] * mood['rest_need']
            eat_desire = outputs[5]
            
            # PERSONALITY-DRIVEN DECISIONS (v2.3)
            
            # Patient predators with high stealth skill might choose to ambush
            should_ambush = (
                self.personality.patience > 0.6 and
                self.skills.stealth > 0.2 and
                self.stamina > 60 and
                self.emotions.hunger > 0.3 and
                self.emotions.hunger < 0.8 and  # Not desperate
                random.random() < self.personality.patience * 0.1
            )
            
            # Bold predators might stalk instead of chase
            should_stalk = (
                self.target and
                self.personality.boldness < 0.5 and  # Cautious
                self.skills.stealth > 0.15 and
                self.position.distance_to(self.target.position) > self.config.predator_attack_range * 2
            )
            
            # INSTINCTIVE HUNTING: Always hunt if prey is nearby and we're hungry!
            if self.target and self.energy < 80:
                dist_to_prey = self.position.distance_to(self.target.position)
                if dist_to_prey < self.config.predator_attack_range:
                    self.state = CreatureState.ATTACKING
                elif should_stalk and dist_to_prey > 100:
                    self.stalk_target = self.target
                    self.state = CreatureState.STALKING
                else:
                    self.state = CreatureState.HUNTING
            # Set up ambush if conditions are right
            elif should_ambush and not self.target:
                # Find a good ambush spot (near remembered hunting ground or food)
                hunting_spot = self.memory.get_nearest(self.position, 'hunting', 300)
                if hunting_spot:
                    self.ambush_position = hunting_spot.position
                else:
                    # Random spot near food sources
                    self.ambush_position = self.position.copy()
                self.ambush_timer = 200 + self.personality.patience * 100
                self.state = CreatureState.AMBUSHING
            # Brain-driven hunting for well-fed predators  
            elif self.target and hunt_desire > 0.3:
                if should_stalk:
                    self.stalk_target = self.target
                    self.state = CreatureState.STALKING
                else:
                    self.state = CreatureState.HUNTING
            elif self.target and attack_desire > 0.4 and self.attack_cooldown <= 0:
                self.state = CreatureState.ATTACKING
            elif rest_desire > 0.7 and self.stamina < 30:
                self.state = CreatureState.RESTING
            elif self.target_food and eat_desire > 0.4 and self.energy < 40:
                # Predators can scavenge when desperate
                self.state = CreatureState.FORAGING
            elif move_forward > 0.2:
                self.state = CreatureState.WANDERING
        else:
            # PREY DECISIONS (v2.3 - with alerts and personality)
            eat_desire = outputs[2]
            flee_desire = outputs[3] * mood['flee_urgency']
            rest_desire = outputs[4] * mood['rest_need']
            flock_desire = outputs[5] if len(outputs) > 5 else 0.5
            
            # Modify by personality
            flee_desire *= (1.0 + (1.0 - self.personality.boldness) * 0.5)
            flock_desire *= (1.0 + self.personality.sociability * 0.5)
            
            # Check if received alert from ally
            if self.received_alert and self.alert_source:
                # React to alert based on personality
                if self.personality.boldness < 0.7 or random.random() < 0.8:
                    self.state = CreatureState.FLEEING
                    self.times_fled += 1
                    # Flee away from alert source
                    self.nearest_threat = None  # Will flee from alert_source instead
            
            # INSTINCTIVE FLEEING: Always flee if predator is close!
            elif self.nearest_threat:
                dist_to_threat = self.position.distance_to(self.nearest_threat.position)
                
                # Should we alert others? (social prey)
                should_alert = (
                    self.personality.sociability > 0.5 and
                    self.alert_cooldown <= 0 and
                    len(self.nearby_allies) > 0 and
                    dist_to_threat < 120
                )
                
                if dist_to_threat < 80:  # Close danger - ALWAYS flee!
                    if should_alert:
                        self.state = CreatureState.ALERTING
                    else:
                        self.state = CreatureState.FLEEING
                        self.times_fled += 1
                    # Remember danger location
                    self.memory.remember(self.nearest_threat.position, 'danger', self.age)
                elif flee_desire > 0.25:  # Far danger - brain decides
                    if should_alert and random.random() < self.personality.sociability:
                        self.state = CreatureState.ALERTING
                    else:
                        self.state = CreatureState.FLEEING
                        self.times_fled += 1
                elif self.target_food and eat_desire > 0.4 and self.energy < 80:
                    self.state = CreatureState.FORAGING
                elif rest_desire > 0.7 and self.stamina < 30:
                    self.state = CreatureState.RESTING
                elif flock_desire > 0.5 and len(self.nearby_allies) > 0:
                    self.state = CreatureState.FLOCKING
                else:
                    self.state = CreatureState.WANDERING
            elif self.target_food and eat_desire > 0.4 and self.energy < 80:
                self.state = CreatureState.FORAGING
            elif rest_desire > 0.7 and self.stamina < 30:
                self.state = CreatureState.RESTING
            elif flock_desire > 0.6 and len(self.nearby_allies) > 2:
                self.state = CreatureState.FLOCKING
            # Curious prey might investigate interesting spots
            elif self.emotions.curiosity > 0.7 and random.random() < 0.01:
                food_mem = self.memory.get_nearest(self.position, 'food', 200)
                if food_mem:
                    self.investigate_target = food_mem.position
                    self.state = CreatureState.INVESTIGATING
                else:
                    self.state = CreatureState.WANDERING
            elif move_forward > 0.3:
                self.state = CreatureState.WANDERING
        
        # Track last action for logging
        self.last_action = self.state.name.lower()
        
        # Apply turning
        if abs(turn) > 0.1:
            perp = Vector2(-self.velocity.y, self.velocity.x)
            if perp.magnitude > 0:
                perp = perp.normalized() * turn * 0.15
                self.apply_force(perp)
    
    def act(self, creatures: List['Creature'], foods: List[FoodSource]) -> Optional['Creature']:
        """Execute current state behavior. Returns killed creature if any."""
        killed = None
        self.last_reward = 0.0
        
        if self.state == CreatureState.WANDERING:
            self.apply_force(self.wander())
            self._drain_stamina(0.2)
            self.last_action = 'move'
            
        elif self.state == CreatureState.FLOCKING:  # NEW
            if self.nearby_allies:
                flock_force = self.flock(self.nearby_allies)
                self.apply_force(flock_force * self.config.prey_flock_weight)
                self._drain_stamina(0.25)
                self.last_action = 'flock'
                self.last_reward = 0.1  # Small reward for staying with group
            else:
                self.state = CreatureState.WANDERING
            
        elif self.state == CreatureState.FORAGING:
            if self.target_food and not self.target_food.is_depleted:
                dist = self.position.distance_to(self.target_food.position)
                if dist < self.size + self.target_food.radius:
                    self.state = CreatureState.EATING
                else:
                    self.apply_force(self.seek(self.target_food.position))
                    self._drain_stamina(0.3)
                    self.last_action = 'move'
            else:
                self.state = CreatureState.WANDERING
                
        elif self.state == CreatureState.EATING:
            if self.target_food and not self.target_food.is_depleted:
                eaten = self.target_food.consume(1.5)
                self.energy = min(self.max_energy, self.energy + eaten * 0.8)
                self.food_eaten += eaten
                self.last_action = 'eat'
                self.last_reward = eaten * 0.1  # Reward for eating
                
                # Evolution points for eating (prey progress)
                if not self.is_predator and eaten > 0:
                    self.earn_evolution_points(eaten * 0.5)
                
                # Remember good food location
                if eaten > 0.5:
                    self.memory.remember(self.target_food.position, 'food', self.age)
                
                if self.energy >= self.max_energy * 0.95 or self.target_food.is_depleted:
                    self.state = CreatureState.WANDERING
            else:
                self.state = CreatureState.WANDERING
                
        elif self.state == CreatureState.FLEEING:
            if self.nearest_threat:
                flee_force = self.evade(self.nearest_threat) * 1.5
                
                # Evasion skill bonus (v2.3)
                flee_force = flee_force * self.skills.get_bonus('evasion')
                
                # Also flee from alert source if we received an alert
                if self.received_alert and self.alert_source:
                    alert_flee = self.flee(self.alert_source) * 0.8
                    flee_force = flee_force + alert_flee
                
                # Check memory for danger zones and add avoidance force
                danger_level = self.memory.get_danger_level(self.position, 80)
                if danger_level > 0.3:
                    # Already in dangerous area - flee faster!
                    flee_force = flee_force * (1.0 + danger_level * 0.5)
                    self.last_reward = -0.1  # Negative reward for being in danger zone
                
                self.apply_force(flee_force)
                # Speed boost when fleeing (enhanced by skill)
                flee_boost = self.config.prey_flee_boost * (1.0 + self.skills.evasion * 0.2)
                self.velocity = self.velocity * flee_boost
                if self.velocity.magnitude > self.max_speed * flee_boost:
                    self.velocity = self.velocity.normalized() * self.max_speed * flee_boost
                self._drain_stamina(1.2)
                self.last_action = 'flee'
                
                # Skill improvement for fleeing (v2.3)
                self.skills.improve('evasion', 0.005 * self.personality.learning_rate)
                
                if danger_level <= 0.3:
                    self.last_reward = 0.05  # Small reward for surviving
                    
                # Check if we escaped (threat far away now)
                dist_to_threat = self.position.distance_to(self.nearest_threat.position)
                if dist_to_threat > self.vision_range * 0.9:
                    self.skills.improve('evasion', 0.02)  # Bonus for successful escape
                    self.last_reward = 0.2
                    # Evolution points for successful escape!
                    self.earn_evolution_points(3)
            else:
                self.state = CreatureState.WANDERING
        
        elif self.state == CreatureState.PACK_HUNTING:  # NEW: Coordinated pack hunting
            if self.target and self.target.alive:
                allies = self.find_pack_allies(creatures)
                pack_force = self.pack_hunt(allies, self.target)
                self.apply_force(pack_force * 1.2)
                self._drain_stamina(0.7)
                self.last_action = 'hunt'
                
                # Check if close enough to attack
                dist = self.position.distance_to(self.target.position)
                if dist < self.config.predator_attack_range:
                    self.state = CreatureState.ATTACKING
            else:
                self.target = None
                self.state = CreatureState.WANDERING
                
        elif self.state == CreatureState.HUNTING:
            if self.target and self.target.alive:
                dist = self.position.distance_to(self.target.position)
                
                # Check if should switch to pack hunting
                allies = self.find_pack_allies(creatures)
                if len(allies) >= 2 and dist > self.config.predator_attack_range:
                    self.state = CreatureState.PACK_HUNTING
                    self.pack_target = self.target
                    # Share target with nearby allies
                    for ally in allies:
                        if ally.pack_target is None:
                            ally.pack_target = self.target
                elif dist < self.config.predator_attack_range:
                    self.state = CreatureState.ATTACKING
                else:
                    self.apply_force(self.pursue(self.target) * 1.3)
                    self._drain_stamina(0.8)
                    self.last_action = 'hunt'
            else:
                self.target = None
                self.state = CreatureState.WANDERING
                
        elif self.state == CreatureState.ATTACKING:
            if self.target and self.target.alive and self.attack_cooldown <= 0:
                dist = self.position.distance_to(self.target.position)
                if dist < self.config.predator_attack_range:
                    # Attack with pack bonus AND skill bonus!
                    allies = self.find_pack_allies(creatures)
                    pack_bonus = 1.0 + len(allies) * self.config.predator_pack_bonus
                    skill_bonus = self.skills.get_bonus('combat')
                    damage = self.config.predator_attack_damage * random.uniform(0.8, 1.2) * pack_bonus * skill_bonus
                    
                    # Prey's evasion skill reduces damage
                    if self.target.skills.evasion > 0:
                        evasion_reduction = 1.0 - (self.target.skills.evasion * 0.3)
                        damage *= evasion_reduction
                    
                    self.target.health -= damage
                    self._drain_stamina(5.0)
                    self.attack_cooldown = 25.0
                    self.last_action = 'attack'
                    
                    # SKILL IMPROVEMENTS (v2.3)
                    self.skills.improve('combat', 0.01)
                    self.skills.improve('hunting', 0.005)
                    self.target.skills.improve('evasion', 0.02)  # Prey learns from attacks
                    
                    # Prey remembers this location as dangerous!
                    self.target.memory.remember(self.position, 'danger', int(self.age))
                    self.target.last_reward = -0.5  # Negative reward for getting hit
                    
                    if self.target.health <= 0:
                        self.target.die()
                        killed = self.target
                        self.kills += 1
                        self.successful_hunts += 1
                        self.last_reward = 1.0  # Big reward for kill
                        self.energy = min(self.max_energy, self.energy + self.config.predator_kill_energy)
                        
                        # Evolution points for killing!
                        self.earn_evolution_points(15)
                        
                        # Big skill improvement for successful kill
                        self.skills.improve('hunting', 0.03)
                        self.skills.improve('combat', 0.02)
                        
                        # Remember successful hunting ground
                        self.memory.remember(self.position, 'hunting', int(self.age))
                        
                        self.target = None
                        self.state = CreatureState.WANDERING
            else:
                self.state = CreatureState.HUNTING if self.target else CreatureState.WANDERING
                
        elif self.state == CreatureState.RESTING:
            self.stamina = min(self.max_stamina, self.stamina + 2.0)
            self.velocity = self.velocity * 0.9
            if self.stamina > self.max_stamina * 0.8:
                self.state = CreatureState.WANDERING
        
        # =================================================================
        # NEW v2.3: ADVANCED PREDATOR BEHAVIORS
        # =================================================================
        
        elif self.state == CreatureState.AMBUSHING:
            """Predator lies in wait at a strategic location."""
            self.velocity = self.velocity * 0.1  # Almost stationary
            self.ambush_timer -= 1
            self._drain_stamina(-0.5)  # Actually recover stamina while waiting
            self.last_action = 'ambush'
            
            # Check for nearby prey
            for creature in creatures:
                if not creature.is_predator and creature.alive:
                    dist = self.position.distance_to(creature.position)
                    if dist < self.vision_range * 0.6:  # Prey getting close!
                        # POUNCE!
                        self.target = creature
                        self.state = CreatureState.ATTACKING
                        self.skills.improve('stealth', 0.02)
                        self.last_reward = 0.3  # Reward for successful ambush setup
                        break
            
            # Give up ambush after timeout or if too hungry
            if self.ambush_timer <= 0 or self.emotions.hunger > 0.8:
                self.state = CreatureState.HUNTING
                self.ambush_position = None
        
        elif self.state == CreatureState.STALKING:
            """Predator sneaks up on prey slowly."""
            if self.stalk_target and self.stalk_target.alive:
                dist = self.position.distance_to(self.stalk_target.position)
                
                # Move slowly toward prey
                stalk_speed = self.max_speed * 0.3 * (1.0 + self.skills.get_bonus('stealth') * 0.2)
                stalk_force = self.seek(self.stalk_target.position) * 0.5
                self.apply_force(stalk_force)
                
                # Limit speed while stalking
                if self.velocity.magnitude > stalk_speed:
                    self.velocity = self.velocity.normalized() * stalk_speed
                
                self._drain_stamina(0.2)
                self.last_action = 'stalk'
                
                # Close enough to pounce?
                if dist < self.config.predator_attack_range * 1.5:
                    self.target = self.stalk_target
                    self.state = CreatureState.ATTACKING
                    self.skills.improve('stealth', 0.02)
                    self.skills.improve('hunting', 0.01)
                
                # Prey detected us? Switch to chase
                if self.stalk_target.state == CreatureState.FLEEING:
                    self.target = self.stalk_target
                    self.state = CreatureState.HUNTING
                    
            else:
                self.stalk_target = None
                self.state = CreatureState.WANDERING
        
        # =================================================================
        # NEW v2.3: PREY ALERT/COMMUNICATION SYSTEM
        # =================================================================
        
        elif self.state == CreatureState.ALERTING:
            """Prey is warning others of danger."""
            # Brief pause to alert
            self.velocity = self.velocity * 0.3
            self.last_action = 'alert'
            
            # Alert nearby allies
            for creature in creatures:
                if not creature.is_predator and creature.alive and creature.id != self.id:
                    dist = self.position.distance_to(creature.position)
                    if dist < self.vision_range * 0.8:
                        creature.received_alert = True
                        creature.alert_source = self.nearest_threat.position if self.nearest_threat else self.position
                        creature.alert_cooldown = 50.0  # Alert lasts for a while
            
            self.alert_cooldown = 100.0  # Can't alert again too soon
            self.skills.improve('evasion', 0.01)  # Learn from danger
            
            # Now flee
            self.state = CreatureState.FLEEING
        
        elif self.state == CreatureState.INVESTIGATING:
            """Creature investigating an interesting location (from memory or curiosity)."""
            if hasattr(self, 'investigate_target') and self.investigate_target:
                dist = self.position.distance_to(self.investigate_target)
                if dist < 20:
                    # Arrived at location
                    self.skills.improve('foraging', 0.005)
                    self.state = CreatureState.WANDERING
                    self.investigate_target = None
                else:
                    self.apply_force(self.seek(self.investigate_target) * 0.8)
                    self._drain_stamina(0.3)
                    self.last_action = 'investigate'
            else:
                self.state = CreatureState.WANDERING
        
        # =================================================================
        # NEW v3.1: STAMPEDE, TERRITORY, AND HAZARD BEHAVIORS
        # =================================================================
        
        elif self.state == CreatureState.STAMPEDING:
            """Mass panic flee - prey run together in same direction."""
            if self.stampede_direction:
                # Run in stampede direction with nearby prey
                stampede_force = self.stampede_direction.normalized() * 2.0
                
                # Add slight randomness to prevent perfect lines
                stampede_force.x += random.uniform(-0.2, 0.2)
                stampede_force.y += random.uniform(-0.2, 0.2)
                
                self.apply_force(stampede_force)
                
                # Stampede speed bonus!
                self.velocity = self.velocity * 1.3
                if self.velocity.magnitude > self.max_speed * 1.5:
                    self.velocity = self.velocity.normalized() * self.max_speed * 1.5
                
                self._drain_stamina(1.5)
                self.last_action = 'stampede'
                
                # Stampede ends when stamina low or no longer scared
                if self.stamina < 20 or (not self.nearest_threat and random.random() < 0.02):
                    self.is_stampeding = False
                    self.stampede_direction = None
                    self.state = CreatureState.WANDERING
            else:
                self.is_stampeding = False
                self.state = CreatureState.FLEEING if self.nearest_threat else CreatureState.WANDERING
        
        elif self.state == CreatureState.DEFENDING_TERRITORY:
            """Predator defending their territory from intruders."""
            if self.territory and self.rival:
                dist = self.position.distance_to(self.rival.position)
                
                if dist < 30:
                    # Close enough - challenge or fight!
                    self.state = CreatureState.CHALLENGING
                else:
                    # Move toward intruder aggressively
                    self.apply_force(self.pursue(self.rival) * 1.2)
                    self._drain_stamina(0.6)
                    self.last_action = 'defend'
                    
                    # Roar/display (increases fear)
                    self.reputation.increase_fear(0.005)
            else:
                self.state = CreatureState.WANDERING
        
        elif self.state == CreatureState.CHALLENGING:
            """Dominance challenge between two predators."""
            if self.rival and self.rival.alive:
                # Calculate challenge outcome based on stats
                my_power = (
                    self.health / self.max_health * 0.3 +
                    self.stamina / self.max_stamina * 0.2 +
                    self.skills.combat * 0.3 +
                    self.reputation.fear_rating * 0.2
                )
                rival_power = (
                    self.rival.health / self.rival.max_health * 0.3 +
                    self.rival.stamina / self.rival.max_stamina * 0.2 +
                    self.rival.skills.combat * 0.3 +
                    self.rival.reputation.fear_rating * 0.2
                )
                
                # Some randomness
                my_power *= random.uniform(0.8, 1.2)
                rival_power *= random.uniform(0.8, 1.2)
                
                if my_power > rival_power * 1.2:
                    # Clear victory - rival backs down
                    self.reputation.increase_fear(0.1)
                    self.reputation.increase_respect(0.05)
                    self.skills.improve('combat', 0.02)
                    self.rival.state = CreatureState.FLEEING
                    self.rival.rival = None
                    self.rival = None
                    self.state = CreatureState.WANDERING
                elif rival_power > my_power * 1.2:
                    # Clear loss - we back down
                    self.rival.reputation.increase_fear(0.1)
                    self.state = CreatureState.FLEEING
                    self.rival = None
                else:
                    # Fight! Both take damage
                    damage = random.uniform(5, 15)
                    self.health -= damage
                    self.rival.health -= damage * random.uniform(0.8, 1.2)
                    self._drain_stamina(10)
                    self.skills.improve('combat', 0.01)
                    
                    # Keep fighting until one gives up
                    if self.health < 30 or self.stamina < 10:
                        self.state = CreatureState.FLEEING
                        self.rival = None
            else:
                self.rival = None
                self.state = CreatureState.WANDERING
        
        elif self.state == CreatureState.FLEEING_HAZARD:
            """Fleeing from environmental hazard."""
            if hasattr(self, 'hazard_position') and self.hazard_position:
                flee_force = self.flee(self.hazard_position) * 2.0
                self.apply_force(flee_force)
                
                # Max speed escape!
                if self.velocity.magnitude > self.max_speed * 1.3:
                    self.velocity = self.velocity.normalized() * self.max_speed * 1.3
                
                self._drain_stamina(1.0)
                self.last_action = 'flee_hazard'
                
                # Check if we're far enough from hazard
                dist = self.position.distance_to(self.hazard_position)
                if dist > 200:
                    self.hazard_position = None
                    self.state = CreatureState.WANDERING
            else:
                self.state = CreatureState.WANDERING
        
        return killed
    
    def learn_from_experience(self, creatures: List['Creature'], foods: List[FoodSource], 
                               is_day: bool) -> None:
        """Update Q-learning based on current state and reward."""
        # Calculate current state
        nearby_threats = 0
        nearby_allies = 0
        nearest_threat_dist = 999.0
        nearest_food_dist = 999.0
        nearest_ally_dist = 999.0
        
        for other in creatures:
            if other.id == self.id or not other.alive:
                continue
            dist = self.position.distance_to(other.position)
            
            if self.is_predator:
                # For predators: prey are targets, other predators are allies
                if not other.is_predator:
                    nearest_threat_dist = min(nearest_threat_dist, dist)  # Actually targets for predator
                else:
                    if dist < self.config.predator_pack_range:
                        nearby_allies += 1
                        nearest_ally_dist = min(nearest_ally_dist, dist)
            else:
                # For prey: predators are threats, other prey are allies
                if other.is_predator:
                    if dist < self.vision_range:
                        nearby_threats += 1
                        nearest_threat_dist = min(nearest_threat_dist, dist)
                else:
                    if dist < 100:
                        nearby_allies += 1
                        nearest_ally_dist = min(nearest_ally_dist, dist)
        
        # Find nearest food
        for food in foods:
            if not food.is_depleted:
                dist = self.position.distance_to(food.position)
                nearest_food_dist = min(nearest_food_dist, dist)
        
        # Generate state key
        new_state_key = PersistentLearner.get_state_key(
            self, nearby_threats, nearby_allies,
            nearest_threat_dist, nearest_food_dist, nearest_ally_dist, is_day
        )
        
        # Calculate reward
        reward = self.last_reward
        
        # Bonus rewards based on survival and reproduction
        if self.alive:
            reward += 0.01  # Small reward for surviving
        if self.energy > 75:
            reward += 0.02
        if self.health < 30:
            reward -= 0.1  # Penalty for low health
        
        # If we have a previous state, update Q-value
        if self.previous_state_key and self.previous_action:
            PersistentLearner.update_q_value(
                self.previous_state_key, self.previous_action,
                reward, new_state_key, self.is_predator
            )
            
            # Record experience for replay
            PersistentLearner.record_experience(
                self.previous_state_key, self.previous_action,
                reward, new_state_key, self.is_predator, not self.alive
            )
        
        # Save for next iteration
        self.previous_state_key = new_state_key
        self.previous_action = self.last_action
        self.cumulative_reward += reward
    
    def _drain_stamina(self, amount: float) -> None:
        """Consume stamina."""
        self.stamina = max(0, self.stamina - amount)
    
    def update(self, width: float, height: float, dt: float = 1.0) -> None:
        """Update physics and stats."""
        if not self.alive:
            return
        
        # Physics update
        self.velocity = self.velocity + self.acceleration
        if self.velocity.magnitude > self.max_speed:
            self.velocity = self.velocity.normalized() * self.max_speed
        
        old_pos = self.position.copy()
        self.position = self.position + self.velocity * dt
        self.acceleration = Vector2(0, 0)
        
        # Bounce off walls
        if self.position.x < self.size:
            self.position.x = self.size
            self.velocity.x *= -0.8
        elif self.position.x > width - self.size:
            self.position.x = width - self.size
            self.velocity.x *= -0.8
            
        if self.position.y < self.size:
            self.position.y = self.size
            self.velocity.y *= -0.8
        elif self.position.y > height - self.size:
            self.position.y = height - self.size
            self.velocity.y *= -0.8
        
        # Friction
        self.velocity = self.velocity * 0.98
        
        # Trail
        self.trail.append(self.position.copy())
        
        # Distance tracking
        self.distance_traveled += old_pos.distance_to(self.position)
        
        # Energy drain
        drain = self.config.predator_energy_drain if self.is_predator else self.config.prey_energy_drain
        drain += self.speed / self.max_speed * 0.02
        self.energy = max(0, self.energy - drain * dt)
        
        # Stamina regen
        if self.stamina < self.max_stamina:
            self.stamina = min(self.max_stamina, self.stamina + 0.3 * dt)
        
        # Cooldowns
        if self.attack_cooldown > 0:
            self.attack_cooldown -= dt
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= dt
        if self.alert_cooldown > 0:
            self.alert_cooldown -= dt
        if self.ambush_timer > 0:
            self.ambush_timer -= dt
        
        # Clear received alert after some time
        if self.received_alert and self.alert_cooldown <= 0:
            self.received_alert = False
            self.alert_source = None
        
        # UPDATE EMOTIONS (v2.3)
        self.emotions.update(dt, self)
        
        # Age
        self.age += dt
        
        # Evolution points for surviving (every 500 age)
        if int(self.age) % 500 == 0 and int(self.age) > 0:
            self.earn_evolution_points(5)
        
        # Death from starvation
        if self.energy <= 0:
            self.health -= 0.3 * dt
        
        # Death from old age (very slow)
        if self.age > 4000:
            self.health -= 0.1 * dt
        
        if self.health <= 0:
            self.die()
    
    def die(self) -> None:
        """Handle death."""
        self.alive = False
        self.state = CreatureState.DEAD
        self.health = 0
    
    def can_reproduce(self) -> bool:
        """Check if can reproduce."""
        threshold = self.config.predator_reproduction_energy if self.is_predator else self.config.prey_reproduction_energy
        return (
            self.alive and
            self.energy >= threshold and
            self.health > 60 and
            self.age > 150 and
            self.reproduction_cooldown <= 0
        )
    
    def reproduce(self, partner: 'Creature') -> Optional['Creature']:
        """Create offspring with partner - now with personality & genetic memory inheritance!"""
        if not self.can_reproduce() or not partner.can_reproduce():
            return None
        
        # Create child brain
        child_brain = CreatureBrain.crossover(self.brain, partner.brain)
        child_brain.mutate(self.config.mutation_rate, self.config.mutation_strength)
        
        # Position near parents
        offset = Vector2.random_unit() * 25
        child_pos = self.position + offset
        
        # Create child
        child = Creature(
            position=child_pos,
            is_predator=self.is_predator,
            brain=child_brain,
            generation=max(self.generation, partner.generation) + 1,
            config=self.config,
        )
        
        # INHERIT PERSONALITY (v2.3)
        child.personality = Personality.inherit(
            self.personality, 
            partner.personality,
            self.config.mutation_rate
        )
        
        # Inherit some skill aptitude (not actual skills - those must be learned)
        # But children of skilled parents learn faster
        avg_parent_skill = (
            (self.skills.hunting + partner.skills.hunting) / 2 +
            (self.skills.evasion + partner.skills.evasion) / 2
        ) / 2
        child.personality.learning_rate = min(1.0, child.personality.learning_rate + avg_parent_skill * 0.1)
        
        # GENETIC MEMORY INHERITANCE (v3.3)
        # Child inherits some of parents' important memories
        self._inherit_memories(child)
        partner._inherit_memories(child)
        
        # Inherit pack role tendency
        if self.pack_role == PackRole.ALPHA or partner.pack_role == PackRole.ALPHA:
            # Children of alphas have leadership potential
            if random.random() < 0.3:
                child.pack_role = PackRole.BETA  # Start as beta, can rise to alpha
        
        # Energy cost
        cost = 25.0
        self.energy -= cost
        partner.energy -= cost * 0.5
        
        self.children += 1
        partner.children += 1
        
        self.reproduction_cooldown = 100.0
        partner.reproduction_cooldown = 100.0
        
        return child
    
    def _inherit_memories(self, child: 'Creature') -> None:
        """Pass important memories to offspring (genetic memory)."""
        # Pass on knowledge of dangerous areas
        dangerous_spots = self.memory.get_danger_zones()
        for spot in dangerous_spots[:5]:  # Top 5 most dangerous spots
            if random.random() < 0.5:  # 50% chance to inherit each memory
                child.memory.add_predator_sighting(spot['position'], spot['threat_level'] * 0.7)
        
        # Pass on knowledge of good food/water areas
        good_spots = self.memory.get_good_resource_spots()
        for spot in good_spots[:3]:  # Top 3 resource spots
            if random.random() < 0.4:  # 40% chance to inherit
                child.memory.add_food_memory(spot['position'])
        
        # Inherited instincts from parents' experiences
        # Child starts with a baseline fear if parent has high fear
        if self.emotions.fear > 0.5:
            child.emotions.fear = min(0.3, self.emotions.fear * 0.3)  # Inherited caution
    
    def get_fitness(self) -> float:
        """Calculate fitness score."""
        fitness = 0.0
        fitness += self.age * 0.01
        fitness += self.food_eaten * 0.3
        fitness += self.children * 100
        fitness += self.kills * 50
        fitness += self.distance_traveled * 0.001
        if self.alive:
            fitness += 50
        return fitness


# =============================================================================
# WORLD
# =============================================================================

class World:
    """The simulation world."""
    
    def __init__(self, config: SimConfig):
        self.config = config
        self.width = config.width
        self.height = config.height
        
        # Time
        self.time = 0
        self.day = 0
        
        # Weather
        self.weather = Weather.CLEAR
        self.weather_timer = 0
        
        # Resources
        self.foods: List[FoodSource] = []
        self.waters: List[WaterSource] = []
        self._spawn_resources()
    
    def _spawn_resources(self) -> None:
        """Spawn initial resources."""
        for _ in range(self.config.food_count):
            self.foods.append(FoodSource(
                position=Vector2(
                    random.uniform(30, self.width - 30),
                    random.uniform(30, self.height - 30)
                ),
                amount=random.uniform(50, 100),
            ))
        
        for _ in range(self.config.water_count):
            self.waters.append(WaterSource(
                position=Vector2(
                    random.uniform(50, self.width - 50),
                    random.uniform(50, self.height - 50)
                ),
            ))
    
    @property
    def is_day(self) -> bool:
        cycle_pos = (self.time % self.config.day_length) / self.config.day_length
        return 0.25 < cycle_pos < 0.75
    
    @property
    def light_level(self) -> float:
        cycle_pos = (self.time % self.config.day_length) / self.config.day_length
        if 0.25 < cycle_pos < 0.75:
            return 1.0
        elif cycle_pos < 0.25:
            return 0.3 + (cycle_pos / 0.25) * 0.7
        else:
            return 0.3 + ((1.0 - cycle_pos) / 0.25) * 0.7
    
    def update(self) -> None:
        """Update world state."""
        self.time += 1
        
        if self.time % self.config.day_length == 0:
            self.day += 1
        
        # Weather changes
        if random.random() < self.config.weather_change_rate:
            self.weather = random.choice(list(Weather))
        
        # Regenerate food
        for food in self.foods:
            food.regenerate()
        
        # Respawn depleted food occasionally
        if random.random() < self.config.food_regen_rate:
            depleted = [f for f in self.foods if f.is_depleted]
            if depleted:
                food = random.choice(depleted)
                food.amount = food.max_amount * 0.5


# =============================================================================
# RENDERER
# =============================================================================

class UltimateRenderer:
    """Beautiful rendering with all effects."""
    
    def __init__(self, width: int, height: int):
        pygame.init()
        pygame.display.set_caption("🧬 NOPAINNOGAIN - Ultimate Ecosystem v3.4 - Evolution Branches & Disasters")
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        self.font_tiny = pygame.font.Font(None, 16)
        
        self.show_trails = False
        self.show_vision = False
        self.show_stats = True
        self.show_heatmap = False  # NEW: Activity heatmap
        
        # Heatmap data
        self.heatmap_grid_size = 20
        self.heatmap: Dict[Tuple[int, int], float] = {}
        
        # Particles
        self.particles: List[Dict] = []
        
        # Population graph
        self.prey_history: deque = deque(maxlen=200)
        self.predator_history: deque = deque(maxlen=200)
        
        # NEW v3.2: Hover/Selection system
        self.hovered_creature: Optional[Creature] = None
        self.selected_creature: Optional[Creature] = None
        
        # NEW v3.3: Communication waves and burrows
        self.communication_waves: List[CommunicationWave] = []
        
        # Kill cam effect
        self.kill_cam_timer: int = 0
        self.kill_cam_pos: Optional[Vector2] = None
        
        # NEW v3.4: Thought bubbles toggle
        self.show_thoughts: bool = True
    
    def spawn_kill_particles(self, pos: Vector2) -> None:
        """EPIC blood splatter effect with shockwave."""
        # Blood splatter
        for _ in range(25):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            self.particles.append({
                'x': pos.x,
                'y': pos.y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': 50,
                'color': (random.randint(180, 220), random.randint(20, 50), random.randint(20, 50)),
                'size': random.uniform(2, 7),
            })
        
        # Skull indicator briefly
        self.particles.append({
            'x': pos.x,
            'y': pos.y - 20,
            'vx': 0,
            'vy': -1,
            'life': 60,
            'color': (255, 255, 255),
            'size': 12,
            'type': 'skull',
        })
        
        # Shockwave ring
        self.communication_waves.append(CommunicationWave(
            position=Vector2(pos.x, pos.y),
            wave_type='kill',
            color=(200, 50, 50),
            max_radius=100,
            speed=8,
        ))
        
        # Kill cam effect
        self.kill_cam_timer = 15
        self.kill_cam_pos = Vector2(pos.x, pos.y)
    
    def spawn_birth_particles(self, pos: Vector2, is_predator: bool) -> None:
        """Birth sparkle effect."""
        color = (255, 150, 150) if is_predator else (150, 255, 150)
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                'x': pos.x,
                'y': pos.y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': 30,
                'color': color,
                'size': random.uniform(2, 4),
            })
    
    def spawn_death_particles(self, pos: Vector2, is_predator: bool) -> None:
        """Death fade effect - ghost rising."""
        base_color = (150, 80, 80) if is_predator else (80, 150, 80)
        # Rising soul particles
        for _ in range(8):
            self.particles.append({
                'x': pos.x + random.uniform(-10, 10),
                'y': pos.y,
                'vx': random.uniform(-0.5, 0.5),
                'vy': random.uniform(-2, -0.5),  # Float upward
                'life': 50,
                'color': base_color,
                'size': random.uniform(3, 6),
            })
        # Scatter particles
        for _ in range(12):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                'x': pos.x,
                'y': pos.y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': 25,
                'color': (100, 100, 100),
                'size': random.uniform(1, 3),
            })
    
    def spawn_communication_wave(self, pos: Vector2, wave_type: str) -> None:
        """Spawn a visual communication wave."""
        colors = {
            'roar': (255, 100, 50),
            'alert': (255, 255, 0),
            'call': (100, 200, 255),
            'howl': (200, 150, 255),
            'frenzy': (255, 0, 0),
        }
        max_radii = {
            'roar': 200,
            'alert': 120,
            'call': 100,
            'howl': 250,
            'frenzy': 150,
        }
        self.communication_waves.append(CommunicationWave(
            position=pos.copy(),
            wave_type=wave_type,
            color=colors.get(wave_type, (255, 255, 255)),
            max_radius=max_radii.get(wave_type, 150),
        ))
    
    def update_communication_waves(self) -> None:
        """Update and clean up communication waves."""
        self.communication_waves = [w for w in self.communication_waves if w.update()]
    
    def _draw_communication_waves(self) -> None:
        """Draw all active communication waves."""
        for wave in self.communication_waves:
            if wave.alpha > 10:
                x, y = int(wave.position.x), int(wave.position.y)
                radius = int(wave.radius)
                surface = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(surface, (*wave.color, wave.alpha), (radius + 2, radius + 2), radius, 2)
                self.screen.blit(surface, (x - radius - 2, y - radius - 2))
    
    def _draw_burrows(self, burrows: List['Burrow']) -> None:
        """Draw prey burrows."""
        for burrow in burrows:
            x, y = int(burrow.position.x), int(burrow.position.y)
            # Hole effect
            pygame.draw.circle(self.screen, (40, 30, 20), (x, y), 12)
            pygame.draw.circle(self.screen, (30, 20, 10), (x, y), 8)
            # Show occupancy
            if burrow.occupants:
                # Little eyes peeking out
                for i, _ in enumerate(burrow.occupants[:3]):
                    offset = (i - 1) * 6
                    pygame.draw.circle(self.screen, (200, 200, 100), (x + offset - 2, y - 3), 2)
                    pygame.draw.circle(self.screen, (200, 200, 100), (x + offset + 2, y - 3), 2)
    
    def _draw_scent_marks(self, scent_marks: List['ScentMark']) -> None:
        """Draw territorial scent marks."""
        for scent in scent_marks:
            x, y = int(scent.position.x), int(scent.position.y)
            alpha = int(scent.strength * 100)
            radius = int(scent.radius * scent.strength)
            
            # Scent cloud effect
            surface = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
            # Gradient circles for scent cloud
            for r_offset in range(0, radius, 8):
                r = radius - r_offset
                a = max(5, int(alpha * (1 - r_offset / radius) * 0.5))
                pygame.draw.circle(surface, (*scent.color, a), (radius + 2, radius + 2), r)
            self.screen.blit(surface, (x - radius - 2, y - radius - 2))
            
            # Scent particles
            if scent.strength > 0.5 and random.random() < 0.05:
                self.particles.append({
                    'x': x + random.uniform(-radius, radius),
                    'y': y + random.uniform(-radius, radius),
                    'vx': random.uniform(-0.5, 0.5),
                    'vy': random.uniform(-1, -0.3),
                    'life': 20,
                    'color': (139, 90, 43),
                    'size': 2,
                })
    
    def _draw_hunting_formations(self, formations: List['HuntingFormation'], creatures: List['Creature']) -> None:
        """Draw hunting formation lines connecting pack members."""
        for formation in formations:
            if not formation.active:
                continue
            
            # Get leader and members
            leader = next((c for c in creatures if c.id == formation.leader_id and c.alive), None)
            if not leader:
                continue
            
            target = next((c for c in creatures if c.id == formation.target_id and c.alive), None)
            
            # Draw formation type indicator
            lx, ly = int(leader.position.x), int(leader.position.y)
            if formation.formation_type == 'surround':
                # Circle around target
                if target:
                    tx, ty = int(target.position.x), int(target.position.y)
                    pygame.draw.circle(self.screen, (255, 50, 50, 100), (tx, ty), 80, 1)
            elif formation.formation_type == 'chase':
                # Arrow pointing at target
                if target:
                    pygame.draw.line(self.screen, (255, 100, 50), (lx, ly), 
                                   (int(target.position.x), int(target.position.y)), 1)
            
            # Draw lines from leader to pack members
            for member_id in formation.member_ids:
                member = next((c for c in creatures if c.id == member_id and c.alive), None)
                if member:
                    mx, my = int(member.position.x), int(member.position.y)
                    pygame.draw.line(self.screen, (255, 150, 50), (lx, ly), (mx, my), 1)
                    # Small diamond at member position
                    pygame.draw.polygon(self.screen, (255, 200, 50), [
                        (mx, my - 5), (mx + 5, my), (mx, my + 5), (mx - 5, my)
                    ], 1)
    
    def _draw_disasters(self, disasters: List['NaturalDisaster']) -> None:
        """Draw natural disaster effects (v3.4) - OPTIMIZED."""
        for disaster in disasters:
            x, y = int(disaster.position.x), int(disaster.position.y)
            radius = int(disaster.radius)
            
            if disaster.disaster_type == 'earthquake':
                # Simple brown circles (no alpha)
                pygame.draw.circle(self.screen, (139, 90, 43), (x, y), radius, 3)
                pygame.draw.circle(self.screen, (100, 70, 30), (x, y), radius // 2, 2)
            
            elif disaster.disaster_type == 'meteor':
                # Simple fire circles
                pygame.draw.circle(self.screen, (255, 100, 0), (x, y), radius, 0)
                pygame.draw.circle(self.screen, (255, 200, 0), (x, y), radius // 2, 0)
                self._draw_text("☄️", x - 10, y - 10, self.font_small)
            
            elif disaster.disaster_type == 'volcanic':
                # Simple lava
                pygame.draw.circle(self.screen, (255, 100, 0), (x, y), radius, 3)
                pygame.draw.circle(self.screen, (100, 100, 100), (x, y), radius + 20, 2)
                self._draw_text("🌋", x - 10, y - 10, self.font_small)
            
            elif disaster.disaster_type == 'blizzard':
                # Simple ice circle
                pygame.draw.circle(self.screen, (200, 220, 255), (x, y), radius, 2)
                self._draw_text("❄️", x - 8, y - 8, self.font_small)
    
    def _draw_thought_bubbles(self, thought_bubbles: List['ThoughtBubble'], creatures: List['Creature']) -> None:
        """Draw thought bubbles above creatures (v3.4)."""
        if not self.show_thoughts:
            return
        
        for bubble in thought_bubbles:
            creature = next((c for c in creatures if c.id == bubble.creature_id and c.alive), None)
            if not creature:
                continue
            
            x = int(creature.position.x)
            y = int(creature.position.y - creature.size - 20)
            
            # Draw cloud bubble
            pygame.draw.ellipse(self.screen, (255, 255, 255), (x - 25, y - 12, 50, 20))
            pygame.draw.ellipse(self.screen, (200, 200, 200), (x - 25, y - 12, 50, 20), 1)
            
            # Little bubbles leading to creature
            pygame.draw.circle(self.screen, (255, 255, 255), (x - 8, y + 5), 4)
            pygame.draw.circle(self.screen, (255, 255, 255), (x - 3, y + 10), 3)
            
            # Thought text with emoji
            self._draw_text(bubble.emoji, x - 10, y - 10, self.font_tiny)
    
    def _draw_boss_indicator(self, creature: 'Creature') -> None:
        """Draw special indicator for boss creatures (v3.4)."""
        if not creature.is_boss or not creature.boss_data:
            return
        
        x, y = int(creature.position.x), int(creature.position.y)
        
        # Glowing aura
        pulse = abs(math.sin(pygame.time.get_ticks() / 200)) * 10
        glow_size = int(creature.size + 15 + pulse)
        color = (255, 215, 0) if creature.is_predator else (100, 255, 100)
        
        surface = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(surface, (*color, 80), (glow_size, glow_size), glow_size)
        pygame.draw.circle(surface, (*color, 150), (glow_size, glow_size), glow_size, 3)
        self.screen.blit(surface, (x - glow_size, y - glow_size))
        
        # Title above
        self._draw_text(f"👑 {creature.boss_data.title}", x - 40, y - creature.size - 35, self.font_small, color)
        
        # Health bar (larger for boss)
        bar_width = 60
        bar_height = 6
        health_ratio = creature.health / creature.max_health
        bar_y = y - creature.size - 15
        pygame.draw.rect(self.screen, (50, 50, 50), (x - bar_width//2, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, (200, 50, 50), (x - bar_width//2, bar_y, int(bar_width * health_ratio), bar_height))
        pygame.draw.rect(self.screen, color, (x - bar_width//2, bar_y, bar_width, bar_height), 1)
    
    def _draw_evolution_indicator(self, creature: 'Creature') -> None:
        """Draw evolution branch indicator (v3.4)."""
        if creature.evolution_branch == EvolutionBranch.NONE:
            return
        
        x, y = int(creature.position.x), int(creature.position.y)
        
        branch_icons = {
            EvolutionBranch.AMBUSHER: "🗡️",
            EvolutionBranch.PACK_LEADER: "👑",
            EvolutionBranch.BERSERKER: "💢",
            EvolutionBranch.STALKER: "👤",
            EvolutionBranch.SPEEDSTER: "⚡",
            EvolutionBranch.TANK: "🛡️",
            EvolutionBranch.SCOUT: "👁️",
            EvolutionBranch.SWARM_MIND: "🐝",
        }
        
        icon = branch_icons.get(creature.evolution_branch, "✨")
        self._draw_text(icon, x + creature.size + 2, y - creature.size, self.font_tiny)
    
    # =========================================================================
    # NEW v3.5: SEASONAL VISUALS, LEGENDARY AURAS, ARTIFACTS, EVENTS
    # =========================================================================
    
    def _draw_season_indicator(self, season: 'Season') -> None:
        """Draw season indicator in corner (v3.5)."""
        season_info = {
            Season.SPRING: ("🌸 SPRING", (255, 200, 220)),
            Season.SUMMER: ("☀️ SUMMER", (255, 220, 100)),
            Season.AUTUMN: ("🍂 AUTUMN", (200, 150, 100)),
            Season.WINTER: ("❄️ WINTER", (200, 220, 255))
        }
        
        name, color = season_info.get(season, ("", (255, 255, 255)))
        
        # Draw season box
        x, y = self.width - 120, 10
        pygame.draw.rect(self.screen, (30, 30, 40), (x, y, 110, 25), border_radius=5)
        pygame.draw.rect(self.screen, color, (x, y, 110, 25), 1, border_radius=5)
        self._draw_text(name, x + 8, y + 5, self.font_small)
    
    def _draw_legendary_aura(self, creature: 'Creature', legendary: 'LegendaryCreature') -> None:
        """Draw simple aura for legendary creatures (v3.5) - OPTIMIZED."""
        x, y = int(creature.position.x), int(creature.position.y)
        
        # Simple single ring (no alpha surfaces - much faster)
        size = int(creature.size + 15)
        pygame.draw.circle(self.screen, legendary.aura_color, (x, y), size, 3)
        pygame.draw.circle(self.screen, legendary.aura_color, (x, y), size + 8, 2)
        
        # Simple title
        self._draw_text(legendary.title, x - 40, y - creature.size - 25, self.font_small, legendary.aura_color)
    
    def _draw_artifacts(self, artifacts: List['Artifact']) -> None:
        """Draw mystical artifacts on the map (v3.5) - OPTIMIZED."""
        for artifact in artifacts:
            x, y = int(artifact.position.x), int(artifact.position.y)
            
            # Simple glow (no alpha surfaces)
            pygame.draw.circle(self.screen, artifact.glow_color, (x, y), int(artifact.radius), 2)
            pygame.draw.circle(self.screen, artifact.glow_color, (x, y), int(artifact.radius // 2), 1)
            
            # Draw artifact icon
            icons = {
                'ancient_bone': "🦴",
                'mystic_stone': "💎",
                'life_spring': "💚",
                'death_mark': "💀"
            }
            icon = icons.get(artifact.artifact_type, "✨")
            self._draw_text(icon, x - 8, y - 8, self.font_small)
    
    def _draw_world_events(self, events: List['WorldEvent']) -> None:
        """Draw active world event banners (v3.5)."""
        y_offset = 50
        for event in events:
            # Event banner
            banner_width = 250
            banner_height = 40
            x = (self.width - banner_width) // 2
            
            # Background
            surface = pygame.Surface((banner_width, banner_height), pygame.SRCALPHA)
            pygame.draw.rect(surface, (0, 0, 0, 180), (0, 0, banner_width, banner_height), border_radius=8)
            
            # Border glow
            progress = event.duration / max(1, event.duration + 100)
            glow_color = (255, int(200 * progress), int(100 * progress))
            pygame.draw.rect(surface, glow_color, (0, 0, banner_width, banner_height), 2, border_radius=8)
            
            self.screen.blit(surface, (x, y_offset))
            
            # Event name
            self._draw_text(event.name, x + 10, y_offset + 5, self.font_small)
            self._draw_text(f"({event.duration} steps left)", x + 10, y_offset + 22, self.font_tiny, (150, 150, 150))
            
            y_offset += 50
    
    def _draw_spectator_ui(self, creature: 'Creature') -> None:
        """Draw spectator mode UI when following a creature (v3.5)."""
        if not creature or not creature.alive:
            return
        
        # Darkened border effect
        border = 40
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.width, border))
        pygame.draw.rect(self.screen, (0, 0, 0), (0, self.height - border, self.width, border))
        
        # Creature info banner
        species = "🦁 PREDATOR" if creature.is_predator else "🐰 PREY"
        status = creature.state.name if creature.state else "UNKNOWN"
        
        info_text = f"📺 SPECTATING: {species} #{creature.id} | HP: {creature.health:.0f}/{creature.max_health:.0f} | Energy: {creature.energy:.0f} | State: {status}"
        self._draw_text(info_text, 10, 10, self.font_small, (255, 255, 255))
        
        # Draw target reticle on creature
        x, y = int(creature.position.x), int(creature.position.y)
        t = pygame.time.get_ticks() / 500
        size = creature.size + 10 + math.sin(t) * 3
        
        # Rotating brackets
        for i in range(4):
            angle = t + i * math.pi / 2
            bx = x + math.cos(angle) * size
            by = y + math.sin(angle) * size
            pygame.draw.line(self.screen, (255, 255, 0), (bx, by), 
                           (bx + math.cos(angle) * 8, by + math.sin(angle) * 8), 2)
    
    def _draw_mutation_glow(self, creature: 'Creature') -> None:
        """Draw special glow for mutated creatures."""
        if not creature.mutation:
            return
        
        x, y = int(creature.position.x), int(creature.position.y)
        glow_size = int(creature.size + 6 + math.sin(pygame.time.get_ticks() / 150) * 2)
        
        # Mutation-specific glow
        if creature.mutation.name == "Apex":
            # Rainbow aura
            hue = (pygame.time.get_ticks() / 10) % 360
            color = pygame.Color(0)
            color.hsva = (hue, 100, 100, 50)
            pygame.draw.circle(self.screen, color, (x, y), glow_size, 3)
        elif creature.mutation.name == "Ghost":
            # Fading ghost effect
            alpha = int(80 + math.sin(pygame.time.get_ticks() / 200) * 40)
            surface = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surface, (200, 200, 255, alpha), (glow_size, glow_size), glow_size)
            self.screen.blit(surface, (x - glow_size, y - glow_size))
        elif creature.mutation.name == "Berserker":
            # Red pulse
            pulse = abs(math.sin(pygame.time.get_ticks() / 100))
            pygame.draw.circle(self.screen, (255, int(50 * pulse), int(50 * pulse)), (x, y), glow_size, 2)
        elif creature.mutation.name == "Genius":
            # Golden sparkle
            pygame.draw.circle(self.screen, (255, 215, 0), (x, y), glow_size, 1)
            # Sparkle particles
            if random.random() < 0.1:
                self.particles.append({
                    'x': x + random.uniform(-glow_size, glow_size),
                    'y': y + random.uniform(-glow_size, glow_size),
                    'vx': 0, 'vy': -1,
                    'life': 15,
                    'color': (255, 215, 0),
                    'size': 2,
                })
        else:
            # Generic mutation glow
            pygame.draw.circle(self.screen, creature.mutation.color_modifier, (x, y), glow_size, 1)
    
    def update_hover(self, creatures: List['Creature'], mouse_pos: Tuple[int, int]) -> None:
        """Detect creature under mouse cursor."""
        self.hovered_creature = None
        mx, my = mouse_pos
        
        for creature in creatures:
            if not creature.alive:
                continue
            dx = creature.position.x - mx
            dy = creature.position.y - my
            if dx * dx + dy * dy < (creature.size + 5) ** 2:
                self.hovered_creature = creature
                break
    
    def handle_click(self, creatures: List['Creature'], mouse_pos: Tuple[int, int]) -> None:
        """Handle mouse click to select creature."""
        mx, my = mouse_pos
        self.selected_creature = None
        
        for creature in creatures:
            if not creature.alive:
                continue
            dx = creature.position.x - mx
            dy = creature.position.y - my
            if dx * dx + dy * dy < (creature.size + 5) ** 2:
                self.selected_creature = creature
                break
    
    def _draw_creature_tooltip(self, creature: 'Creature') -> None:
        """Draw detailed tooltip for hovered/selected creature."""
        x = int(creature.position.x)
        y = int(creature.position.y)
        
        # Tooltip positioning (above creature, or flip if near top)
        tooltip_width = 180
        tooltip_height = 200
        tooltip_x = x - tooltip_width // 2
        tooltip_y = y - creature.size - tooltip_height - 15
        
        if tooltip_y < 5:
            tooltip_y = y + creature.size + 15
        if tooltip_x < 5:
            tooltip_x = 5
        if tooltip_x + tooltip_width > self.width - 5:
            tooltip_x = self.width - tooltip_width - 5
        
        # Background
        surface = pygame.Surface((tooltip_width, tooltip_height), pygame.SRCALPHA)
        border_color = (200, 100, 100) if creature.is_predator else (100, 200, 100)
        surface.fill((20, 20, 30, 220))
        pygame.draw.rect(surface, border_color, (0, 0, tooltip_width, tooltip_height), 2)
        self.screen.blit(surface, (tooltip_x, tooltip_y))
        
        # Content
        ty = tooltip_y + 8
        tx = tooltip_x + 8
        
        # Type & ID
        type_str = "🦁 PREDATOR" if creature.is_predator else "🐰 PREY"
        self._draw_text(f"{type_str} #{creature.id % 1000}", tx, ty, self.font_small, border_color)
        ty += 18
        
        # Life Stage
        stage_icons = {
            LifeStage.BABY: "👶",
            LifeStage.JUVENILE: "🧒",
            LifeStage.ADULT: "🧑",
            LifeStage.ELDER: "👴",
        }
        stage_icon = stage_icons.get(creature.life_stage, "")
        self._draw_text(f"{stage_icon} {creature.life_stage.name} | Gen {creature.generation}", tx, ty, self.font_tiny)
        ty += 16
        
        # Health/Energy/Stamina bars
        self._draw_stat_bar(tx, ty, tooltip_width - 16, "HP", creature.health, creature.max_health, (200, 50, 50))
        ty += 14
        self._draw_stat_bar(tx, ty, tooltip_width - 16, "EN", creature.energy, creature.max_energy, (50, 200, 50))
        ty += 14
        self._draw_stat_bar(tx, ty, tooltip_width - 16, "ST", creature.stamina, creature.max_stamina, (50, 150, 200))
        ty += 18
        
        # Emotions (compact)
        emotions_str = f"😱{creature.emotions.fear:.1f} 🍖{creature.emotions.hunger:.1f} 😠{creature.emotions.aggression:.1f}"
        self._draw_text(emotions_str, tx, ty, self.font_tiny, (200, 200, 150))
        ty += 14
        
        # Personality (compact)
        pers = creature.personality
        pers_str = f"B:{pers.boldness:.1f} A:{pers.aggression:.1f} S:{pers.sociability:.1f}"
        self._draw_text(pers_str, tx, ty, self.font_tiny, (150, 200, 200))
        ty += 16
        
        # Top Skills
        skills = creature.skills
        if creature.is_predator:
            self._draw_text(f"🎯Hunt:{skills.hunting:.2f} ⚔️Combat:{skills.combat:.2f}", tx, ty, self.font_tiny, (255, 200, 150))
        else:
            self._draw_text(f"🏃Evade:{skills.evasion:.2f} 🌿Forage:{skills.foraging:.2f}", tx, ty, self.font_tiny, (150, 255, 200))
        ty += 16
        
        # Reputation
        rep = creature.reputation
        if creature.is_predator:
            self._draw_text(f"⭐Fear Rating: {rep.fear_rating:.2f}", tx, ty, self.font_tiny, (255, 150, 150))
        else:
            self._draw_text(f"⭐Respect: {rep.respect_rating:.2f}", tx, ty, self.font_tiny, (150, 255, 150))
        ty += 16
        
        # Special Ability
        if creature.special_ability != SpecialAbility.NONE:
            ability_icons = {
                SpecialAbility.NIGHT_VISION: "👁️",
                SpecialAbility.AMBUSH_MASTER: "🗡️",
                SpecialAbility.PACK_CALLER: "📯",
                SpecialAbility.BLOOD_FRENZY: "🩸",
                SpecialAbility.INTIMIDATING_ROAR: "🦁",
                SpecialAbility.CAMOUFLAGE: "🌿",
                SpecialAbility.BURROW_EXPERT: "🕳️",
                SpecialAbility.DANGER_SENSE: "⚠️",
                SpecialAbility.SPEED_BURST: "⚡",
                SpecialAbility.HERD_MIND: "🐑",
            }
            icon = ability_icons.get(creature.special_ability, "✨")
            ability_name = creature.special_ability.name.replace("_", " ").title()
            cooldown_str = f" ({int(creature.ability_cooldown)})" if creature.ability_cooldown > 0 else " ✓"
            self._draw_text(f"{icon} {ability_name}{cooldown_str}", tx, ty, self.font_tiny, (255, 215, 0))
            ty += 16
        
        # State
        state_name = creature.state.name.replace("_", " ")
        self._draw_text(f"State: {state_name}", tx, ty, self.font_tiny, (200, 200, 255))
        ty += 14
        
        # Kills/Age
        self._draw_text(f"Kills: {creature.kills} | Age: {creature.age}", tx, ty, self.font_tiny)
    
    def _draw_stat_bar(self, x: int, y: int, width: int, label: str, value: float, max_val: float, color: Tuple) -> None:
        """Draw a labeled stat bar."""
        ratio = min(1.0, value / max_val) if max_val > 0 else 0
        bar_width = width - 25
        
        # Label
        label_surf = self.font_tiny.render(label, True, (180, 180, 180))
        self.screen.blit(label_surf, (x, y))
        
        # Background
        pygame.draw.rect(self.screen, (50, 50, 50), (x + 22, y + 2, bar_width, 8))
        # Fill
        pygame.draw.rect(self.screen, color, (x + 22, y + 2, int(bar_width * ratio), 8))
    
    def update_particles(self) -> None:
        """Update particle positions and lifetimes."""
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1  # gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
    
    def update_heatmap(self, creatures: List['Creature']) -> None:
        """Update activity heatmap based on creature positions."""
        # Decay existing values
        for key in self.heatmap:
            self.heatmap[key] *= 0.98
        
        # Add creature activity
        for creature in creatures:
            if creature.alive:
                gx = int(creature.position.x // self.heatmap_grid_size)
                gy = int(creature.position.y // self.heatmap_grid_size)
                key = (gx, gy)
                self.heatmap[key] = self.heatmap.get(key, 0) + (0.5 if creature.is_predator else 0.3)
        
        # Clean up low values
        self.heatmap = {k: v for k, v in self.heatmap.items() if v > 0.1}
    
    def _draw_heatmap(self) -> None:
        """Draw activity heatmap overlay."""
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        max_val = max(self.heatmap.values()) if self.heatmap else 1
        
        for (gx, gy), value in self.heatmap.items():
            intensity = min(1.0, value / max_val)
            # Color: blue (low) -> yellow (medium) -> red (high)
            if intensity < 0.5:
                r = int(255 * intensity * 2)
                g = int(255 * intensity * 2)
                b = int(255 * (1 - intensity * 2))
            else:
                r = 255
                g = int(255 * (1 - (intensity - 0.5) * 2))
                b = 0
            
            alpha = int(50 + intensity * 100)
            rect = pygame.Rect(
                gx * self.heatmap_grid_size,
                gy * self.heatmap_grid_size,
                self.heatmap_grid_size,
                self.heatmap_grid_size
            )
            pygame.draw.rect(surface, (r, g, b, alpha), rect)
        
        self.screen.blit(surface, (0, 0))
    
    def _draw_hazards(self, hazards: List['EnvironmentalHazard']) -> None:
        """Draw environmental hazards."""
        for hazard in hazards:
            x, y = int(hazard.position.x), int(hazard.position.y)
            radius = int(hazard.radius)
            intensity = hazard.intensity
            
            if hazard.hazard_type == 'fire':
                # Flickering fire effect
                for i in range(3):
                    r = radius - i * 10
                    if r > 0:
                        alpha = int(100 + random.randint(-20, 20))
                        color = (255, 100 + i * 50, 0, alpha)
                        surface = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                        pygame.draw.circle(surface, color, (r, r), r)
                        self.screen.blit(surface, (x - r, y - r))
                # Fire particles
                if random.random() < 0.3:
                    self.particles.append({
                        'x': x + random.uniform(-radius/2, radius/2),
                        'y': y + random.uniform(-radius/2, radius/2),
                        'vx': random.uniform(-1, 1),
                        'vy': random.uniform(-3, -1),
                        'life': 20,
                        'color': (255, random.randint(100, 200), 0),
                        'size': random.uniform(2, 4),
                    })
            
            elif hazard.hazard_type == 'flood':
                # Blue water effect
                surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(surface, (50, 100, 200, 80), (radius, radius), radius)
                pygame.draw.circle(surface, (80, 150, 255, 60), (radius, radius), int(radius * 0.7))
                self.screen.blit(surface, (x - radius, y - radius))
                # Wave effect
                wave_offset = int(math.sin(pygame.time.get_ticks() / 200) * 5)
                pygame.draw.circle(self.screen, (100, 180, 255), (x + wave_offset, y), int(radius * 0.3), 2)
            
            elif hazard.hazard_type == 'disease':
                # Sickly green cloud
                surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(surface, (100, 150, 50, 60), (radius, radius), radius)
                pygame.draw.circle(surface, (150, 200, 50, 40), (radius, radius), int(radius * 0.6))
                self.screen.blit(surface, (x - radius, y - radius))
                # Toxic particles
                if random.random() < 0.1:
                    angle = random.uniform(0, 2 * math.pi)
                    dist = random.uniform(0, radius)
                    self.particles.append({
                        'x': x + math.cos(angle) * dist,
                        'y': y + math.sin(angle) * dist,
                        'vx': random.uniform(-0.5, 0.5),
                        'vy': random.uniform(-1, 0),
                        'life': 30,
                        'color': (100, 180, 50),
                        'size': random.uniform(1, 3),
                    })
    
    def _draw_territories(self, territories: List['Territory']) -> None:
        """Draw predator territories."""
        for territory in territories:
            x, y = int(territory.center.x), int(territory.center.y)
            radius = int(territory.radius)
            strength = territory.strength
            
            # Territory circle (faint red)
            alpha = int(30 + strength * 40)
            surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(surface, (200, 50, 50, alpha), (radius, radius), radius)
            pygame.draw.circle(surface, (255, 100, 100, alpha + 20), (radius, radius), radius, 2)
            self.screen.blit(surface, (x - radius, y - radius))
    
    def _draw_creature_highlight(self, creature: 'Creature', color: Tuple) -> None:
        """Draw highlight ring around creature."""
        x, y = int(creature.position.x), int(creature.position.y)
        size = creature.size + 8
        pygame.draw.circle(self.screen, color, (x, y), size, 2)
        # Pulsing effect
        pulse = abs(math.sin(pygame.time.get_ticks() / 200)) * 4
        pygame.draw.circle(self.screen, (*color[:3], 100), (x, y), int(size + pulse), 1)
    
    def render(self, world: World, creatures: List[Creature], stats: Dict,
               hazards: List['EnvironmentalHazard'] = None,
               territories: List['Territory'] = None,
               burrows: List['Burrow'] = None,
               scent_marks: List['ScentMark'] = None,
               hunting_formations: List['HuntingFormation'] = None,
               disasters: List['NaturalDisaster'] = None,
               thought_bubbles: List['ThoughtBubble'] = None,
               # v3.5 additions
               season: 'Season' = None,
               world_events: List['WorldEvent'] = None,
               artifacts: List['Artifact'] = None,
               legendary_creatures: List['LegendaryCreature'] = None,
               spectator_creature: 'Creature' = None) -> None:
        """Render everything."""
        hazards = hazards or []
        territories = territories or []
        burrows = burrows or []
        scent_marks = scent_marks or []
        hunting_formations = hunting_formations or []
        disasters = disasters or []
        thought_bubbles = thought_bubbles or []
        world_events = world_events or []
        artifacts = artifacts or []
        legendary_creatures = legendary_creatures or []
        
        # Background based on time of day
        bg_light = int(40 + world.light_level * 60)
        # Season tinting
        if season == Season.SPRING:
            bg_color = (bg_light, bg_light + 15, bg_light + 20)
        elif season == Season.SUMMER:
            bg_color = (bg_light + 10, bg_light + 10, bg_light + 20)
        elif season == Season.AUTUMN:
            bg_color = (bg_light + 15, bg_light + 5, bg_light)
        elif season == Season.WINTER:
            bg_color = (bg_light + 5, bg_light + 10, bg_light + 25)
        else:
            bg_color = (bg_light, bg_light + 10, bg_light + 30)
        self.screen.fill(bg_color)
        
        # Weather effects
        self._draw_weather(world)
        
        # Draw natural disasters (v3.4) - behind everything
        self._draw_disasters(disasters)
        
        # Heatmap (behind everything)
        if self.show_heatmap:
            self.update_heatmap(creatures)
            self._draw_heatmap()
        
        # Draw hazards (v3.1)
        self._draw_hazards(hazards)
        
        # Draw territories (v3.1)
        self._draw_territories(territories)
        
        # Draw burrows (v3.3)
        self._draw_burrows(burrows)
        
        # Draw scent marks (v3.3)
        self._draw_scent_marks(scent_marks)
        
        # Resources
        self._draw_resources(world)
        
        # Trails
        if self.show_trails:
            self._draw_trails(creatures)
        
        # Draw hunting formations (v3.3)
        self._draw_hunting_formations(hunting_formations, creatures)
        
        # Draw mutation glows first (behind creatures)
        for creature in creatures:
            if creature.alive and creature.mutation:
                self._draw_mutation_glow(creature)
        
        # Draw boss indicators behind creatures (v3.4)
        for creature in creatures:
            if creature.alive and creature.is_boss:
                self._draw_boss_indicator(creature)
        
        # Creatures
        for creature in creatures:
            if creature.alive:
                self._draw_creature(creature, world)
                # Draw evolution indicator (v3.4)
                if creature.evolution_branch:
                    self._draw_evolution_indicator(creature)
        
        # Communication waves (v3.3)
        self.update_communication_waves()
        self._draw_communication_waves()
        
        # Draw thought bubbles (v3.4)
        if self.show_thoughts:
            self._draw_thought_bubbles(thought_bubbles, creatures)
        
        # Highlight selected/hovered creature
        if self.selected_creature and self.selected_creature.alive:
            self._draw_creature_highlight(self.selected_creature, (255, 255, 0))
        elif self.hovered_creature and self.hovered_creature.alive:
            self._draw_creature_highlight(self.hovered_creature, (200, 200, 200))
        
        # Particles
        self.update_particles()
        self._draw_particles()
        
        # Kill cam effect (screen shake + flash)
        self._draw_kill_cam_effect()
        
        # NEW v3.5: Draw artifacts
        self._draw_artifacts(artifacts)
        
        # NEW v3.5: Draw legendary creature auras
        for leg_data in legendary_creatures:
            leg_creature = next((c for c in creatures if c.id == leg_data.creature_id and c.alive), None)
            if leg_creature:
                self._draw_legendary_aura(leg_creature, leg_data)
        
        # UI
        if self.show_stats:
            self._draw_ui(world, creatures, stats)
        
        # NEW v3.5: Draw season indicator
        if season:
            self._draw_season_indicator(season)
        
        # NEW v3.5: Draw world event banners
        self._draw_world_events(world_events)
        
        # NEW v3.5: Spectator mode UI
        if spectator_creature:
            self._draw_spectator_ui(spectator_creature)
        
        # Tooltip for hovered/selected creature (draw last, on top)
        if self.selected_creature and self.selected_creature.alive:
            self._draw_creature_tooltip(self.selected_creature)
        elif self.hovered_creature and self.hovered_creature.alive:
            self._draw_creature_tooltip(self.hovered_creature)
        
        pygame.display.flip()
    
    def _draw_weather(self, world: World) -> None:
        """Draw weather effects."""
        if world.weather == Weather.RAIN:
            for _ in range(50):
                x = random.randint(0, self.width)
                y = random.randint(0, self.height)
                pygame.draw.line(self.screen, (100, 100, 150), (x, y), (x - 2, y + 10), 1)
        elif world.weather == Weather.FOG:
            fog_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            fog_surface.fill((200, 200, 200, 50))
            self.screen.blit(fog_surface, (0, 0))
        elif world.weather == Weather.STORM:
            for _ in range(80):
                x = random.randint(0, self.width)
                y = random.randint(0, self.height)
                pygame.draw.line(self.screen, (150, 150, 200), (x, y), (x - 4, y + 15), 1)
            if random.random() < 0.02:
                pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, self.width, self.height))
    
    def _draw_resources(self, world: World) -> None:
        """Draw food and water."""
        for food in world.foods:
            if not food.is_depleted:
                alpha = int(100 + (food.amount / food.max_amount) * 155)
                size = int(4 + (food.amount / food.max_amount) * 8)
                pygame.draw.circle(
                    self.screen,
                    (50, min(255, 100 + alpha), 50),
                    (int(food.position.x), int(food.position.y)),
                    size
                )
        
        for water in world.waters:
            pygame.draw.circle(
                self.screen,
                (50, 100, 200),
                (int(water.position.x), int(water.position.y)),
                int(water.radius),
            )
            pygame.draw.circle(
                self.screen,
                (100, 150, 255),
                (int(water.position.x), int(water.position.y)),
                int(water.radius * 0.6),
            )
    
    def _draw_trails(self, creatures: List[Creature]) -> None:
        """Draw creature movement trails."""
        for creature in creatures:
            if len(creature.trail) < 2:
                continue
            points = [(int(p.x), int(p.y)) for p in creature.trail]
            color = (100, 50, 50) if creature.is_predator else (50, 100, 50)
            pygame.draw.lines(self.screen, color, False, points, 1)
    
    def _draw_creature(self, creature: Creature, world: World) -> None:
        """Draw a single creature with enhanced visuals."""
        # Skip hidden creatures (in burrow)
        if creature.state == CreatureState.HIDING:
            return
        
        x, y = int(creature.position.x), int(creature.position.y)
        
        # Vision cone (if enabled)
        if self.show_vision:
            vision_color = (80, 40, 40, 30) if creature.is_predator else (40, 80, 40, 30)
            self._draw_vision_cone(creature, vision_color)
        
        # Body
        size = int(creature.size * (0.8 + creature.health / creature.max_health * 0.4))
        
        # Frenzy effect (v3.3)
        if creature.is_frenzied:
            # Red pulsing aura
            pulse = abs(math.sin(pygame.time.get_ticks() / 50)) * 8
            pygame.draw.circle(self.screen, (255, 0, 0), (x, y), int(size + pulse + 5), 2)
        
        # Sleeping effect (v3.3)
        if creature.state == CreatureState.SLEEPING:
            # Draw Zzz
            self._draw_text("💤", x + size, y - size - 5, self.font_tiny)
        
        # Glow effect for high generation creatures
        if creature.generation >= 5:
            glow_intensity = min(creature.generation - 4, 10) * 10
            glow_color = (glow_intensity + 100, 50, 50) if creature.is_predator else (50, glow_intensity + 100, 50)
            pygame.draw.circle(self.screen, glow_color, (x, y), size + 3, 2)
        
        pygame.draw.circle(self.screen, creature.color, (x, y), size)
        
        # Pack role indicator (v3.3) - crown for alpha, dot for beta
        if creature.pack_role == PackRole.ALPHA:
            # Crown
            pygame.draw.polygon(self.screen, (255, 215, 0), [
                (x - 6, y - size - 8),
                (x - 3, y - size - 14),
                (x, y - size - 10),
                (x + 3, y - size - 14),
                (x + 6, y - size - 8),
            ])
        elif creature.pack_role == PackRole.BETA:
            pygame.draw.circle(self.screen, (192, 192, 192), (x, y - size - 10), 3)
        
        # Outline based on state (v3.0 - more states!)
        outline_color = (255, 255, 255)
        if creature.state == CreatureState.FLEEING:
            outline_color = (255, 255, 0)  # Yellow - scared
        elif creature.state == CreatureState.HUNTING:
            outline_color = (255, 100, 0)  # Orange - hunting
        elif creature.state == CreatureState.ATTACKING:
            outline_color = (255, 0, 0)    # Red - attacking
        elif creature.state == CreatureState.EATING:
            outline_color = (0, 255, 0)    # Green - eating
        elif creature.state == CreatureState.DRINKING:
            outline_color = (0, 150, 255)  # Blue - drinking
        elif creature.state == CreatureState.RESTING:
            outline_color = (150, 150, 255)  # Light blue - resting
        elif creature.state == CreatureState.AMBUSHING:
            outline_color = (128, 0, 128)  # Purple - ambushing
        elif creature.state == CreatureState.STALKING:
            outline_color = (75, 0, 130)   # Indigo - stalking
        elif creature.state == CreatureState.ALERTING:
            outline_color = (255, 165, 0)  # Orange - alerting
        elif creature.state == CreatureState.PACK_HUNTING:
            outline_color = (255, 50, 50)  # Bright red - pack hunting
        elif creature.state == CreatureState.FLOCKING:
            outline_color = (100, 255, 100)  # Light green - flocking
        elif creature.state == CreatureState.INVESTIGATING:
            outline_color = (0, 255, 255)  # Cyan - curious
        elif creature.state == CreatureState.FRENZY:
            outline_color = (255, 0, 50)   # Deep red - frenzy
        elif creature.state == CreatureState.LEADING_PACK:
            outline_color = (255, 215, 0)  # Gold - leading
        elif creature.state == CreatureState.SLEEPING:
            outline_color = (100, 100, 150)  # Dim - sleeping
        
        pygame.draw.circle(self.screen, outline_color, (x, y), size, 1)
        
        # Show ambush indicator (crouching predator)
        if creature.state == CreatureState.AMBUSHING:
            # Draw "..." above creature
            pygame.draw.circle(self.screen, (128, 0, 128), (x - 6, y - size - 12), 2)
            pygame.draw.circle(self.screen, (128, 0, 128), (x, y - size - 12), 2)
            pygame.draw.circle(self.screen, (128, 0, 128), (x + 6, y - size - 12), 2)
        
        # Show alert indicator (prey warning others)
        if creature.state == CreatureState.ALERTING:
            # Draw "!" above creature
            pygame.draw.line(self.screen, (255, 165, 0), (x, y - size - 15), (x, y - size - 8), 2)
            pygame.draw.circle(self.screen, (255, 165, 0), (x, y - size - 5), 2)
        
        # Special Ability visual effects (v3.3)
        if creature.special_ability != SpecialAbility.NONE:
            # Night vision - glowing eyes
            if creature.special_ability == SpecialAbility.NIGHT_VISION:
                pygame.draw.circle(self.screen, (0, 255, 0), (x - 3, y - 2), 2)
                pygame.draw.circle(self.screen, (0, 255, 0), (x + 3, y - 2), 2)
            # Speed burst active - lightning effect
            elif creature.has_active_effect(SpecialAbility.SPEED_BURST):
                for i in range(3):
                    offset = (pygame.time.get_ticks() // 50 + i * 5) % 20 - 10
                    pygame.draw.line(self.screen, (255, 255, 0), (x - offset, y + size), (x + offset, y + size + 10), 2)
            # Camouflage active - semi-transparent
            elif creature.has_active_effect(SpecialAbility.CAMOUFLAGE):
                # Draw leaves around creature
                for i in range(3):
                    angle = (pygame.time.get_ticks() / 500 + i * 2.1) % (2 * math.pi)
                    lx = x + math.cos(angle) * (size + 5)
                    ly = y + math.sin(angle) * (size + 5)
                    pygame.draw.circle(self.screen, (50, 150, 50), (int(lx), int(ly)), 3)
            # Danger sense - radar pulse
            elif creature.special_ability == SpecialAbility.DANGER_SENSE and not world.is_day:
                pulse = (pygame.time.get_ticks() // 100) % 30
                pygame.draw.circle(self.screen, (255, 200, 0), (x, y), pulse + size, 1)
        
        # Direction indicator
        heading = creature.heading
        end_x = x + math.cos(heading) * size * 1.5
        end_y = y + math.sin(heading) * size * 1.5
        pygame.draw.line(self.screen, outline_color, (x, y), (int(end_x), int(end_y)), 2)
        
        # Energy bar
        bar_width = size * 2
        bar_height = 3
        energy_ratio = creature.energy / creature.max_energy
        pygame.draw.rect(self.screen, (50, 50, 50), (x - bar_width//2, y - size - 8, bar_width, bar_height))
        pygame.draw.rect(self.screen, (50, 200, 50), (x - bar_width//2, y - size - 8, int(bar_width * energy_ratio), bar_height))
    
    def _draw_vision_cone(self, creature: Creature, color: Tuple) -> None:
        """Draw creature's vision cone."""
        x, y = int(creature.position.x), int(creature.position.y)
        heading = creature.heading
        vision_angle = math.pi * 0.7
        vision_range = creature.vision_range
        
        points = [(x, y)]
        for a in range(-15, 16):
            angle = heading + (a / 15) * (vision_angle / 2)
            px = x + math.cos(angle) * vision_range
            py = y + math.sin(angle) * vision_range
            points.append((int(px), int(py)))
        
        if len(points) > 2:
            surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.polygon(surface, (*color[:3], 30), points)
            self.screen.blit(surface, (0, 0))
    
    def _draw_particles(self) -> None:
        """Draw all particles."""
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            color = (*p['color'][:3],)
            
            # Special skull particle for kills
            if p.get('type') == 'skull':
                self._draw_text("💀", int(p['x']) - 8, int(p['y']), self.font_medium)
            else:
                pygame.draw.circle(
                    self.screen,
                    color,
                    (int(p['x']), int(p['y'])),
                    int(p['size'])
                )
    
    def _draw_kill_cam_effect(self) -> None:
        """Draw dramatic kill camera effect - REDUCED INTENSITY."""
        if self.kill_cam_timer <= 0:
            return
        
        self.kill_cam_timer -= 1
        
        # Screen shake effect by drawing red vignette
        intensity = self.kill_cam_timer / 15.0
        
        # Red flash overlay - MUCH SMALLER AND LESS INTENSE
        if self.kill_cam_timer > 12:
            flash_alpha = min(30, int((self.kill_cam_timer - 12) * 10))  # Max 30 alpha
            # Only flash edges, not whole screen
            edge_surface = pygame.Surface((self.width, 30), pygame.SRCALPHA)
            edge_surface.fill((255, 50, 50, flash_alpha))
            self.screen.blit(edge_surface, (0, 0))
            self.screen.blit(edge_surface, (0, self.height - 30))
        
        # Kill indicator at location - smaller
        if self.kill_cam_pos:
            x, y = int(self.kill_cam_pos.x), int(self.kill_cam_pos.y)
            # Small expanding circle
            radius = int((15 - self.kill_cam_timer) * 4)  # Smaller radius
            alpha = int(intensity * 100)
            if alpha > 0 and radius < 60:
                surface = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(surface, (255, 100, 100, alpha), (radius + 2, radius + 2), radius, 2)
                self.screen.blit(surface, (x - radius - 2, y - radius - 2))
    
    def _draw_ui(self, world: World, creatures: List[Creature], stats: Dict) -> None:
        """Draw UI overlay."""
        prey = [c for c in creatures if not c.is_predator and c.alive]
        predators = [c for c in creatures if c.is_predator and c.alive]
        
        # Update history
        self.prey_history.append(len(prey))
        self.predator_history.append(len(predators))
        
        # Stats panel
        panel_width = 200
        panel_height = 180
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 150))
        self.screen.blit(panel_surface, (10, 10))
        
        # Time info
        time_icon = "☀️" if world.is_day else "🌙"
        weather_icons = {
            Weather.CLEAR: "☀️",
            Weather.CLOUDY: "☁️",
            Weather.RAIN: "🌧️",
            Weather.STORM: "⛈️",
            Weather.FOG: "🌫️",
        }
        weather_icon = weather_icons.get(world.weather, "")
        
        y_offset = 15
        self._draw_text(f"Day {world.day} {time_icon} {weather_icon}", 20, y_offset, self.font_medium)
        y_offset += 25
        
        # Population
        self._draw_text(f"🐰 Prey: {len(prey)}", 20, y_offset, self.font_small, (200, 200, 100))
        y_offset += 20
        self._draw_text(f"🦁 Predators: {len(predators)}", 20, y_offset, self.font_small, (255, 100, 100))
        y_offset += 25
        
        # Stats
        self._draw_text(f"Kills: {stats['kills']}", 20, y_offset, self.font_small)
        y_offset += 20
        self._draw_text(f"Births: {stats['births']}", 20, y_offset, self.font_small)
        y_offset += 20
        
        # Best fitness
        if prey:
            best_prey = max(prey, key=lambda c: c.get_fitness())
            self._draw_text(f"Best Prey Gen: {best_prey.generation}", 20, y_offset, self.font_small, (150, 200, 150))
        y_offset += 20
        if predators:
            best_pred = max(predators, key=lambda c: c.get_fitness())
            self._draw_text(f"Best Pred Gen: {best_pred.generation}", 20, y_offset, self.font_small, (200, 150, 150))
        
        # Population graph
        self._draw_population_graph()
        
        # Controls hint
        self._draw_text("ESC:Quit  SPACE:Pause  T:Trails  V:Vision  H:Heatmap", 
                       self.width - 420, self.height - 25, self.font_small, (150, 150, 150))
    
    def _draw_text(self, text: str, x: int, y: int, font: pygame.font.Font, color: Tuple = (255, 255, 255)) -> None:
        """Draw text with shadow."""
        shadow = font.render(text, True, (0, 0, 0))
        self.screen.blit(shadow, (x + 1, y + 1))
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))
    
    def _draw_population_graph(self) -> None:
        """Draw mini population graph."""
        if len(self.prey_history) < 2:
            return
        
        graph_width = 150
        graph_height = 60
        graph_x = self.width - graph_width - 15
        graph_y = 15
        
        # Background
        graph_surface = pygame.Surface((graph_width, graph_height), pygame.SRCALPHA)
        graph_surface.fill((0, 0, 0, 150))
        self.screen.blit(graph_surface, (graph_x, graph_y))
        
        # Find max for scaling
        max_pop = max(max(self.prey_history, default=1), max(self.predator_history, default=1), 1)
        
        # Draw prey line
        if len(self.prey_history) > 1:
            points = []
            for i, pop in enumerate(self.prey_history):
                x = graph_x + (i / len(self.prey_history)) * graph_width
                y = graph_y + graph_height - (pop / max_pop) * graph_height
                points.append((int(x), int(y)))
            pygame.draw.lines(self.screen, (200, 200, 100), False, points, 2)
        
        # Draw predator line
        if len(self.predator_history) > 1:
            points = []
            for i, pop in enumerate(self.predator_history):
                x = graph_x + (i / len(self.predator_history)) * graph_width
                y = graph_y + graph_height - (pop / max_pop) * graph_height
                points.append((int(x), int(y)))
            pygame.draw.lines(self.screen, (255, 100, 100), False, points, 2)
    
    def quit(self) -> None:
        """Cleanup."""
        pygame.quit()


# =============================================================================
# SIMULATION
# =============================================================================

class UltimateSimulation:
    """The ultimate ecosystem simulation with full data logging and persistent learning."""
    
    def __init__(self, config: SimConfig = None):
        self.config = config or SimConfig()
        self.world = World(self.config)
        self.renderer = UltimateRenderer(self.config.width, self.config.height)
        self.clock = pygame.time.Clock()
        
        # Initialize data logger
        self.logger = SimulationLogger(self.config.log_dir)
        
        # Load persistent learning knowledge
        PersistentLearner.load_knowledge()
        
        # Initialize stats BEFORE spawning population (used by mutations)
        self.stats = {
            'kills': 0,
            'births': 0,
            'deaths': 0,
            'max_generation_prey': 0,
            'max_generation_predator': 0,
            'kills_this_step': 0,
            'births_this_step': 0,
            'stampedes': 0,
            'territory_fights': 0,
            'hazards_spawned': 0,
            'mutations': 0,
            'frenzy_activations': 0,
            'burrow_uses': 0,
            'scent_marks': 0,
            'pack_hunts': 0,
        }
        
        self.creatures: List[Creature] = []
        self._spawn_initial_population()
        
        # NEW v3.1: Environmental hazards and territories
        self.hazards: List[EnvironmentalHazard] = []
        self.territories: List[Territory] = []
        
        # NEW v3.3: Burrows for prey
        self.burrows: List[Burrow] = []
        self._spawn_initial_burrows()
        
        # NEW v3.3: Scent marks and hunting formations
        self.scent_marks: List[ScentMark] = []
        self.hunting_formations: List[HuntingFormation] = []
        
        # NEW v3.4: Natural disasters, boss creatures, thought bubbles
        self.disasters: List[NaturalDisaster] = []
        self.boss_creatures: List[BossCreature] = []
        self.thought_bubbles: List[ThoughtBubble] = []
        self.total_achievements_unlocked: int = 0
        
        # NEW v3.5: Seasons, families, legendaries, world events
        self.current_season: Season = Season.SPRING
        self.season_day_counter: int = 0
        self.days_per_season: int = 5  # Each season lasts 5 days
        self.family_trees: Dict[int, CreatureLineage] = {}
        self.relationships: List[CreatureRelationship] = []
        self.legendary_creatures: List[LegendaryCreature] = []
        self.world_events: List[WorldEvent] = []
        self.hall_of_legends: List[CreatureMemorial] = []
        self.artifacts: List[Artifact] = []
        self.spectator_mode: bool = False
        self.followed_creature: Optional[Creature] = None
        self.camera_offset: Vector2 = Vector2(0, 0)
        self.cinematic_moment: bool = False
        self.cinematic_timer: int = 0
        
        self.running = True
        self.paused = False
        self.step = 0
        
        # Dynamic difficulty tracking
        self.ecosystem_health = 1.0
        self.prey_predator_ratio_history: deque = deque(maxlen=50)
        
        # Learning stats
        self.learning_updates = 0
        self.last_replay_step = 0
        
        self._print_banner()
    
    def _print_banner(self) -> None:
        """Print startup banner."""
        learn_stats = PersistentLearner.get_stats()
        print("\n" + "=" * 70)
        print("  🧬 NOPAINNOGAIN - ULTIMATE EVOLUTIONARY ECOSYSTEM v3.5 🧬")
        print("=" * 70)
        print(f"  World: {self.config.width} x {self.config.height}")
        print(f"  Initial Prey: {self.config.initial_prey}")
        print(f"  Initial Predators: {self.config.initial_predators}")
        print(f"  Food Sources: {len(self.world.foods)}")
        print(f"  Water Sources: {len(self.world.waters)}")
        print("=" * 70)
        print("  🧠 Neural Network Brains: 20 inputs → 32 → 24 → 8 outputs")
        print("  🎯 Evolution: Crossover + Mutation on reproduction")
        print("  🌍 Day/Night cycle with weather effects")
        print("  🐺 Pack Hunting & 🐰 Flocking Behaviors")
        print("  🧠 Spatial Memory System")
        print("  📊 Full Data Logging for ML Analysis")
        print("  🎓 Persistent Q-Learning (learns across runs!)")
        print("=" * 70)
        print("  🆕 NEW in v3.0:")
        print("     😱 Emotions | 🎭 Personality | 📈 Skills")
        print("     🥷 Ambush & Stalking | 📢 Alert Communication")
        print("  🆕 NEW in v3.1:")
        print("     � Life Stages (Baby→Juvenile→Adult→Elder)")
        print("     ⭐ Reputation System (Fear & Respect)")
        print("     � Territory Control & Dominance Fights")
        print("     � Prey Stampedes")
        print("     🔥 Environmental Hazards (Fire, Flood, Disease)")
        print("  🆕 NEW in v3.3:")
        print("     👑 Pack Hierarchy (Alpha/Beta/Omega/Lone)")
        print("     🧬 Random Mutations with Visual Effects")
        print("     🕳️ Prey Burrows (Hiding Spots)")
        print("     📢 Communication Waves (Roar/Alert/Call)")
        print("     😴 Sleep & Frenzy States")
        print("     🦶 Territorial Scent Marking")
        print("     🎯 Coordinated Pack Hunt Formations")
        print("     🧠 Genetic Memory Inheritance")
        print("     ⚡ Special Abilities (10 types!)")
        print("  🆕 NEW in v3.4:")
        print("     🌿 Evolution Branches (8 specializations)")
        print("     🏆 Achievement System (track accomplishments)")
        print("     🌋 Natural Disasters (Earthquake/Meteor/Volcanic/Blizzard)")
        print("     👑 Boss Creatures (rare powerful spawns)")
        print("     💭 Creature Thoughts (see what they think)")
        print("  🆕 NEW in v3.5:")
        print("     � Four Seasons (Spring/Summer/Autumn/Winter)")
        print("     👨‍👩‍👧 Family Trees & Dynasties")
        print("     ⚔️ Legendary Creatures (Phoenix/Shadow Wolf/Storm Bringer)")
        print("     🌙 World Events (Blood Moon/Migration/Famine)")
        print("     🏺 Mystical Artifacts & Power Zones")
        print("     🪦 Hall of Legends (Creature Memorials)")
        print("     📺 Spectator Mode (F to follow creature)")
        print("     🎬 Cinematic Camera for epic moments")
        print("=" * 70)
        print(f"  📚 Loaded Knowledge:")
        print(f"     Predator states: {learn_stats['predator_states']}")
        print(f"     Prey states: {learn_stats['prey_states']}")
        print(f"     Experiences: {learn_stats['experiences']}")
        print("=" * 70)
        print("  Controls:")
        print("    ESC   - Quit          SPACE - Pause")
        print("    T     - Trails        V     - Vision cones")
        print("    H     - Heatmap       B     - Thought bubbles")
        print("    F     - Follow mode   L     - Hall of Legends")
        print("    N     - Next season   E     - Trigger event")
        print("=" * 70 + "\n")
    
    def _spawn_initial_population(self) -> None:
        """Spawn starting creatures."""
        for _ in range(self.config.initial_prey):
            pos = Vector2(
                random.uniform(50, self.world.width - 50),
                random.uniform(50, self.world.height - 50)
            )
            creature = Creature(
                position=pos,
                is_predator=False,
                config=self.config,
            )
            if creature.mutation:
                self.stats['mutations'] += 1
            self.creatures.append(creature)
        
        for _ in range(self.config.initial_predators):
            pos = Vector2(
                random.uniform(50, self.world.width - 50),
                random.uniform(50, self.world.height - 50)
            )
            creature = Creature(
                position=pos,
                is_predator=True,
                config=self.config,
            )
            if creature.mutation:
                self.stats['mutations'] += 1
            self.creatures.append(creature)
        
        # Assign initial pack roles for predators
        self._update_pack_hierarchy()
    
    def _spawn_initial_burrows(self) -> None:
        """Spawn burrows around the map for prey to hide in."""
        num_burrows = 15
        for _ in range(num_burrows):
            pos = Vector2(
                random.uniform(100, self.world.width - 100),
                random.uniform(100, self.world.height - 100)
            )
            self.burrows.append(Burrow(position=pos))
    
    def _update_pack_hierarchy(self) -> None:
        """Update predator pack roles based on kills and reputation."""
        predators = [c for c in self.creatures if c.is_predator and c.alive]
        if len(predators) < 2:
            return
        
        # Sort by fitness (kills + reputation)
        predators.sort(key=lambda p: p.kills + p.reputation.fear_rating * 10, reverse=True)
        
        # Reset all roles
        for p in predators:
            p.pack_role = PackRole.LONE
            p.pack_leader = None
        
        # Top predator is Alpha
        if predators:
            predators[0].pack_role = PackRole.ALPHA
            
        # Next few are Betas
        for p in predators[1:min(4, len(predators))]:
            p.pack_role = PackRole.BETA
            p.pack_leader = predators[0]
        
        # Weak ones are Omegas
        for p in predators[-2:]:
            if p.pack_role == PackRole.LONE:
                p.pack_role = PackRole.OMEGA
    
    def run(self) -> None:
        """Main loop."""
        while self.running:
            self._handle_events()
            
            if not self.paused:
                self._update()
            
            self._render()
            self.clock.tick(self.config.fps)
            self.step += 1
        
        self._cleanup()
    
    def _handle_events(self) -> None:
        """Handle input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("⏸️ PAUSED" if self.paused else "▶️ RESUMED")
                elif event.key == pygame.K_t:
                    self.renderer.show_trails = not self.renderer.show_trails
                elif event.key == pygame.K_v:
                    self.renderer.show_vision = not self.renderer.show_vision
                elif event.key == pygame.K_h:
                    self.renderer.show_heatmap = not self.renderer.show_heatmap
                elif event.key == pygame.K_b:
                    self.renderer.show_thoughts = not self.renderer.show_thoughts
                    print("💭 Thoughts: " + ("ON" if self.renderer.show_thoughts else "OFF"))
                # NEW v3.5 controls
                elif event.key == pygame.K_f:
                    # Follow selected creature
                    if self.renderer.selected_creature:
                        self.spectator_mode = not self.spectator_mode
                        if self.spectator_mode:
                            self.followed_creature = self.renderer.selected_creature
                            print(f"📺 Following {'Predator' if self.followed_creature.is_predator else 'Prey'} #{self.followed_creature.id}")
                        else:
                            self.followed_creature = None
                            print("📺 Spectator mode OFF")
                elif event.key == pygame.K_l:
                    # Show Hall of Legends
                    self._print_hall_of_legends()
                elif event.key == pygame.K_n:
                    # Force next season
                    self.season_day_counter = 0
                    self._update_seasons()
                elif event.key == pygame.K_e:
                    # Trigger random world event
                    self._trigger_world_event()
                elif event.key == pygame.K_1:
                    # Spawn legendary
                    self.stats['kills'] = 100  # Ensure threshold
                    self._spawn_legendary_creature()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.renderer.handle_click(self.creatures, event.pos)
                elif event.button == 3:  # Right click - deselect
                    self.renderer.selected_creature = None
                    self.spectator_mode = False
                    self.followed_creature = None
    
    def _print_hall_of_legends(self) -> None:
        """Print the Hall of Legends."""
        print("\n" + "=" * 60)
        print("  🪦 HALL OF LEGENDS 🪦")
        print("=" * 60)
        if not self.hall_of_legends:
            print("  No legendary creatures have fallen yet...")
        else:
            for i, memorial in enumerate(self.hall_of_legends[-10:], 1):  # Last 10
                title = f" ({memorial.legendary_title})" if memorial.legendary_title else ""
                print(f"  {i}. {memorial.name}{title}")
                print(f"     Kills: {memorial.kills} | Days: {memorial.survived_days:.1f}")
                print(f"     Cause: {memorial.cause_of_death}")
        print("=" * 60 + "\n")
    
    def _update(self) -> None:
        """Update simulation with data logging."""
        self.world.update()
        
        # Apply weather effects to creatures (v3.2)
        self._apply_weather_effects()
        
        # Reset per-step counters
        self.stats['kills_this_step'] = 0
        self.stats['births_this_step'] = 0
        
        prey = [c for c in self.creatures if not c.is_predator and c.alive]
        predators = [c for c in self.creatures if c.is_predator and c.alive]
        
        # Update nearby allies for all creatures (for flocking/pack hunting)
        self._update_nearby_allies()
        
        # Shuffle for fairness
        random.shuffle(self.creatures)
        
        # Update each creature
        for creature in self.creatures:
            if not creature.alive:
                continue
            
            # Decay memories
            creature.memory.decay_memories()
            
            # Sense
            inputs = creature.sense(self.creatures, self.world.foods, self.world.is_day)
            
            # Think
            creature.think(inputs)
            
            # Act
            killed = creature.act(self.creatures, self.world.foods)
            if killed:
                self.stats['kills'] += 1
                self.stats['kills_this_step'] += 1
                self.stats['deaths'] += 1
                self.renderer.spawn_kill_particles(killed.position)
                
                # Log the kill event
                if self.step % self.config.log_interval == 0:
                    self.logger.log_creature(self.step, killed, 'died', -1.0)
            
            # Physics update
            creature.update(self.world.width, self.world.height)
            
            # Learn from experience (Q-learning update)
            creature.learn_from_experience(self.creatures, self.world.foods, self.world.is_day)
            self.learning_updates += 1
            
            # Log creature state
            if self.step % self.config.log_interval == 0 and creature.alive:
                self.logger.log_creature(self.step, creature, creature.last_action, creature.last_reward)
            
            # Check death
            if not creature.alive:
                self.stats['deaths'] += 1
        
        # Experience replay learning (batch learning from stored experiences)
        if self.step - self.last_replay_step >= 50:
            PersistentLearner.replay_batch(batch_size=32)
            self.last_replay_step = self.step
        
        # Handle reproduction
        births = self._handle_reproduction()
        self.stats['births_this_step'] = births
        
        # Population balance with dynamic difficulty
        self._balance_population()
        self._adjust_difficulty()
        
        # Spawn death particles for creatures that just died
        for c in self.creatures:
            if not c.alive:
                self.renderer.spawn_death_particles(c.position, c.is_predator)
                self.stats['deaths'] += 1
        
        # Remove dead
        self.creatures = [c for c in self.creatures if c.alive]
        
        # Update generation stats
        for c in self.creatures:
            if c.is_predator:
                self.stats['max_generation_predator'] = max(
                    self.stats['max_generation_predator'], c.generation)
            else:
                self.stats['max_generation_prey'] = max(
                    self.stats['max_generation_prey'], c.generation)
        
        # Log species state
        if self.step % self.config.log_interval == 0:
            self.logger.log_species_state(
                self.step, self.creatures,
                self.stats['kills'],  # Cumulative total kills
                self.stats['births'],  # Cumulative total births
                self.world
            )
        
        # NEW v3.1: Update hazards, territories, and check for stampedes
        self._update_hazards()
        self._update_territories()
        self._check_stampedes()
        self._update_reputations()
        
        # NEW v3.3: Update scent marks and hunting formations
        self._update_scent_marks()
        self._update_hunting_formations()
        
        # NEW v3.4: Update disasters, achievements, thoughts, boss spawns
        self._update_disasters()
        self._update_achievements()
        self._update_thought_bubbles()
        self._spawn_boss_creature()
        
        # NEW v3.5: Update seasons, world events, legendaries, artifacts
        self._update_seasons()
        self._update_world_events()
        self._spawn_legendary_creature()
        self._spawn_artifact()
        
        # Status print
        if self.step % 100 == 0:
            self._print_status()
    
    def _update_nearby_allies(self) -> None:
        """Update nearby allies for all creatures."""
        prey = [c for c in self.creatures if not c.is_predator and c.alive]
        predators = [c for c in self.creatures if c.is_predator and c.alive]
        
        # For prey - find nearby prey for flocking
        for creature in prey:
            creature.nearby_allies = []
            for other in prey:
                if other.id != creature.id:
                    dist = creature.position.distance_to(other.position)
                    if dist < 80:  # Flock radius
                        creature.nearby_allies.append(other)
        
        # For predators - find nearby predators for pack hunting
        for creature in predators:
            creature.nearby_allies = []
            for other in predators:
                if other.id != creature.id:
                    dist = creature.position.distance_to(other.position)
                    if dist < self.config.predator_pack_range:
                        creature.nearby_allies.append(other)
    
    def _adjust_difficulty(self) -> None:
        """Dynamically adjust ecosystem parameters to prevent extinction."""
        prey_count = sum(1 for c in self.creatures if not c.is_predator and c.alive)
        pred_count = sum(1 for c in self.creatures if c.is_predator and c.alive)
        
        if pred_count == 0:
            return
        
        ratio = prey_count / max(pred_count, 1)
        self.prey_predator_ratio_history.append(ratio)
        
        # Calculate ecosystem health
        target = self.config.ecosystem_target_ratio
        deviation = abs(ratio - target) / target
        self.ecosystem_health = max(0, 1 - deviation)
        
        # Adjust parameters if needed
        if ratio < target * 0.5:  # Too few prey
            # Boost prey survival
            self.config.prey_reproduction_rate = min(0.15, self.config.prey_reproduction_rate * 1.01)
        elif ratio > target * 2:  # Too many prey
            # Boost predator effectiveness
            self.config.predator_reproduction_rate = min(0.08, self.config.predator_reproduction_rate * 1.01)

    def _handle_reproduction(self) -> int:
        """Handle breeding. Returns number of births."""
        new_creatures = []
        births = 0
        
        prey = [c for c in self.creatures if not c.is_predator and c.can_reproduce()]
        predators = [c for c in self.creatures if c.is_predator and c.can_reproduce()]
        
        # Prey reproduction
        for creature in prey:
            if random.random() > self.config.prey_reproduction_rate:
                continue
            
            # Find nearby partner
            partners = [
                c for c in prey
                if c.id != creature.id
                and c.position.distance_to(creature.position) < 60
            ]
            
            if partners:
                partner = random.choice(partners)
                child = creature.reproduce(partner)
                if child:
                    new_creatures.append(child)
                    self.stats['births'] += 1
                    births += 1
                    self.renderer.spawn_birth_particles(child.position, False)
        
        # Predator reproduction
        for creature in predators:
            if random.random() > self.config.predator_reproduction_rate:
                continue
            
            partners = [
                c for c in predators
                if c.id != creature.id
                and c.position.distance_to(creature.position) < 60
            ]
            
            if partners:
                partner = random.choice(partners)
                child = creature.reproduce(partner)
                if child:
                    new_creatures.append(child)
                    self.stats['births'] += 1
                    births += 1
                    self.renderer.spawn_birth_particles(child.position, True)
        
        self.creatures.extend(new_creatures)
        return births
    
    def _balance_population(self) -> None:
        """Keep populations in balance."""
        prey = [c for c in self.creatures if not c.is_predator and c.alive]
        predators = [c for c in self.creatures if c.is_predator and c.alive]
        
        # Respawn prey if too few
        if len(prey) < self.config.min_prey:
            for _ in range(self.config.min_prey - len(prey)):
                pos = Vector2(
                    random.uniform(50, self.world.width - 50),
                    random.uniform(50, self.world.height - 50)
                )
                self.creatures.append(Creature(
                    position=pos,
                    is_predator=False,
                    config=self.config,
                ))
        
        # Respawn predators if extinct and prey available
        # Reintroduce predators when they drop below minimum and prey exists
        if len(predators) < self.config.min_predators and len(prey) > self.config.min_prey:
            # Use best prey brain structure as inspiration (like natural adaptation)
            for _ in range(self.config.min_predators - len(predators)):
                pos = Vector2(
                    random.uniform(50, self.world.width - 50),
                    random.uniform(50, self.world.height - 50)
                )
                self.creatures.append(Creature(
                    position=pos,
                    is_predator=True,
                    config=self.config,
                ))
        
        # Cull excess if too many (keep fittest)
        if len(prey) > self.config.max_prey:
            prey.sort(key=lambda c: c.get_fitness(), reverse=True)
            for c in prey[self.config.max_prey:]:
                c.die()
        
        if len(predators) > self.config.max_predators:
            predators.sort(key=lambda c: c.get_fitness(), reverse=True)
            for c in predators[self.config.max_predators:]:
                c.die()
    
    # =========================================================================
    # NEW v3.1: Hazards, Territories, Stampedes, Reputations
    # =========================================================================
    
    def _update_hazards(self) -> None:
        """Update environmental hazards and spawn new ones."""
        # Update existing hazards
        self.hazards = [h for h in self.hazards if h.update()]
        
        # Randomly spawn new hazards (rare)
        if random.random() < 0.0005:  # About once every 2000 steps
            hazard_type = random.choice(['fire', 'flood', 'disease'])
            pos = Vector2(
                random.uniform(100, self.world.width - 100),
                random.uniform(100, self.world.height - 100)
            )
            self.hazards.append(EnvironmentalHazard(hazard_type, pos))
            self.stats['hazards_spawned'] += 1
        
        # Apply hazard damage to creatures
        for hazard in self.hazards:
            for creature in self.creatures:
                if not creature.alive:
                    continue
                
                damage = hazard.affects(creature.position)
                if damage > 0:
                    # Different hazard effects
                    if hazard.hazard_type == 'fire':
                        creature.health -= damage * 0.5
                        creature.energy -= damage * 0.3
                    elif hazard.hazard_type == 'flood':
                        creature._drain_stamina(damage * 2)
                        # Slow down in flood
                        creature.velocity = creature.velocity * (1.0 - damage * 0.3)
                    elif hazard.hazard_type == 'disease':
                        creature.health -= damage * 0.2
                        creature.max_stamina *= (1.0 - damage * 0.01)
                    
                    # Creature notices hazard and tries to flee
                    if damage > 0.3 and creature.state not in [CreatureState.FLEEING_HAZARD, CreatureState.FLEEING]:
                        creature.hazard_position = hazard.position
                        creature.state = CreatureState.FLEEING_HAZARD
    
    def _update_territories(self) -> None:
        """Update predator territories and handle intrusions."""
        predators = [c for c in self.creatures if c.is_predator and c.alive]
        
        # Strong adult predators can claim territory
        for pred in predators:
            if pred.life_stage == LifeStage.ADULT and pred.territory is None:
                # Check if this predator is strong enough to claim territory
                if pred.kills > 3 and pred.skills.combat > 0.2 and random.random() < 0.001:
                    # Claim territory around current position
                    pred.territory = Territory(
                        center=pred.position.copy(),
                        radius=120 + pred.reputation.fear_rating * 50,
                        owner_id=pred.id
                    )
                    self.territories.append(pred.territory)
        
        # Check for territory intrusions
        for pred in predators:
            if pred.territory:
                # Decay territory claim
                pred.territory.decay()
                if pred.territory.strength <= 0:
                    self.territories.remove(pred.territory)
                    pred.territory = None
                    continue
                
                # Check for intruders
                for other in predators:
                    if other.id == pred.id or other.rival is not None:
                        continue
                    
                    if pred.territory.contains(other.position):
                        # Intruder detected!
                        # Decide if worth fighting based on personality
                        if pred.personality.aggression > 0.4 and random.random() < 0.1:
                            pred.rival = other
                            other.rival = pred
                            pred.state = CreatureState.DEFENDING_TERRITORY
                            other.state = CreatureState.CHALLENGING
                            self.stats['territory_fights'] += 1
    
    def _check_stampedes(self) -> None:
        """Check if conditions are right for prey stampede."""
        prey = [c for c in self.creatures if not c.is_predator and c.alive]
        
        # Count panicked prey
        panicked = [p for p in prey if p.state == CreatureState.FLEEING and p.emotions.fear > 0.7]
        
        # Stampede triggers when many prey are panicking together
        if len(panicked) >= 5:
            # Check if they're close together
            center = Vector2(
                sum(p.position.x for p in panicked) / len(panicked),
                sum(p.position.y for p in panicked) / len(panicked)
            )
            
            nearby_panicked = [
                p for p in panicked 
                if p.position.distance_to(center) < 150
            ]
            
            if len(nearby_panicked) >= 4:
                # STAMPEDE!
                # Calculate stampede direction (away from nearest predator)
                predators = [c for c in self.creatures if c.is_predator and c.alive]
                if predators:
                    nearest_pred = min(predators, key=lambda p: p.position.distance_to(center))
                    stampede_dir = center - nearest_pred.position
                    if stampede_dir.magnitude > 0:
                        stampede_dir = stampede_dir.normalized()
                    else:
                        stampede_dir = Vector2.random_unit()
                else:
                    stampede_dir = Vector2.random_unit()
                
                # Trigger stampede in all nearby prey
                for p in prey:
                    if p.position.distance_to(center) < 200:
                        p.is_stampeding = True
                        p.stampede_direction = stampede_dir
                        p.state = CreatureState.STAMPEDING
                
                self.stats['stampedes'] += 1
    
    def _update_reputations(self) -> None:
        """Update creature reputations over time."""
        for creature in self.creatures:
            if creature.alive:
                creature.reputation.decay()
    
    def _update_scent_marks(self) -> None:
        """Update scent marks - decay old ones and create new ones."""
        # Decay existing scent marks
        self.scent_marks = [s for s in self.scent_marks if s.update()]
        
        # Predators leave scent marks
        for creature in self.creatures:
            if creature.alive and creature.should_mark_territory():
                mark = ScentMark(
                    position=Vector2(creature.position.x, creature.position.y),
                    owner_id=creature.id,
                    strength=1.0 if creature.pack_role == PackRole.ALPHA else 0.7,
                )
                self.scent_marks.append(mark)
                self.stats['scent_marks'] += 1
                
                # Prey react to scent marks
                for prey in self.creatures:
                    if not prey.is_predator and prey.alive:
                        dist = prey.position.distance_to(mark.position)
                        if dist < mark.radius * 2:
                            prey.emotions.fear = min(1.0, prey.emotions.fear + 0.1)
                            prey.memory.add_predator_sighting(mark.position, threat_level=0.5)
    
    def _update_hunting_formations(self) -> None:
        """Manage pack hunting formations."""
        # Remove inactive formations
        self.hunting_formations = [f for f in self.hunting_formations if f.active]
        
        # Check if alphas should start a hunt
        predators = [c for c in self.creatures if c.is_predator and c.alive]
        prey = [c for c in self.creatures if not c.is_predator and c.alive]
        
        for pred in predators:
            if pred.pack_role != PackRole.ALPHA:
                continue
            
            # Check if alpha is already in a formation
            in_formation = any(f.leader_id == pred.id for f in self.hunting_formations)
            if in_formation:
                continue
            
            # Alpha can start formation hunt if hungry and nearby pack members exist
            if pred.energy < 50 and len(pred.nearby_allies) >= 2:
                # Find nearest prey
                nearest_prey = None
                min_dist = float('inf')
                for p in prey:
                    dist = pred.position.distance_to(p.position)
                    if dist < 200 and dist < min_dist:
                        min_dist = dist
                        nearest_prey = p
                
                if nearest_prey:
                    # Create hunting formation
                    members = [a.id for a in pred.nearby_allies[:4] if a.can_join_hunt()]
                    if len(members) >= 2:
                        formation = HuntingFormation(
                            leader_id=pred.id,
                            member_ids=members,
                            target_id=nearest_prey.id,
                            formation_type=random.choice(['surround', 'chase', 'ambush']),
                        )
                        self.hunting_formations.append(formation)
                        self.stats['pack_hunts'] += 1
                        
                        # Signal the hunt
                        self.renderer.communication_waves.append(CommunicationWave(
                            position=Vector2(pred.position.x, pred.position.y),
                            wave_type='howl',
                            color=(200, 50, 50),
                            max_radius=180,
                        ))
        
        # Update active formations
        for formation in self.hunting_formations:
            # Check if target still alive
            target = next((c for c in self.creatures if c.id == formation.target_id and c.alive), None)
            if not target:
                formation.active = False
                continue
            
            # Check if leader still alive
            leader = next((c for c in self.creatures if c.id == formation.leader_id and c.alive), None)
            if not leader:
                formation.active = False
                continue
            
            # Guide pack members to formation positions
            for member_id in formation.member_ids:
                member = next((c for c in self.creatures if c.id == member_id and c.alive), None)
                if member:
                    ideal_pos = formation.get_formation_position(member_id, target.position, leader.position)
                    direction = ideal_pos - member.position
                    if direction.length() > 10:
                        member.velocity = member.velocity + direction.normalize() * 0.3
                        member.state = CreatureState.PACK_HUNTING
    
    def _update_disasters(self) -> None:
        """Update natural disasters and spawn new ones (v3.4) - REDUCED FREQUENCY."""
        # Update existing disasters
        self.disasters = [d for d in self.disasters if d.update()]
        
        # Very rare chance to spawn disaster (about every 20000 steps)
        # Only spawn if no active disasters
        if len(self.disasters) == 0 and random.random() < 0.00005:
            disaster_type = random.choice(['earthquake', 'blizzard'])  # Only mild disasters
            pos = Vector2(
                random.uniform(200, self.world.width - 200),
                random.uniform(200, self.world.height - 200)
            )
            
            # Smaller, shorter disasters
            if disaster_type == 'earthquake':
                disaster = NaturalDisaster(disaster_type, pos, 150, 0.5, 100, 100)
            else:  # blizzard
                disaster = NaturalDisaster(disaster_type, pos, 200, 0.4, 150, 150)
            
            self.disasters.append(disaster)
            print(f"⚠️  DISASTER: {disaster_type.upper()} at ({int(pos.x)}, {int(pos.y)})!")
        
        # Apply MINIMAL disaster effects to creatures
        for disaster in self.disasters:
            for creature in self.creatures:
                if not creature.alive:
                    continue
                
                effects = disaster.get_effect_at(creature.position)
                if not effects:
                    continue
                
                # Apply very reduced damage
                if 'damage' in effects:
                    creature.health -= effects['damage'] * 0.02  # 5x less damage
                
                # Apply minor speed penalty
                if 'speed_penalty' in effects:
                    creature.velocity = creature.velocity * (1.0 - effects['speed_penalty'] * 0.02)
    
    def _update_achievements(self) -> None:
        """Check achievements for all creatures (v3.4)."""
        for creature in self.creatures:
            if not creature.alive:
                continue
            
            unlocked = creature.check_achievements()
            for ach_name in unlocked:
                self.total_achievements_unlocked += 1
                # Earn evolution points for achievements
                creature.earn_evolution_points(25)
    
    def _update_thought_bubbles(self) -> None:
        """Update thought bubbles for rendering (v3.4)."""
        self.thought_bubbles = []
        for creature in self.creatures:
            if not creature.alive or not creature.current_thought:
                continue
            
            thought, emoji = creature.current_thought
            self.thought_bubbles.append(ThoughtBubble(
                creature_id=creature.id,
                thought=thought,
                emoji=emoji,
                position=Vector2(creature.position.x, creature.position.y - creature.size - 25)
            ))
    
    def _spawn_boss_creature(self) -> None:
        """Rare chance to spawn a boss creature (v3.4)."""
        if random.random() > 0.0001:  # Very rare
            return
        
        is_predator = random.random() < 0.7  # 70% predator boss
        pos = Vector2(
            random.uniform(100, self.world.width - 100),
            random.uniform(100, self.world.height - 100)
        )
        
        boss = Creature(position=pos, is_predator=is_predator, config=self.config)
        
        # Make it POWERFUL
        boss.is_boss = True
        boss.max_health *= 3.0
        boss.health = boss.max_health
        boss.max_speed *= 1.3
        boss.vision_range *= 1.5
        boss.size *= 1.8
        
        # Give boss a title
        if is_predator:
            titles = ["Ancient Hunter", "Dire Wolf", "Apex Destroyer", "Blood King"]
        else:
            titles = ["Ancient One", "Herd Guardian", "Great Survivor", "Elder Spirit"]
        
        boss.boss_data = BossCreature(
            boss_type="apex_predator" if is_predator else "ancient_prey",
            creature_id=boss.id,
            title=random.choice(titles),
            power_level=3.0,
            special_moves=["charge", "roar", "heal"]
        )
        
        self.creatures.append(boss)
        self.boss_creatures.append(boss.boss_data)
        
        print(f"👑 BOSS SPAWNED: {boss.boss_data.title}!")
    
    # =========================================================================
    # NEW v3.5: SEASONS, FAMILIES, LEGENDARIES, WORLD EVENTS
    # =========================================================================
    
    def _update_seasons(self) -> None:
        """Update seasonal effects (v3.5)."""
        # Check for season change
        if self.world.day > self.season_day_counter + self.days_per_season:
            self.season_day_counter = self.world.day
            seasons = [Season.SPRING, Season.SUMMER, Season.AUTUMN, Season.WINTER]
            current_idx = seasons.index(self.current_season)
            self.current_season = seasons[(current_idx + 1) % 4]
            self._announce_season_change()
        
        # Apply seasonal effects only every 10 steps (OPTIMIZATION)
        if self.step % 10 == 0:
            self._apply_season_effects()
    
    def _announce_season_change(self) -> None:
        """Announce new season with effects."""
        season_info = {
            Season.SPRING: ("🌸 SPRING", "Life blooms! +30% food, +20% reproduction"),
            Season.SUMMER: ("☀️ SUMMER", "Peak activity! +10% speed for all"),
            Season.AUTUMN: ("🍂 AUTUMN", "Preparing for winter... -20% food"),
            Season.WINTER: ("❄️ WINTER", "Survival mode! -50% food, -30% speed")
        }
        name, desc = season_info[self.current_season]
        print(f"\n{'='*50}")
        print(f"  {name} HAS ARRIVED!")
        print(f"  {desc}")
        print(f"{'='*50}\n")
    
    def _apply_season_effects(self) -> None:
        """Apply current season effects to world and creatures (OPTIMIZED)."""
        # Only process a subset of creatures per call for performance
        sample_size = min(30, len(self.creatures))
        sample = random.sample([c for c in self.creatures if c.alive], min(sample_size, len([c for c in self.creatures if c.alive])))
        
        for creature in sample:
            if self.current_season == Season.WINTER:
                creature.energy -= 0.1  # Slightly stronger but less frequent
            elif self.current_season == Season.SPRING:
                creature.energy = min(creature.max_energy, creature.energy + 0.1)
    
    def _update_family_trees(self, parent: 'Creature', child: 'Creature') -> None:
        """Track family relationships (v3.5)."""
        # Create lineage for child
        child_lineage = CreatureLineage(
            creature_id=child.id,
            parent_ids=[parent.id],
            generation=parent.generation,
            family_name=self._get_family_name(parent)
        )
        self.family_trees[child.id] = child_lineage
        
        # Update parent's lineage
        if parent.id in self.family_trees:
            self.family_trees[parent.id].children_ids.append(child.id)
        
        # Create relationship
        self.relationships.append(CreatureRelationship(
            creature_a_id=parent.id,
            creature_b_id=child.id,
            relationship_type='parent',
            bond_strength=0.8
        ))
    
    def _get_family_name(self, creature: 'Creature') -> str:
        """Generate or get family dynasty name."""
        if creature.id in self.family_trees and self.family_trees[creature.id].family_name:
            return self.family_trees[creature.id].family_name
        
        # Generate new dynasty name
        prefixes = ["Shadow", "Storm", "Swift", "Ancient", "Golden", "Iron", "Blood", "Frost"]
        suffixes = ["claw", "fang", "heart", "spirit", "hunter", "runner", "striker", "walker"]
        return f"{random.choice(prefixes)}{random.choice(suffixes)}"
    
    def _spawn_legendary_creature(self) -> None:
        """Spawn a legendary creature when conditions are met (v3.5)."""
        # Check if legendary should spawn (very rare, based on total kills)
        if self.stats['kills'] < 100 or random.random() > 0.0005:
            return
        
        # Already have max legendaries?
        if len(self.legendary_creatures) >= 3:
            return
        
        # Pick legendary type
        leg_type = random.choice(list(LEGENDARY_TYPES.keys()))
        info = LEGENDARY_TYPES[leg_type]
        
        is_predator = not info.get('prey_only', False)
        pos = Vector2(
            random.uniform(100, self.world.width - 100),
            random.uniform(100, self.world.height - 100)
        )
        
        legendary = Creature(position=pos, is_predator=is_predator, config=self.config)
        
        # Make it LEGENDARY
        legendary.is_boss = True
        legendary.max_health *= 5.0
        legendary.health = legendary.max_health
        legendary.max_speed *= 1.5
        legendary.vision_range *= 2.0
        legendary.size *= 2.5
        
        legendary_data = LegendaryCreature(
            creature_id=legendary.id,
            legendary_type=leg_type,
            title=info['title'],
            aura_color=info['aura'],
            special_powers=info['powers']
        )
        
        self.creatures.append(legendary)
        self.legendary_creatures.append(legendary_data)
        
        print(f"\n{'⚡'*20}")
        print(f"  🌟 LEGENDARY CREATURE APPEARED! 🌟")
        print(f"  {info['title'].upper()} - The {leg_type.replace('_', ' ').title()}")
        print(f"{'⚡'*20}\n")
    
    def _trigger_world_event(self, event_type: str = None) -> None:
        """Trigger a world event (v3.5)."""
        if event_type is None:
            event_type = random.choice(list(WORLD_EVENTS.keys()))
        
        if event_type not in WORLD_EVENTS:
            return
        
        info = WORLD_EVENTS[event_type]
        event = WorldEvent(
            event_type=event_type,
            name=info['name'],
            description=info['description'],
            duration=info['duration'],
            effects=info['effects'].copy()
        )
        
        self.world_events.append(event)
        
        print(f"\n{'🌟'*15}")
        print(f"  WORLD EVENT: {info['name']}")
        print(f"  {info['description']}")
        print(f"  Duration: {info['duration']} steps")
        print(f"{'🌟'*15}\n")
    
    def _update_world_events(self) -> None:
        """Update active world events (v3.5) - OPTIMIZED."""
        # Only check every 50 steps
        if self.step % 50 != 0:
            return
            
        # Random chance for new event
        if random.random() < 0.005 and len(self.world_events) < 2:
            self._trigger_world_event()
        
        # Update existing events
        self.world_events = [e for e in self.world_events if e.update()]
        
        # Apply event effects to small sample only
        if self.world_events:
            sample = random.sample([c for c in self.creatures if c.alive], 
                                   min(20, len([c for c in self.creatures if c.alive])))
            for event in self.world_events:
                for creature in sample:
                    if 'energy_drain' in event.effects:
                        creature.energy -= 0.5 * event.effects['energy_drain']
    
    def _spawn_artifact(self) -> None:
        """Spawn mystical artifacts in the world (v3.5)."""
        if len(self.artifacts) >= 5 or random.random() > 0.001:
            return
        
        artifact_types = {
            'ancient_bone': {'effects': {'damage': 1.2}, 'color': (200, 180, 150)},
            'mystic_stone': {'effects': {'vision': 1.3}, 'color': (100, 150, 255)},
            'life_spring': {'effects': {'health_regen': 2.0}, 'color': (100, 255, 150)},
            'death_mark': {'effects': {'damage': 1.5, 'defense': 0.8}, 'color': (150, 0, 50)}
        }
        
        art_type = random.choice(list(artifact_types.keys()))
        info = artifact_types[art_type]
        
        artifact = Artifact(
            position=Vector2(
                random.uniform(100, self.world.width - 100),
                random.uniform(100, self.world.height - 100)
            ),
            artifact_type=art_type,
            effects=info['effects'],
            glow_color=info['color']
        )
        
        self.artifacts.append(artifact)
    
    def _add_to_hall_of_legends(self, creature: 'Creature', cause: str) -> None:
        """Add notable creature to hall of legends (v3.5)."""
        # Only add noteworthy creatures
        if creature.kills < 5 and creature.age < 1000:
            return
        
        species = "Predator" if creature.is_predator else "Prey"
        title = None
        
        if creature.is_boss:
            title = getattr(creature, 'boss_data', None)
            title = title.title if title else "Boss"
        elif creature.kills >= 20:
            title = "Legendary Hunter"
        elif creature.age >= 2000:
            title = "Ancient One"
        
        memorial = CreatureMemorial(
            name=f"{species} #{creature.id}",
            species=species,
            kills=creature.kills,
            survived_days=creature.age / 100,
            cause_of_death=cause,
            legendary_title=title
        )
        
        self.hall_of_legends.append(memorial)
        
        if title:
            print(f"🪦 {title} has fallen. Kills: {creature.kills}, Days: {creature.age/100:.1f}")

    def _apply_weather_effects(self) -> None:
        """Apply weather effects to all creatures (v3.2)."""
        weather = self.world.weather
        
        for creature in self.creatures:
            if not creature.alive:
                continue
            
            if weather == Weather.RAIN:
                # Rain: Slightly slower movement, harder to see
                creature.velocity = creature.velocity * 0.98
                # Predators less effective at hunting
                if creature.is_predator:
                    creature.emotions.aggression *= 0.95
                # Prey more alert (can hear splashing)
                else:
                    creature.emotions.fear = min(1.0, creature.emotions.fear + 0.01)
            
            elif weather == Weather.STORM:
                # Storm: Much slower, increased fear, possible damage
                creature.velocity = creature.velocity * 0.9
                creature.emotions.fear = min(1.0, creature.emotions.fear + 0.05)
                creature.emotions.exhaustion = min(1.0, creature.emotions.exhaustion + 0.02)
                # Random lightning damage (very rare)
                if random.random() < 0.0001:
                    creature.health -= 30
                    self.renderer.particles.append({
                        'x': creature.position.x,
                        'y': creature.position.y,
                        'vx': 0, 'vy': -5,
                        'life': 10,
                        'color': (255, 255, 100),
                        'size': 8,
                    })
            
            elif weather == Weather.FOG:
                # Fog: Reduced vision
                # (vision reduction is handled in sense method via world.light_level)
                # Creatures more cautious
                if creature.is_predator:
                    # Predators use more stalking
                    if creature.state == CreatureState.HUNTING and random.random() < 0.05:
                        creature.state = CreatureState.STALKING
                else:
                    # Prey more bunched up
                    creature.emotions.fear = min(1.0, creature.emotions.fear + 0.02)
            
            elif weather == Weather.CLEAR:
                # Clear: Energy regenerates faster
                creature.energy = min(creature.max_energy, creature.energy + 0.05)
                creature.emotions.exhaustion = max(0, creature.emotions.exhaustion - 0.01)
    
    def _print_status(self) -> None:
        """Print console status."""
        prey = sum(1 for c in self.creatures if not c.is_predator and c.alive)
        predators = sum(1 for c in self.creatures if c.is_predator and c.alive)
        
        time_icon = "☀️" if self.world.is_day else "🌙"
        weather_icons = {
            Weather.CLEAR: "☀️",
            Weather.CLOUDY: "☁️",
            Weather.RAIN: "🌧️",
            Weather.STORM: "⛈️",
            Weather.FOG: "🌫️",
        }
        weather_icon = weather_icons.get(self.world.weather, "")
        
        print(
            f"Step {self.step:6d} | "
            f"Day {self.world.day:3d} {time_icon} {weather_icon} | "
            f"🐰 {prey:3d} (Gen {self.stats['max_generation_prey']}) "
            f"🦁 {predators:2d} (Gen {self.stats['max_generation_predator']}) | "
            f"Kills: {self.stats['kills']:4d} | "
            f"Births: {self.stats['births']:4d}"
        )
    
    def _render(self) -> None:
        """Render frame."""
        # Update hover detection
        mouse_pos = pygame.mouse.get_pos()
        self.renderer.update_hover(self.creatures, mouse_pos)
        
        self.renderer.render(
            self.world, self.creatures, self.stats,
            hazards=self.hazards,
            territories=self.territories,
            burrows=self.burrows,
            scent_marks=self.scent_marks,
            hunting_formations=self.hunting_formations,
            disasters=self.disasters,
            thought_bubbles=self.thought_bubbles,
            # v3.5 additions
            season=self.current_season,
            world_events=self.world_events,
            artifacts=self.artifacts,
            legendary_creatures=self.legendary_creatures,
            spectator_creature=self.followed_creature if self.spectator_mode else None
        )
    
    def _cleanup(self) -> None:
        """Cleanup and save."""
        print("\n" + "=" * 70)
        print("  🏁 Simulation Complete!")
        print("=" * 70)
        print(f"\n📊 Final Statistics:")
        print(f"  Total Steps: {self.step}")
        print(f"  Total Days: {self.world.day}")
        print(f"  Total Kills: {self.stats['kills']}")
        print(f"  Total Births: {self.stats['births']}")
        print(f"  Total Deaths: {self.stats['deaths']}")
        print(f"  Max Prey Generation: {self.stats['max_generation_prey']}")
        print(f"  Max Predator Generation: {self.stats['max_generation_predator']}")
        
        # Save simulation logs
        self.logger.save()
        print(f"\n💾 Data saved to:")
        print(f"   - {self.logger.agent_log_path}")
        print(f"   - {self.logger.species_log_path}")
        
        # Save persistent learning knowledge
        print(f"\n🧠 Saving learned knowledge...")
        PersistentLearner.save_knowledge()
        learn_stats = PersistentLearner.get_stats()
        print(f"   Learning updates this session: {self.learning_updates:,}")
        print(f"   Total predator states: {learn_stats['predator_states']:,}")
        print(f"   Total prey states: {learn_stats['prey_states']:,}")
        print(f"   Total experiences: {learn_stats['experiences']:,}")
        
        self.renderer.quit()
        print("\n✅ Goodbye! Creatures will remember their lessons next time!")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Launch the ultimate simulation."""
    config = SimConfig(
        width=1400,
        height=900,
        initial_prey=80,
        initial_predators=12,
        fps=60,
    )
    
    sim = UltimateSimulation(config)
    sim.run()


if __name__ == "__main__":
    main()
