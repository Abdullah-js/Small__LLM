"""
NOPAINNOGAIN - Advanced Simulation
====================================
Evolutionary AI Ecosystem with:
- Physics-based movement
- Neural network brains
- Flocking behavior
- Day/night cycles
- Weather effects
- Advanced predator-prey dynamics
"""

from __future__ import annotations
import sys
import os
import random
import json
from typing import List, Dict, Optional
from datetime import datetime

import pygame
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.physics import Vector2
from modules.world import World, Weather
from modules.creature import Creature, CreatureState
from modules.renderer import Renderer


class AdvancedSimulation:
    """Main simulation with all advanced features."""
    
    def __init__(
        self,
        world_width: float = 1200,
        world_height: float = 900,
        initial_prey: int = 50,
        initial_predators: int = 8,
        max_steps: int = 50000,
        fps: int = 60,
    ):
        self.step = 0
        self.max_steps = max_steps
        self.fps = fps
        self.running = True
        self.paused = False
        
        self.world = World(width=world_width, height=world_height)
        self.renderer = Renderer(int(world_width), int(world_height))
        self.clock = pygame.time.Clock()
        
        self.creatures: List[Creature] = []
        self._spawn_initial_population(initial_prey, initial_predators)
        
        self.stats = {
            "kills": 0,
            "births": 0,
            "deaths": 0,
            "prey_births": 0,
            "predator_births": 0,
            "max_prey": initial_prey,
            "max_predators": initial_predators,
            "best_prey_fitness": 0,
            "best_predator_fitness": 0,
        }
        
        self.logs: List[Dict] = []
        self.population_history: List[Dict] = []
        
        self.min_prey = 10
        self.min_predators = 3
        self.max_prey = 200
        self.max_predators = 30
        
        self._print_banner()
    
    def _print_banner(self) -> None:
        """Print startup banner."""
        print("\n" + "=" * 60)
        print("  🧬 NOPAINNOGAIN - Evolutionary AI Ecosystem 🧬")
        print("=" * 60)
        print(f"  World: {self.world.width:.0f} x {self.world.height:.0f}")
        print(f"  Prey: {sum(1 for c in self.creatures if not c.is_predator)}")
        print(f"  Predators: {sum(1 for c in self.creatures if c.is_predator)}")
        print(f"  Food sources: {len(self.world.food_sources)}")
        print(f"  Water sources: {len(self.world.water_sources)}")
        print("=" * 60)
        print("  Controls:")
        print("    ESC    - Quit")
        print("    SPACE  - Pause/Resume")
        print("    T      - Toggle trails")
        print("    V      - Toggle vision cones")
        print("    D      - Toggle debug info")
        print("=" * 60 + "\n")
    
    def _spawn_initial_population(self, num_prey: int, num_predators: int) -> None:
        """Spawn initial creatures."""
        for i in range(num_prey):
            pos = Vector2(
                random.uniform(50, self.world.width - 50),
                random.uniform(50, self.world.height - 50)
            )
            self.creatures.append(Creature(
                position=pos,
                is_predator=False,
                name=f"Prey_{i}"
            ))
        
        for i in range(num_predators):
            pos = Vector2(
                random.uniform(50, self.world.width - 50),
                random.uniform(50, self.world.height - 50)
            )
            self.creatures.append(Creature(
                position=pos,
                is_predator=True,
                name=f"Predator_{i}"
            ))
    
    def run(self) -> None:
        """Main simulation loop."""
        while self.running and self.step < self.max_steps:
            self._handle_events()
            
            if not self.paused:
                self._update()
            
            self._render()
            
            self.clock.tick(self.fps)
            self.step += 1
        
        self._cleanup()
    
    def _handle_events(self) -> None:
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("PAUSED" if self.paused else "RESUMED")
                elif event.key == pygame.K_t:
                    self.renderer.show_trails = not self.renderer.show_trails
                elif event.key == pygame.K_v:
                    self.renderer.show_vision = not self.renderer.show_vision
                elif event.key == pygame.K_d:
                    self.renderer.show_debug = not self.renderer.show_debug
    
    def _update(self) -> None:
        """Update simulation state."""
        self.world.update()
        
        random.shuffle(self.creatures)
        
        for creature in self.creatures:
            if not creature.alive:
                continue
            
            creature.sense_environment(self.world, self.creatures)
            creature.think(self.world)
            creature.act(self.world, self.creatures)
            creature.update(self.world)
            
            if not creature.alive:
                self.stats["deaths"] += 1
                self.renderer.spawn_death_particles(creature.position)
        
        self._handle_attacks()
        
        self._handle_reproduction()
        
        self._balance_population()
        
        self.creatures = [c for c in self.creatures if c.alive]
        
        self._update_stats()
        
        if self.step % 100 == 0:
            self._print_status()
    
    def _handle_attacks(self) -> None:
        """Process predator attacks."""
        predators = [c for c in self.creatures if c.is_predator and c.alive]
        prey = [c for c in self.creatures if not c.is_predator and c.alive]
        
        for predator in predators:
            if predator.state != CreatureState.ATTACKING:
                continue
            
            for target in prey:
                if not target.alive:
                    continue
                
                dist = predator.position.distance_to(target.position)
                if dist < predator.traits.attack_range:
                    kills_before = predator.stats.kills
                    
                    if predator.attack_cooldown <= 0:
                        damage = predator.traits.attack_power
                        target.take_damage(damage, predator)
                        predator.attack_cooldown = 20
                        
                        self.renderer.spawn_attack_particles(target.position)
                        
                        if not target.alive:
                            predator.stats.kills += 1
                            predator.energy = min(predator.max_energy, predator.energy + 50)
                            self.stats["kills"] += 1
                    break
    
    def _handle_reproduction(self) -> None:
        """Handle creature reproduction."""
        new_creatures: List[Creature] = []
        
        reproducers = [c for c in self.creatures if c.can_reproduce()]
        
        for creature in reproducers:
            same_species = [
                c for c in reproducers
                if c.id != creature.id
                and c.is_predator == creature.is_predator
                and c.position.distance_to(creature.position) < 80
            ]
            
            if same_species and random.random() < 0.02:
                partner = random.choice(same_species)
                child = creature.reproduce(partner)
                
                if child:
                    new_creatures.append(child)
                    self.stats["births"] += 1
                    
                    if child.is_predator:
                        self.stats["predator_births"] += 1
                    else:
                        self.stats["prey_births"] += 1
                    
                    self.renderer.spawn_birth_particles(child.position)
        
        self.creatures.extend(new_creatures)
    
    def _balance_population(self) -> None:
        """Keep populations within bounds."""
        prey = [c for c in self.creatures if not c.is_predator and c.alive]
        predators = [c for c in self.creatures if c.is_predator and c.alive]
        
        if len(prey) < self.min_prey:
            for _ in range(self.min_prey - len(prey)):
                pos = Vector2(
                    random.uniform(50, self.world.width - 50),
                    random.uniform(50, self.world.height - 50)
                )
                self.creatures.append(Creature(position=pos, is_predator=False))
        
        if len(predators) < self.min_predators and len(prey) > 20:
            for _ in range(self.min_predators - len(predators)):
                pos = Vector2(
                    random.uniform(50, self.world.width - 50),
                    random.uniform(50, self.world.height - 50)
                )
                self.creatures.append(Creature(position=pos, is_predator=True))
        
        if len(prey) > self.max_prey:
            prey.sort(key=lambda c: c.get_fitness())
            for c in prey[:len(prey) - self.max_prey]:
                c.die()
        
        if len(predators) > self.max_predators:
            predators.sort(key=lambda c: c.get_fitness())
            for c in predators[:len(predators) - self.max_predators]:
                c.die()
    
    def _update_stats(self) -> None:
        """Update statistics."""
        prey = [c for c in self.creatures if not c.is_predator and c.alive]
        predators = [c for c in self.creatures if c.is_predator and c.alive]
        
        self.stats["max_prey"] = max(self.stats["max_prey"], len(prey))
        self.stats["max_predators"] = max(self.stats["max_predators"], len(predators))
        
        if prey:
            best_prey = max(prey, key=lambda c: c.get_fitness())
            self.stats["best_prey_fitness"] = max(
                self.stats["best_prey_fitness"],
                best_prey.get_fitness()
            )
        
        if predators:
            best_pred = max(predators, key=lambda c: c.get_fitness())
            self.stats["best_predator_fitness"] = max(
                self.stats["best_predator_fitness"],
                best_pred.get_fitness()
            )
        
        if self.step % 50 == 0:
            self.population_history.append({
                "step": self.step,
                "prey": len(prey),
                "predators": len(predators),
                "day": self.world.current_day,
                "weather": self.world.weather.name,
                "kills": self.stats["kills"],
                "births": self.stats["births"],
            })
    
    def _print_status(self) -> None:
        """Print status to console."""
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
            f"Day {self.world.current_day:3d} {time_icon} {weather_icon} | "
            f"🐰 {prey:3d} 🦁 {predators:2d} | "
            f"Kills: {self.stats['kills']:4d} | "
            f"Births: {self.stats['births']:4d}"
        )
    
    def _render(self) -> None:
        """Render the simulation."""
        self.renderer.render(self.world, self.creatures, self.stats)
    
    def _cleanup(self) -> None:
        """Save data and cleanup."""
        print("\n" + "=" * 60)
        print("  Simulation Complete!")
        print("=" * 60)
        
        print(f"\n📊 Final Statistics:")
        print(f"  Total steps: {self.step}")
        print(f"  Total days: {self.world.current_day}")
        print(f"  Total kills: {self.stats['kills']}")
        print(f"  Total births: {self.stats['births']}")
        print(f"  Total deaths: {self.stats['deaths']}")
        print(f"  Max prey population: {self.stats['max_prey']}")
        print(f"  Max predator population: {self.stats['max_predators']}")
        print(f"  Best prey fitness: {self.stats['best_prey_fitness']:.1f}")
        print(f"  Best predator fitness: {self.stats['best_predator_fitness']:.1f}")
        
        self._save_logs()
        
        self.renderer.quit()
        pygame.quit()
        
        print("\n✅ Goodbye!")
    
    def _save_logs(self) -> None:
        """Save simulation logs."""
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        if self.population_history:
            df = pd.DataFrame(self.population_history)
            filepath = os.path.join(log_dir, "population_history.csv")
            df.to_csv(filepath, index=False)
            print(f"\n📁 Saved population history to {filepath}")


def main():
    """Entry point."""
    sim = AdvancedSimulation(
        world_width=1200,
        world_height=900,
        initial_prey=60,
        initial_predators=10,
        max_steps=100000,
        fps=60,
    )
    sim.run()


if __name__ == "__main__":
    main()
