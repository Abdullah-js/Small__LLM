"""
NOPAINNOGAIN - Main Entry Point
================================
Evolutionary AI Ecosystem Simulator with Predator-Prey Dynamics
"""

from __future__ import annotations
import sys
import os
import random
from typing import List, Dict

import pygame
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings, constants
from modules.agent import Agent
from modules.environment import Environment
from modules.decision_engine import DecisionEngine
from modules.evolution import reproduce, select_partner
from modules.visualization import Visualization
from modules.utils import setup_directories, save_logs, get_population_stats


class Simulation:
    """Main simulation controller with predator-prey dynamics."""
    
    def __init__(self):
        self.step = 0
        self.running = True
        
        setup_directories()
        
        self.environment = Environment(settings.GRID_WIDTH, settings.GRID_HEIGHT)
        self.decision_engine = DecisionEngine()
        self.visualization = Visualization(self.environment)
        self.clock = pygame.time.Clock()
        
        self.agents: List[Agent] = []
        self._spawn_initial_population()
        
        self.simulation_logs: List[Dict] = []
        self.species_logs: List[Dict] = []
        
        self.total_kills = 0
        self.total_births = 0
        self.total_deaths = 0
        
        print("=" * 60)
        print("   NOPAINNOGAIN - AI Ecosystem Simulator")
        print("   🦁 Predator-Prey Learning System 🐰")
        print("=" * 60)
        print(f"Grid: {settings.GRID_WIDTH}x{settings.GRID_HEIGHT}")
        print(f"Initial Prey: {settings.INITIAL_PREY_POPULATION}")
        print(f"Initial Predators: {settings.INITIAL_PREDATOR_POPULATION}")
        print("=" * 60)
        print("Controls: ESC=quit, SPACE=pause (not implemented)")
        print("=" * 60)
    
    def _spawn_initial_population(self) -> None:
        """Create initial prey and predator populations."""
        for i in range(settings.INITIAL_PREY_POPULATION):
            agent = Agent(
                name=f"Prey_{i}",
                species_id="prey",
                position=(
                    random.randint(0, settings.GRID_WIDTH - 1),
                    random.randint(0, settings.GRID_HEIGHT - 1)
                )
            )
            self.agents.append(agent)
        
        for i in range(settings.INITIAL_PREDATOR_POPULATION):
            agent = Agent(
                name=f"Predator_{i}",
                species_id="predator",
                position=(
                    random.randint(0, settings.GRID_WIDTH - 1),
                    random.randint(0, settings.GRID_HEIGHT - 1)
                )
            )
            self.agents.append(agent)
    
    def run(self) -> None:
        """Main simulation loop."""
        while self.running and self.step < settings.MAX_STEPS:
            self._handle_events()
            
            if not self.running:
                break
            
            self._update()
            self._render()
            
            self.clock.tick(settings.FPS)
            self.step += 1
        
        self._cleanup()
    
    def _handle_events(self) -> None:
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    pass
    
    def _update(self) -> None:
        """Update simulation state with predator-prey interactions."""
        new_agents: List[Agent] = []
        dead_positions: List[tuple] = []
        attack_positions: List[tuple] = []
        flee_positions: List[tuple] = []
        
        random.shuffle(self.agents)
        
        for agent in self.agents:
            if not agent.alive:
                continue
            
            state = self.decision_engine.get_state(agent, self.environment, self.agents)
            action = self.decision_engine.choose_action(agent, self.environment, self.agents)
            
            kills_before = agent.kills
            agent.perform_action(action, self.environment, self.agents)
            
            if agent.kills > kills_before:
                self.total_kills += 1
                attack_positions.append(agent.position)
            
            if action == "flee" and agent.last_action_success:
                flee_positions.append(agent.position)
            
            reward = self.decision_engine.get_reward(agent, self.environment, action, self.agents)
            
            next_state = self.decision_engine.get_state(agent, self.environment, self.agents)
            self.decision_engine.update_q_table(agent, state, action, reward, next_state, not agent.alive)
            
            self._log_agent(agent, action, reward)
            
            if agent.can_reproduce():
                if random.random() < agent.traits.reproduction_rate:
                    partner = select_partner(agent, self.agents)
                    if partner:
                        child = reproduce(agent, partner)
                        if child:
                            new_agents.append(child)
                            self.total_births += 1
                            self.visualization.spawn_birth_particles(child.position)
            
            agent.step()
            
            if not agent.alive:
                dead_positions.append(agent.position)
                self.total_deaths += 1
        
        for pos in dead_positions:
            self.visualization.spawn_death_particles(pos)
        
        for pos in attack_positions:
            self.visualization.spawn_attack_particles(pos)
        
        for pos in flee_positions:
            self.visualization.spawn_flee_particles(pos)
        
        self.agents = [a for a in self.agents if a.alive]
        self.agents.extend(new_agents)
        
        self._balance_populations()
        
        self._log_species()
        
        self.environment.update_state()
        
        self.decision_engine.decay_exploration()
        
        if self.step % 100 == 0:
            self.decision_engine.replay_experiences(is_predator=True)
            self.decision_engine.replay_experiences(is_predator=False)
        
        self._check_end_conditions()
        
        if self.step % 50 == 0:
            self._print_status()
    
    def _balance_populations(self) -> None:
        """Ensure minimum predator population for interesting dynamics."""
        prey_count = sum(1 for a in self.agents if not a.is_predator and a.alive)
        predator_count = sum(1 for a in self.agents if a.is_predator and a.alive)
        
        if predator_count < settings.POP.MIN_PREDATOR_POPULATION and prey_count > 20:
            for i in range(settings.POP.MIN_PREDATOR_POPULATION - predator_count):
                self.agents.append(Agent(
                    name=f"Predator_spawn_{self.step}_{i}",
                    species_id="predator",
                    position=(
                        random.randint(0, settings.GRID_WIDTH - 1),
                        random.randint(0, settings.GRID_HEIGHT - 1)
                    )
                ))
        
        if predator_count > settings.POP.MAX_PREDATOR_POPULATION:
            predators = [a for a in self.agents if a.is_predator and a.alive]
            predators.sort(key=lambda a: a.energy)
            for pred in predators[:predator_count - settings.POP.MAX_PREDATOR_POPULATION]:
                pred.alive = False
    
    def _print_status(self) -> None:
        """Print simulation status."""
        stats = get_population_stats(self.agents)
        prey_count = sum(1 for a in self.agents if not a.is_predator and a.alive)
        predator_count = sum(1 for a in self.agents if a.is_predator and a.alive)
        
        engine_stats = self.decision_engine.get_stats()
        
        print(
            f"Step {self.step:5d} | "
            f"🐰 {prey_count:3d} 🦁 {predator_count:2d} | "
            f"Kills: {self.total_kills:3d} | "
            f"Births: {self.total_births:3d} | "
            f"Prey ε: {engine_stats['prey_exploration']:.3f} | "
            f"Pred ε: {engine_stats['predator_exploration']:.3f}"
        )
    
    def _render(self) -> None:
        """Render current state."""
        self.visualization.render(self.agents)
    
    def _log_agent(self, agent: Agent, action: str, reward: float) -> None:
        """Log agent data."""
        traits = agent.traits.to_dict() if hasattr(agent.traits, "to_dict") else {}
        
        self.simulation_logs.append({
            "step": self.step,
            "agent_id": agent.id,
            "agent_name": agent.name,
            "species_id": agent.species_id,
            "is_predator": agent.is_predator,
            "health": agent.health,
            "energy": agent.energy,
            "position_x": agent.position[0],
            "position_y": agent.position[1],
            "generation": agent.generation,
            "age": agent.age,
            "speed": traits.get("speed", 0),
            "intelligence": traits.get("intelligence", 0),
            "aggression": traits.get("aggression", 0),
            "vision": traits.get("vision", 0),
            "action": action,
            "reward": reward,
            "kills": agent.kills,
            "times_fled": agent.times_fled,
        })
    
    def _log_species(self) -> None:
        """Log species population data."""
        prey_count = sum(1 for a in self.agents if not a.is_predator and a.alive)
        predator_count = sum(1 for a in self.agents if a.is_predator and a.alive)
        
        self.species_logs.append({
            "step": self.step,
            "prey_population": prey_count,
            "predator_population": predator_count,
            "total_population": prey_count + predator_count,
            "total_kills": self.total_kills,
            "total_births": self.total_births,
        })
    
    def _check_end_conditions(self) -> None:
        """Check for victory or extinction."""
        prey_count = sum(1 for a in self.agents if not a.is_predator and a.alive)
        predator_count = sum(1 for a in self.agents if a.is_predator and a.alive)
        
        if prey_count >= settings.VICTORY_POPULATION:
            print("\n" + "=" * 60)
            print("🎉 VICTORY! Prey population target reached!")
            print("=" * 60)
            self.running = False
        
        if prey_count == 0:
            print("\n" + "=" * 60)
            print("💀 EXTINCTION. All prey have been hunted.")
            print("=" * 60)
            self.running = False
    
    def _cleanup(self) -> None:
        """Save logs and cleanup."""
        print("\n" + "=" * 60)
        print("Saving simulation data...")
        
        if self.simulation_logs:
            sim_df = pd.DataFrame(self.simulation_logs)
            save_logs(sim_df, os.path.join(constants.LOG_DIR, "simulation_logs.csv"))
        
        if self.species_logs:
            species_df = pd.DataFrame(self.species_logs)
            save_logs(species_df, os.path.join(constants.LOG_DIR, "species_logs.csv"))
        
        self.decision_engine.save_q_table()
        print(f"Q-tables saved to {constants.MODELS_DIR}")
        
        stats = self.decision_engine.get_stats()
        print(f"\n📊 Learning Statistics:")
        print(f"  Prey states learned: {stats['prey_states']}")
        print(f"  Predator states learned: {stats['predator_states']}")
        print(f"  Prey avg reward: {stats['prey_avg_reward']:.3f}")
        print(f"  Predator avg reward: {stats['predator_avg_reward']:.3f}")
        print(f"  Total kills: {self.total_kills}")
        print(f"  Total births: {self.total_births}")
        print(f"  Total deaths: {self.total_deaths}")
        
        self.visualization.quit()
        pygame.quit()
        
        print("\n✅ Simulation complete!")
        print("=" * 60)


def main():
    """Entry point."""
    simulation = Simulation()
    simulation.run()


if __name__ == "__main__":
    main()
