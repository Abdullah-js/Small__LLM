"""
NOPAINNOGAIN - Decision Engine (Enhanced Q-Learning)
With separate learning for predators and prey, eligibility traces, and experience replay
"""

from __future__ import annotations
import random
import json
import os
from collections import deque
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

from config import settings, constants

if TYPE_CHECKING:
    from modules.agent import Agent
    from modules.environment import Environment


class Experience:
    """Single experience tuple for replay."""
    def __init__(self, state: Tuple, action: str, reward: float, next_state: Tuple, done: bool):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class DecisionEngine:
    """Enhanced Q-Learning with separate tables for predator/prey."""
    
    def __init__(self):
        self.prey_q_table: Dict[Tuple, Dict[str, float]] = {}
        self.predator_q_table: Dict[Tuple, Dict[str, float]] = {}
        
        self.prey_exploration_rate = settings.MAX_EXPLORATION_RATE
        self.predator_exploration_rate = settings.MAX_EXPLORATION_RATE
        self.learning_rate = settings.LEARNING_RATE
        self.discount_factor = settings.DISCOUNT_FACTOR
        
        self.prey_eligibility: Dict[Tuple, Dict[str, float]] = {}
        self.predator_eligibility: Dict[Tuple, Dict[str, float]] = {}
        
        self.prey_memory: deque = deque(maxlen=settings.RL.MEMORY_SIZE)
        self.predator_memory: deque = deque(maxlen=settings.RL.MEMORY_SIZE)
        
        self.total_updates = 0
        self.total_rewards = 0.0
        self.prey_updates = 0
        self.predator_updates = 0
        self.prey_rewards = 0.0
        self.predator_rewards = 0.0
        
        self._load_existing_tables()
    
    def _load_existing_tables(self) -> None:
        """Load existing Q-tables if available."""
        self.load_q_table(constants.PREY_Q_TABLE_PATH, is_predator=False)
        self.load_q_table(constants.PREDATOR_Q_TABLE_PATH, is_predator=True)
    
    def get_state(self, agent: Agent, environment: Environment, all_agents: List[Agent] = None) -> Tuple:
        """Convert agent and environment state to hashable tuple with richer features."""
        x, y = agent.position
        cell = environment.grid[y][x]
        
        total_resources = sum(
            r.nutritional_value for r in cell["resources"].values()
            if hasattr(r, "nutritional_value")
        )
        
        health_bin = min(3, int(agent.health / 25))
        energy_bin = min(3, int(agent.energy / 25))
        resource_bin = min(3, int(total_resources / 25))
        hazard_level = min(3, int(cell["hazards"] / 3))
        
        nearby_predators = 0
        nearby_prey = 0
        nearest_threat_dist = 10
        nearest_prey_dist = 10
        
        if all_agents:
            features = agent.get_state_features(all_agents)
            nearby_predators = features["nearby_predators"]
            nearby_prey = features["nearby_prey"]
            
            for other in all_agents:
                if other.id == agent.id or not other.alive:
                    continue
                dist = agent.distance_to(other.position)
                if other.is_predator and not agent.is_predator:
                    nearest_threat_dist = min(nearest_threat_dist, int(dist))
                elif not other.is_predator and agent.is_predator:
                    nearest_prey_dist = min(nearest_prey_dist, int(dist))
        
        threat_bin = min(3, nearest_threat_dist // 3)
        prey_bin = min(3, nearest_prey_dist // 3)
        
        if agent.is_predator:
            return (health_bin, energy_bin, nearby_prey, prey_bin, hazard_level)
        else:
            return (health_bin, energy_bin, resource_bin, nearby_predators, threat_bin)
    
    def get_actions(self, agent: Agent) -> List[str]:
        """Get available actions based on agent type."""
        if agent.is_predator:
            return constants.PREDATOR_ACTIONS
        return constants.PREY_ACTIONS
    
    def get_q_table(self, agent: Agent) -> Dict[Tuple, Dict[str, float]]:
        """Get appropriate Q-table for agent type."""
        return self.predator_q_table if agent.is_predator else self.prey_q_table
    
    def choose_action(
        self,
        agent: Agent,
        environment: Environment,
        all_agents: List[Agent] = None
    ) -> str:
        """Choose action using epsilon-greedy with intelligence modifier."""
        state = self.get_state(agent, environment, all_agents)
        actions = self.get_actions(agent)
        q_table = self.get_q_table(agent)
        
        exploration_rate = self.predator_exploration_rate if agent.is_predator else self.prey_exploration_rate
        
        intelligence_bonus = agent.traits.intelligence * 0.2
        effective_exploration = max(0.01, exploration_rate - intelligence_bonus)
        
        if random.random() < effective_exploration:
            if agent.is_predator and agent.nearby_prey_count > 0 and agent.energy > 30:
                return random.choice(["hunt", "attack", "move"])
            elif not agent.is_predator and agent.nearby_threat:
                return random.choice(["flee", "flee", "move"])
            return random.choice(actions)
        
        if state not in q_table:
            q_table[state] = {action: 0.0 for action in actions}
        
        q_values = q_table[state]
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        
        return random.choice(best_actions)
    
    def update_q_table(
        self,
        agent: Agent,
        state: Tuple,
        action: str,
        reward: float,
        next_state: Tuple,
        done: bool = False
    ) -> None:
        """Update Q-value with eligibility traces."""
        q_table = self.get_q_table(agent)
        actions = self.get_actions(agent)
        eligibility = self.predator_eligibility if agent.is_predator else self.prey_eligibility
        
        if state not in q_table:
            q_table[state] = {a: 0.0 for a in actions}
        if next_state not in q_table:
            q_table[next_state] = {a: 0.0 for a in actions}
        
        old_q = q_table[state].get(action, 0.0)
        next_max_q = 0.0 if done else max(q_table[next_state].values())
        
        td_error = reward + self.discount_factor * next_max_q - old_q
        
        if state not in eligibility:
            eligibility[state] = {a: 0.0 for a in actions}
        eligibility[state][action] = 1.0
        
        if settings.RL.USE_ELIGIBILITY_TRACES:
            for s in list(eligibility.keys()):
                for a in eligibility[s]:
                    q_table[s][a] = q_table[s].get(a, 0.0) + self.learning_rate * td_error * eligibility[s][a]
                    eligibility[s][a] *= self.discount_factor * settings.RL.ELIGIBILITY_DECAY
                
                if all(e < 0.01 for e in eligibility[s].values()):
                    del eligibility[s]
        else:
            new_q = old_q + self.learning_rate * td_error
            q_table[state][action] = new_q
        
        memory = self.predator_memory if agent.is_predator else self.prey_memory
        memory.append(Experience(state, action, reward, next_state, done))
        
        self.total_updates += 1
        self.total_rewards += reward
        
        if agent.is_predator:
            self.predator_updates += 1
            self.predator_rewards += reward
        else:
            self.prey_updates += 1
            self.prey_rewards += reward
    
    def replay_experiences(self, is_predator: bool, batch_size: int = None) -> None:
        """Replay random batch of experiences to improve learning."""
        memory = self.predator_memory if is_predator else self.prey_memory
        q_table = self.predator_q_table if is_predator else self.prey_q_table
        actions = constants.PREDATOR_ACTIONS if is_predator else constants.PREY_ACTIONS
        
        batch_size = batch_size or settings.RL.BATCH_SIZE
        if len(memory) < batch_size:
            return
        
        batch = random.sample(list(memory), batch_size)
        
        for exp in batch:
            if exp.state not in q_table:
                q_table[exp.state] = {a: 0.0 for a in actions}
            if exp.next_state not in q_table:
                q_table[exp.next_state] = {a: 0.0 for a in actions}
            
            old_q = q_table[exp.state].get(exp.action, 0.0)
            next_max_q = 0.0 if exp.done else max(q_table[exp.next_state].values())
            
            new_q = old_q + self.learning_rate * 0.5 * (
                exp.reward + self.discount_factor * next_max_q - old_q
            )
            q_table[exp.state][exp.action] = new_q
    
    def get_reward(
        self,
        agent: Agent,
        environment: Environment,
        action: str,
        all_agents: List[Agent] = None
    ) -> float:
        """Calculate reward based on action outcome and agent type."""
        reward = 0.0
        x, y = agent.position
        cell = environment.grid[y][x]
        
        if not agent.alive:
            return settings.REWARD.PENALTY_DEATH
        
        reward += settings.REWARD.REWARD_SURVIVAL * 0.1
        
        if agent.was_attacked:
            reward += settings.REWARD.PENALTY_ATTACKED
        
        if agent.is_predator:
            reward += self._get_predator_reward(agent, action, cell)
        else:
            reward += self._get_prey_reward(agent, action, cell, all_agents)
        
        return reward
    
    def _get_predator_reward(self, agent: Agent, action: str, cell: dict) -> float:
        """Calculate reward for predator actions."""
        reward = 0.0
        
        if action == "attack":
            if agent.last_action_success:
                reward += settings.REWARD.REWARD_SUCCESSFUL_ATTACK
                if agent.kills > 0:
                    reward += settings.REWARD.REWARD_SUCCESSFUL_HUNT
            else:
                reward += settings.REWARD.PENALTY_FAILED_ACTION * 2
        
        elif action == "hunt":
            if agent.nearby_prey_count > 0:
                reward += settings.REWARD.REWARD_NEAR_PREY
            else:
                reward += settings.REWARD.PENALTY_FAILED_HUNT * 0.5
        
        elif action == "drink":
            has_water = "water" in cell["resources"] and cell["resources"]["water"].quantity > 0
            reward += settings.REWARD.REWARD_DRINK_SUCCESS if has_water else settings.REWARD.PENALTY_FAILED_ACTION
        
        elif action == "rest":
            if agent.energy < 40:
                reward += 1.0
            else:
                reward -= 0.5
        
        elif action == "move":
            if agent.nearby_prey_count > 0:
                reward += settings.REWARD.REWARD_MOVE_TO_RESOURCE * 0.5
        
        return reward
    
    def _get_prey_reward(self, agent: Agent, action: str, cell: dict, all_agents: List[Agent]) -> float:
        """Calculate reward for prey actions."""
        reward = 0.0
        
        if agent.nearby_threat:
            reward += settings.REWARD.PENALTY_NEAR_PREDATOR
        
        if action == "flee":
            if agent.nearby_threat:
                reward += settings.REWARD.REWARD_SUCCESSFUL_FLEE
            else:
                reward += settings.REWARD.PENALTY_FAILED_ACTION
        
        elif action == "eat":
            has_plant = "plant" in cell["resources"] and cell["resources"]["plant"].quantity > 0
            reward += settings.REWARD.REWARD_EAT_SUCCESS if has_plant else settings.REWARD.PENALTY_FAILED_ACTION
        
        elif action == "drink":
            has_water = "water" in cell["resources"] and cell["resources"]["water"].quantity > 0
            reward += settings.REWARD.REWARD_DRINK_SUCCESS if has_water else settings.REWARD.PENALTY_FAILED_ACTION
        
        elif action == "move":
            total_resources = sum(
                r.quantity for r in cell["resources"].values()
                if hasattr(r, "quantity")
            )
            if total_resources > 0:
                reward += settings.REWARD.REWARD_MOVE_TO_RESOURCE
            if cell["hazards"] > 5:
                reward += settings.REWARD.PENALTY_MOVE_TO_HAZARD
        
        elif action == "rest":
            if agent.energy < 30:
                reward += 2.0
            elif agent.nearby_threat:
                reward -= 5.0
            else:
                reward -= 0.5
        
        return reward
    
    def decay_exploration(self) -> None:
        """Decay exploration rates separately for predators and prey."""
        self.prey_exploration_rate = max(
            settings.MIN_EXPLORATION_RATE,
            self.prey_exploration_rate * settings.EXPLORATION_DECAY
        )
        self.predator_exploration_rate = max(
            settings.MIN_EXPLORATION_RATE,
            self.predator_exploration_rate * settings.EXPLORATION_DECAY
        )
    
    def clear_eligibility_traces(self) -> None:
        """Clear eligibility traces (call at episode end)."""
        self.prey_eligibility.clear()
        self.predator_eligibility.clear()
    
    def save_q_table(self, filepath: str = None, is_predator: bool = None) -> None:
        """Save Q-tables to JSON files."""
        os.makedirs(os.path.dirname(constants.PREY_Q_TABLE_PATH), exist_ok=True)
        
        prey_serializable = {str(k): v for k, v in self.prey_q_table.items()}
        with open(constants.PREY_Q_TABLE_PATH, "w") as f:
            json.dump(prey_serializable, f, indent=2)
        
        predator_serializable = {str(k): v for k, v in self.predator_q_table.items()}
        with open(constants.PREDATOR_Q_TABLE_PATH, "w") as f:
            json.dump(predator_serializable, f, indent=2)
    
    def load_q_table(self, filepath: str, is_predator: bool = False) -> None:
        """Load Q-table from JSON file."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            table = {eval(k): v for k, v in data.items()}
            if is_predator:
                self.predator_q_table = table
            else:
                self.prey_q_table = table
            print(f"Loaded Q-table from {filepath} ({len(table)} states)")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error loading Q-table: {e}")
    
    def get_stats(self) -> Dict:
        """Get comprehensive learning statistics."""
        return {
            "total_states": len(self.prey_q_table) + len(self.predator_q_table),
            "prey_states": len(self.prey_q_table),
            "predator_states": len(self.predator_q_table),
            "total_updates": self.total_updates,
            "prey_updates": self.prey_updates,
            "predator_updates": self.predator_updates,
            "total_rewards": self.total_rewards,
            "prey_exploration": self.prey_exploration_rate,
            "predator_exploration": self.predator_exploration_rate,
            "avg_reward": self.total_rewards / max(1, self.total_updates),
            "prey_avg_reward": self.prey_rewards / max(1, self.prey_updates),
            "predator_avg_reward": self.predator_rewards / max(1, self.predator_updates),
            "prey_memory_size": len(self.prey_memory),
            "predator_memory_size": len(self.predator_memory),
        }
