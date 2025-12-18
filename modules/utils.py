"""
NOPAINNOGAIN - Utility Functions
"""

from __future__ import annotations
import os
import uuid
from typing import Dict, List, TYPE_CHECKING

import pandas as pd

from config import constants

if TYPE_CHECKING:
    from modules.agent import Agent


def setup_directories() -> None:
    """Create required data directories."""
    directories = [
        constants.DATA_DIR,
        constants.LOG_DIR,
        constants.REPORTS_DIR,
        constants.MODELS_DIR,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def log_agent_data(
    df: pd.DataFrame,
    agent: Agent,
    step: int,
    action: str,
    reward: float
) -> pd.DataFrame:
    """Add agent data row to dataframe."""
    traits = agent.traits.to_dict() if hasattr(agent.traits, "to_dict") else {}
    
    new_row = {
        "step": step,
        "agent_id": str(agent.id),
        "agent_name": agent.name,
        "species_id": agent.species_id,
        "health": agent.health,
        "energy": agent.energy,
        "position_x": agent.position[0],
        "position_y": agent.position[1],
        "generation": agent.generation,
        "age": agent.age,
        "speed": traits.get("speed", 0),
        "intelligence": traits.get("intelligence", 0),
        "aggression": traits.get("aggression", 0),
        "cooperation": traits.get("cooperation", 0),
        "pollution_tolerance": traits.get("pollution_tolerance", 0),
        "action": action,
        "reward": reward,
    }
    
    df.loc[len(df)] = new_row
    return df


def log_species_data(
    df: pd.DataFrame,
    species_id: str,
    step: int,
    population_size: int
) -> pd.DataFrame:
    """Add species population data row to dataframe."""
    new_row = {
        "step": step,
        "species_id": species_id,
        "population_size": population_size,
    }
    df.loc[len(df)] = new_row
    return df


def save_logs(df: pd.DataFrame, filepath: str) -> None:
    """Save dataframe to CSV file."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Log saved: {filepath}")


def generate_unique_id() -> str:
    """Generate unique identifier."""
    return str(uuid.uuid4())


def get_population_stats(agents: List[Agent]) -> Dict:
    """Calculate population statistics."""
    if not agents:
        return {
            "total": 0,
            "alive": 0,
            "avg_energy": 0,
            "avg_health": 0,
            "avg_age": 0,
            "species_count": 0,
        }
    
    alive_agents = [a for a in agents if a.alive]
    species = set(a.species_id for a in alive_agents)
    
    return {
        "total": len(agents),
        "alive": len(alive_agents),
        "avg_energy": sum(a.energy for a in alive_agents) / max(1, len(alive_agents)),
        "avg_health": sum(a.health for a in alive_agents) / max(1, len(alive_agents)),
        "avg_age": sum(a.age for a in alive_agents) / max(1, len(alive_agents)),
        "species_count": len(species),
    }


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))
