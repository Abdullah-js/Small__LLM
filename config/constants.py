"""
NOPAINNOGAIN - Constants
"""

import os
from enum import Enum, auto
from typing import List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOG_DIR = os.path.join(DATA_DIR, "logs")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
MODELS_DIR = os.path.join(DATA_DIR, "models")

SIMULATION_LOG = os.path.join(LOG_DIR, "simulation_logs.csv")
SPECIES_LOG = os.path.join(LOG_DIR, "species_logs.csv")
REWARDS_LOG = os.path.join(LOG_DIR, "rewards_logs.csv")
Q_TABLE_PATH = os.path.join(MODELS_DIR, "q_table.json")
PREY_Q_TABLE_PATH = os.path.join(MODELS_DIR, "prey_q_table.json")
PREDATOR_Q_TABLE_PATH = os.path.join(MODELS_DIR, "predator_q_table.json")


class Action(Enum):
    MOVE = auto()
    EAT = auto()
    DRINK = auto()
    REST = auto()
    HUNT = auto()
    FLEE = auto()
    ATTACK = auto()
    REPRODUCE = auto()


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class AgentState(Enum):
    IDLE = auto()
    FORAGING = auto()
    HUNTING = auto()
    FLEEING = auto()
    RESTING = auto()
    REPRODUCING = auto()
    ATTACKING = auto()
    DEAD = auto()


class ResourceType(Enum):
    PLANT = "plant"
    WATER = "water"
    MINERAL = "mineral"


PREY_ACTIONS: List[str] = ["move", "eat", "drink", "rest", "flee"]
PREDATOR_ACTIONS: List[str] = ["move", "hunt", "attack", "rest", "drink"]
ACTIONS: List[str] = ["move", "eat", "drink", "rest", "flee", "hunt", "attack"]
ACTIONS_MOVE: List[str] = ["up", "down", "left", "right"]
RESOURCE_TYPES: List[str] = ["plant", "water", "mineral"]

MAX_HEALTH = 100
MAX_ENERGY = 100
MIN_ENERGY = 0
GENOME_LENGTH = 10

PREDATOR_VISION_RANGE = 5
PREY_VISION_RANGE = 4
ATTACK_RANGE = 1
FLEE_SPEED_BONUS = 1.5
