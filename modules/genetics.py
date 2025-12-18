"""
NOPAINNOGAIN - Genetics Module
Alias for evolution module for backward compatibility.
"""

from modules.evolution import (
    reproduce,
    genetic_distance,
    crossover,
    mutate,
    select_partner,
    calculate_fitness,
)

__all__ = [
    "reproduce",
    "genetic_distance",
    "crossover",
    "mutate",
    "select_partner",
    "calculate_fitness",
]
