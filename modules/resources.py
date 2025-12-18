"""
NOPAINNOGAIN - Resource Classes
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Tuple
from config import settings


@dataclass
class Resource:
    """Base class for all environmental resources."""
    position: Tuple[int, int]
    resource_type: str = "generic"
    quantity: float = 100.0
    nutritional_value: float = 10.0
    regeneration_rate: float = 1.0
    is_depleted: bool = False
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def regenerate(self) -> None:
        """Regenerate resource quantity over time."""
        if self.quantity < settings.MAX_RESOURCE_CAPACITY:
            self.quantity = min(
                settings.MAX_RESOURCE_CAPACITY,
                self.quantity + self.regeneration_rate
            )
        self.is_depleted = self.quantity <= 0

    def consume(self, amount: float) -> float:
        """Consume resource and return actual amount consumed."""
        actual = min(amount, self.quantity)
        self.quantity -= actual
        self.is_depleted = self.quantity <= 0
        return actual

    def __repr__(self) -> str:
        return f"{self.resource_type}(qty={self.quantity:.1f}, pos={self.position})"


@dataclass
class Plant(Resource):
    """Vegetation resource - high nutritional value."""
    resource_type: str = "plant"
    quantity: float = 100.0
    nutritional_value: float = field(default_factory=lambda: float(settings.PLANT_NUTRITIONAL_VALUE))
    regeneration_rate: float = field(default_factory=lambda: float(settings.PLANT_REGEN_RATE))


@dataclass
class Water(Resource):
    """Water resource - essential for survival."""
    resource_type: str = "water"
    quantity: float = 100.0
    nutritional_value: float = field(default_factory=lambda: float(settings.WATER_NUTRITIONAL_VALUE))
    regeneration_rate: float = field(default_factory=lambda: float(settings.WATER_REGEN_RATE))


@dataclass
class Mineral(Resource):
    """Mineral resource - lower value but long-lasting."""
    resource_type: str = "mineral"
    quantity: float = 100.0
    nutritional_value: float = field(default_factory=lambda: float(settings.MINERAL_NUTRITIONAL_VALUE))
    regeneration_rate: float = field(default_factory=lambda: float(settings.MINERAL_REGEN_RATE))


def create_resource(resource_type: str, position: Tuple[int, int]) -> Resource:
    """Factory function to create resources by type."""
    resource_map = {
        "plant": Plant,
        "water": Water,
        "mineral": Mineral,
    }
    resource_class = resource_map.get(resource_type, Resource)
    return resource_class(position=position)
