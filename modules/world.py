"""
NOPAINNOGAIN - World System
Continuous world with resources, weather, day/night cycle
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum, auto

from modules.physics import Vector2


class Weather(Enum):
    CLEAR = auto()
    CLOUDY = auto()
    RAIN = auto()
    STORM = auto()
    FOG = auto()


class Season(Enum):
    SPRING = auto()
    SUMMER = auto()
    AUTUMN = auto()
    WINTER = auto()


@dataclass
class FoodSource:
    """A food resource in the world."""
    position: Vector2
    quantity: float = 100.0
    max_quantity: float = 100.0
    regen_rate: float = 0.5
    nutrition_value: float = 20.0
    radius: float = 15.0
    food_type: str = "plant"
    
    def consume(self, amount: float) -> float:
        """Consume food and return actual amount eaten."""
        eaten = min(amount, self.quantity)
        self.quantity -= eaten
        return eaten * self.nutrition_value
    
    def regenerate(self, dt: float = 1.0, season_modifier: float = 1.0) -> None:
        """Regenerate food over time."""
        if self.quantity < self.max_quantity:
            self.quantity = min(
                self.max_quantity,
                self.quantity + self.regen_rate * dt * season_modifier
            )
    
    @property
    def is_depleted(self) -> bool:
        return self.quantity <= 0


@dataclass
class WaterSource:
    """A water source in the world."""
    position: Vector2
    radius: float = 30.0
    hydration_value: float = 15.0


@dataclass
class Obstacle:
    """An obstacle that blocks movement."""
    position: Vector2
    radius: float = 20.0
    obstacle_type: str = "rock"


@dataclass
class ScentTrail:
    """A smell trail left by creatures."""
    position: Vector2
    strength: float = 1.0
    decay_rate: float = 0.02
    scent_type: str = "prey"
    
    def update(self, dt: float = 1.0) -> bool:
        """Update scent, return False if should be removed."""
        self.strength -= self.decay_rate * dt
        return self.strength > 0.05


@dataclass
class World:
    """The simulation world with all environmental features."""
    
    width: float = 1200.0
    height: float = 900.0
    
    time_of_day: float = 0.5
    day_length: float = 2000.0
    current_day: int = 0
    
    weather: Weather = Weather.CLEAR
    weather_duration: float = 500.0
    weather_timer: float = 0.0
    
    season: Season = Season.SPRING
    season_length: float = 10000.0
    season_timer: float = 0.0
    
    food_sources: List[FoodSource] = field(default_factory=list)
    water_sources: List[WaterSource] = field(default_factory=list)
    obstacles: List[Obstacle] = field(default_factory=list)
    scent_trails: List[ScentTrail] = field(default_factory=list)
    
    ambient_light: float = 1.0
    temperature: float = 20.0
    wind_direction: Vector2 = field(default_factory=lambda: Vector2(1, 0))
    wind_strength: float = 0.0
    
    def __post_init__(self):
        if not self.food_sources:
            self._generate_resources()
    
    def _generate_resources(self) -> None:
        """Generate initial food and water sources."""
        num_food = int((self.width * self.height) / 8000)
        for _ in range(num_food):
            self.food_sources.append(FoodSource(
                position=Vector2(
                    random.uniform(50, self.width - 50),
                    random.uniform(50, self.height - 50)
                ),
                quantity=random.uniform(50, 100),
                regen_rate=random.uniform(0.3, 0.8),
            ))
        
        num_water = max(3, int((self.width * self.height) / 50000))
        for _ in range(num_water):
            self.water_sources.append(WaterSource(
                position=Vector2(
                    random.uniform(100, self.width - 100),
                    random.uniform(100, self.height - 100)
                ),
                radius=random.uniform(40, 80),
            ))
        
        num_obstacles = int((self.width * self.height) / 30000)
        for _ in range(num_obstacles):
            self.obstacles.append(Obstacle(
                position=Vector2(
                    random.uniform(50, self.width - 50),
                    random.uniform(50, self.height - 50)
                ),
                radius=random.uniform(15, 40),
            ))
    
    def update(self, dt: float = 1.0) -> None:
        """Update world state."""
        self._update_time(dt)
        self._update_weather(dt)
        self._update_season(dt)
        self._update_resources(dt)
        self._update_scents(dt)
        self._update_environment()
    
    def _update_time(self, dt: float) -> None:
        """Update time of day."""
        self.time_of_day += dt / self.day_length
        if self.time_of_day >= 1.0:
            self.time_of_day = 0.0
            self.current_day += 1
    
    def _update_weather(self, dt: float) -> None:
        """Update weather conditions."""
        self.weather_timer += dt
        if self.weather_timer >= self.weather_duration:
            self.weather_timer = 0
            self.weather_duration = random.uniform(300, 800)
            
            weights = [0.4, 0.25, 0.2, 0.05, 0.1]
            if self.season == Season.SPRING:
                weights = [0.3, 0.2, 0.35, 0.05, 0.1]
            elif self.season == Season.SUMMER:
                weights = [0.5, 0.25, 0.1, 0.1, 0.05]
            elif self.season == Season.AUTUMN:
                weights = [0.3, 0.3, 0.2, 0.1, 0.1]
            elif self.season == Season.WINTER:
                weights = [0.3, 0.3, 0.15, 0.05, 0.2]
            
            self.weather = random.choices(list(Weather), weights=weights)[0]
            
            if self.weather == Weather.CLEAR:
                self.wind_strength = random.uniform(0, 0.3)
            elif self.weather == Weather.STORM:
                self.wind_strength = random.uniform(0.5, 1.0)
                self.wind_direction = Vector2.random_unit()
            else:
                self.wind_strength = random.uniform(0.1, 0.5)
    
    def _update_season(self, dt: float) -> None:
        """Update season."""
        self.season_timer += dt
        if self.season_timer >= self.season_length:
            self.season_timer = 0
            seasons = list(Season)
            current_idx = seasons.index(self.season)
            self.season = seasons[(current_idx + 1) % len(seasons)]
    
    def _update_resources(self, dt: float) -> None:
        """Update food regeneration."""
        season_modifier = {
            Season.SPRING: 1.5,
            Season.SUMMER: 1.2,
            Season.AUTUMN: 0.8,
            Season.WINTER: 0.3,
        }.get(self.season, 1.0)
        
        for food in self.food_sources:
            food.regenerate(dt, season_modifier)
        
        if random.random() < 0.001 * dt:
            self.food_sources.append(FoodSource(
                position=Vector2(
                    random.uniform(50, self.width - 50),
                    random.uniform(50, self.height - 50)
                ),
            ))
    
    def _update_scents(self, dt: float) -> None:
        """Update and decay scent trails."""
        self.scent_trails = [s for s in self.scent_trails if s.update(dt)]
        
        if self.wind_strength > 0:
            for scent in self.scent_trails:
                scent.position = scent.position + (self.wind_direction * self.wind_strength * dt * 0.5)
    
    def _update_environment(self) -> None:
        """Update ambient conditions based on time and weather."""
        day_progress = self.time_of_day * 2 * math.pi
        base_light = (math.sin(day_progress - math.pi/2) + 1) / 2
        
        weather_light = {
            Weather.CLEAR: 1.0,
            Weather.CLOUDY: 0.7,
            Weather.RAIN: 0.5,
            Weather.STORM: 0.3,
            Weather.FOG: 0.6,
        }.get(self.weather, 1.0)
        
        self.ambient_light = base_light * weather_light
        self.ambient_light = max(0.1, min(1.0, self.ambient_light))
        
        base_temp = {
            Season.SPRING: 15,
            Season.SUMMER: 25,
            Season.AUTUMN: 12,
            Season.WINTER: 0,
        }.get(self.season, 15)
        
        day_temp_mod = (base_light - 0.5) * 10
        weather_temp_mod = {
            Weather.CLEAR: 2,
            Weather.CLOUDY: -2,
            Weather.RAIN: -5,
            Weather.STORM: -8,
            Weather.FOG: -3,
        }.get(self.weather, 0)
        
        self.temperature = base_temp + day_temp_mod + weather_temp_mod
    
    def add_scent(self, position: Vector2, scent_type: str = "prey", strength: float = 1.0) -> None:
        """Add a scent trail at position."""
        if len(self.scent_trails) < 500:
            self.scent_trails.append(ScentTrail(
                position=position.copy(),
                strength=strength,
                scent_type=scent_type,
            ))
    
    def find_nearest_food(self, position: Vector2, max_range: float = 200.0) -> Optional[FoodSource]:
        """Find nearest food source within range."""
        nearest = None
        nearest_dist = max_range
        
        for food in self.food_sources:
            if food.is_depleted:
                continue
            dist = position.distance_to(food.position)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = food
        
        return nearest
    
    def find_nearest_water(self, position: Vector2, max_range: float = 300.0) -> Optional[WaterSource]:
        """Find nearest water source within range."""
        nearest = None
        nearest_dist = max_range
        
        for water in self.water_sources:
            dist = position.distance_to(water.position)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = water
        
        return nearest
    
    def find_scents(self, position: Vector2, scent_type: str, 
                   max_range: float = 100.0) -> List[Tuple[ScentTrail, float]]:
        """Find all scents of type within range, with distances."""
        results = []
        for scent in self.scent_trails:
            if scent.scent_type == scent_type:
                dist = position.distance_to(scent.position)
                if dist < max_range:
                    results.append((scent, dist))
        return sorted(results, key=lambda x: x[1])
    
    def check_collision(self, position: Vector2, radius: float = 10.0) -> Optional[Obstacle]:
        """Check if position collides with any obstacle."""
        for obstacle in self.obstacles:
            dist = position.distance_to(obstacle.position)
            if dist < radius + obstacle.radius:
                return obstacle
        return None
    
    @property
    def is_day(self) -> bool:
        """Check if it's daytime."""
        return 0.25 <= self.time_of_day <= 0.75
    
    @property
    def is_night(self) -> bool:
        """Check if it's nighttime."""
        return not self.is_day
    
    def get_vision_modifier(self) -> float:
        """Get vision range modifier based on conditions."""
        light_mod = 0.5 + 0.5 * self.ambient_light
        weather_mod = {
            Weather.CLEAR: 1.0,
            Weather.CLOUDY: 0.9,
            Weather.RAIN: 0.7,
            Weather.STORM: 0.5,
            Weather.FOG: 0.4,
        }.get(self.weather, 1.0)
        return light_mod * weather_mod
    
    def get_hearing_modifier(self) -> float:
        """Get hearing range modifier based on conditions."""
        return {
            Weather.CLEAR: 1.0,
            Weather.CLOUDY: 1.0,
            Weather.RAIN: 0.6,
            Weather.STORM: 0.3,
            Weather.FOG: 1.1,
        }.get(self.weather, 1.0)
