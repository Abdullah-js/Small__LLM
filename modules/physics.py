"""
NOPAINNOGAIN - Physics Engine
Continuous movement with velocity, acceleration, and forces
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from config import settings


@dataclass
class Vector2:
    """2D Vector for physics calculations."""
    x: float = 0.0
    y: float = 0.0
    
    def __add__(self, other: Vector2) -> Vector2:
        return Vector2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: Vector2) -> Vector2:
        return Vector2(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> Vector2:
        return Vector2(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar: float) -> Vector2:
        if scalar == 0:
            return Vector2(0, 0)
        return Vector2(self.x / scalar, self.y / scalar)
    
    def __neg__(self) -> Vector2:
        return Vector2(-self.x, -self.y)
    
    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    @property
    def magnitude_squared(self) -> float:
        return self.x * self.x + self.y * self.y
    
    def normalized(self) -> Vector2:
        mag = self.magnitude
        if mag == 0:
            return Vector2(0, 0)
        return Vector2(self.x / mag, self.y / mag)
    
    def limit(self, max_mag: float) -> Vector2:
        if self.magnitude_squared > max_mag * max_mag:
            return self.normalized() * max_mag
        return Vector2(self.x, self.y)
    
    def distance_to(self, other: Vector2) -> float:
        return (self - other).magnitude
    
    def angle_to(self, other: Vector2) -> float:
        diff = other - self
        return math.atan2(diff.y, diff.x)
    
    def rotate(self, angle: float) -> Vector2:
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vector2(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )
    
    def dot(self, other: Vector2) -> float:
        return self.x * other.x + self.y * other.y
    
    def copy(self) -> Vector2:
        return Vector2(self.x, self.y)
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    @staticmethod
    def random_unit() -> Vector2:
        angle = random.uniform(0, 2 * math.pi)
        return Vector2(math.cos(angle), math.sin(angle))
    
    @staticmethod
    def from_angle(angle: float) -> Vector2:
        return Vector2(math.cos(angle), math.sin(angle))


@dataclass
class PhysicsBody:
    """Physics body with position, velocity, acceleration."""
    position: Vector2 = field(default_factory=Vector2)
    velocity: Vector2 = field(default_factory=Vector2)
    acceleration: Vector2 = field(default_factory=Vector2)
    
    mass: float = 1.0
    max_speed: float = 5.0
    max_force: float = 0.5
    friction: float = 0.98
    
    heading: float = 0.0
    angular_velocity: float = 0.0
    
    def apply_force(self, force: Vector2) -> None:
        """Apply force to body (F = ma, so a = F/m)."""
        self.acceleration = self.acceleration + (force / self.mass)
    
    def update(self, dt: float = 1.0) -> None:
        """Update physics state."""
        self.velocity = self.velocity + (self.acceleration * dt)
        self.velocity = self.velocity.limit(self.max_speed)
        self.velocity = self.velocity * self.friction
        
        self.position = self.position + (self.velocity * dt)
        
        if self.velocity.magnitude > 0.01:
            self.heading = math.atan2(self.velocity.y, self.velocity.x)
        
        self.acceleration = Vector2(0, 0)
    
    def seek(self, target: Vector2) -> Vector2:
        """Calculate steering force toward target."""
        desired = target - self.position
        desired = desired.normalized() * self.max_speed
        steer = desired - self.velocity
        return steer.limit(self.max_force)
    
    def flee(self, target: Vector2) -> Vector2:
        """Calculate steering force away from target."""
        return -self.seek(target)
    
    def arrive(self, target: Vector2, slowing_radius: float = 100.0) -> Vector2:
        """Seek with slowing near target."""
        desired = target - self.position
        distance = desired.magnitude
        
        if distance < slowing_radius:
            speed = self.max_speed * (distance / slowing_radius)
        else:
            speed = self.max_speed
        
        desired = desired.normalized() * speed
        steer = desired - self.velocity
        return steer.limit(self.max_force)
    
    def pursue(self, target_body: PhysicsBody, prediction_time: float = 1.0) -> Vector2:
        """Pursue moving target by predicting future position."""
        future_pos = target_body.position + (target_body.velocity * prediction_time)
        return self.seek(future_pos)
    
    def evade(self, pursuer_body: PhysicsBody, prediction_time: float = 1.0) -> Vector2:
        """Evade by fleeing from predicted position."""
        future_pos = pursuer_body.position + (pursuer_body.velocity * prediction_time)
        return self.flee(future_pos)
    
    def wander(self, wander_radius: float = 50.0, wander_distance: float = 80.0, 
               wander_jitter: float = 0.3) -> Vector2:
        """Random wandering behavior."""
        jitter = Vector2.random_unit() * wander_jitter
        
        wander_target = Vector2.from_angle(self.heading) * wander_distance
        wander_target = wander_target + (jitter * wander_radius)
        
        target = self.position + wander_target
        return self.seek(target)
    
    def wrap_around(self, width: float, height: float) -> None:
        """Wrap position around world boundaries."""
        if self.position.x < 0:
            self.position.x = width
        elif self.position.x > width:
            self.position.x = 0
        
        if self.position.y < 0:
            self.position.y = height
        elif self.position.y > height:
            self.position.y = 0
    
    def bounce(self, width: float, height: float, damping: float = 0.8) -> None:
        """Bounce off world boundaries."""
        if self.position.x < 0:
            self.position.x = 0
            self.velocity.x *= -damping
        elif self.position.x > width:
            self.position.x = width
            self.velocity.x *= -damping
        
        if self.position.y < 0:
            self.position.y = 0
            self.velocity.y *= -damping
        elif self.position.y > height:
            self.position.y = height
            self.velocity.y *= -damping


def separation(body: PhysicsBody, neighbors: List[PhysicsBody], 
               desired_separation: float = 25.0) -> Vector2:
    """Steer to avoid crowding local flockmates."""
    steer = Vector2(0, 0)
    count = 0
    
    for other in neighbors:
        d = body.position.distance_to(other.position)
        if 0 < d < desired_separation:
            diff = body.position - other.position
            diff = diff.normalized() / d
            steer = steer + diff
            count += 1
    
    if count > 0:
        steer = steer / count
        if steer.magnitude > 0:
            steer = steer.normalized() * body.max_speed - body.velocity
            steer = steer.limit(body.max_force)
    
    return steer


def alignment(body: PhysicsBody, neighbors: List[PhysicsBody]) -> Vector2:
    """Steer towards average heading of local flockmates."""
    avg_velocity = Vector2(0, 0)
    count = 0
    
    for other in neighbors:
        avg_velocity = avg_velocity + other.velocity
        count += 1
    
    if count > 0:
        avg_velocity = avg_velocity / count
        avg_velocity = avg_velocity.normalized() * body.max_speed
        steer = avg_velocity - body.velocity
        return steer.limit(body.max_force)
    
    return Vector2(0, 0)


def cohesion(body: PhysicsBody, neighbors: List[PhysicsBody]) -> Vector2:
    """Steer toward average position of local flockmates."""
    center = Vector2(0, 0)
    count = 0
    
    for other in neighbors:
        center = center + other.position
        count += 1
    
    if count > 0:
        center = center / count
        return body.seek(center)
    
    return Vector2(0, 0)


def flock(body: PhysicsBody, neighbors: List[PhysicsBody],
          sep_weight: float = 1.5, align_weight: float = 1.0, 
          coh_weight: float = 1.0) -> Vector2:
    """Combined flocking behavior."""
    sep = separation(body, neighbors) * sep_weight
    ali = alignment(body, neighbors) * align_weight
    coh = cohesion(body, neighbors) * coh_weight
    return sep + ali + coh
