"""
NOPAINNOGAIN - Advanced Renderer
Beautiful graphics with particles, trails, weather effects, and UI
"""

from __future__ import annotations
import math
import random
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

import pygame
from pygame import gfxdraw

from modules.physics import Vector2
from modules.world import Weather, Season

if TYPE_CHECKING:
    from modules.creature import Creature
    from modules.world import World


@dataclass
class Particle:
    """Visual particle effect."""
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    size: float = 3.0
    lifetime: float = 1.0
    max_lifetime: float = 1.0
    gravity: float = 0.0
    fade: bool = True
    shrink: bool = True
    
    def update(self, dt: float = 1.0) -> bool:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += self.gravity * dt
        self.lifetime -= dt * 0.02
        return self.lifetime > 0
    
    def draw(self, screen: pygame.Surface) -> None:
        if self.lifetime <= 0:
            return
        
        alpha = int(255 * (self.lifetime / self.max_lifetime)) if self.fade else 255
        size = self.size * (self.lifetime / self.max_lifetime) if self.shrink else self.size
        size = max(1, int(size))
        
        if alpha > 0 and size > 0:
            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            color_alpha = (*self.color, alpha)
            pygame.draw.circle(surf, color_alpha, (size, size), size)
            screen.blit(surf, (int(self.x) - size, int(self.y) - size))


@dataclass
class RainDrop:
    """Rain particle."""
    x: float
    y: float
    speed: float = 8.0
    length: float = 10.0
    
    def update(self, wind_x: float = 0) -> bool:
        self.y += self.speed
        self.x += wind_x
        return self.y < 1000
    
    def draw(self, screen: pygame.Surface) -> None:
        color = (150, 150, 200, 100)
        end_y = self.y + self.length
        pygame.draw.line(screen, color[:3], (self.x, self.y), (self.x + 2, end_y), 1)


class Renderer:
    """Advanced rendering system."""
    
    def __init__(self, width: int = 1200, height: int = 900):
        pygame.init()
        pygame.display.set_caption("NOPAINNOGAIN - Evolutionary AI Ecosystem")
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        
        self.particles: List[Particle] = []
        self.rain_drops: List[RainDrop] = []
        
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 42)
        
        self.camera_x = 0
        self.camera_y = 0
        self.zoom = 1.0
        
        self.show_debug = False
        self.show_trails = True
        self.show_vision = False
        self.show_stats = True
        
        self.time = 0
    
    def world_to_screen(self, pos: Vector2) -> Tuple[int, int]:
        """Convert world position to screen coordinates."""
        x = (pos.x - self.camera_x) * self.zoom
        y = (pos.y - self.camera_y) * self.zoom
        return (int(x), int(y))
    
    def render(self, world: World, creatures: List[Creature], stats: Dict = None) -> None:
        """Render full frame."""
        self.time += 1
        
        bg_color = self._get_background_color(world)
        self.screen.fill(bg_color)
        
        self._draw_environment(world)
        
        self._draw_food_sources(world)
        self._draw_water_sources(world)
        
        if self.show_trails:
            self._draw_trails(creatures)
        
        prey = [c for c in creatures if not c.is_predator and c.alive]
        predators = [c for c in creatures if c.is_predator and c.alive]
        
        for creature in prey:
            self._draw_creature(creature, world)
        
        for creature in predators:
            self._draw_creature(creature, world)
        
        self._update_weather(world)
        self._update_particles()
        
        if self.show_stats:
            self._draw_ui(world, creatures, stats)
        
        pygame.display.flip()
    
    def _get_background_color(self, world: World) -> Tuple[int, int, int]:
        """Get background color based on time and weather."""
        base_day = (45, 85, 45)
        base_night = (15, 25, 35)
        
        t = world.ambient_light
        r = int(base_night[0] + (base_day[0] - base_night[0]) * t)
        g = int(base_night[1] + (base_day[1] - base_night[1]) * t)
        b = int(base_night[2] + (base_day[2] - base_night[2]) * t)
        
        if world.weather == Weather.RAIN:
            r = int(r * 0.7)
            g = int(g * 0.75)
            b = int(b * 0.9)
        elif world.weather == Weather.STORM:
            r = int(r * 0.5)
            g = int(g * 0.5)
            b = int(b * 0.6)
        elif world.weather == Weather.FOG:
            fog = (180, 180, 190)
            r = int(r * 0.5 + fog[0] * 0.5)
            g = int(g * 0.5 + fog[1] * 0.5)
            b = int(b * 0.5 + fog[2] * 0.5)
        
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    
    def _draw_environment(self, world: World) -> None:
        """Draw environmental features."""
        for obstacle in world.obstacles:
            pos = self.world_to_screen(obstacle.position)
            radius = int(obstacle.radius * self.zoom)
            
            pygame.draw.circle(self.screen, (80, 70, 60), pos, radius)
            pygame.draw.circle(self.screen, (60, 50, 40), pos, radius, 2)
            
            highlight = (pos[0] - radius // 3, pos[1] - radius // 3)
            pygame.draw.circle(self.screen, (100, 90, 80), highlight, radius // 4)
    
    def _draw_food_sources(self, world: World) -> None:
        """Draw food sources."""
        for food in world.food_sources:
            if food.is_depleted:
                continue
            
            pos = self.world_to_screen(food.position)
            fullness = food.quantity / food.max_quantity
            radius = int(food.radius * fullness * self.zoom)
            radius = max(3, radius)
            
            green = int(100 + 155 * fullness)
            color = (30, green, 30)
            
            pygame.draw.circle(self.screen, color, pos, radius)
            
            if radius > 5:
                highlight = (pos[0] - radius // 3, pos[1] - radius // 3)
                pygame.draw.circle(self.screen, (50, min(255, green + 50), 50), highlight, radius // 3)
    
    def _draw_water_sources(self, world: World) -> None:
        """Draw water sources."""
        for water in world.water_sources:
            pos = self.world_to_screen(water.position)
            radius = int(water.radius * self.zoom)
            
            wave = math.sin(self.time * 0.05) * 3
            
            pygame.draw.circle(self.screen, (40, 80, 150), pos, radius + int(wave))
            pygame.draw.circle(self.screen, (60, 100, 180), pos, int(radius * 0.7))
            
            for i in range(3):
                wave_r = int(radius * (0.3 + i * 0.2) + math.sin(self.time * 0.03 + i) * 5)
                pygame.draw.circle(self.screen, (80, 120, 200), pos, wave_r, 1)
    
    def _draw_trails(self, creatures: List[Creature]) -> None:
        """Draw creature movement trails."""
        for creature in creatures:
            if not creature.alive or len(creature.trail) < 2:
                continue
            
            points = [self.world_to_screen(p) for p in creature.trail]
            
            for i in range(len(points) - 1):
                alpha = int(100 * (i / len(points)))
                thickness = max(1, int(2 * (i / len(points))))
                
                color = creature.color if creature.is_predator else (200, 200, 150)
                color = (
                    max(0, color[0] - 50),
                    max(0, color[1] - 50),
                    max(0, color[2] - 50)
                )
                
                pygame.draw.line(self.screen, color, points[i], points[i + 1], thickness)
    
    def _draw_creature(self, creature: Creature, world: World) -> None:
        """Draw a single creature with all effects."""
        pos = self.world_to_screen(creature.position)
        size = int(creature.traits.size * self.zoom)
        
        if creature.is_predator:
            self._draw_predator(creature, pos, size, world)
        else:
            self._draw_prey(creature, pos, size, world)
        
        self._draw_health_bar(creature, pos, size)
        
        if self.show_vision and creature.is_predator:
            self._draw_vision_cone(creature, pos, world)
    
    def _draw_predator(self, creature: Creature, pos: Tuple[int, int], size: int, world: World) -> None:
        """Draw predator with special effects."""
        color = creature.color
        
        if creature.state.name == "HUNTING":
            pulse = abs(math.sin(self.time * 0.15)) * 0.3
            color = (
                min(255, int(color[0] * (1 + pulse))),
                int(color[1] * (1 - pulse * 0.5)),
                int(color[2] * (1 - pulse * 0.5))
            )
        
        body_rect = pygame.Rect(pos[0] - size, pos[1] - size // 2, size * 2, size)
        pygame.draw.ellipse(self.screen, color, body_rect)
        
        head_x = pos[0] + int(math.cos(creature.heading) * size)
        head_y = pos[1] + int(math.sin(creature.heading) * size)
        head_size = size // 2
        pygame.draw.circle(self.screen, color, (head_x, head_y), head_size)
        
        horn_len = size // 2
        for angle_offset in [-0.4, 0.4]:
            horn_angle = creature.heading + angle_offset - math.pi / 4
            horn_end_x = head_x + int(math.cos(horn_angle) * horn_len)
            horn_end_y = head_y + int(math.sin(horn_angle) * horn_len)
            pygame.draw.line(self.screen, (255, 255, 255), (head_x, head_y), (horn_end_x, horn_end_y), 2)
        
        eye_offset = head_size // 2
        for side in [-1, 1]:
            perp_angle = creature.heading + math.pi / 2
            eye_x = head_x + int(math.cos(perp_angle) * eye_offset * side * 0.5)
            eye_y = head_y + int(math.sin(perp_angle) * eye_offset * side * 0.5)
            pygame.draw.circle(self.screen, (255, 255, 0), (eye_x, eye_y), 3)
            pygame.draw.circle(self.screen, (0, 0, 0), (eye_x, eye_y), 1)
        
        pygame.draw.ellipse(self.screen, (max(0, color[0] - 30), max(0, color[1] - 20), max(0, color[2] - 20)), body_rect, 2)
    
    def _draw_prey(self, creature: Creature, pos: Tuple[int, int], size: int, world: World) -> None:
        """Draw prey with special effects."""
        color = creature.color
        
        if creature.state.name == "FLEEING":
            color = (
                min(255, color[0] + 30),
                min(255, color[1] + 30),
                color[2]
            )
        
        pygame.draw.circle(self.screen, color, pos, size)
        
        ear_size = size // 2
        for side in [-1, 1]:
            ear_x = pos[0] + side * (size // 2)
            ear_y = pos[1] - size
            ear_rect = pygame.Rect(ear_x - ear_size // 3, ear_y - ear_size, ear_size // 1.5, ear_size * 1.5)
            pygame.draw.ellipse(self.screen, color, ear_rect)
            
            inner_rect = pygame.Rect(ear_x - ear_size // 5, ear_y - ear_size + 3, ear_size // 2.5, ear_size)
            pygame.draw.ellipse(self.screen, (255, 200, 200), inner_rect)
        
        eye_offset = size // 3
        for side in [-1, 1]:
            eye_x = pos[0] + side * eye_offset
            eye_y = pos[1] - size // 4
            pygame.draw.circle(self.screen, (0, 0, 0), (eye_x, eye_y), 3)
            pygame.draw.circle(self.screen, (255, 255, 255), (eye_x - 1, eye_y - 1), 1)
        
        nose_y = pos[1] + size // 4
        pygame.draw.circle(self.screen, (255, 150, 150), (pos[0], nose_y), 2)
        
        tail_x = pos[0] - int(math.cos(creature.heading) * size * 1.2)
        tail_y = pos[1] - int(math.sin(creature.heading) * size * 1.2)
        pygame.draw.circle(self.screen, (255, 255, 255), (tail_x, tail_y), size // 3)
        
        pygame.draw.circle(self.screen, (max(0, color[0] - 40), max(0, color[1] - 40), max(0, color[2] - 30)), pos, size, 1)
    
    def _draw_health_bar(self, creature: Creature, pos: Tuple[int, int], size: int) -> None:
        """Draw health and energy bars."""
        bar_width = size * 2
        bar_height = 4
        bar_y = pos[1] + size + 5
        
        health_pct = creature.health / creature.max_health
        pygame.draw.rect(self.screen, (60, 60, 60), (pos[0] - bar_width // 2, bar_y, bar_width, bar_height))
        health_color = (50, 200, 50) if health_pct > 0.3 else (200, 50, 50)
        pygame.draw.rect(self.screen, health_color, (pos[0] - bar_width // 2, bar_y, int(bar_width * health_pct), bar_height))
        
        energy_pct = creature.energy / creature.max_energy
        energy_y = bar_y + bar_height + 2
        pygame.draw.rect(self.screen, (40, 40, 40), (pos[0] - bar_width // 2, energy_y, bar_width, 3))
        pygame.draw.rect(self.screen, (50, 150, 255), (pos[0] - bar_width // 2, energy_y, int(bar_width * energy_pct), 3))
    
    def _draw_vision_cone(self, creature: Creature, pos: Tuple[int, int], world: World) -> None:
        """Draw creature's vision cone."""
        range_px = int(creature.traits.vision_range * world.get_vision_modifier() * self.zoom)
        half_angle = creature.traits.vision_angle / 2
        
        points = [pos]
        num_segments = 20
        for i in range(num_segments + 1):
            angle = creature.heading - half_angle + (half_angle * 2 * i / num_segments)
            x = pos[0] + int(math.cos(angle) * range_px)
            y = pos[1] + int(math.sin(angle) * range_px)
            points.append((x, y))
        
        if len(points) > 2:
            surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.polygon(surf, (255, 255, 0, 30), points)
            self.screen.blit(surf, (0, 0))
    
    def _update_weather(self, world: World) -> None:
        """Update and draw weather effects."""
        if world.weather == Weather.RAIN:
            if random.random() < 0.3:
                self.rain_drops.append(RainDrop(
                    x=random.uniform(0, self.width),
                    y=0,
                    speed=random.uniform(6, 12)
                ))
        elif world.weather == Weather.STORM:
            if random.random() < 0.5:
                self.rain_drops.append(RainDrop(
                    x=random.uniform(0, self.width),
                    y=0,
                    speed=random.uniform(10, 18),
                    length=random.uniform(15, 25)
                ))
            
            if random.random() < 0.002:
                self.screen.fill((255, 255, 255))
        
        wind_x = world.wind_direction.x * world.wind_strength * 3
        self.rain_drops = [drop for drop in self.rain_drops if drop.update(wind_x)]
        for drop in self.rain_drops:
            drop.draw(self.screen)
        
        if world.weather == Weather.FOG:
            fog_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            fog_surf.fill((200, 200, 210, int(60 * (1 - world.ambient_light))))
            self.screen.blit(fog_surf, (0, 0))
    
    def _update_particles(self) -> None:
        """Update and draw particles."""
        self.particles = [p for p in self.particles if p.update()]
        for particle in self.particles:
            particle.draw(self.screen)
    
    def _draw_ui(self, world: World, creatures: List[Creature], stats: Dict = None) -> None:
        """Draw UI overlay."""
        panel = pygame.Surface((220, 200), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        self.screen.blit(panel, (10, 10))
        
        y = 20
        prey_count = sum(1 for c in creatures if not c.is_predator and c.alive)
        predator_count = sum(1 for c in creatures if c.is_predator and c.alive)
        
        title = self.font_large.render("NOPAINNOGAIN", True, (255, 255, 255))
        self.screen.blit(title, (20, y))
        y += 35
        
        time_str = "Day" if world.is_day else "Night"
        time_text = self.font_medium.render(f"Day {world.current_day} - {time_str}", True, (200, 200, 200))
        self.screen.blit(time_text, (20, y))
        y += 25
        
        weather_text = self.font_small.render(f"Weather: {world.weather.name}", True, (180, 180, 180))
        self.screen.blit(weather_text, (20, y))
        y += 20
        
        season_text = self.font_small.render(f"Season: {world.season.name}", True, (180, 180, 180))
        self.screen.blit(season_text, (20, y))
        y += 25
        
        prey_text = self.font_medium.render(f"🐰 Prey: {prey_count}", True, (255, 215, 0))
        self.screen.blit(prey_text, (20, y))
        y += 25
        
        pred_text = self.font_medium.render(f"🦁 Predators: {predator_count}", True, (220, 60, 60))
        self.screen.blit(pred_text, (20, y))
        y += 25
        
        if stats:
            kills_text = self.font_small.render(f"Kills: {stats.get('kills', 0)}", True, (200, 100, 100))
            self.screen.blit(kills_text, (20, y))
            y += 18
            
            births_text = self.font_small.render(f"Births: {stats.get('births', 0)}", True, (100, 200, 100))
            self.screen.blit(births_text, (20, y))
        
        controls = [
            "ESC - Quit",
            "T - Toggle trails",
            "V - Toggle vision",
            "D - Toggle debug"
        ]
        
        ctrl_y = self.height - 80
        for ctrl in controls:
            text = self.font_small.render(ctrl, True, (150, 150, 150))
            self.screen.blit(text, (10, ctrl_y))
            ctrl_y += 18
    
    def spawn_death_particles(self, pos: Vector2) -> None:
        """Spawn death effect."""
        screen_pos = self.world_to_screen(pos)
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            self.particles.append(Particle(
                x=screen_pos[0],
                y=screen_pos[1],
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                color=(200, 50, 50),
                size=random.uniform(3, 6),
                lifetime=1.0,
                gravity=0.1
            ))
    
    def spawn_birth_particles(self, pos: Vector2) -> None:
        """Spawn birth effect."""
        screen_pos = self.world_to_screen(pos)
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append(Particle(
                x=screen_pos[0],
                y=screen_pos[1],
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed - 2,
                color=(100, 255, 100),
                size=random.uniform(2, 5),
                lifetime=1.0,
                gravity=-0.05
            ))
    
    def spawn_attack_particles(self, pos: Vector2) -> None:
        """Spawn attack effect."""
        screen_pos = self.world_to_screen(pos)
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(3, 8)
            self.particles.append(Particle(
                x=screen_pos[0],
                y=screen_pos[1],
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                color=(255, 100, 50),
                size=random.uniform(4, 8),
                lifetime=0.5,
                shrink=True
            ))
    
    def quit(self) -> None:
        """Clean up."""
        pygame.quit()
