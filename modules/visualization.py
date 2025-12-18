"""
NOPAINNOGAIN - Visualization Module
Enhanced with predator-prey visual effects
"""

from __future__ import annotations
import math
import random
from typing import List, Dict, TYPE_CHECKING

import pygame

from config import settings, constants

if TYPE_CHECKING:
    from modules.agent import Agent
    from modules.environment import Environment


class Particle:
    """Visual particle for effects."""
    
    def __init__(self, x: float, y: float, color: tuple, velocity: tuple = None, size: int = 3):
        self.x = x
        self.y = y
        self.color = color
        self.velocity = velocity or (
            random.uniform(-3, 3),
            random.uniform(-3, 3)
        )
        self.lifetime = 255
        self.decay_rate = random.randint(5, 15)
        self.size = size
    
    def update(self) -> bool:
        """Update particle and return True if still alive."""
        self.x += self.velocity[0]
        self.y += self.velocity[1]
        self.lifetime -= self.decay_rate
        return self.lifetime > 0
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw particle with fading alpha."""
        if self.lifetime <= 0:
            return
        
        alpha = max(0, min(255, self.lifetime))
        surface = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        color_with_alpha = (*self.color, alpha)
        pygame.draw.circle(surface, color_with_alpha, (self.size, self.size), self.size)
        screen.blit(surface, (int(self.x) - self.size, int(self.y) - self.size))


class Visualization:
    """Pygame-based visualization with predator-prey effects."""
    
    def __init__(self, environment: Environment):
        pygame.init()
        pygame.display.set_caption("NOPAINNOGAIN - Predator-Prey AI Ecosystem")
        
        self.screen = pygame.display.set_mode((
            settings.SCREEN_WIDTH,
            settings.SCREEN_HEIGHT
        ))
        
        self.environment = environment
        self.particles: List[Particle] = []
        
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        
        self.camera_x = 0
        self.camera_y = 0
    
    def render(self, agents: List[Agent]) -> None:
        """Render full frame."""
        self.screen.fill(settings.COLORS["background"])
        
        self._draw_grid()
        
        prey = [a for a in agents if not a.is_predator and a.alive]
        predators = [a for a in agents if a.is_predator and a.alive]
        
        for agent in prey:
            self._draw_agent(agent)
        
        for agent in predators:
            self._draw_agent(agent)
        
        self._update_particles()
        
        pygame.display.flip()
    
    def _draw_grid(self) -> None:
        """Draw environment grid with resources and hazards."""
        tile = settings.TILE_SIZE
        
        for y in range(self.environment.height):
            for x in range(self.environment.width):
                screen_x = x * tile - self.camera_x
                screen_y = y * tile - self.camera_y
                
                if not (-tile < screen_x < settings.SCREEN_WIDTH and
                        -tile < screen_y < settings.SCREEN_HEIGHT):
                    continue
                
                cell = self.environment.grid[y][x]
                rect = pygame.Rect(screen_x, screen_y, tile, tile)
                
                color = settings.COLORS["ground"]
                
                resources = cell.get("resources", {})
                if "plant" in resources and resources["plant"].quantity > 0:
                    color = settings.COLORS["resource_plant"]
                elif "water" in resources and resources["water"].quantity > 0:
                    color = settings.COLORS["resource_water"]
                elif "mineral" in resources and resources["mineral"].quantity > 0:
                    color = settings.COLORS["resource_mineral"]
                
                hazard = cell.get("hazards", 0)
                if hazard > 0:
                    intensity = min(1.0, hazard / settings.MAX_HAZARD_VALUE)
                    hazard_color = settings.COLORS["hazard"]
                    color = (
                        int(color[0] * (1 - intensity) + hazard_color[0] * intensity),
                        int(color[1] * (1 - intensity) + hazard_color[1] * intensity),
                        int(color[2] * (1 - intensity) + hazard_color[2] * intensity),
                    )
                
                pygame.draw.rect(self.screen, color, rect)
    
    def _draw_agent(self, agent: Agent) -> None:
        """Draw single agent with species-specific appearance."""
        tile = settings.TILE_SIZE
        x, y = agent.position
        
        screen_x = x * tile - self.camera_x
        screen_y = y * tile - self.camera_y
        
        center_x = screen_x + tile // 2
        center_y = screen_y + tile // 2
        
        energy_ratio = agent.energy / settings.MAX_ENERGY
        base_radius = tile // 2 - 2
        radius = max(3, int(base_radius * (0.5 + 0.5 * energy_ratio)))
        
        if agent.is_predator:
            self._draw_predator(center_x, center_y, radius, agent)
        else:
            self._draw_prey(center_x, center_y, radius, agent)
        
        if settings.VIS.SHOW_HEALTH_BARS:
            self._draw_health_bar(screen_x, screen_y + tile - 6, tile, agent)
    
    def _draw_predator(self, cx: int, cy: int, radius: int, agent: Agent) -> None:
        """Draw predator with distinctive features."""
        color = agent.color
        
        if agent.state == constants.AgentState.HUNTING:
            pulse = abs(math.sin(pygame.time.get_ticks() * 0.01)) * 30
            color = (
                min(255, color[0] + int(pulse)),
                color[1],
                color[2]
            )
        
        pygame.draw.circle(self.screen, color, (cx, cy), radius)
        
        pygame.draw.circle(self.screen, (100, 0, 0), (cx, cy), radius, 2)
        
        self._draw_horns(cx, cy, radius)
        
        eye_offset = radius // 3
        eye_radius = max(2, radius // 4)
        pygame.draw.circle(self.screen, (255, 255, 255), (cx - eye_offset, cy - eye_offset), eye_radius)
        pygame.draw.circle(self.screen, (255, 255, 255), (cx + eye_offset, cy - eye_offset), eye_radius)
        pygame.draw.circle(self.screen, (0, 0, 0), (cx - eye_offset, cy - eye_offset), eye_radius // 2)
        pygame.draw.circle(self.screen, (0, 0, 0), (cx + eye_offset, cy - eye_offset), eye_radius // 2)
    
    def _draw_prey(self, cx: int, cy: int, radius: int, agent: Agent) -> None:
        """Draw prey with distinctive features."""
        color = agent.color
        
        if agent.state == constants.AgentState.FLEEING:
            pulse = abs(math.sin(pygame.time.get_ticks() * 0.02)) * 50
            color = (
                min(255, color[0] + int(pulse * 0.5)),
                min(255, color[1] + int(pulse * 0.5)),
                color[2]
            )
        
        pygame.draw.circle(self.screen, color, (cx, cy), radius)
        
        pygame.draw.circle(self.screen, (180, 160, 0), (cx, cy), radius, 1)
        
        ear_size = radius // 2
        pygame.draw.ellipse(
            self.screen, color,
            (cx - radius, cy - radius - ear_size, ear_size, ear_size * 2)
        )
        pygame.draw.ellipse(
            self.screen, color,
            (cx + radius - ear_size, cy - radius - ear_size, ear_size, ear_size * 2)
        )
    
    def _draw_horns(self, cx: int, cy: int, radius: int) -> None:
        """Draw predator horns."""
        horn_color = settings.COLORS["horns"]
        horn_len = int(radius * 0.8)
        
        pygame.draw.line(
            self.screen, horn_color,
            (cx - radius // 2, cy - radius // 2),
            (cx - radius // 2 - horn_len // 2, cy - radius - horn_len),
            2
        )
        pygame.draw.line(
            self.screen, horn_color,
            (cx + radius // 2, cy - radius // 2),
            (cx + radius // 2 + horn_len // 2, cy - radius - horn_len),
            2
        )
    
    def _draw_health_bar(self, x: int, y: int, width: int, agent: Agent) -> None:
        """Draw health bar below agent."""
        bar_height = 4
        bar_width = width - 4
        bar_x = x + 2
        
        pygame.draw.rect(
            self.screen,
            settings.COLORS["health_bar_bg"],
            (bar_x, y, bar_width, bar_height)
        )
        
        health_pct = max(0, min(1, agent.health / settings.MAX_HEALTH))
        fill_width = int(bar_width * health_pct)
        
        health_color = settings.COLORS["health_bar_full"] if health_pct > 0.3 else settings.COLORS["health_bar_low"]
        
        if fill_width > 0:
            pygame.draw.rect(
                self.screen,
                health_color,
                (bar_x, y, fill_width, bar_height)
            )
    
    def spawn_death_particles(self, position: tuple) -> None:
        """Spawn death effect particles."""
        x, y = position
        tile = settings.TILE_SIZE
        center_x = x * tile + tile // 2 - self.camera_x
        center_y = y * tile + tile // 2 - self.camera_y
        
        for _ in range(20):
            self.particles.append(Particle(
                center_x, center_y,
                settings.COLORS["death_particle"],
                size=random.randint(2, 4)
            ))
    
    def spawn_birth_particles(self, position: tuple) -> None:
        """Spawn birth effect particles."""
        x, y = position
        tile = settings.TILE_SIZE
        center_x = x * tile + tile // 2 - self.camera_x
        center_y = y * tile + tile // 2 - self.camera_y
        
        for _ in range(15):
            self.particles.append(Particle(
                center_x, center_y,
                settings.COLORS["birth_particle"],
                size=random.randint(2, 3)
            ))
    
    def spawn_attack_particles(self, position: tuple) -> None:
        """Spawn attack effect particles (red burst)."""
        x, y = position
        tile = settings.TILE_SIZE
        center_x = x * tile + tile // 2 - self.camera_x
        center_y = y * tile + tile // 2 - self.camera_y
        
        for _ in range(25):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append(Particle(
                center_x, center_y,
                settings.COLORS["attack_particle"],
                velocity=velocity,
                size=random.randint(3, 5)
            ))
    
    def spawn_flee_particles(self, position: tuple) -> None:
        """Spawn flee effect particles (blue trail)."""
        x, y = position
        tile = settings.TILE_SIZE
        center_x = x * tile + tile // 2 - self.camera_x
        center_y = y * tile + tile // 2 - self.camera_y
        
        for _ in range(8):
            velocity = (random.uniform(-1, 1), random.uniform(1, 3))
            self.particles.append(Particle(
                center_x, center_y,
                settings.COLORS["flee_particle"],
                velocity=velocity,
                size=random.randint(2, 3)
            ))
    
    def spawn_hunt_particles(self, position: tuple) -> None:
        """Spawn hunt effect particles (orange trail)."""
        x, y = position
        tile = settings.TILE_SIZE
        center_x = x * tile + tile // 2 - self.camera_x
        center_y = y * tile + tile // 2 - self.camera_y
        
        for _ in range(5):
            velocity = (random.uniform(-2, 2), random.uniform(-2, 2))
            self.particles.append(Particle(
                center_x, center_y,
                settings.COLORS["hunt_particle"],
                velocity=velocity,
                size=2
            ))
    
    def _update_particles(self) -> None:
        """Update and draw all particles."""
        alive_particles = []
        for particle in self.particles:
            if particle.update():
                particle.draw(self.screen)
                alive_particles.append(particle)
        self.particles = alive_particles
    
    def quit(self) -> None:
        """Clean up pygame."""
        pygame.quit()
