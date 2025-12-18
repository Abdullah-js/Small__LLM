"""
NOPAINNOGAIN - Neural Network Brain
Simple feedforward neural network that evolves through natural selection
"""

from __future__ import annotations
import math
import random
import json
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


def sigmoid(x: float) -> float:
    """Sigmoid activation function."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def tanh(x: float) -> float:
    """Tanh activation function."""
    return math.tanh(x)


def relu(x: float) -> float:
    """ReLU activation function."""
    return max(0, x)


def leaky_relu(x: float, alpha: float = 0.01) -> float:
    """Leaky ReLU activation function."""
    return x if x > 0 else alpha * x


ACTIVATIONS = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
}


@dataclass
class NeuralNetwork:
    """Simple feedforward neural network with evolvable weights."""
    
    layer_sizes: List[int] = field(default_factory=lambda: [8, 16, 8, 4])
    weights: List[List[List[float]]] = field(default_factory=list)
    biases: List[List[float]] = field(default_factory=list)
    activation: str = "tanh"
    
    def __post_init__(self):
        if not self.weights:
            self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights with Xavier initialization."""
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            
            limit = math.sqrt(6.0 / (input_size + output_size))
            
            layer_weights = [
                [random.uniform(-limit, limit) for _ in range(input_size)]
                for _ in range(output_size)
            ]
            layer_biases = [random.uniform(-0.1, 0.1) for _ in range(output_size)]
            
            self.weights.append(layer_weights)
            self.biases.append(layer_biases)
    
    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass through the network."""
        if len(inputs) != self.layer_sizes[0]:
            raise ValueError(f"Expected {self.layer_sizes[0]} inputs, got {len(inputs)}")
        
        activation_fn = ACTIVATIONS.get(self.activation, tanh)
        current = inputs
        
        for layer_idx in range(len(self.weights)):
            layer_weights = self.weights[layer_idx]
            layer_biases = self.biases[layer_idx]
            
            next_layer = []
            for neuron_idx in range(len(layer_weights)):
                total = layer_biases[neuron_idx]
                for input_idx in range(len(current)):
                    total += current[input_idx] * layer_weights[neuron_idx][input_idx]
                
                if layer_idx == len(self.weights) - 1:
                    next_layer.append(sigmoid(total))
                else:
                    next_layer.append(activation_fn(total))
            
            current = next_layer
        
        return current
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.3) -> None:
        """Mutate weights and biases."""
        for layer_idx in range(len(self.weights)):
            for neuron_idx in range(len(self.weights[layer_idx])):
                for weight_idx in range(len(self.weights[layer_idx][neuron_idx])):
                    if random.random() < mutation_rate:
                        self.weights[layer_idx][neuron_idx][weight_idx] += random.gauss(0, mutation_strength)
            
            for bias_idx in range(len(self.biases[layer_idx])):
                if random.random() < mutation_rate:
                    self.biases[layer_idx][bias_idx] += random.gauss(0, mutation_strength)
    
    def copy(self) -> NeuralNetwork:
        """Create a deep copy of this network."""
        new_weights = [
            [[w for w in neuron] for neuron in layer]
            for layer in self.weights
        ]
        new_biases = [
            [b for b in layer]
            for layer in self.biases
        ]
        
        return NeuralNetwork(
            layer_sizes=self.layer_sizes.copy(),
            weights=new_weights,
            biases=new_biases,
            activation=self.activation
        )
    
    @staticmethod
    def crossover(parent1: NeuralNetwork, parent2: NeuralNetwork) -> NeuralNetwork:
        """Create child network from two parents."""
        if parent1.layer_sizes != parent2.layer_sizes:
            raise ValueError("Parents must have same architecture")
        
        child = parent1.copy()
        
        for layer_idx in range(len(child.weights)):
            for neuron_idx in range(len(child.weights[layer_idx])):
                for weight_idx in range(len(child.weights[layer_idx][neuron_idx])):
                    if random.random() < 0.5:
                        child.weights[layer_idx][neuron_idx][weight_idx] = \
                            parent2.weights[layer_idx][neuron_idx][weight_idx]
            
            for bias_idx in range(len(child.biases[layer_idx])):
                if random.random() < 0.5:
                    child.biases[layer_idx][bias_idx] = parent2.biases[layer_idx][bias_idx]
        
        return child
    
    def get_weight_count(self) -> int:
        """Get total number of weights in network."""
        count = 0
        for layer in self.weights:
            for neuron in layer:
                count += len(neuron)
        for layer in self.biases:
            count += len(layer)
        return count
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "layer_sizes": self.layer_sizes,
            "weights": self.weights,
            "biases": self.biases,
            "activation": self.activation
        }
    
    @staticmethod
    def from_dict(data: Dict) -> NeuralNetwork:
        """Deserialize from dictionary."""
        return NeuralNetwork(
            layer_sizes=data["layer_sizes"],
            weights=data["weights"],
            biases=data["biases"],
            activation=data.get("activation", "tanh")
        )


@dataclass  
class CreatureBrain:
    """High-level brain that uses neural network for decisions."""
    
    network: NeuralNetwork = field(default_factory=lambda: NeuralNetwork(
        layer_sizes=[12, 24, 16, 6]
    ))
    
    memory: List[float] = field(default_factory=lambda: [0.0] * 4)
    
    def think(
        self,
        health: float,
        energy: float,
        stamina: float,
        nearest_food_dist: float,
        nearest_food_angle: float,
        nearest_threat_dist: float,
        nearest_threat_angle: float,
        nearest_ally_dist: float,
        nearest_ally_angle: float,
        is_day: float,
        num_nearby_threats: int,
        num_nearby_allies: int
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Process inputs and return action weights.
        
        Returns: (move_force, turn_angle, eat_desire, flee_desire, attack_desire, rest_desire)
        """
        inputs = [
            health,
            energy,
            stamina,
            nearest_food_dist,
            nearest_food_angle / math.pi,
            nearest_threat_dist,
            nearest_threat_angle / math.pi,
            nearest_ally_dist,
            nearest_ally_angle / math.pi,
            is_day,
            min(1.0, num_nearby_threats / 5.0),
            min(1.0, num_nearby_allies / 10.0),
        ]
        
        outputs = self.network.forward(inputs)
        
        return tuple(outputs)
    
    def mutate(self, rate: float = 0.1, strength: float = 0.3) -> None:
        """Mutate the brain."""
        self.network.mutate(rate, strength)
    
    def copy(self) -> CreatureBrain:
        """Create copy of brain."""
        return CreatureBrain(network=self.network.copy())
    
    @staticmethod
    def crossover(parent1: CreatureBrain, parent2: CreatureBrain) -> CreatureBrain:
        """Create child brain from parents."""
        child_network = NeuralNetwork.crossover(parent1.network, parent2.network)
        return CreatureBrain(network=child_network)


class PreyBrain(CreatureBrain):
    """Specialized brain for prey with flee-focused outputs."""
    
    def __init__(self):
        super().__init__(network=NeuralNetwork(layer_sizes=[12, 20, 12, 5]))
    
    def decide(self, inputs: List[float]) -> Dict[str, float]:
        """Get action decisions as dictionary."""
        outputs = self.network.forward(inputs)
        return {
            "move": outputs[0],
            "turn": (outputs[1] - 0.5) * 2,
            "eat": outputs[2],
            "flee": outputs[3],
            "rest": outputs[4],
        }


class PredatorBrain(CreatureBrain):
    """Specialized brain for predators with hunt-focused outputs."""
    
    def __init__(self):
        super().__init__(network=NeuralNetwork(layer_sizes=[12, 20, 12, 5]))
    
    def decide(self, inputs: List[float]) -> Dict[str, float]:
        """Get action decisions as dictionary."""
        outputs = self.network.forward(inputs)
        return {
            "move": outputs[0],
            "turn": (outputs[1] - 0.5) * 2,
            "hunt": outputs[2],
            "attack": outputs[3],
            "rest": outputs[4],
        }
