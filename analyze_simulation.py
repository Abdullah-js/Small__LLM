"""
🔬 PREDATOR-PREY ECOSYSTEM ANALYZER v3.0 - AI ENHANCED
═══════════════════════════════════════════════════════════════════════════════
Advanced analytics suite with Machine Learning, Predictive Modeling,
and Deep Statistical Analysis for simulation data.

Features:
  • 🤖 Machine Learning predictive models (survival prediction, behavior clustering)
  • 📈 Time-series forecasting with trend analysis
  • 🧬 Genetic trait correlation analysis
  • 🎯 Agent behavior pattern recognition
  • 📊 Advanced statistical modeling (Lotka-Volterra fitting)
  • 🔮 Future population prediction
  • 📉 Survival analysis with Kaplan-Meier curves
  • 🧠 Decision pattern analysis from simulation logs
  • 🌐 Full agent-level data analysis (not just species aggregates)
  • 📜 Comprehensive run history with timestamped logs
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any, Callable, Union
from enum import Enum, auto
from abc import ABC, abstractmethod
import warnings
from datetime import datetime, timedelta
import json
import hashlib
import logging
import sys
import os
import time
import textwrap
import traceback
from functools import wraps
from collections import Counter, defaultdict
from scipy import stats, signal
from scipy.optimize import curve_fit
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# Suppress warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for non-interactive use
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set beautiful style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('ggplot')


# ═══════════════════════════════════════════════════════════════════════════════
# MACHINE LEARNING MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class MLPredictor:
    """Machine Learning prediction engine for ecosystem analysis."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def prepare_features(self, df: pd.DataFrame, target_col: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix from agent data."""
        feature_cols = ['health', 'energy', 'speed', 'intelligence', 'aggression', 'vision', 'age', 'generation']
        available_cols = [c for c in feature_cols if c in df.columns]
        
        X = df[available_cols].values
        y = df[target_col].values if target_col and target_col in df.columns else None
        
        # Simple normalization
        X = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-8)
        X = np.nan_to_num(X, 0)
        
        return X, y
    
    def logistic_regression_predict(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Simple logistic regression prediction."""
        z = np.dot(X, weights[:-1]) + weights[-1]
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def train_survival_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train a survival prediction model based on agent traits."""
        # Create survival labels (did agent survive to end?)
        if 'step' not in df.columns:
            return {'error': 'No step column found'}
            
        max_step = df['step'].max()
        
        # Group by agent and determine if they survived
        agent_data = []
        for agent_id, group in df.groupby('agent_id'):
            final_row = group.iloc[-1]
            survived = group['step'].max() >= max_step * 0.9  # Survived to 90%+ of simulation
            
            agent_data.append({
                'agent_id': agent_id,
                'is_predator': final_row.get('is_predator', False),
                'health': group['health'].mean(),
                'energy': group['energy'].mean(),
                'speed': final_row.get('speed', 0.5),
                'intelligence': final_row.get('intelligence', 0.5),
                'aggression': final_row.get('aggression', 0.5),
                'vision': final_row.get('vision', 0.5),
                'generation': final_row.get('generation', 0),
                'age': group['age'].max(),
                'survived': survived,
                'lifespan': group['step'].max() - group['step'].min()
            })
        
        agent_df = pd.DataFrame(agent_data)
        
        if len(agent_df) < 10:
            return {'error': 'Not enough agents for analysis'}
        
        # Prepare features
        X, y = self.prepare_features(agent_df, 'survived')
        
        # Simple gradient descent for logistic regression
        n_features = X.shape[1]
        weights = np.zeros(n_features + 1)
        learning_rate = 0.1
        
        for _ in range(1000):
            predictions = self.logistic_regression_predict(X, weights)
            errors = y.astype(float) - predictions
            
            gradient = np.zeros(n_features + 1)
            gradient[:-1] = np.dot(X.T, errors) / len(y)
            gradient[-1] = np.mean(errors)
            
            weights += learning_rate * gradient
        
        # Calculate feature importance
        feature_names = ['health', 'energy', 'speed', 'intelligence', 'aggression', 'vision', 'age', 'generation']
        importance = dict(zip(feature_names[:len(weights)-1], np.abs(weights[:-1])))
        
        # Normalize importance
        total = sum(importance.values()) + 1e-8
        importance = {k: v/total * 100 for k, v in importance.items()}
        
        self.models['survival'] = weights
        self.feature_importance['survival'] = importance
        
        # Calculate accuracy
        final_predictions = (self.logistic_regression_predict(X, weights) > 0.5).astype(int)
        accuracy = np.mean(final_predictions == y.astype(int))
        
        return {
            'accuracy': accuracy,
            'feature_importance': importance,
            'n_agents': len(agent_df),
            'survival_rate': y.mean(),
            'weights': weights.tolist()
        }
    
    def predict_survival_probability(self, agent_traits: Dict[str, float]) -> float:
        """Predict survival probability for a single agent."""
        if 'survival' not in self.models:
            return 0.5
            
        feature_order = ['health', 'energy', 'speed', 'intelligence', 'aggression', 'vision', 'age', 'generation']
        X = np.array([[agent_traits.get(f, 0.5) for f in feature_order]])
        X = (X - 0.5) / 0.3  # Rough normalization
        
        return float(self.logistic_regression_predict(X, self.models['survival'])[0])


class BehaviorAnalyzer:
    """Analyzes agent behavior patterns from simulation logs."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.action_patterns = {}
        self.behavior_clusters = {}
        
    def analyze_action_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of actions taken by agents."""
        if 'action' not in self.df.columns:
            return {}
            
        # Overall action distribution
        action_counts = self.df['action'].value_counts()
        total = action_counts.sum()
        
        # By species
        prey_actions = self.df[self.df['is_predator'] == False]['action'].value_counts() if 'is_predator' in self.df.columns else pd.Series()
        pred_actions = self.df[self.df['is_predator'] == True]['action'].value_counts() if 'is_predator' in self.df.columns else pd.Series()
        
        # Action success rates (based on reward)
        if 'reward' in self.df.columns:
            action_rewards = self.df.groupby('action')['reward'].agg(['mean', 'std', 'count'])
            success_rates = {
                action: {
                    'avg_reward': float(row['mean']),
                    'reward_std': float(row['std']) if not pd.isna(row['std']) else 0,
                    'count': int(row['count'])
                }
                for action, row in action_rewards.iterrows()
            }
        else:
            success_rates = {}
        
        return {
            'overall_distribution': (action_counts / total * 100).to_dict(),
            'prey_distribution': (prey_actions / prey_actions.sum() * 100).to_dict() if len(prey_actions) > 0 else {},
            'predator_distribution': (pred_actions / pred_actions.sum() * 100).to_dict() if len(pred_actions) > 0 else {},
            'action_effectiveness': success_rates,
            'total_actions': int(total)
        }
    
    def analyze_behavior_over_time(self) -> Dict[str, Any]:
        """Analyze how behavior changes over simulation time."""
        if 'action' not in self.df.columns or 'step' not in self.df.columns:
            return {}
        
        # Divide simulation into phases
        max_step = self.df['step'].max()
        phases = {
            'early': (0, max_step * 0.33),
            'mid': (max_step * 0.33, max_step * 0.66),
            'late': (max_step * 0.66, max_step)
        }
        
        phase_behaviors = {}
        for phase_name, (start, end) in phases.items():
            phase_df = self.df[(self.df['step'] >= start) & (self.df['step'] < end)]
            if len(phase_df) > 0:
                action_dist = phase_df['action'].value_counts()
                phase_behaviors[phase_name] = (action_dist / action_dist.sum() * 100).to_dict()
        
        # Detect behavior shifts
        shifts = []
        if 'early' in phase_behaviors and 'late' in phase_behaviors:
            for action in set(phase_behaviors['early'].keys()) | set(phase_behaviors['late'].keys()):
                early_pct = phase_behaviors['early'].get(action, 0)
                late_pct = phase_behaviors['late'].get(action, 0)
                change = late_pct - early_pct
                if abs(change) > 5:  # Significant shift
                    shifts.append({
                        'action': action,
                        'early_pct': early_pct,
                        'late_pct': late_pct,
                        'change': change,
                        'direction': 'increased' if change > 0 else 'decreased'
                    })
        
        return {
            'phase_behaviors': phase_behaviors,
            'behavior_shifts': sorted(shifts, key=lambda x: abs(x['change']), reverse=True)
        }
    
    def cluster_agent_behaviors(self, n_clusters: int = 4) -> Dict[str, Any]:
        """Cluster agents by their behavior patterns."""
        if 'agent_id' not in self.df.columns or 'action' not in self.df.columns:
            return {}
        
        # Create behavior profile for each agent
        agent_profiles = []
        agent_ids = []
        
        for agent_id, group in self.df.groupby('agent_id'):
            if len(group) < 5:  # Need minimum data
                continue
                
            action_counts = group['action'].value_counts()
            total = action_counts.sum()
            
            profile = {
                'agent_id': agent_id,
                'is_predator': group['is_predator'].iloc[0] if 'is_predator' in group.columns else False,
                'avg_health': group['health'].mean() if 'health' in group.columns else 0,
                'avg_energy': group['energy'].mean() if 'energy' in group.columns else 0,
                'avg_reward': group['reward'].mean() if 'reward' in group.columns else 0,
                'lifespan': group['step'].max() - group['step'].min(),
                'total_actions': total
            }
            
            # Add action percentages
            all_actions = ['move', 'eat', 'drink', 'rest', 'hunt', 'flee', 'reproduce', 'attack']
            for action in all_actions:
                profile[f'pct_{action}'] = action_counts.get(action, 0) / total * 100 if total > 0 else 0
            
            agent_profiles.append(profile)
            agent_ids.append(agent_id)
        
        if len(agent_profiles) < n_clusters:
            return {'error': 'Not enough agents for clustering'}
        
        profiles_df = pd.DataFrame(agent_profiles)
        
        # Features for clustering (action percentages)
        feature_cols = [c for c in profiles_df.columns if c.startswith('pct_')]
        X = profiles_df[feature_cols].values
        
        # Normalize
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # Hierarchical clustering
        try:
            Z = linkage(X, method='ward')
            cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
            profiles_df['cluster'] = cluster_labels
            
            # Analyze each cluster
            cluster_analysis = {}
            for cluster_id in range(1, n_clusters + 1):
                cluster_df = profiles_df[profiles_df['cluster'] == cluster_id]
                
                # Find dominant behavior
                action_avgs = {col: cluster_df[col].mean() for col in feature_cols}
                dominant_action = max(action_avgs, key=action_avgs.get).replace('pct_', '')
                
                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_df),
                    'dominant_action': dominant_action,
                    'avg_lifespan': float(cluster_df['lifespan'].mean()),
                    'avg_reward': float(cluster_df['avg_reward'].mean()),
                    'predator_ratio': float(cluster_df['is_predator'].mean()),
                    'action_profile': {k.replace('pct_', ''): float(v) for k, v in action_avgs.items()}
                }
            
            self.behavior_clusters = cluster_analysis
            return {
                'n_clusters': n_clusters,
                'clusters': cluster_analysis,
                'agent_cluster_map': dict(zip(agent_ids, cluster_labels.tolist()))
            }
            
        except Exception as e:
            return {'error': str(e)}


class GeneticAnalyzer:
    """Analyzes genetic traits and their evolution across generations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.trait_cols = ['speed', 'intelligence', 'aggression', 'vision']
        
    def analyze_trait_evolution(self) -> Dict[str, Any]:
        """Analyze how traits evolve across generations."""
        if 'generation' not in self.df.columns:
            return {}
        
        # Get unique generations
        generations = sorted(self.df['generation'].unique())
        
        trait_evolution = {trait: [] for trait in self.trait_cols}
        generation_sizes = []
        
        for gen in generations:
            gen_df = self.df[self.df['generation'] == gen]
            generation_sizes.append(len(gen_df['agent_id'].unique()) if 'agent_id' in gen_df.columns else len(gen_df))
            
            for trait in self.trait_cols:
                if trait in gen_df.columns:
                    trait_evolution[trait].append({
                        'generation': int(gen),
                        'mean': float(gen_df[trait].mean()),
                        'std': float(gen_df[trait].std()),
                        'min': float(gen_df[trait].min()),
                        'max': float(gen_df[trait].max())
                    })
        
        # Detect trait trends
        trait_trends = {}
        for trait, data in trait_evolution.items():
            if len(data) >= 2:
                means = [d['mean'] for d in data]
                gens = [d['generation'] for d in data]
                
                # Linear regression for trend
                if len(means) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(gens, means)
                    trait_trends[trait] = {
                        'slope': float(slope),
                        'r_squared': float(r_value ** 2),
                        'p_value': float(p_value),
                        'trend': 'increasing' if slope > 0.001 else 'decreasing' if slope < -0.001 else 'stable',
                        'significant': p_value < 0.05
                    }
        
        return {
            'generations': list(generations),
            'generation_sizes': generation_sizes,
            'trait_evolution': trait_evolution,
            'trait_trends': trait_trends
        }
    
    def analyze_trait_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between traits and outcomes."""
        available_traits = [t for t in self.trait_cols if t in self.df.columns]
        
        if len(available_traits) < 2:
            return {}
        
        # Trait-trait correlations
        trait_df = self.df[available_traits].drop_duplicates()
        corr_matrix = trait_df.corr()
        
        correlations = {}
        for i, t1 in enumerate(available_traits):
            for t2 in available_traits[i+1:]:
                corr = corr_matrix.loc[t1, t2]
                correlations[f'{t1}_vs_{t2}'] = {
                    'correlation': float(corr),
                    'strength': 'strong' if abs(corr) > 0.7 else 'moderate' if abs(corr) > 0.4 else 'weak',
                    'direction': 'positive' if corr > 0 else 'negative'
                }
        
        # Trait-survival correlations
        if 'step' in self.df.columns and 'agent_id' in self.df.columns:
            agent_lifespans = self.df.groupby('agent_id').agg({
                'step': lambda x: x.max() - x.min(),
                **{t: 'first' for t in available_traits}
            }).rename(columns={'step': 'lifespan'})
            
            survival_correlations = {}
            for trait in available_traits:
                if trait in agent_lifespans.columns:
                    corr, p_value = stats.pearsonr(
                        agent_lifespans[trait].fillna(0),
                        agent_lifespans['lifespan'].fillna(0)
                    )
                    survival_correlations[trait] = {
                        'correlation': float(corr),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'effect': 'beneficial' if corr > 0.1 else 'detrimental' if corr < -0.1 else 'neutral'
                    }
        else:
            survival_correlations = {}
        
        return {
            'trait_correlations': correlations,
            'survival_correlations': survival_correlations
        }
    
    def identify_optimal_traits(self) -> Dict[str, Any]:
        """Identify optimal trait combinations for survival."""
        if 'agent_id' not in self.df.columns or 'step' not in self.df.columns:
            return {}
        
        # Calculate lifespan for each agent
        agent_data = self.df.groupby('agent_id').agg({
            'step': lambda x: x.max() - x.min(),
            'is_predator': 'first',
            **{t: 'first' for t in self.trait_cols if t in self.df.columns}
        }).rename(columns={'step': 'lifespan'})
        
        # Split into long-lived and short-lived
        median_lifespan = agent_data['lifespan'].median()
        long_lived = agent_data[agent_data['lifespan'] >= median_lifespan]
        short_lived = agent_data[agent_data['lifespan'] < median_lifespan]
        
        optimal_traits = {}
        available_traits = [t for t in self.trait_cols if t in agent_data.columns]
        
        for trait in available_traits:
            long_mean = long_lived[trait].mean()
            short_mean = short_lived[trait].mean()
            
            # Statistical test
            t_stat, p_value = stats.ttest_ind(
                long_lived[trait].dropna(),
                short_lived[trait].dropna()
            )
            
            optimal_traits[trait] = {
                'long_lived_mean': float(long_mean),
                'short_lived_mean': float(short_mean),
                'difference': float(long_mean - short_mean),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'optimal_direction': 'higher' if long_mean > short_mean else 'lower'
            }
        
        # Separate analysis for predators and prey
        species_optimal = {}
        for is_pred in [True, False]:
            species_name = 'predator' if is_pred else 'prey'
            species_data = agent_data[agent_data['is_predator'] == is_pred] if 'is_predator' in agent_data.columns else agent_data
            
            if len(species_data) > 10:
                species_median = species_data['lifespan'].median()
                species_long = species_data[species_data['lifespan'] >= species_median]
                
                species_optimal[species_name] = {
                    trait: float(species_long[trait].mean())
                    for trait in available_traits
                }
        
        return {
            'optimal_traits': optimal_traits,
            'species_optimal_profiles': species_optimal,
            'median_lifespan': float(median_lifespan),
            'n_long_lived': len(long_lived),
            'n_short_lived': len(short_lived)
        }


class TimeSeriesForecaster:
    """Time series analysis and forecasting for population dynamics."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def lotka_volterra_model(self, t, N0, P0, alpha, beta, gamma, delta):
        """Lotka-Volterra differential equation numerical solution."""
        dt = 0.1
        N, P = N0, P0
        N_series, P_series = [N], [P]
        
        for _ in range(int(t / dt)):
            dN = alpha * N - beta * N * P
            dP = delta * N * P - gamma * P
            
            N = max(0, N + dN * dt)
            P = max(0, P + dP * dt)
            
            N_series.append(N)
            P_series.append(P)
        
        return N_series, P_series
    
    def fit_lotka_volterra(self) -> Dict[str, Any]:
        """Fit Lotka-Volterra model to population data."""
        if 'prey_population' not in self.df.columns or 'predator_population' not in self.df.columns:
            return {}
        
        prey = self.df['prey_population'].values
        pred = self.df['predator_population'].values
        
        # Estimate parameters using least squares
        # dN/dt = alpha*N - beta*N*P
        # dP/dt = delta*N*P - gamma*P
        
        try:
            # Calculate derivatives
            dN = np.diff(prey)
            dP = np.diff(pred)
            
            N = prey[:-1]
            P = pred[:-1]
            
            # Avoid division by zero
            N = np.where(N == 0, 1e-6, N)
            P = np.where(P == 0, 1e-6, P)
            
            # Solve for parameters (simplified)
            # For prey: dN/N = alpha - beta*P
            prey_growth_rate = dN / N
            
            # Linear regression: growth_rate = alpha - beta*P
            valid_idx = np.isfinite(prey_growth_rate) & np.isfinite(P)
            if valid_idx.sum() > 10:
                slope_prey, intercept_prey, r_prey, _, _ = stats.linregress(P[valid_idx], prey_growth_rate[valid_idx])
                alpha = float(intercept_prey)
                beta = float(-slope_prey) if slope_prey < 0 else 0.01
            else:
                alpha, beta = 0.1, 0.01
            
            # For predator: dP/P = delta*N - gamma
            pred_growth_rate = dP / P
            valid_idx = np.isfinite(pred_growth_rate) & np.isfinite(N)
            if valid_idx.sum() > 10:
                slope_pred, intercept_pred, r_pred, _, _ = stats.linregress(N[valid_idx], pred_growth_rate[valid_idx])
                delta = float(slope_pred) if slope_pred > 0 else 0.01
                gamma = float(-intercept_pred) if intercept_pred < 0 else 0.1
            else:
                delta, gamma = 0.01, 0.1
            
            # Calculate model fit
            N_model, P_model = self.lotka_volterra_model(
                len(prey), prey[0], pred[0], alpha, beta, gamma, delta
            )
            
            # Truncate to match length
            N_model = N_model[:len(prey)]
            P_model = P_model[:len(pred)]
            
            # R-squared
            ss_res_prey = np.sum((prey - N_model) ** 2)
            ss_tot_prey = np.sum((prey - np.mean(prey)) ** 2)
            r2_prey = 1 - ss_res_prey / ss_tot_prey if ss_tot_prey > 0 else 0
            
            ss_res_pred = np.sum((pred - P_model) ** 2)
            ss_tot_pred = np.sum((pred - np.mean(pred)) ** 2)
            r2_pred = 1 - ss_res_pred / ss_tot_pred if ss_tot_pred > 0 else 0
            
            return {
                'parameters': {
                    'alpha': alpha,  # Prey growth rate
                    'beta': beta,    # Predation rate
                    'gamma': gamma,  # Predator death rate
                    'delta': delta   # Predator reproduction rate
                },
                'fit_quality': {
                    'prey_r2': float(max(0, r2_prey)),
                    'predator_r2': float(max(0, r2_pred)),
                    'overall_r2': float(max(0, (r2_prey + r2_pred) / 2))
                },
                'model_prediction': {
                    'prey': [float(x) for x in N_model],
                    'predator': [float(x) for x in P_model]
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def forecast_population(self, steps_ahead: int = 50) -> Dict[str, Any]:
        """Forecast future population using multiple methods."""
        if 'prey_population' not in self.df.columns:
            return {}
        
        prey = self.df['prey_population'].values
        pred = self.df['predator_population'].values if 'predator_population' in self.df.columns else None
        
        forecasts = {}
        
        # Method 1: Linear trend extrapolation
        x = np.arange(len(prey))
        slope, intercept, _, _, _ = stats.linregress(x, prey)
        linear_forecast = [max(0, intercept + slope * (len(prey) + i)) for i in range(steps_ahead)]
        forecasts['linear'] = {
            'prey': linear_forecast,
            'method': 'Linear Extrapolation'
        }
        
        # Method 2: Exponential smoothing
        alpha_smooth = 0.3
        smoothed = [prey[0]]
        for val in prey[1:]:
            smoothed.append(alpha_smooth * val + (1 - alpha_smooth) * smoothed[-1])
        
        # Project forward
        exp_forecast = [smoothed[-1]] * steps_ahead
        trend = (smoothed[-1] - smoothed[-min(10, len(smoothed))]) / min(10, len(smoothed))
        for i in range(steps_ahead):
            exp_forecast[i] = max(0, smoothed[-1] + trend * i)
        
        forecasts['exponential_smoothing'] = {
            'prey': exp_forecast,
            'method': 'Exponential Smoothing'
        }
        
        # Method 3: Moving average
        window = min(20, len(prey) // 3)
        if window > 0:
            ma = np.convolve(prey, np.ones(window)/window, mode='valid')
            ma_trend = (ma[-1] - ma[0]) / len(ma) if len(ma) > 1 else 0
            ma_forecast = [max(0, ma[-1] + ma_trend * i) for i in range(steps_ahead)]
            forecasts['moving_average'] = {
                'prey': ma_forecast,
                'method': f'Moving Average (window={window})'
            }
        
        # Ensemble forecast (average of methods)
        ensemble = []
        for i in range(steps_ahead):
            values = [f['prey'][i] for f in forecasts.values() if 'prey' in f]
            ensemble.append(float(np.mean(values)))
        
        forecasts['ensemble'] = {
            'prey': ensemble,
            'method': 'Ensemble Average'
        }
        
        # Confidence bounds (based on historical volatility)
        volatility = np.std(prey)
        forecasts['confidence_bounds'] = {
            'upper': [e + 2 * volatility for e in ensemble],
            'lower': [max(0, e - 2 * volatility) for e in ensemble]
        }
        
        return forecasts
    
    def detect_cycles(self) -> Dict[str, Any]:
        """Detect population cycles using FFT."""
        if 'prey_population' not in self.df.columns:
            return {}
        
        prey = self.df['prey_population'].values
        
        # Detrend
        detrended = signal.detrend(prey)
        
        # FFT
        fft = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(detrended))
        
        # Find dominant frequencies
        magnitudes = np.abs(fft[1:len(fft)//2])
        freqs_positive = freqs[1:len(freqs)//2]
        
        if len(magnitudes) > 0:
            # Top 3 frequencies
            top_indices = np.argsort(magnitudes)[-3:][::-1]
            
            cycles = []
            for idx in top_indices:
                if freqs_positive[idx] > 0:
                    period = 1 / freqs_positive[idx]
                    if period < len(prey) / 2:  # Reasonable cycle length
                        cycles.append({
                            'period': float(period),
                            'strength': float(magnitudes[idx] / magnitudes.sum() * 100)
                        })
            
            return {
                'detected_cycles': cycles,
                'primary_cycle': cycles[0] if cycles else None,
                'is_cyclic': len(cycles) > 0 and cycles[0]['strength'] > 10
            }
        
        return {'detected_cycles': [], 'is_cyclic': False}


# ═══════════════════════════════════════════════════════════════════════════════
# AI INSIGHTS GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class AIInsightsGenerator:
    """Generates natural language insights from analysis results."""
    
    def __init__(self):
        self.insights = []
        
    def add_insight(self, category: str, severity: str, message: str, data: Dict = None):
        """Add an insight."""
        self.insights.append({
            'category': category,
            'severity': severity,
            'message': message,
            'data': data or {},
            'timestamp': datetime.now().isoformat()
        })
    
    def generate_survival_insights(self, survival_model: Dict) -> List[str]:
        """Generate insights from survival model."""
        insights = []
        
        if 'feature_importance' in survival_model:
            imp = survival_model['feature_importance']
            top_factor = max(imp, key=imp.get)
            
            insights.append(f"🧬 **Key Survival Factor**: {top_factor.title()} is the most important trait for survival ({imp[top_factor]:.1f}% importance)")
            
            if imp.get('energy', 0) > 20:
                insights.append("⚡ High energy levels strongly correlate with survival - consider adjusting energy mechanics")
            
            if imp.get('speed', 0) > 20:
                insights.append("🏃 Speed is crucial for survival - faster agents significantly outlive slower ones")
            
        if 'accuracy' in survival_model:
            acc = survival_model['accuracy']
            if acc > 0.8:
                insights.append(f"🎯 Survival is highly predictable from traits (model accuracy: {acc:.1%}) - suggests deterministic selection")
            elif acc < 0.6:
                insights.append(f"🎲 Survival has high randomness component (model accuracy: {acc:.1%}) - environmental factors dominate")
        
        return insights
    
    def generate_behavior_insights(self, behavior_analysis: Dict) -> List[str]:
        """Generate insights from behavior analysis."""
        insights = []
        
        if 'overall_distribution' in behavior_analysis:
            dist = behavior_analysis['overall_distribution']
            top_action = max(dist, key=dist.get)
            
            insights.append(f"🎮 **Dominant Behavior**: Agents spend most time on '{top_action}' ({dist[top_action]:.1f}%)")
            
            if dist.get('flee', 0) > 30:
                insights.append("😰 High flee rate suggests predator pressure is very strong")
            
            if dist.get('rest', 0) > 40:
                insights.append("😴 Excessive resting indicates energy recovery is a bottleneck")
            
        if 'behavior_shifts' in behavior_analysis:
            shifts = behavior_analysis['behavior_shifts']
            for shift in shifts[:2]:
                direction = "📈" if shift['direction'] == 'increased' else "📉"
                insights.append(f"{direction} '{shift['action']}' behavior {shift['direction']} by {abs(shift['change']):.1f}% from early to late simulation")
        
        return insights
    
    def generate_genetic_insights(self, genetic_analysis: Dict) -> List[str]:
        """Generate insights from genetic analysis."""
        insights = []
        
        if 'trait_evolution' in genetic_analysis:
            trait_evol = genetic_analysis['trait_evolution']
            if isinstance(trait_evol, dict) and 'trait_trends' in trait_evol:
                trends = trait_evol['trait_trends']
                for trait, data in trends.items():
                    if isinstance(data, dict) and data.get('significant', False):
                        direction = "📈" if data.get('trend') == 'increasing' else "📉"
                        p_val = data.get('p_value', 1.0)
                        insights.append(f"{direction} **Evolution Detected**: {trait.title()} is {data.get('trend', 'changing')} across generations (p={p_val:.3f})")
        
        if 'optimal_traits' in genetic_analysis:
            opt = genetic_analysis['optimal_traits']
            if isinstance(opt, dict):
                # Look for 'optimal_traits' sub-key which contains the trait details
                traits_detail = opt.get('optimal_traits', opt)
                if isinstance(traits_detail, dict):
                    beneficial = []
                    for trait, data in traits_detail.items():
                        if isinstance(data, dict) and data.get('significant') and data.get('optimal_direction') == 'higher':
                            beneficial.append(trait)
                    if beneficial:
                        insights.append(f"🏆 Traits favoring survival: {', '.join(beneficial)}")
                
                # Also check species profiles
                species_profiles = opt.get('species_optimal_profiles', {})
                if species_profiles:
                    for species, traits in species_profiles.items():
                        if isinstance(traits, dict):
                            top_trait = max(traits.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
                            insights.append(f"🎯 Best {species} trait: {top_trait[0]} ({top_trait[1]:.2f})")
        
        return insights
    
    def generate_forecast_insights(self, forecast_data: Dict) -> List[str]:
        """Generate insights from population forecasts."""
        insights = []
        
        if 'ensemble' in forecast_data:
            ensemble = forecast_data['ensemble']
            prey_data = ensemble.get('prey', [])
            final_pop = prey_data[-1] if prey_data else 0
            
            if isinstance(final_pop, (int, float)):
                if final_pop < 10:
                    insights.append("🔴 **Extinction Warning**: Forecasts predict near-extinction within forecast horizon")
                elif final_pop > 200:
                    insights.append("📈 **Growth Trend**: Population expected to grow significantly")
                else:
                    insights.append(f"📊 Forecasted prey population: ~{final_pop:.0f} at end of prediction window")
        
        return insights
    
    def compile_executive_summary(self, all_analyses: Dict) -> str:
        """Compile all insights into an executive summary."""
        summary_parts = []
        
        summary_parts.append("\n" + "═" * 70)
        summary_parts.append("  🤖 AI-GENERATED EXECUTIVE SUMMARY")
        summary_parts.append("═" * 70)
        
        # Overall ecosystem health
        metrics = all_analyses.get('metrics', {})
        if isinstance(metrics, SimulationMetrics):
            metrics = metrics.to_dict()
        
        health_score = metrics.get('ecosystem_stability_score', 50)
        risk_score = metrics.get('extinction_risk_score', 50)
        
        if risk_score > 70:
            summary_parts.append("\n  ⚠️  CRITICAL: Ecosystem shows high extinction risk")
        elif health_score > 60:
            summary_parts.append("\n  ✅ HEALTHY: Ecosystem demonstrates good stability")
        else:
            summary_parts.append("\n  🟡 CAUTION: Ecosystem stability could be improved")
        
        # Key findings
        summary_parts.append("\n  📋 KEY FINDINGS:")
        
        for category, insights in all_analyses.get('insights', {}).items():
            if insights:
                summary_parts.append(f"\n  {category.upper()}:")
                for insight in insights[:3]:
                    summary_parts.append(f"    • {insight}")
        
        summary_parts.append("\n" + "═" * 70 + "\n")
        
        return '\n'.join(summary_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    ICONS = {
        'DEBUG': '🔍',
        'INFO': '📊',
        'WARNING': '⚠️ ',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        icon = self.ICONS.get(record.levelname, '')
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        return f"{color}{icon} [{timestamp}] {record.getMessage()}{reset}"


class AnalysisLogger:
    """Comprehensive logging system for analysis runs."""
    
    def __init__(self, log_dir: Path, run_id: str):
        self.log_dir = log_dir
        self.run_id = run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(f'EcosystemAnalyzer_{run_id}')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        # File handler (detailed logs)
        log_file = self.log_dir / f'analysis_{run_id}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler (colored output)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(ColoredFormatter())
        self.logger.addHandler(console_handler)
        
        self.log_file = log_file
        self.start_time = datetime.now()
        
    def debug(self, msg: str): self.logger.debug(msg)
    def info(self, msg: str): self.logger.info(msg)
    def warning(self, msg: str): self.logger.warning(msg)
    def error(self, msg: str): self.logger.error(msg)
    def critical(self, msg: str): self.logger.critical(msg)
    
    def section(self, title: str):
        """Log a section header."""
        self.logger.info("─" * 50)
        self.logger.info(f"  {title}")
        self.logger.info("─" * 50)
        
    def elapsed(self) -> str:
        """Get elapsed time since start."""
        delta = datetime.now() - self.start_time
        return str(delta).split('.')[0]


def timed_operation(operation_name: str):
    """Decorator to time and log operations."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.time()
            self.log.debug(f"Starting: {operation_name}")
            try:
                result = func(self, *args, **kwargs)
                elapsed = time.time() - start
                self.log.debug(f"Completed: {operation_name} ({elapsed:.2f}s)")
                return result
            except Exception as e:
                self.log.error(f"Failed: {operation_name} - {str(e)}")
                raise
        return wrapper
    return decorator


class AnalysisMode(Enum):
    """Analysis modes for different use cases."""
    QUICK = auto()      # Fast overview
    STANDARD = auto()   # Normal analysis
    DEEP = auto()       # Full statistical analysis
    RESEARCH = auto()   # Academic-grade with all metrics


@dataclass
class SimulationMetrics:
    """Container for computed simulation metrics."""
    duration: int = 0
    peak_prey: int = 0
    peak_prey_step: int = 0
    peak_predator: int = 0
    peak_predator_step: int = 0
    total_kills: int = 0
    total_births: int = 0
    prey_extinction_step: Optional[int] = None
    predator_extinction_step: Optional[int] = None
    total_extinction_step: Optional[int] = None
    avg_prey_growth_rate: float = 0.0
    avg_predator_growth_rate: float = 0.0
    prey_volatility: float = 0.0
    predator_volatility: float = 0.0
    kill_efficiency: float = 0.0
    ecosystem_stability_score: float = 0.0
    lotka_volterra_fit: float = 0.0
    
    # New advanced metrics
    prey_carrying_capacity: float = 0.0
    predator_carrying_capacity: float = 0.0
    population_oscillation_period: float = 0.0
    ecosystem_resilience: float = 0.0
    biodiversity_index: float = 0.0
    trophic_efficiency: float = 0.0
    extinction_risk_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass 
class RunHistoryEntry:
    """Single entry in the analysis run history."""
    run_id: str
    timestamp: str
    source_file: str
    source_file_hash: str
    duration_steps: int
    analysis_mode: str
    metrics: Dict[str, Any]
    output_files: List[str]
    elapsed_time: str
    status: str  # 'success', 'warning', 'error'
    notes: List[str] = field(default_factory=list)


@dataclass
class AnalysisConfig:
    """Configuration for analysis behavior."""
    mode: AnalysisMode = AnalysisMode.STANDARD
    output_format: str = "png"
    dpi: int = 150
    figure_size: Tuple[int, int] = (16, 12)
    color_scheme: Dict[str, str] = field(default_factory=lambda: {
        'prey': '#2ecc71',
        'predator': '#e74c3c', 
        'neutral': '#3498db',
        'warning': '#f39c12',
        'background': '#2c3e50',
        'grid': '#95a5a6',
        'success': '#27ae60',
        'danger': '#c0392b',
        'info': '#2980b9'
    })
    smoothing_window: int = 5
    show_plots: bool = True
    save_plots: bool = True
    export_json: bool = True
    log_runs: bool = True
    compare_runs: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# RUN HISTORY MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class RunHistoryManager:
    """Manages analysis run history for tracking and comparison."""
    
    HISTORY_FILE = "analysis_history.json"
    MAX_HISTORY_ENTRIES = 100
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.history_file = output_dir / self.HISTORY_FILE
        self.history: List[Dict] = []
        self._load_history()
        
    def _load_history(self) -> None:
        """Load existing history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.history = data.get('runs', [])
            except (json.JSONDecodeError, KeyError):
                self.history = []
                
    def _save_history(self) -> None:
        """Save history to file."""
        # Trim to max entries
        self.history = self.history[-self.MAX_HISTORY_ENTRIES:]
        
        data = {
            'version': '2.0',
            'last_updated': datetime.now().isoformat(),
            'total_runs': len(self.history),
            'runs': self.history
        }
        
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
    def add_run(self, entry: RunHistoryEntry) -> None:
        """Add a new run to history."""
        self.history.append(asdict(entry))
        self._save_history()
        
    def get_recent_runs(self, n: int = 10) -> List[Dict]:
        """Get the n most recent runs."""
        return self.history[-n:][::-1]
    
    def get_run_by_id(self, run_id: str) -> Optional[Dict]:
        """Get a specific run by ID."""
        for run in self.history:
            if run['run_id'] == run_id:
                return run
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics across all runs."""
        if not self.history:
            return {}
            
        durations = [r['duration_steps'] for r in self.history]
        success_count = sum(1 for r in self.history if r['status'] == 'success')
        
        return {
            'total_runs': len(self.history),
            'success_rate': success_count / len(self.history) * 100,
            'avg_duration': np.mean(durations),
            'max_duration': max(durations),
            'min_duration': min(durations),
            'first_run': self.history[0]['timestamp'] if self.history else None,
            'last_run': self.history[-1]['timestamp'] if self.history else None
        }
    
    def compare_metrics(self, metric_name: str) -> pd.DataFrame:
        """Compare a specific metric across all runs."""
        data = []
        for run in self.history:
            if metric_name in run.get('metrics', {}):
                data.append({
                    'run_id': run['run_id'][:8],
                    'timestamp': run['timestamp'],
                    metric_name: run['metrics'][metric_name]
                })
        return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """Detects anomalies and interesting patterns in simulation data."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.anomalies: List[Dict] = []
        
    def detect_all(self) -> List[Dict]:
        """Run all anomaly detection methods."""
        self.anomalies = []
        self._detect_population_crashes()
        self._detect_population_explosions()
        self._detect_oscillation_damping()
        self._detect_equilibrium_shifts()
        self._detect_extinction_cascade()
        return self.anomalies
    
    def _detect_population_crashes(self, threshold: float = 0.5) -> None:
        """Detect sudden population crashes (>50% drop in 10 steps)."""
        for species in ['prey', 'predator']:
            col = f'{species}_population'
            for i in range(10, len(self.data)):
                prev = self.data[col].iloc[i-10]
                curr = self.data[col].iloc[i]
                if prev > 0 and (prev - curr) / prev > threshold:
                    self.anomalies.append({
                        'type': 'population_crash',
                        'species': species,
                        'step': int(self.data['step'].iloc[i]),
                        'severity': 'high',
                        'description': f'{species.title()} population crashed from {prev} to {curr} ({((prev-curr)/prev)*100:.1f}% drop)'
                    })
                    break  # Only report first crash per species
                    
    def _detect_population_explosions(self, threshold: float = 2.0) -> None:
        """Detect sudden population explosions (>100% increase in 20 steps)."""
        for species in ['prey', 'predator']:
            col = f'{species}_population'
            for i in range(20, len(self.data)):
                prev = self.data[col].iloc[i-20]
                curr = self.data[col].iloc[i]
                if prev > 0 and curr / prev > threshold:
                    self.anomalies.append({
                        'type': 'population_explosion',
                        'species': species,
                        'step': int(self.data['step'].iloc[i]),
                        'severity': 'medium',
                        'description': f'{species.title()} population exploded from {prev} to {curr} ({(curr/prev)*100:.1f}% increase)'
                    })
                    break
                    
    def _detect_oscillation_damping(self) -> None:
        """Detect if population oscillations are damping over time."""
        prey_std_early = self.data['prey_population'].iloc[:len(self.data)//3].std()
        prey_std_late = self.data['prey_population'].iloc[-len(self.data)//3:].std()
        
        if prey_std_early > 0 and prey_std_late / prey_std_early < 0.3:
            self.anomalies.append({
                'type': 'oscillation_damping',
                'species': 'ecosystem',
                'step': len(self.data) // 2,
                'severity': 'info',
                'description': 'Population oscillations are damping - system approaching equilibrium'
            })
            
    def _detect_equilibrium_shifts(self) -> None:
        """Detect major shifts in ecosystem equilibrium."""
        # Compare first and last third averages
        n = len(self.data) // 3
        if n < 10:
            return
            
        early_ratio = (self.data['prey_population'].iloc[:n].mean() / 
                      max(self.data['predator_population'].iloc[:n].mean(), 1))
        late_ratio = (self.data['prey_population'].iloc[-n:].mean() / 
                     max(self.data['predator_population'].iloc[-n:].mean(), 1))
        
        if abs(early_ratio - late_ratio) > 5:
            self.anomalies.append({
                'type': 'equilibrium_shift',
                'species': 'ecosystem',
                'step': n,
                'severity': 'medium',
                'description': f'Ecosystem equilibrium shifted: prey:predator ratio changed from {early_ratio:.1f}:1 to {late_ratio:.1f}:1'
            })
            
    def _detect_extinction_cascade(self) -> None:
        """Detect if predator extinction led to prey explosion."""
        pred_extinct = self.data[self.data['predator_population'] == 0]
        if len(pred_extinct) > 0:
            extinct_step = pred_extinct['step'].iloc[0]
            prey_at_extinct = self.data[self.data['step'] == extinct_step]['prey_population'].iloc[0]
            max_prey_after = self.data[self.data['step'] >= extinct_step]['prey_population'].max()
            
            if max_prey_after > prey_at_extinct * 1.5:
                self.anomalies.append({
                    'type': 'extinction_cascade',
                    'species': 'ecosystem',
                    'step': int(extinct_step),
                    'severity': 'high',
                    'description': f'Predator extinction at step {extinct_step} triggered prey population explosion ({prey_at_extinct} → {max_prey_after})'
                })


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class StatisticalAnalyzer:
    """Advanced statistical analysis for ecosystem data."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def compute_carrying_capacity(self) -> Tuple[float, float]:
        """Estimate carrying capacity using logistic growth model."""
        # Use 95th percentile as rough estimate
        prey_k = np.percentile(self.data['prey_population'], 95)
        pred_k = np.percentile(self.data['predator_population'], 95)
        return float(prey_k), float(pred_k)
    
    def compute_oscillation_period(self) -> float:
        """Estimate population oscillation period using FFT."""
        try:
            prey_centered = self.data['prey_population'] - self.data['prey_population'].mean()
            fft = np.fft.fft(prey_centered)
            freqs = np.fft.fftfreq(len(prey_centered))
            
            # Find dominant frequency (excluding DC component)
            magnitudes = np.abs(fft[1:len(fft)//2])
            if len(magnitudes) > 0 and max(magnitudes) > 0:
                dominant_idx = np.argmax(magnitudes) + 1
                if freqs[dominant_idx] != 0:
                    period = abs(1 / freqs[dominant_idx])
                    return float(period) if period < len(self.data) / 2 else 0.0
        except Exception:
            pass
        return 0.0
    
    def compute_resilience(self) -> float:
        """
        Compute ecosystem resilience score based on recovery from perturbations.
        Higher score = faster recovery to equilibrium.
        """
        # Measure autocorrelation decay rate
        try:
            prey = self.data['prey_population'].values
            if len(prey) < 20:
                return 0.0
                
            autocorr = np.correlate(prey - prey.mean(), prey - prey.mean(), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Find where autocorrelation drops below 0.5
            half_life = np.where(autocorr < 0.5)[0]
            if len(half_life) > 0:
                resilience = 100 / (1 + half_life[0] / 10)
                return float(min(100, resilience))
        except Exception:
            pass
        return 50.0
    
    def compute_trophic_efficiency(self) -> float:
        """Compute energy transfer efficiency between trophic levels."""
        # Approximated by predator biomass supported per prey biomass
        avg_prey = self.data['prey_population'].mean()
        avg_pred = self.data['predator_population'].mean()
        
        if avg_prey > 0:
            efficiency = (avg_pred / avg_prey) * 100
            return float(min(100, efficiency))
        return 0.0
    
    def compute_extinction_risk(self) -> float:
        """Compute extinction risk score based on population trends."""
        risk_score = 0.0
        
        # Factor 1: Final population levels
        final_prey = self.data['prey_population'].iloc[-1]
        final_pred = self.data['predator_population'].iloc[-1]
        
        if final_prey == 0:
            risk_score += 50
        elif final_prey < 10:
            risk_score += 30
        elif final_prey < 20:
            risk_score += 15
            
        if final_pred == 0:
            risk_score += 50
        elif final_pred < 5:
            risk_score += 30
        elif final_pred < 10:
            risk_score += 15
            
        # Factor 2: Negative growth trends
        prey_trend = np.polyfit(range(len(self.data)), self.data['prey_population'], 1)[0]
        pred_trend = np.polyfit(range(len(self.data)), self.data['predator_population'], 1)[0]
        
        if prey_trend < -0.1:
            risk_score += 10
        if pred_trend < -0.05:
            risk_score += 10
            
        return float(min(100, risk_score))


class EcosystemAnalyzer:
    """
    Advanced ecosystem simulation analyzer with comprehensive
    statistical analysis, visualization, predictive modeling,
    ML-based insights, and full run logging.
    
    Now supports:
    - Species-level data (species_logs.csv)
    - Agent-level detailed data (simulation_logs.csv)
    """
    
    def __init__(
        self, 
        csv_path: str | Path,
        config: Optional[AnalysisConfig] = None,
        simulation_logs_path: Optional[str | Path] = None
    ):
        self.csv_path = Path(csv_path)
        self.config = config or AnalysisConfig()
        self.data: pd.DataFrame = pd.DataFrame()
        self.agent_data: pd.DataFrame = pd.DataFrame()  # Detailed agent-level data
        self.metrics = SimulationMetrics()
        self.output_dir = self.csv_path.parent / "analysis"
        self.run_id = self._generate_run_id()
        self.output_files: List[str] = []
        self.notes: List[str] = []
        self.status = 'success'
        
        # ML components
        self.ml_predictor = MLPredictor()
        self.ai_insights = AIInsightsGenerator()
        self.ml_results: Dict[str, Any] = {}
        
        # Initialize logging
        self.log = AnalysisLogger(self.output_dir / "logs", self.run_id)
        self.log.info(f"Initializing EcosystemAnalyzer v3.0 AI-ENHANCED (Run ID: {self.run_id[:8]})")
        
        # Initialize history manager
        self.history = RunHistoryManager(self.output_dir)
        
        # Determine simulation logs path
        if simulation_logs_path:
            self.simulation_logs_path = Path(simulation_logs_path)
        else:
            # Try to find it automatically
            self.simulation_logs_path = self.csv_path.parent / 'simulation_logs.csv'
        
        # Load data
        self._load_and_validate_data()
        
    def _generate_run_id(self) -> str:
        """Generate unique run ID based on timestamp and file."""
        timestamp = datetime.now().isoformat()
        content = f"{timestamp}_{self.csv_path}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _compute_file_hash(self) -> str:
        """Compute hash of source file for tracking changes."""
        with open(self.csv_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:12]
        
    @timed_operation("Data Loading & Validation")
    def _load_and_validate_data(self) -> None:
        """Load CSV data with validation and preprocessing."""
        if not self.csv_path.exists():
            self.log.error(f"Data file not found: {self.csv_path}")
            raise FileNotFoundError(f"Data file not found: {self.csv_path}")
        
        self.data = pd.read_csv(self.csv_path)
        self.log.info(f"Loaded {len(self.data)} species data points from {self.csv_path.name}")
        
        required_cols = ['step', 'prey_population', 'predator_population', 
                        'total_population', 'total_kills', 'total_births']
        
        missing = set(required_cols) - set(self.data.columns)
        if missing:
            self.log.error(f"Missing required columns: {missing}")
            raise ValueError(f"Missing required columns: {missing}")
        
        # Log data summary
        self.log.debug(f"Columns: {list(self.data.columns)}")
        self.log.debug(f"Step range: {self.data['step'].min()} - {self.data['step'].max()}")
        
        # Load detailed agent-level simulation logs if available
        self._load_agent_data()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Precompute derived columns
        self._compute_derived_metrics()
    
    @timed_operation("Agent Data Loading")
    def _load_agent_data(self) -> None:
        """Load detailed agent-level simulation logs if available."""
        if self.simulation_logs_path.exists():
            try:
                self.agent_data = pd.read_csv(self.simulation_logs_path)
                self.log.info(f"🤖 Loaded {len(self.agent_data):,} agent-level records from {self.simulation_logs_path.name}")
                
                # Basic stats
                if 'agent_id' in self.agent_data.columns:
                    n_agents = self.agent_data['agent_id'].nunique()
                    self.log.info(f"   └── {n_agents:,} unique agents tracked")
                    
                if 'action' in self.agent_data.columns:
                    n_actions = len(self.agent_data['action'].unique())
                    self.log.info(f"   └── {n_actions} different action types recorded")
                    
            except Exception as e:
                self.log.warning(f"Could not load agent data: {e}")
                self.agent_data = pd.DataFrame()
        else:
            self.log.debug(f"No agent-level data found at {self.simulation_logs_path}")
            self.agent_data = pd.DataFrame()
        
    def _compute_derived_metrics(self) -> None:
        """Compute all derived metrics and rates."""
        df = self.data
        
        # Growth rates (percentage change)
        df['prey_growth_rate'] = df['prey_population'].pct_change() * 100
        df['predator_growth_rate'] = df['predator_population'].pct_change() * 100
        
        # Per-step events
        df['kills_per_step'] = df['total_kills'].diff().fillna(0).clip(lower=0)
        df['births_per_step'] = df['total_births'].diff().fillna(0).clip(lower=0)
        
        # Smoothed populations (moving average)
        window = self.config.smoothing_window
        df['prey_smooth'] = df['prey_population'].rolling(window=window, center=True).mean()
        df['predator_smooth'] = df['predator_population'].rolling(window=window, center=True).mean()
        
        # Population ratios
        df['prey_predator_ratio'] = df['prey_population'] / df['predator_population'].replace(0, np.nan)
        
        # Ecosystem health indicator (balanced when close to 1)
        optimal_ratio = 7  # Typical healthy prey:predator ratio
        df['ecosystem_health'] = 1 - np.abs(np.log(df['prey_predator_ratio'] / optimal_ratio)).clip(upper=3) / 3
        
        # Momentum indicators
        df['prey_momentum'] = df['prey_population'].diff().rolling(window=3).mean()
        df['predator_momentum'] = df['predator_population'].diff().rolling(window=3).mean()
        
        # Fill NaN values
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        self.log.debug("Computed derived metrics for all columns")
        
    @timed_operation("Metrics Computation")
    def compute_metrics(self) -> SimulationMetrics:
        """Calculate comprehensive simulation metrics."""
        df = self.data
        
        self.log.info("Computing simulation metrics...")
        
        self.metrics.duration = int(df['step'].max())
        self.metrics.peak_prey = int(df['prey_population'].max())
        self.metrics.peak_prey_step = int(df['prey_population'].idxmax())
        self.metrics.peak_predator = int(df['predator_population'].max())
        self.metrics.peak_predator_step = int(df['predator_population'].idxmax())
        self.metrics.total_kills = int(df['total_kills'].iloc[-1])
        self.metrics.total_births = int(df['total_births'].iloc[-1])
        
        # Extinction detection
        prey_extinct = df[df['prey_population'] == 0]['step']
        pred_extinct = df[df['predator_population'] == 0]['step']
        total_extinct = df[df['total_population'] == 0]['step']
        
        self.metrics.prey_extinction_step = int(prey_extinct.min()) if len(prey_extinct) > 0 else None
        self.metrics.predator_extinction_step = int(pred_extinct.min()) if len(pred_extinct) > 0 else None
        self.metrics.total_extinction_step = int(total_extinct.min()) if len(total_extinct) > 0 else None
        
        # Growth rates (excluding infinite values)
        prey_growth = df['prey_growth_rate'].replace([np.inf, -np.inf], np.nan).dropna()
        pred_growth = df['predator_growth_rate'].replace([np.inf, -np.inf], np.nan).dropna()
        
        self.metrics.avg_prey_growth_rate = float(prey_growth.mean()) if len(prey_growth) > 0 else 0
        self.metrics.avg_predator_growth_rate = float(pred_growth.mean()) if len(pred_growth) > 0 else 0
        
        # Volatility (standard deviation of growth rates)
        self.metrics.prey_volatility = float(prey_growth.std()) if len(prey_growth) > 0 else 0
        self.metrics.predator_volatility = float(pred_growth.std()) if len(pred_growth) > 0 else 0
        
        # Kill efficiency (kills per predator-step)
        total_pred_steps = df['predator_population'].sum()
        self.metrics.kill_efficiency = self.metrics.total_kills / total_pred_steps if total_pred_steps > 0 else 0
        
        # Ecosystem stability score (inverse of combined volatility, normalized)
        combined_volatility = self.metrics.prey_volatility + self.metrics.predator_volatility
        self.metrics.ecosystem_stability_score = max(0, min(100, 100 / (1 + combined_volatility / 10)))
        
        # Lotka-Volterra model fit (simplified correlation-based)
        self.metrics.lotka_volterra_fit = self._compute_lotka_volterra_correlation()
        
        # Advanced metrics using StatisticalAnalyzer
        self.log.debug("Computing advanced statistical metrics...")
        stats = StatisticalAnalyzer(self.data)
        
        prey_k, pred_k = stats.compute_carrying_capacity()
        self.metrics.prey_carrying_capacity = prey_k
        self.metrics.predator_carrying_capacity = pred_k
        
        self.metrics.population_oscillation_period = stats.compute_oscillation_period()
        self.metrics.ecosystem_resilience = stats.compute_resilience()
        self.metrics.trophic_efficiency = stats.compute_trophic_efficiency()
        self.metrics.extinction_risk_score = stats.compute_extinction_risk()
        
        # Biodiversity index (Shannon-like, based on population balance)
        total = df['total_population'].mean()
        if total > 0:
            p_prey = df['prey_population'].mean() / total
            p_pred = df['predator_population'].mean() / total
            if p_prey > 0 and p_pred > 0:
                self.metrics.biodiversity_index = -((p_prey * np.log(p_prey)) + (p_pred * np.log(p_pred))) / np.log(2) * 100
        
        # Log warnings for concerning metrics
        if self.metrics.extinction_risk_score > 50:
            self.log.warning(f"High extinction risk detected: {self.metrics.extinction_risk_score:.1f}%")
            self.notes.append(f"High extinction risk: {self.metrics.extinction_risk_score:.1f}%")
            
        if self.metrics.ecosystem_stability_score < 30:
            self.log.warning(f"Low ecosystem stability: {self.metrics.ecosystem_stability_score:.1f}")
            self.notes.append(f"Low ecosystem stability: {self.metrics.ecosystem_stability_score:.1f}")
        
        self.log.info(f"Computed {len(asdict(self.metrics))} metrics")
        
        return self.metrics
        self.metrics.ecosystem_stability_score = max(0, min(100, 100 / (1 + combined_volatility / 10)))
        
        # Lotka-Volterra model fit (simplified correlation-based)
        self.metrics.lotka_volterra_fit = self._compute_lotka_volterra_correlation()
        
        return self.metrics
    
    def _compute_lotka_volterra_correlation(self) -> float:
        """
        Compute how well the data fits classic Lotka-Volterra dynamics.
        In L-V model, predator growth lags behind prey growth.
        """
        try:
            # Shift prey population and correlate with predator
            prey_shifted = self.data['prey_population'].shift(5)
            correlation = self.data['predator_population'].corr(prey_shifted)
            return float(abs(correlation)) if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0
    
    @timed_operation("Anomaly Detection")
    def detect_anomalies(self) -> List[Dict]:
        """Run anomaly detection on simulation data."""
        self.log.info("Running anomaly detection...")
        detector = AnomalyDetector(self.data)
        anomalies = detector.detect_all()
        
        for anomaly in anomalies:
            level = anomaly.get('severity', 'info')
            msg = f"[{anomaly['type']}] {anomaly['description']}"
            
            if level == 'high':
                self.log.warning(msg)
                self.notes.append(msg)
            elif level == 'medium':
                self.log.info(msg)
            else:
                self.log.debug(msg)
                
        self.log.info(f"Detected {len(anomalies)} anomalies")
        return anomalies
    
    def print_summary(self) -> None:
        """Print a beautiful summary report to console."""
        m = self.metrics if self.metrics.duration > 0 else self.compute_metrics()
        
        # Header
        print("\n" + "═" * 70)
        print("  🔬 PREDATOR-PREY ECOSYSTEM ANALYSIS REPORT v2.0")
        print("═" * 70)
        print(f"  🆔 Run ID: {self.run_id[:8]}")
        print(f"  📁 Data Source: {self.csv_path.name}")
        print(f"  📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("─" * 70)
        
        # Duration & Overview
        print("\n  📊 SIMULATION OVERVIEW")
        print(f"  {'Duration:':<25} {m.duration:,} steps")
        print(f"  {'Final Population:':<25} {int(self.data['total_population'].iloc[-1]):,}")
        
        # Population Peaks
        print("\n  🏔️  POPULATION PEAKS")
        print(f"  {'Peak Prey:':<25} {m.peak_prey:,} (step {m.peak_prey_step:,})")
        print(f"  {'Peak Predator:':<25} {m.peak_predator:,} (step {m.peak_predator_step:,})")
        print(f"  {'Prey Carrying Capacity:':<25} ~{m.prey_carrying_capacity:.0f}")
        print(f"  {'Predator Carrying Cap.:':<25} ~{m.predator_carrying_capacity:.0f}")
        
        # Events
        print("\n  📈 LIFECYCLE EVENTS")
        print(f"  {'Total Kills:':<25} {m.total_kills:,}")
        print(f"  {'Total Births:':<25} {m.total_births:,}")
        print(f"  {'Kill Efficiency:':<25} {m.kill_efficiency:.4f} kills/predator-step")
        print(f"  {'Trophic Efficiency:':<25} {m.trophic_efficiency:.2f}%")
        
        # Extinction Events
        print("\n  ⚰️  EXTINCTION EVENTS")
        if m.predator_extinction_step:
            print(f"  {'Predators Extinct:':<25} Step {m.predator_extinction_step:,}")
        else:
            print(f"  {'Predators:':<25} ✅ Survived")
            
        if m.prey_extinction_step:
            print(f"  {'Prey Extinct:':<25} Step {m.prey_extinction_step:,}")
        else:
            print(f"  {'Prey:':<25} ✅ Survived")
        
        # Growth Analysis
        print("\n  📉 GROWTH DYNAMICS")
        print(f"  {'Avg Prey Growth:':<25} {m.avg_prey_growth_rate:+.2f}%")
        print(f"  {'Avg Predator Growth:':<25} {m.avg_predator_growth_rate:+.2f}%")
        print(f"  {'Prey Volatility:':<25} {m.prey_volatility:.2f}%")
        print(f"  {'Predator Volatility:':<25} {m.predator_volatility:.2f}%")
        if m.population_oscillation_period > 0:
            print(f"  {'Oscillation Period:':<25} ~{m.population_oscillation_period:.0f} steps")
        
        # Ecosystem Health
        print("\n  🌿 ECOSYSTEM HEALTH")
        stability_bar = self._make_progress_bar(m.ecosystem_stability_score, 100)
        lv_bar = self._make_progress_bar(m.lotka_volterra_fit * 100, 100)
        resilience_bar = self._make_progress_bar(m.ecosystem_resilience, 100)
        biodiv_bar = self._make_progress_bar(m.biodiversity_index, 100)
        risk_bar = self._make_progress_bar(m.extinction_risk_score, 100)
        
        print(f"  {'Stability Score:':<25} {stability_bar} {m.ecosystem_stability_score:.1f}/100")
        print(f"  {'Lotka-Volterra Fit:':<25} {lv_bar} {m.lotka_volterra_fit:.2%}")
        print(f"  {'Ecosystem Resilience:':<25} {resilience_bar} {m.ecosystem_resilience:.1f}/100")
        print(f"  {'Biodiversity Index:':<25} {biodiv_bar} {m.biodiversity_index:.1f}/100")
        
        # Risk Assessment
        print("\n  ⚠️  RISK ASSESSMENT")
        risk_color = "🔴" if m.extinction_risk_score > 70 else "🟡" if m.extinction_risk_score > 40 else "🟢"
        print(f"  {'Extinction Risk:':<25} {risk_bar} {risk_color} {m.extinction_risk_score:.1f}/100")
        
        # Notes/Warnings
        if self.notes:
            print("\n  📝 NOTES & WARNINGS")
            for note in self.notes[:5]:  # Show max 5 notes
                print(f"  • {note}")
        
        print("\n" + "═" * 70 + "\n")
    
    @staticmethod
    def _make_progress_bar(value: float, max_value: float, width: int = 20) -> str:
        """Create a Unicode progress bar."""
        ratio = min(value / max_value, 1.0) if max_value > 0 else 0
        filled = int(width * ratio)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"
    
    @timed_operation("Dashboard Generation")
    def plot_comprehensive_dashboard(self) -> plt.Figure:
        """Create a comprehensive multi-panel dashboard."""
        self.log.info("Generating visualization dashboard...")
        
        colors = self.config.color_scheme
        fig = plt.figure(figsize=self.config.figure_size)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # ═══════════════════════════════════════════════════════════════
        # Panel 1: Population Time Series (Top-left, spans 2 columns)
        # ═══════════════════════════════════════════════════════════════
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_population_timeseries(ax1, colors)
        
        # ═══════════════════════════════════════════════════════════════
        # Panel 2: Phase Space Diagram (Top-right)
        # ═══════════════════════════════════════════════════════════════
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_phase_space(ax2, colors)
        
        # ═══════════════════════════════════════════════════════════════
        # Panel 3: Population Ratio Over Time (Middle-left)
        # ═══════════════════════════════════════════════════════════════
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_population_ratio(ax3, colors)
        
        # ═══════════════════════════════════════════════════════════════
        # Panel 4: Events Timeline (Middle-center)
        # ═══════════════════════════════════════════════════════════════
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_events_timeline(ax4, colors)
        
        # ═══════════════════════════════════════════════════════════════
        # Panel 5: Ecosystem Health (Middle-right)
        # ═══════════════════════════════════════════════════════════════
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_ecosystem_health(ax5, colors)
        
        # ═══════════════════════════════════════════════════════════════
        # Panel 6: Growth Rate Distribution (Bottom-left)
        # ═══════════════════════════════════════════════════════════════
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_growth_distribution(ax6, colors)
        
        # ═══════════════════════════════════════════════════════════════
        # Panel 7: Cumulative Events (Bottom-center)
        # ═══════════════════════════════════════════════════════════════
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_cumulative_events(ax7, colors)
        
        # ═══════════════════════════════════════════════════════════════
        # Panel 8: Population Momentum (Bottom-right)
        # ═══════════════════════════════════════════════════════════════
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_momentum(ax8, colors)
        
        # Main title
        fig.suptitle(
            '🔬 Predator-Prey Ecosystem Analysis Dashboard',
            fontsize=18, fontweight='bold', y=0.98
        )
        
        if self.config.save_plots:
            filepath = self.output_dir / f'ecosystem_dashboard.{self.config.output_format}'
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"  📊 Saved: {filepath}")
        
        if self.config.show_plots:
            plt.show()
        
        return fig
    
    def _plot_population_timeseries(self, ax: plt.Axes, colors: Dict[str, str]) -> None:
        """Plot population over time with filled areas."""
        df = self.data
        
        # Plot smoothed lines
        ax.plot(df['step'], df['prey_smooth'], color=colors['prey'], 
                linewidth=2.5, label='Prey (smoothed)', zorder=3)
        ax.plot(df['step'], df['predator_smooth'], color=colors['predator'], 
                linewidth=2.5, label='Predator (smoothed)', zorder=3)
        
        # Fill areas (raw data)
        ax.fill_between(df['step'], df['prey_population'], alpha=0.2, color=colors['prey'])
        ax.fill_between(df['step'], df['predator_population'], alpha=0.2, color=colors['predator'])
        
        # Mark extinction events
        if self.metrics.predator_extinction_step:
            ax.axvline(x=self.metrics.predator_extinction_step, color=colors['predator'], 
                      linestyle='--', alpha=0.7, label='Predator Extinction')
        
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Population', fontsize=10)
        ax.set_title('Population Dynamics Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0, df['step'].max())
        ax.set_ylim(0, None)
        
    def _plot_phase_space(self, ax: plt.Axes, colors: Dict[str, str]) -> None:
        """Plot phase space diagram with trajectory coloring."""
        df = self.data
        
        # Create color gradient based on time
        n_points = len(df)
        time_colors = plt.cm.viridis(np.linspace(0, 1, n_points))
        
        # Plot trajectory with time-based coloring
        for i in range(n_points - 1):
            ax.plot(df['prey_population'].iloc[i:i+2], 
                   df['predator_population'].iloc[i:i+2],
                   color=time_colors[i], linewidth=1.5, alpha=0.7)
        
        # Mark start and end
        ax.scatter(df['prey_population'].iloc[0], df['predator_population'].iloc[0],
                  color=colors['prey'], s=150, marker='o', zorder=5, 
                  edgecolors='white', linewidths=2, label='Start')
        ax.scatter(df['prey_population'].iloc[-1], df['predator_population'].iloc[-1],
                  color=colors['predator'], s=150, marker='X', zorder=5,
                  edgecolors='white', linewidths=2, label='End')
        
        ax.set_xlabel('Prey Population', fontsize=10)
        ax.set_ylabel('Predator Population', fontsize=10)
        ax.set_title('Phase Space Trajectory', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        
    def _plot_population_ratio(self, ax: plt.Axes, colors: Dict[str, str]) -> None:
        """Plot prey-to-predator ratio over time."""
        df = self.data
        ratio = df['prey_predator_ratio'].replace([np.inf, -np.inf], np.nan)
        
        ax.plot(df['step'], ratio, color=colors['neutral'], linewidth=1.5)
        ax.axhline(y=7, color=colors['warning'], linestyle='--', 
                  alpha=0.7, label='Optimal Ratio (~7:1)')
        
        # Shade danger zones
        ax.axhspan(0, 3, alpha=0.1, color=colors['predator'], label='Predator Dominance')
        ax.axhspan(15, ratio.max() if ratio.max() > 15 else 20, alpha=0.1, 
                  color=colors['prey'], label='Prey Dominance')
        
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Prey:Predator Ratio', fontsize=10)
        ax.set_title('Population Balance', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7)
        ax.set_ylim(0, min(ratio.quantile(0.95) * 1.2, 50))
        
    def _plot_events_timeline(self, ax: plt.Axes, colors: Dict[str, str]) -> None:
        """Plot kills and births as a timeline."""
        df = self.data
        
        ax.bar(df['step'], df['kills_per_step'], color=colors['predator'], 
               alpha=0.7, label='Kills', width=1.0)
        ax.bar(df['step'], -df['births_per_step'], color=colors['prey'], 
               alpha=0.7, label='Births', width=1.0)
        
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Events (Kills ↑ / Births ↓)', fontsize=10)
        ax.set_title('Life & Death Events', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        
    def _plot_ecosystem_health(self, ax: plt.Axes, colors: Dict[str, str]) -> None:
        """Plot ecosystem health indicator over time."""
        df = self.data
        
        # Color based on health value
        health = df['ecosystem_health']
        ax.fill_between(df['step'], health, alpha=0.5, 
                       color=colors['neutral'], label='Health Score')
        ax.plot(df['step'], health, color=colors['neutral'], linewidth=1.5)
        
        # Rolling average
        rolling_health = health.rolling(window=20).mean()
        ax.plot(df['step'], rolling_health, color=colors['warning'], 
               linewidth=2, linestyle='--', label='Trend (20-step avg)')
        
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Health Score', fontsize=10)
        ax.set_title('Ecosystem Health Index', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=8)
        
    def _plot_growth_distribution(self, ax: plt.Axes, colors: Dict[str, str]) -> None:
        """Plot distribution of growth rates."""
        prey_growth = self.data['prey_growth_rate'].replace([np.inf, -np.inf], np.nan).dropna()
        pred_growth = self.data['predator_growth_rate'].replace([np.inf, -np.inf], np.nan).dropna()
        
        # Clip extreme values for visualization
        prey_growth = prey_growth.clip(-50, 50)
        pred_growth = pred_growth.clip(-50, 50)
        
        ax.hist(prey_growth, bins=30, alpha=0.6, color=colors['prey'], 
               label='Prey', density=True)
        ax.hist(pred_growth, bins=30, alpha=0.6, color=colors['predator'], 
               label='Predator', density=True)
        
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel('Growth Rate (%)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title('Growth Rate Distribution', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        
    def _plot_cumulative_events(self, ax: plt.Axes, colors: Dict[str, str]) -> None:
        """Plot cumulative births vs kills."""
        df = self.data
        
        ax.plot(df['step'], df['total_births'], color=colors['prey'], 
               linewidth=2, label='Cumulative Births')
        ax.plot(df['step'], df['total_kills'], color=colors['predator'], 
               linewidth=2, label='Cumulative Kills')
        
        # Net population change area
        net_change = df['total_births'] - df['total_kills']
        ax.fill_between(df['step'], 0, net_change, alpha=0.3, 
                       color=colors['neutral'], label='Net Growth')
        
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Cumulative Count', fontsize=10)
        ax.set_title('Births vs Deaths (Cumulative)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        
    def _plot_momentum(self, ax: plt.Axes, colors: Dict[str, str]) -> None:
        """Plot population momentum indicators."""
        df = self.data
        
        ax.plot(df['step'], df['prey_momentum'], color=colors['prey'], 
               linewidth=1.5, alpha=0.8, label='Prey Momentum')
        ax.plot(df['step'], df['predator_momentum'], color=colors['predator'], 
               linewidth=1.5, alpha=0.8, label='Predator Momentum')
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(df['step'], df['prey_momentum'], 0, alpha=0.2, color=colors['prey'])
        ax.fill_between(df['step'], df['predator_momentum'], 0, alpha=0.2, color=colors['predator'])
        
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Momentum (Δ Population)', fontsize=10)
        ax.set_title('Population Momentum', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        
    def export_report(self) -> Dict[str, Any]:
        """Export analysis as JSON report."""
        m = self.metrics if self.metrics.duration > 0 else self.compute_metrics()
        
        report = {
            'metadata': {
                'source_file': str(self.csv_path),
                'generated_at': datetime.now().isoformat(),
                'analysis_mode': self.config.mode.name
            },
            'simulation_overview': {
                'duration_steps': m.duration,
                'final_prey_population': int(self.data['prey_population'].iloc[-1]),
                'final_predator_population': int(self.data['predator_population'].iloc[-1]),
                'final_total_population': int(self.data['total_population'].iloc[-1])
            },
            'population_peaks': {
                'peak_prey': m.peak_prey,
                'peak_prey_step': m.peak_prey_step,
                'peak_predator': m.peak_predator,
                'peak_predator_step': m.peak_predator_step
            },
            'events': {
                'total_kills': m.total_kills,
                'total_births': m.total_births,
                'kill_efficiency': round(m.kill_efficiency, 6)
            },
            'extinction_events': {
                'prey_extinction_step': m.prey_extinction_step,
                'predator_extinction_step': m.predator_extinction_step,
                'total_extinction_step': m.total_extinction_step
            },
            'growth_dynamics': {
                'avg_prey_growth_rate': round(m.avg_prey_growth_rate, 4),
                'avg_predator_growth_rate': round(m.avg_predator_growth_rate, 4),
                'prey_volatility': round(m.prey_volatility, 4),
                'predator_volatility': round(m.predator_volatility, 4)
            },
            'ecosystem_health': {
                'stability_score': round(m.ecosystem_stability_score, 2),
                'lotka_volterra_fit': round(m.lotka_volterra_fit, 4),
                'ecosystem_resilience': round(m.ecosystem_resilience, 2),
                'biodiversity_index': round(m.biodiversity_index, 2),
                'extinction_risk_score': round(m.extinction_risk_score, 2)
            },
            'advanced_metrics': {
                'prey_carrying_capacity': round(m.prey_carrying_capacity, 2),
                'predator_carrying_capacity': round(m.predator_carrying_capacity, 2),
                'population_oscillation_period': round(m.population_oscillation_period, 2),
                'trophic_efficiency': round(m.trophic_efficiency, 4)
            }
        }
        
        if self.config.export_json:
            # Save timestamped report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_path = self.output_dir / f'analysis_report_{timestamp}.json'
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.output_files.append(str(json_path))
            self.log.info(f"Saved report: {json_path.name}")
            
            # Also save as latest
            latest_path = self.output_dir / 'analysis_report_latest.json'
            with open(latest_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    @timed_operation("Data Export")
    def export_enhanced_data(self) -> Path:
        """Export enhanced dataset with all computed columns."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        enhanced_path = self.output_dir / f'enhanced_species_logs_{timestamp}.csv'
        self.data.to_csv(enhanced_path, index=False)
        self.output_files.append(str(enhanced_path))
        self.log.info(f"Saved enhanced data: {enhanced_path.name}")
        
        # Also save as latest
        latest_path = self.output_dir / 'enhanced_species_logs_latest.csv'
        self.data.to_csv(latest_path, index=False)
        
        return enhanced_path
    
    def _save_run_to_history(self) -> None:
        """Save this run to the history log."""
        entry = RunHistoryEntry(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            source_file=str(self.csv_path),
            source_file_hash=self._compute_file_hash(),
            duration_steps=self.metrics.duration,
            analysis_mode=self.config.mode.name,
            metrics=self.metrics.to_dict(),
            output_files=self.output_files,
            elapsed_time=self.log.elapsed(),
            status=self.status,
            notes=self.notes
        )
        self.history.add_run(entry)
        self.log.info(f"Run logged to history (ID: {self.run_id[:8]})")
    
    def print_run_history(self, n: int = 5) -> None:
        """Print recent run history."""
        recent = self.history.get_recent_runs(n)
        
        if not recent:
            print("  No previous runs found.")
            return
            
        print("\n" + "─" * 70)
        print("  📜 RECENT ANALYSIS RUNS")
        print("─" * 70)
        
        for i, run in enumerate(recent, 1):
            status_icon = "✅" if run['status'] == 'success' else "⚠️" if run['status'] == 'warning' else "❌"
            ts = datetime.fromisoformat(run['timestamp']).strftime('%Y-%m-%d %H:%M')
            print(f"  {i}. [{run['run_id'][:8]}] {ts} | {run['duration_steps']:,} steps | {status_icon} {run['status']}")
            
        # Show statistics
        stats = self.history.get_statistics()
        if stats:
            print("\n  📊 AGGREGATE STATISTICS")
            print(f"     Total Runs: {stats['total_runs']}")
            print(f"     Success Rate: {stats['success_rate']:.1f}%")
            print(f"     Avg Duration: {stats['avg_duration']:.0f} steps")
            
        print("─" * 70 + "\n")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # ML/AI ANALYSIS METHODS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    @timed_operation("ML Analysis")
    def run_ml_analysis(self) -> Dict[str, Any]:
        """Run comprehensive ML/AI analysis on agent-level data."""
        if self.agent_data is None or self.agent_data.empty:
            self.log.warning("No agent data available for ML analysis")
            return {}
        
        self.log.section("ML/AI ANALYSIS v3.0")
        self.log.info(f"Analyzing {len(self.agent_data):,} agent records...")
        
        results = {}
        
        # 1. Survival Prediction
        self.log.info("Training survival prediction model...")
        try:
            model_results = self.ml_predictor.train_survival_model(self.agent_data)
            results['survival_prediction'] = model_results
            self.log.info(f"  Model accuracy: {model_results.get('accuracy', 0):.2%}")
            self.log.info(f"  Precision: {model_results.get('precision', 0):.2%}")
            self.log.info(f"  Top feature: {list(model_results.get('feature_importance', {}).keys())[0] if model_results.get('feature_importance') else 'N/A'}")
        except Exception as e:
            self.log.warning(f"Survival prediction failed: {e}")
            results['survival_prediction'] = {'error': str(e)}
        
        # 2. Behavior Analysis
        self.log.info("Analyzing agent behaviors...")
        try:
            behavior_analyzer = BehaviorAnalyzer(self.agent_data)
            
            # Action distribution
            action_dist = behavior_analyzer.analyze_action_distribution()
            results['action_distribution'] = action_dist
            
            # Behavior over time
            behavior_time = behavior_analyzer.analyze_behavior_over_time()
            results['behavior_over_time'] = behavior_time.to_dict() if hasattr(behavior_time, 'to_dict') else behavior_time
            
            # Agent clustering (use correct method name)
            clusters = behavior_analyzer.cluster_agent_behaviors()
            results['agent_clusters'] = clusters
            self.log.info(f"  Found {clusters.get('n_clusters', 0)} behavior clusters")
            
        except Exception as e:
            self.log.warning(f"Behavior analysis failed: {e}")
            results['behavior_analysis'] = {'error': str(e)}
        
        # 3. Genetic Analysis
        self.log.info("Analyzing genetic traits...")
        try:
            genetic_analyzer = GeneticAnalyzer(self.agent_data)
            
            # Trait evolution
            trait_evolution = genetic_analyzer.analyze_trait_evolution()
            results['trait_evolution'] = {k: v.to_dict() if hasattr(v, 'to_dict') else v 
                                          for k, v in trait_evolution.items()}
            
            # Trait correlations
            trait_corr = genetic_analyzer.analyze_trait_correlations()
            results['trait_correlations'] = trait_corr.to_dict() if hasattr(trait_corr, 'to_dict') else trait_corr
            
            # Optimal traits (use correct method name)
            optimal = genetic_analyzer.identify_optimal_traits()
            results['optimal_traits'] = optimal
            
            # Safe logging with type checking
            pred_traits = optimal.get('species_optimal_profiles', {}).get('predator', {})
            prey_traits = optimal.get('species_optimal_profiles', {}).get('prey', {})
            if pred_traits and isinstance(pred_traits.get('speed'), (int, float)):
                self.log.info(f"  Optimal predator traits: speed={pred_traits['speed']:.2f}")
            if prey_traits and isinstance(prey_traits.get('speed'), (int, float)):
                self.log.info(f"  Optimal prey traits: speed={prey_traits['speed']:.2f}")
            
        except Exception as e:
            self.log.warning(f"Genetic analysis failed: {e}")
            results['genetic_analysis'] = {'error': str(e)}
        
        # 4. Time Series Forecasting
        self.log.info("Running time series analysis...")
        try:
            forecaster = TimeSeriesForecaster(self.data)
            
            # Fit Lotka-Volterra model
            lv_params = forecaster.fit_lotka_volterra()
            results['lotka_volterra_params'] = lv_params
            
            # Generate forecast
            forecast = forecaster.forecast_population(steps_ahead=50)
            results['population_forecast'] = {k: v.tolist() if hasattr(v, 'tolist') else v 
                                              for k, v in forecast.items()}
            
            # Detect cycles
            cycles = forecaster.detect_cycles()
            
            # Transform cycle data to expected format
            primary = cycles.get('primary_cycle', {})
            results['population_cycles'] = {
                'prey_period': primary.get('period', 0) if primary else 0,
                'cycle_strength': primary.get('strength', 0) if primary else 0,
                'is_cyclic': cycles.get('is_cyclic', False),
                'all_cycles': cycles.get('detected_cycles', [])
            }
            
            # Safe logging with type checking
            prey_period = results['population_cycles'].get('prey_period', 0)
            if isinstance(prey_period, (int, float)) and prey_period > 0:
                self.log.info(f"  Detected primary cycle: {prey_period:.1f} steps")
            
        except Exception as e:
            self.log.warning(f"Time series analysis failed: {e}")
            results['time_series'] = {'error': str(e)}
        
        # Store results
        self.ml_results = results
        
        return results
    
    def generate_ai_insights(self) -> List[str]:
        """Generate AI-powered insights from analysis results."""
        if not self.ml_results:
            self.log.warning("No ML results available for insight generation")
            return []
        
        self.log.info("Generating AI insights...")
        
        # Generate insights from each category
        all_insights = []
        
        # 1. Survival model insights
        survival_data = self.ml_results.get('survival_prediction', {})
        if survival_data and 'error' not in survival_data:
            all_insights.extend(self.ai_insights.generate_survival_insights(survival_data))
        
        # 2. Behavior insights
        action_dist = self.ml_results.get('action_distribution', {})
        behavior_time = self.ml_results.get('behavior_over_time', {})
        if action_dist or behavior_time:
            combined_behavior = {**action_dist, **behavior_time} if isinstance(behavior_time, dict) else action_dist
            all_insights.extend(self.ai_insights.generate_behavior_insights(combined_behavior))
        
        # 3. Genetic insights
        trait_evol = self.ml_results.get('trait_evolution', {})
        optimal = self.ml_results.get('optimal_traits', {})
        if trait_evol or optimal:
            combined_genetic = {'trait_evolution': trait_evol, 'optimal_traits': optimal}
            all_insights.extend(self.ai_insights.generate_genetic_insights(combined_genetic))
        
        # 4. Forecast insights  
        forecast = self.ml_results.get('population_forecast', {})
        if forecast:
            all_insights.extend(self.ai_insights.generate_forecast_insights({'ensemble': forecast}))
        
        self.log.info(f"  Generated {len(all_insights)} insights")
        
        return all_insights
    
    def print_ml_summary(self) -> None:
        """Print ML analysis summary."""
        if not self.ml_results:
            return
        
        print("\n" + "─" * 70)
        print("  🤖 ML/AI ANALYSIS SUMMARY")
        print("─" * 70)
        
        # Survival Model
        survival = self.ml_results.get('survival_prediction', {})
        if survival and 'accuracy' in survival:
            print("\n  📊 SURVIVAL PREDICTION MODEL")
            print(f"     Accuracy: {survival['accuracy']:.2%}")
            print(f"     Precision: {survival.get('precision', 0):.2%}")
            print(f"     Recall: {survival.get('recall', 0):.2%}")
            print(f"     F1 Score: {survival.get('f1_score', 0):.2%}")
            
            if 'feature_importance' in survival:
                print("\n     📈 Feature Importance (Top 5):")
                for i, (feat, imp) in enumerate(list(survival['feature_importance'].items())[:5], 1):
                    bar = "█" * int(imp * 20)
                    print(f"        {i}. {feat:15} {bar} {imp:.3f}")
        
        # Agent Clusters
        clusters = self.ml_results.get('agent_clusters', {})
        if clusters and 'cluster_profiles' in clusters:
            print("\n  🎯 BEHAVIOR CLUSTERS")
            for cid, profile in clusters['cluster_profiles'].items():
                print(f"     Cluster {cid}: {profile.get('size', 0)} agents")
                if 'dominant_action' in profile:
                    print(f"        Dominant action: {profile['dominant_action']}")
        
        # Optimal Traits
        optimal = self.ml_results.get('optimal_traits', {})
        if optimal:
            print("\n  🧬 OPTIMAL GENETIC TRAITS")
            species_profiles = optimal.get('species_optimal_profiles', {})
            if species_profiles:
                for species, traits in species_profiles.items():
                    if isinstance(traits, dict):
                        print(f"     {species.upper()}:")
                        for trait, value in traits.items():
                            if isinstance(value, (int, float)):
                                print(f"        {trait}: {value:.3f}")
            else:
                # Fallback to old format
                for species, traits in optimal.items():
                    if isinstance(traits, dict) and species != 'optimal_traits':
                        print(f"     {species.upper()}:")
                        for trait, value in traits.items():
                            if isinstance(value, (int, float)):
                                print(f"        {trait}: {value:.3f}")
        
        # Population Cycles
        cycles = self.ml_results.get('population_cycles', {})
        if cycles:
            print("\n  🔄 POPULATION CYCLES")
            prey_period = cycles.get('prey_period', 0)
            if isinstance(prey_period, (int, float)) and prey_period > 0:
                print(f"     Primary cycle period: {prey_period:.1f} steps")
            cycle_strength = cycles.get('cycle_strength', 0)
            if isinstance(cycle_strength, (int, float)) and cycle_strength > 0:
                print(f"     Cycle strength: {cycle_strength:.1f}%")
            if cycles.get('is_cyclic'):
                print(f"     ✅ Ecosystem exhibits cyclic behavior")
            else:
                print(f"     ⚠️ No clear cyclic pattern detected")
        
        print("─" * 70)
    
    def print_ai_insights(self, insights: List[str]) -> None:
        """Print AI-generated insights."""
        if not insights:
            return
        
        print("\n" + "─" * 70)
        print("  💡 AI-GENERATED INSIGHTS")
        print("─" * 70)
        
        for i, insight in enumerate(insights, 1):
            # Add emoji based on content
            if 'warning' in insight.lower() or 'risk' in insight.lower() or 'decline' in insight.lower():
                emoji = "⚠️"
            elif 'strong' in insight.lower() or 'success' in insight.lower() or 'effective' in insight.lower():
                emoji = "✅"
            elif 'evolv' in insight.lower() or 'adapt' in insight.lower():
                emoji = "🧬"
            elif 'predict' in insight.lower() or 'forecast' in insight.lower():
                emoji = "🔮"
            else:
                emoji = "💡"
            
            # Wrap text nicely
            wrapped = textwrap.fill(insight, width=60, initial_indent="     ", subsequent_indent="     ")
            print(f"\n  {emoji} Insight #{i}:")
            print(wrapped)
        
        print("\n" + "─" * 70)
    
    def plot_ml_dashboard(self) -> plt.Figure:
        """Create ML analysis visualization dashboard."""
        if not self.ml_results:
            self.log.warning("No ML results to visualize")
            return None
        
        self.log.info("Generating ML dashboard...")
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('🤖 ML/AI Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3,
                             left=0.06, right=0.94, top=0.92, bottom=0.06)
        
        colors = {
            'predator': '#e74c3c',
            'prey': '#3498db',
            'neutral': '#2ecc71',
            'warning': '#f39c12',
            'cluster1': '#9b59b6',
            'cluster2': '#1abc9c',
            'cluster3': '#e67e22'
        }
        
        # 1. Feature Importance
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_feature_importance(ax1, colors)
        
        # 2. Action Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_action_distribution(ax2, colors)
        
        # 3. Trait Evolution
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_trait_evolution(ax3, colors)
        
        # 4. Survival by Species
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_survival_by_traits(ax4, colors)
        
        # 5. Agent Clusters
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_agent_clusters(ax5, colors)
        
        # 6. Population Forecast
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_population_forecast(ax6, colors)
        
        # 7. Behavior Over Time Heatmap
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_behavior_heatmap(ax7, colors)
        
        # 8. Trait Correlations
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_trait_correlations(ax8, colors)
        
        # 9. Key Insights Box
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_insights_box(ax9, colors)
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ml_dashboard_path = self.output_dir / f'ml_dashboard_{timestamp}.png'
        fig.savefig(ml_dashboard_path, dpi=self.config.dpi, facecolor='white',
                   bbox_inches='tight', pad_inches=0.2)
        self.output_files.append(str(ml_dashboard_path))
        self.log.info(f"Saved ML dashboard: {ml_dashboard_path.name}")
        
        # Also save as latest
        latest_path = self.output_dir / 'ml_dashboard_latest.png'
        fig.savefig(latest_path, dpi=self.config.dpi, facecolor='white',
                   bbox_inches='tight', pad_inches=0.2)
        
        if self.config.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _plot_feature_importance(self, ax: plt.Axes, colors: Dict[str, str]) -> None:
        """Plot feature importance from survival model."""
        survival = self.ml_results.get('survival_prediction', {})
        importance = survival.get('feature_importance', {})
        
        if not importance:
            ax.text(0.5, 0.5, 'No feature importance data', ha='center', va='center')
            ax.set_title('Feature Importance', fontsize=12, fontweight='bold')
            return
        
        # Take top 8 features
        top_features = dict(list(importance.items())[:8])
        features = list(top_features.keys())
        values = list(top_features.values())
        
        # Horizontal bar chart
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, values, color=colors['neutral'], alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Importance', fontsize=10)
        ax.set_title('🎯 Feature Importance', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=8)
    
    def _plot_action_distribution(self, ax: plt.Axes, colors: Dict[str, str]) -> None:
        """Plot action distribution by species."""
        action_dist = self.ml_results.get('action_distribution', {})
        
        if not action_dist:
            ax.text(0.5, 0.5, 'No action data', ha='center', va='center')
            ax.set_title('Action Distribution', fontsize=12, fontweight='bold')
            return
        
        predator_actions = action_dist.get('predator', {})
        prey_actions = action_dist.get('prey', {})
        
        all_actions = list(set(list(predator_actions.keys()) + list(prey_actions.keys())))
        x = np.arange(len(all_actions))
        width = 0.35
        
        pred_vals = [predator_actions.get(a, 0) for a in all_actions]
        prey_vals = [prey_actions.get(a, 0) for a in all_actions]
        
        ax.bar(x - width/2, pred_vals, width, label='Predator', color=colors['predator'], alpha=0.8)
        ax.bar(x + width/2, prey_vals, width, label='Prey', color=colors['prey'], alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(all_actions, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('📊 Action Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
    
    def _plot_trait_evolution(self, ax: plt.Axes, colors: Dict[str, str]) -> None:
        """Plot trait evolution over generations."""
        trait_evol = self.ml_results.get('trait_evolution', {})
        
        if not trait_evol or 'speed' not in trait_evol:
            ax.text(0.5, 0.5, 'No trait evolution data', ha='center', va='center')
            ax.set_title('Trait Evolution', fontsize=12, fontweight='bold')
            return
        
        # Plot speed evolution for both species
        speed_data = trait_evol.get('speed', {})
        
        if isinstance(speed_data, dict):
            for species, data in speed_data.items():
                if isinstance(data, dict) and 'mean' in data:
                    generations = list(data.get('mean', {}).keys())
                    values = list(data.get('mean', {}).values())
                    color = colors['predator'] if 'pred' in species.lower() else colors['prey']
                    ax.plot(generations, values, marker='o', markersize=3,
                           label=species, color=color, linewidth=1.5)
        
        ax.set_xlabel('Generation', fontsize=10)
        ax.set_ylabel('Avg Speed', fontsize=10)
        ax.set_title('🧬 Trait Evolution (Speed)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_survival_by_traits(self, ax: plt.Axes, colors: Dict[str, str]) -> None:
        """Plot survival rates by trait values."""
        if self.agent_data is None or self.agent_data.empty:
            ax.text(0.5, 0.5, 'No agent data', ha='center', va='center')
            ax.set_title('Survival by Traits', fontsize=12, fontweight='bold')
            return
        
        # Get last step for each agent
        agent_final = self.agent_data.groupby('agent_id').last().reset_index()
        
        # Scatter plot: speed vs intelligence, colored by survival (health > 0)
        survived = agent_final['health'] > 0
        
        ax.scatter(agent_final.loc[~survived, 'speed'], 
                  agent_final.loc[~survived, 'intelligence'],
                  c=colors['predator'], alpha=0.3, s=20, label='Died')
        ax.scatter(agent_final.loc[survived, 'speed'], 
                  agent_final.loc[survived, 'intelligence'],
                  c=colors['neutral'], alpha=0.5, s=30, label='Survived')
        
        ax.set_xlabel('Speed', fontsize=10)
        ax.set_ylabel('Intelligence', fontsize=10)
        ax.set_title('🎲 Survival by Traits', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
    
    def _plot_agent_clusters(self, ax: plt.Axes, colors: Dict[str, str]) -> None:
        """Plot agent behavior clusters."""
        clusters = self.ml_results.get('agent_clusters', {})
        
        if not clusters or 'labels' not in clusters:
            ax.text(0.5, 0.5, 'No cluster data', ha='center', va='center')
            ax.set_title('Behavior Clusters', fontsize=12, fontweight='bold')
            return
        
        # Create pie chart of cluster sizes
        cluster_profiles = clusters.get('cluster_profiles', {})
        sizes = []
        labels = []
        cluster_colors = [colors['cluster1'], colors['cluster2'], colors['cluster3'], 
                         colors['neutral'], colors['warning']]
        
        for cid, profile in cluster_profiles.items():
            sizes.append(profile.get('size', 0))
            labels.append(f"Cluster {cid}")
        
        if sizes:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, 
                                               colors=cluster_colors[:len(sizes)],
                                               autopct='%1.1f%%', startangle=90)
            ax.set_title('🎯 Behavior Clusters', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No clusters found', ha='center', va='center')
    
    def _plot_population_forecast(self, ax: plt.Axes, colors: Dict[str, str]) -> None:
        """Plot population forecast."""
        forecast = self.ml_results.get('population_forecast', {})
        
        if not forecast:
            ax.text(0.5, 0.5, 'No forecast data', ha='center', va='center')
            ax.set_title('Population Forecast', fontsize=12, fontweight='bold')
            return
        
        # Plot historical data
        historical_steps = len(self.data)
        ax.plot(self.data['step'], self.data['prey_population'], 
               color=colors['prey'], alpha=0.7, label='Prey (actual)')
        ax.plot(self.data['step'], self.data['predator_population'], 
               color=colors['predator'], alpha=0.7, label='Predator (actual)')
        
        # Plot forecast
        if 'prey' in forecast and 'steps' in forecast:
            forecast_steps = forecast['steps']
            ax.plot(forecast_steps, forecast['prey'], 
                   color=colors['prey'], linestyle='--', linewidth=2, label='Prey (forecast)')
            ax.plot(forecast_steps, forecast['predator'], 
                   color=colors['predator'], linestyle='--', linewidth=2, label='Predator (forecast)')
            
            # Add confidence band
            ax.axvspan(historical_steps, max(forecast_steps), alpha=0.1, color='gray')
        
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Population', fontsize=10)
        ax.set_title('🔮 Population Forecast', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
    
    def _plot_behavior_heatmap(self, ax: plt.Axes, colors: Dict[str, str]) -> None:
        """Plot behavior over time heatmap."""
        behavior_time = self.ml_results.get('behavior_over_time', {})
        
        if not behavior_time or not isinstance(behavior_time, dict):
            ax.text(0.5, 0.5, 'No behavior time data', ha='center', va='center')
            ax.set_title('Behavior Over Time', fontsize=12, fontweight='bold')
            return
        
        try:
            # Convert to DataFrame for heatmap
            df = pd.DataFrame(behavior_time)
            if df.empty:
                raise ValueError("Empty behavior data")
            
            # Downsample for visualization
            if len(df) > 50:
                df = df.iloc[::len(df)//50]
            
            im = ax.imshow(df.T, aspect='auto', cmap='YlOrRd')
            ax.set_xlabel('Time Bin', fontsize=10)
            ax.set_ylabel('Action', fontsize=10)
            ax.set_title('📈 Behavior Heatmap', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax, shrink=0.8)
        except Exception as e:
            ax.text(0.5, 0.5, f'Heatmap error: {str(e)[:30]}', ha='center', va='center')
            ax.set_title('Behavior Over Time', fontsize=12, fontweight='bold')
    
    def _plot_trait_correlations(self, ax: plt.Axes, colors: Dict[str, str]) -> None:
        """Plot trait correlation matrix."""
        corr = self.ml_results.get('trait_correlations', {})
        
        if not corr or not isinstance(corr, dict):
            ax.text(0.5, 0.5, 'No correlation data', ha='center', va='center')
            ax.set_title('Trait Correlations', fontsize=12, fontweight='bold')
            return
        
        try:
            df = pd.DataFrame(corr)
            im = ax.imshow(df, cmap='RdYlBu', vmin=-1, vmax=1)
            
            ax.set_xticks(range(len(df.columns)))
            ax.set_yticks(range(len(df.index)))
            ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(df.index, fontsize=8)
            ax.set_title('🔗 Trait Correlations', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax, shrink=0.8)
        except Exception as e:
            ax.text(0.5, 0.5, f'Correlation error', ha='center', va='center')
    
    def _plot_insights_box(self, ax: plt.Axes, colors: Dict[str, str]) -> None:
        """Plot key insights summary box."""
        ax.axis('off')
        
        # Gather key metrics
        survival = self.ml_results.get('survival_prediction', {})
        cycles = self.ml_results.get('population_cycles', {})
        optimal = self.ml_results.get('optimal_traits', {})
        
        lines = []
        lines.append("═══ KEY AI INSIGHTS ═══")
        lines.append("")
        
        if survival and 'accuracy' in survival:
            lines.append(f"🎯 Model Accuracy: {survival['accuracy']:.1%}")
        
        if cycles and cycles.get('prey_period', 0) > 0:
            lines.append(f"🔄 Prey Cycle: {cycles['prey_period']:.0f} steps")
        
        if optimal:
            pred_opt = optimal.get('predator', {})
            if pred_opt and 'speed' in pred_opt:
                lines.append(f"🦁 Optimal Pred Speed: {pred_opt['speed']:.2f}")
            
            prey_opt = optimal.get('prey', {})
            if prey_opt and 'speed' in prey_opt:
                lines.append(f"🐰 Optimal Prey Speed: {prey_opt['speed']:.2f}")
        
        if self.agent_data is not None:
            lines.append(f"📊 Agents Analyzed: {self.agent_data['agent_id'].nunique():,}")
            lines.append(f"📝 Total Records: {len(self.agent_data):,}")
        
        # Add to plot
        text = '\n'.join(lines)
        ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               family='monospace')
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete analysis pipeline with full logging and ML/AI analysis."""
        try:
            self.log.section("ECOSYSTEM ANALYSIS v3.0 - AI ENHANCED")
            self.log.info(f"Source: {self.csv_path.name}")
            self.log.info(f"Mode: {self.config.mode.name}")
            
            print("\n" + "━" * 70)
            print("  🚀 STARTING ECOSYSTEM ANALYSIS v3.0 (AI ENHANCED)")
            print("━" * 70 + "\n")
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 1: TRADITIONAL ANALYSIS
            # ═══════════════════════════════════════════════════════════════════
            
            # Compute metrics
            self.compute_metrics()
            
            # Detect anomalies
            anomalies = self.detect_anomalies()
            if any(a['severity'] == 'high' for a in anomalies):
                self.status = 'warning'
            
            # Print summary
            self.print_summary()
            
            # Generate visualizations
            self.log.info("Generating species dashboard...")
            self.plot_comprehensive_dashboard()
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 2: ML/AI ANALYSIS (NEW)
            # ═══════════════════════════════════════════════════════════════════
            
            print("\n" + "━" * 70)
            print("  🤖 RUNNING ML/AI ANALYSIS...")
            print("━" * 70 + "\n")
            
            # Run ML analysis on agent-level data
            ml_results = self.run_ml_analysis()
            
            # Generate AI insights
            insights = self.generate_ai_insights() if ml_results else []
            
            # Print ML summary
            self.print_ml_summary()
            
            # Print AI insights
            if insights:
                self.print_ai_insights(insights)
            
            # Generate ML dashboard
            if ml_results:
                self.log.info("Generating ML dashboard...")
                self.plot_ml_dashboard()
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 3: EXPORT & FINALIZE
            # ═══════════════════════════════════════════════════════════════════
            
            # Export data
            self.log.section("EXPORTING DATA")
            report = self.export_report()
            
            # Add ML results to report
            if ml_results:
                report['ml_analysis'] = {
                    'survival_accuracy': ml_results.get('survival_prediction', {}).get('accuracy', 0),
                    'behavior_clusters': len(ml_results.get('agent_clusters', {}).get('cluster_profiles', {})),
                    'ai_insights_count': len(insights),
                    'agent_records_analyzed': len(self.agent_data) if self.agent_data is not None else 0
                }
            
            # Export enhanced ML report
            self._export_ml_report(ml_results, insights)
            
            self.export_enhanced_data()
            
            # Save to run history
            if self.config.log_runs:
                self._save_run_to_history()
            
            # Show run history
            if self.config.compare_runs:
                self.print_run_history()
            
            # Final summary
            print("\n" + "━" * 70)
            print("  ✅ ANALYSIS COMPLETE! (v3.0 AI Enhanced)")
            print(f"  🆔 Run ID: {self.run_id[:8]}")
            print(f"  ⏱️  Elapsed: {self.log.elapsed()}")
            print(f"  📁 Output: {self.output_dir}")
            print(f"  📝 Log: {self.log.log_file.name}")
            if ml_results:
                print(f"  🤖 ML Insights: {len(insights)} generated")
            print("━" * 70 + "\n")
            
            self.log.info(f"Analysis completed successfully in {self.log.elapsed()}")
            
            return report
            
        except Exception as e:
            self.status = 'error'
            self.log.critical(f"Analysis failed: {str(e)}")
            self.log.debug(traceback.format_exc())
            
            if self.config.log_runs:
                self.notes.append(f"Error: {str(e)}")
                self._save_run_to_history()
            
            raise
    
    def _export_ml_report(self, ml_results: Dict[str, Any], insights: List[str]) -> None:
        """Export comprehensive ML analysis report."""
        if not ml_results:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare serializable version of ML results
        serializable_results = {}
        for key, value in ml_results.items():
            if isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict()
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = self._make_serializable(value)
            else:
                serializable_results[key] = value
        
        ml_report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'run_id': self.run_id,
                'analyzer_version': '3.0',
                'agent_records_analyzed': len(self.agent_data) if self.agent_data is not None else 0
            },
            'ml_results': serializable_results,
            'ai_insights': insights
        }
        
        # Save timestamped report
        ml_report_path = self.output_dir / f'ml_analysis_report_{timestamp}.json'
        with open(ml_report_path, 'w') as f:
            json.dump(ml_report, f, indent=2, default=str)
        self.output_files.append(str(ml_report_path))
        self.log.info(f"Saved ML report: {ml_report_path.name}")
        
        # Also save as latest
        latest_path = self.output_dir / 'ml_analysis_report_latest.json'
        with open(latest_path, 'w') as f:
            json.dump(ml_report, f, indent=2, default=str)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Recursively make objects JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class RunComparison:
    """Compare multiple analysis runs."""
    
    def __init__(self, output_dir: Path):
        self.history = RunHistoryManager(output_dir)
        
    def compare_recent(self, n: int = 5) -> pd.DataFrame:
        """Create comparison DataFrame of recent runs."""
        recent = self.history.get_recent_runs(n)
        
        if not recent:
            return pd.DataFrame()
            
        data = []
        for run in recent:
            metrics = run.get('metrics', {})
            data.append({
                'run_id': run['run_id'][:8],
                'timestamp': run['timestamp'],
                'duration': run['duration_steps'],
                'peak_prey': metrics.get('peak_prey', 0),
                'peak_predator': metrics.get('peak_predator', 0),
                'stability': metrics.get('ecosystem_stability_score', 0),
                'extinction_risk': metrics.get('extinction_risk_score', 0),
                'status': run['status']
            })
            
        return pd.DataFrame(data)
    
    def plot_metric_trend(self, metric: str, n: int = 20) -> plt.Figure:
        """Plot a metric across recent runs."""
        df = self.history.compare_metrics(metric)
        
        if df.empty:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(df)), df[metric], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Run #')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Across Recent Runs')
        ax.grid(True, alpha=0.3)
        
        return fig


def find_latest_species_log(base_path: Path) -> Optional[Path]:
    """Find the most recent species log file."""
    possible_paths = [
        base_path / 'data' / 'logs' / 'species_logs.csv',
        base_path / 'data' / 'species_logs.csv',
        base_path / 'logs' / 'species_logs.csv',
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Search recursively
    for path in base_path.rglob('species_logs.csv'):
        return path
    
    return None


def find_simulation_logs(base_path: Path) -> Optional[Path]:
    """Find the agent-level simulation log file."""
    possible_paths = [
        base_path / 'data' / 'logs' / 'simulation_logs.csv',
        base_path / 'data' / 'simulation_logs.csv',
        base_path / 'logs' / 'simulation_logs.csv',
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Search recursively
    for path in base_path.rglob('simulation_logs.csv'):
        return path
    
    return None


def main():
    """Main entry point with smart file discovery and ML analysis."""
    print("\n" + "═" * 70)
    print("  🔬 PREDATOR-PREY ECOSYSTEM ANALYZER v3.0 - AI ENHANCED")
    print("═" * 70)
    print("  🤖 Machine Learning | 📈 Forecasting | 🧬 Genetics | 💡 Insights")
    print("═" * 70)
    
    # Try to find the data files
    base_path = Path(__file__).parent
    csv_path = find_latest_species_log(base_path)
    sim_log_path = find_simulation_logs(base_path)
    
    if csv_path is None:
        print("\n  ❌ Error: Could not find species_logs.csv")
        print("     Please ensure simulation has been run and data exists.\n")
        return
    
    print(f"\n  📂 Species data: {csv_path}")
    
    if sim_log_path:
        # Count records for display
        with open(sim_log_path, 'r') as f:
            line_count = sum(1 for _ in f) - 1  # Subtract header
        print(f"  📂 Agent data: {sim_log_path} ({line_count:,} records)")
    else:
        print("  ⚠️  No simulation_logs.csv found - ML analysis will be limited")
    
    # Configure analysis
    config = AnalysisConfig(
        mode=AnalysisMode.STANDARD,
        dpi=150,
        show_plots=True,
        save_plots=True,
        export_json=True,
        log_runs=True,
        compare_runs=True
    )
    
    # Run analysis
    try:
        analyzer = EcosystemAnalyzer(csv_path, config)
        analyzer.run_full_analysis()
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Analysis interrupted by user.\n")
    except Exception as e:
        print(f"\n  ❌ Analysis failed: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
