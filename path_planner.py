#!/usr/bin/env python3
"""
Path Planner Module
Advanced path planning with obstacle avoidance for autonomous drone navigation
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time
import math
from collections import deque
import heapq

class PlannerType(Enum):
    A_STAR = "a_star"
    RRT = "rrt"
    POTENTIAL_FIELD = "potential_field"
    DYNAMIC_WINDOW = "dynamic_window"

@dataclass
class Waypoint:
    position: Tuple[float, float, float]
    timestamp: float
    velocity: Tuple[float, float, float]
    acceleration: Tuple[float, float, float]

@dataclass
class Path:
    waypoints: List[Waypoint]
    total_distance: float
    total_time: float
    safety_margin: float
    cost: float

class PathPlanner:
    """Advanced path planning system with multiple algorithms"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Planning parameters
        self.planner_type = PlannerType(config.planner_type)
        self.safety_margin = config.safety_margin
        self.max_velocity = config.max_velocity
        self.max_acceleration = config.max_acceleration
        self.planning_horizon = config.planning_horizon
        
        # Grid-based planning
        self.grid_resolution = config.grid_resolution
        self.grid_size = config.grid_size
        self.occupancy_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        
        # Path history and smoothing
        self.path_history = deque(maxlen=100)
        self.current_path = None
        self.path_index = 0
        
        # Dynamic replanning
        self.replan_threshold = config.replan_threshold
        self.last_replan_time = 0
        self.min_replan_interval = config.min_replan_interval
        
        # Emergency behaviors
        self.emergency_stop_distance = config.emergency_stop_distance
        self.collision_avoidance_distance = config.collision_avoidance_distance
        
        # Performance tracking
        self.planning_times = []
        self.path_qualities = []
        
        self.logger.info(f"Path planner initialized with {self.planner_type.value} algorithm")

    def plan_path(self, start_position: Tuple[float, float, float], 
                  target_position: Optional[Tuple[float, float, float]], 
                  obstacles: List[Dict], 
                  depth_map: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """Main path planning function"""
        start_time = time.time()
        
        try:
            # Check if we need emergency behavior
            emergency_velocity = self._check_emergency_conditions(start_position, obstacles)
            if emergency_velocity is not None:
                return emergency_velocity
            
            # Update occupancy grid with obstacles
            self._update_occupancy_grid(obstacles, depth_map)
            
            # Check if replanning is needed
            if not self._should_replan(start_position, target_position, obstacles):
                # Continue with current path
                velocity = self._follow_current_path(start_position)
                if velocity is not None:
                    return velocity
            
            # Plan new path
            if target_position is None:
                # Hover in place
                return (0.0, 0.0, 0.0)
            
            # Choose planning algorithm
            if self.planner_type == PlannerType.A_STAR:
                path = self._plan_a_star(start_position, target_position, obstacles)
            elif self.planner_type == PlannerType.RRT:
                path = self._plan_rrt(start_position, target_position, obstacles)
            elif self.planner_type == PlannerType.POTENTIAL_FIELD:
                path = self._plan_potential_field(start_position, target_position, obstacles)
            elif self.planner_type == PlannerType.DYNAMIC_WINDOW:
                return self._plan_dynamic_window(start_position, target_position, obstacles)
            else:
                path = self._plan_potential_field(start_position, target_position, obstacles)
            
            # Store and smooth path
            if path and path.waypoints:
                self.current_path = self._smooth_path(path)
                self.path_index = 0
                self.last_replan_time = time.time()
                
                # Calculate velocity for first waypoint
                velocity = self._calculate_velocity_command(start_position)
            else:
                # No valid path found - hover
                velocity = (0.0, 0.0, 0.0)
            
            # Track performance
            planning_time = time.time() - start_time
            self._update_performance_metrics(planning_time, path)
            
            return velocity
            
        except Exception as e:
            self.logger.error(f"Error in path planning: {e}")
            return (0.0, 0.0, 0.0)

    def _check_emergency_conditions(self, position: Tuple[float, float, float], 
                                   obstacles: List[Dict]) -> Optional[Tuple[float, float, float]]:
        """Check for emergency conditions requiring immediate action"""
        try:
            for obstacle in obstacles:
                distance = obstacle.get('distance', float('inf'))
                threat_level = obstacle.get('threat_level', 0.0)
                
                # Emergency stop if obstacle too close
                if distance < self.emergency_stop_distance or threat_level > 0.9:
                    self.logger.warning(f"Emergency stop: obstacle at {distance}m")
                    return (0.0, 0.0, 0.0)
                
                # Emergency avoidance maneuver
                if distance < self.collision_avoidance_distance:
                    avoidance_velocity = self._calculate_avoidance_velocity(position, obstacle)
                    self.logger.warning(f"Emergency avoidance: {avoidance_velocity}")
                    return avoidance_velocity
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking emergency conditions: {e}")
            return (0.0, 0.0, 0.0)

    def _calculate_avoidance_velocity(self, position: Tuple[float, float, float], 
                                    obstacle: Dict) -> Tuple[float, float, float]:
        """Calculate emergency avoidance velocity"""
        try:
            pos = np.array(position)
            obs_pos = np.array(obstacle['position'])
            
            # Vector away from obstacle
            avoidance_vector = pos - obs_pos
            distance = np.linalg.norm(avoidance_vector)
            
            if distance > 0:
                # Normalize and scale
                avoidance_vector = avoidance_vector / distance
                avoidance_speed = min(self.max_velocity, 3.0)  # Emergency speed
                
                return tuple(avoidance_vector * avoidance_speed)
            else:
                # Move up if too close
                return (0.0, 0.0, 2.0)
                
        except Exception as e:
            self.logger.error(f"Error calculating avoidance velocity: {e}")
            return (0.0, 0.0, 1.0)

    def _should_replan(self, position: Tuple[float, float, float], 
                      target: Optional[Tuple[float, float, float]], 
                      obstacles: List[Dict]) -> bool:
        """Determine if replanning is necessary"""
        try:
            current_time = time.time()
            
            # Time-based replanning
            if current_time - self.last_replan_time > self.min_replan_interval:
                return True
            
            # No current path
            if self.current_path is None:
                return True
            
            # Target changed significantly
            if target and self.current_path.waypoints:
                last_target = self.current_path.waypoints[-1].position
                target_distance = np.linalg.norm(np.array(target) - np.array(last_target))
                if target_distance > self.replan_threshold:
                    return True
            
            # Path blocked by new obstacles
            if self._is_path_blocked(obstacles):
                return True
            
            # Deviated too far from path
            if self._path_deviation_too_large(position):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking replan conditions: {e}")
            return True

    def _is_path_blocked(self, obstacles: List[Dict]) -> bool:
        """Check if current path is blocked by obstacles"""
        try:
            if not self.current_path or not self.current_path.waypoints:
                return False
            
            # Check upcoming waypoints
            for i in range(self.path_index, min(len(self.current_path.waypoints), self.path_index + 5)):
                waypoint_pos = np.array(self.current_path.waypoints[i].position)
                
                for obstacle in obstacles:
                    obs_pos = np.array(obstacle['position'])
                    distance = np.linalg.norm(waypoint_pos - obs_pos)
                    
                    if distance < self.safety_margin:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking path blockage: {e}")
            return True

    def _path_deviation_too_large(self, position: Tuple[float, float, float]) -> bool:
        """Check if current position deviates too much from planned path"""
        try:
            if not self.current_path or not self.current_path.waypoints:
                return False
            
            if self.path_index >= len(self.current_path.waypoints):
                return True
            
            target_pos = np.array(self.current_path.waypoints[self.path_index].position)
            current_pos = np.array(position)
            deviation = np.linalg.norm(current_pos - target_pos)
            
            return deviation > self.replan_threshold
            
        except Exception as e:
            self.logger.error(f"Error checking path deviation: {e}")
            return True

    def _follow_current_path(self, position: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """Follow the current planned path"""
        try:
            if not self.current_path or not self.current_path.waypoints:
                return None
            
            # Update path index based on position
            self._update_path_index(position)
            
            if self.path_index >= len(self.current_path.waypoints):
                # Reached end of path
                return (0.0, 0.0, 0.0)
            
            # Calculate velocity command
            return self._calculate_velocity_command(position)
            
        except Exception as e:
            self.logger.error(f"Error following current path: {e}")
            return None

    def _update_path_index(self, position: Tuple[float, float, float]):
        """Update the current waypoint index"""
        try:
            if not self.current_path or not self.current_path.waypoints:
                return
            
            current_pos = np.array(position)
            
            # Find closest waypoint ahead
            min_distance = float('inf')
            best_index = self.path_index
            
            for i in range(self.path_index, len(self.current_path.waypoints)):
                waypoint_pos = np.array(self.current_path.waypoints[i].position)
                distance = np.linalg.norm(current_pos - waypoint_pos)
                
                # If we're close enough to this waypoint, advance
                if distance < 1.0:  # 1 meter tolerance
                    best_index = min(i + 1, len(self.current_path.waypoints) - 1)
                    break
            
            self.path_index = best_index
            
        except Exception as e:
            self.logger.error(f"Error updating path index: {e}")

    def _calculate_velocity_command(self, position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Calculate velocity command to follow path"""
        try:
            if not self.current_path or self.path_index >= len(self.current_path.waypoints):
                return (0.0, 0.0, 0.0)
            
            current_pos = np.array(position)
            target_waypoint = self.current_path.waypoints[self.path_index]
            target_pos = np.array(target_waypoint.position)
            
            # Calculate direction vector
            direction = target_pos - current_pos
            distance = np.linalg.norm(direction)
            
            if distance < 0.1:  # Very close
                return (0.0, 0.0, 0.0)
            
            # Normalize direction
            direction = direction / distance
            
            # Calculate desired speed based on distance
            if distance > 5.0:
                speed = self.max_velocity
            elif distance > 2.0:
                speed = self.max_velocity * 0.7
            else:
                speed = self.max_velocity * 0.3
            
            # Look ahead for smoother motion
            if self.path_index + 1 < len(self.current_path.waypoints):
                next_waypoint = self.current_path.waypoints[self.path_index + 1]
                next_pos = np.array(next_waypoint.position)
                look_ahead_direction = next_pos - target_pos
                
                if np.linalg.norm(look_ahead_direction) > 0:
                    look_ahead_direction = look_ahead_direction / np.linalg.norm(look_ahead_direction)
                    # Blend current direction with look-ahead
                    direction = 0.7 * direction + 0.3 * look_ahead_direction
                    direction = direction / np.linalg.norm(direction)
            
            velocity = direction * speed
            
            # Limit velocity
            velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
            
            return tuple(velocity)
            
        except Exception as e:
            self.logger.error(f"Error calculating velocity command: {e}")
            return (0.0, 0.0, 0.0)

    def _plan_a_star(self, start: Tuple[float, float, float], 
                     goal: Tuple[float, float, float], 
                     obstacles: List[Dict]) -> Optional[Path]:
        """A* path planning algorithm"""
        try:
            # Convert to grid coordinates
            start_grid = self._world_to_grid(start)
            goal_grid = self._world_to_grid(goal)
            
            # A* implementation
            open_set = []
            heapq.heappush(open_set, (0, start_grid))
            
            came_from = {}
            g_score = {start_grid: 0}
            f_score = {start_grid: self._heuristic(start_grid, goal_grid)}
            
            while open_set:
                current = heapq.heappop(open_set)[1]
                
                if current == goal_grid:
                    # Reconstruct path
                    path_points = []
                    while current in came_from:
                        path_points.append(self._grid_to_world(current))
                        current = came_from[current]
                    path_points.append(start)
                    path_points.reverse()
                    
                    # Convert to waypoints
                    waypoints = []
                    for i, pos in enumerate(path_points):
                        waypoint = Waypoint(
                            position=pos,
                            timestamp=i * 0.1,
                            velocity=(0.0, 0.0, 0.0),
                            acceleration=(0.0, 0.0, 0.0)
                        )
                        waypoints.append(waypoint)
                    
                    return Path(
                        waypoints=waypoints,
                        total_distance=self._calculate_path_distance(path_points),
                        total_time=len(path_points) * 0.1,
                        safety_margin=self.safety_margin,
                        cost=g_score[goal_grid]
                    )
                
                for neighbor in self._get_neighbors(current):
                    if not self._is_valid_position(neighbor):
                        continue
                    
                    tentative_g = g_score[current] + self._distance(current, neighbor)
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_grid)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
            
            return None  # No path found
            
        except Exception as e:
            self.logger.error(f"Error in A* planning: {e}")
            return None

    def _plan_rrt(self, start: Tuple[float, float, float], 
                  goal: Tuple[float, float, float], 
                  obstacles: List[Dict]) -> Optional[Path]:
        """RRT (Rapidly-exploring Random Tree) path planning"""
        try:
            # RRT implementation
            max_iterations = 1000
            step_size = 1.0
            
            # Tree nodes
            nodes = [start]
            parent = {0: None}
            
            for i in range(max_iterations):
                # Sample random point
                if np.random.random() < 0.1:  # 10% bias toward goal
                    sample = goal
                else:
                    sample = self._random_sample()
                
                # Find nearest node
                nearest_idx = i
        
        return nearest_idx

    def _extend_toward(self, from_pos: Tuple[float, float, float], 
                      to_pos: Tuple[float, float, float], 
                      step_size: float) -> Tuple[float, float, float]:
        """Extend from one position toward another"""
        direction = np.array(to_pos) - np.array(from_pos)
        distance = np.linalg.norm(direction)
        
        if distance <= step_size:
            return to_pos
        
        direction = direction / distance
        new_pos = np.array(from_pos) + direction * step_size
        return tuple(new_pos)

    def _is_path_valid(self, start: Tuple[float, float, float], 
                      end: Tuple[float, float, float], 
                      obstacles: List[Dict]) -> bool:
        """Check if path segment is collision-free"""
        try:
            start_pos = np.array(start)
            end_pos = np.array(end)
            
            # Sample points along the path
            num_samples = int(np.linalg.norm(end_pos - start_pos) / 0.5) + 1
            
            for i in range(num_samples + 1):
                t = i / max(1, num_samples)
                sample_pos = start_pos + t * (end_pos - start_pos)
                
                # Check against obstacles
                for obstacle in obstacles:
                    obs_pos = np.array(obstacle['position'])
                    distance = np.linalg.norm(sample_pos - obs_pos)
                    
                    if distance < self.safety_margin:
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking path validity: {e}")
            return False

    def _calculate_path_distance(self, path_points: List[Tuple[float, float, float]]) -> float:
        """Calculate total distance of path"""
        try:
            total_distance = 0.0
            
            for i in range(1, len(path_points)):
                pos1 = np.array(path_points[i-1])
                pos2 = np.array(path_points[i])
                total_distance += np.linalg.norm(pos2 - pos1)
            
            return total_distance
            
        except Exception as e:
            self.logger.error(f"Error calculating path distance: {e}")
            return 0.0

    def _update_performance_metrics(self, planning_time: float, path: Optional[Path]):
        """Update performance tracking"""
        try:
            self.planning_times.append(planning_time)
            
            if path:
                quality = 1.0 / (1.0 + path.cost)  # Simple quality metric
                self.path_qualities.append(quality)
            
            # Keep limited history
            max_samples = 100
            if len(self.planning_times) > max_samples:
                self.planning_times = self.planning_times[-max_samples:]
            if len(self.path_qualities) > max_samples:
                self.path_qualities = self.path_qualities[-max_samples:]
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    def get_performance_stats(self) -> Dict:
        """Get planning performance statistics"""
        try:
            if not self.planning_times:
                return {}
            
            return {
                'avg_planning_time': float(np.mean(self.planning_times)),
                'max_planning_time': float(np.max(self.planning_times)),
                'planning_frequency': 1.0 / float(np.mean(self.planning_times)) if np.mean(self.planning_times) > 0 else 0.0,
                'avg_path_quality': float(np.mean(self.path_qualities)) if self.path_qualities else 0.0,
                'total_plans': len(self.planning_times)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance stats: {e}")
            return {}

    def get_current_path_info(self) -> Optional[Dict]:
        """Get information about current path"""
        try:
            if not self.current_path:
                return None
            
            return {
                'total_waypoints': len(self.current_path.waypoints),
                'current_waypoint': self.path_index,
                'remaining_waypoints': len(self.current_path.waypoints) - self.path_index,
                'total_distance': self.current_path.total_distance,
                'estimated_time': self.current_path.total_time,
                'safety_margin': self.current_path.safety_margin
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current path info: {e}")
            return None

    def set_planner_type(self, planner_type: PlannerType):
        """Change the planning algorithm"""
        self.planner_type = planner_type
        self.logger.info(f"Planner type changed to: {planner_type.value}")

    def set_safety_margin(self, margin: float):
        """Update safety margin"""
        self.safety_margin = max(0.1, margin)
        self.logger.info(f"Safety margin set to: {self.safety_margin}m")

    def clear_path(self):
        """Clear current path"""
        self.current_path = None
        self.path_index = 0
        self.logger.info("Current path cleared")

    def get_status(self) -> Dict:
        """Get path planner status"""
        return {
            'planner_type': self.planner_type.value,
            'has_current_path': self.current_path is not None,
            'path_index': self.path_index,
            'safety_margin': self.safety_margin,
            'max_velocity': self.max_velocity,
            'performance': self.get_performance_stats(),
            'current_path': self.get_current_path_info()
        } = self._find_nearest_node(nodes, sample)
                nearest = nodes[nearest_idx]
                
                # Extend toward sample
                new_node = self._extend_toward(nearest, sample, step_size)
                
                # Check if valid
                if self._is_path_valid(nearest, new_node, obstacles):
                    nodes.append(new_node)
                    parent[len(nodes) - 1] = nearest_idx
                    
                    # Check if we reached goal
                    if np.linalg.norm(np.array(new_node) - np.array(goal)) < step_size:
                        # Reconstruct path
                        path_points = []
                        current_idx = len(nodes) - 1
                        
                        while current_idx is not None:
                            path_points.append(nodes[current_idx])
                            current_idx = parent[current_idx]
                        
                        path_points.reverse()
                        
                        # Convert to waypoints
                        waypoints = []
                        for i, pos in enumerate(path_points):
                            waypoint = Waypoint(
                                position=pos,
                                timestamp=i * 0.2,
                                velocity=(0.0, 0.0, 0.0),
                                acceleration=(0.0, 0.0, 0.0)
                            )
                            waypoints.append(waypoint)
                        
                        return Path(
                            waypoints=waypoints,
                            total_distance=self._calculate_path_distance(path_points),
                            total_time=len(path_points) * 0.2,
                            safety_margin=self.safety_margin,
                            cost=self._calculate_path_distance(path_points)
                        )
            
            return None  # No path found
            
        except Exception as e:
            self.logger.error(f"Error in RRT planning: {e}")
            return None

    def _plan_potential_field(self, start: Tuple[float, float, float], 
                             goal: Tuple[float, float, float], 
                             obstacles: List[Dict]) -> Optional[Path]:
        """Potential field path planning"""
        try:
            path_points = [start]
            current = np.array(start)
            max_iterations = 100
            step_size = 0.5
            
            for _ in range(max_iterations):
                # Calculate attractive force toward goal
                goal_vec = np.array(goal) - current
                goal_distance = np.linalg.norm(goal_vec)
                
                if goal_distance < 0.5:  # Reached goal
                    path_points.append(goal)
                    break
                
                attractive_force = goal_vec / goal_distance * min(1.0, goal_distance)
                
                # Calculate repulsive forces from obstacles
                repulsive_force = np.array([0.0, 0.0, 0.0])
                
                for obstacle in obstacles:
                    obs_pos = np.array(obstacle['position'])
                    obs_vec = current - obs_pos
                    obs_distance = np.linalg.norm(obs_vec)
                    
                    if obs_distance < 5.0:  # Within influence range
                        if obs_distance > 0.1:
                            repulsive_force += obs_vec / obs_distance**2 * 2.0
                        else:
                            repulsive_force += np.array([1.0, 0.0, 0.0])  # Avoid singularity
                
                # Combine forces
                total_force = attractive_force + repulsive_force
                force_magnitude = np.linalg.norm(total_force)
                
                if force_magnitude > 0:
                    direction = total_force / force_magnitude
                    next_pos = current + direction * step_size
                    
                    # Check if movement is valid
                    if self._is_path_valid(tuple(current), tuple(next_pos), obstacles):
                        current = next_pos
                        path_points.append(tuple(current))
                    else:
                        # Try alternative direction
                        alt_direction = np.array([direction[1], -direction[0], direction[2]])
                        alt_pos = current + alt_direction * step_size
                        if self._is_path_valid(tuple(current), tuple(alt_pos), obstacles):
                            current = alt_pos
                            path_points.append(tuple(current))
            
            if len(path_points) > 1:
                # Convert to waypoints
                waypoints = []
                for i, pos in enumerate(path_points):
                    waypoint = Waypoint(
                        position=pos,
                        timestamp=i * 0.1,
                        velocity=(0.0, 0.0, 0.0),
                        acceleration=(0.0, 0.0, 0.0)
                    )
                    waypoints.append(waypoint)
                
                return Path(
                    waypoints=waypoints,
                    total_distance=self._calculate_path_distance(path_points),
                    total_time=len(path_points) * 0.1,
                    safety_margin=self.safety_margin,
                    cost=self._calculate_path_distance(path_points)
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in potential field planning: {e}")
            return None

    def _plan_dynamic_window(self, start: Tuple[float, float, float], 
                           goal: Tuple[float, float, float], 
                           obstacles: List[Dict]) -> Tuple[float, float, float]:
        """Dynamic Window Approach for local planning"""
        try:
            # Sample velocity space
            best_velocity = (0.0, 0.0, 0.0)
            best_score = -float('inf')
            
            velocity_samples = 20
            
            for vx in np.linspace(-self.max_velocity, self.max_velocity, velocity_samples):
                for vy in np.linspace(-self.max_velocity, self.max_velocity, velocity_samples):
                    for vz in np.linspace(-self.max_velocity/2, self.max_velocity/2, velocity_samples//2):
                        velocity = (vx, vy, vz)
                        
                        # Evaluate this velocity
                        score = self._evaluate_velocity(start, goal, velocity, obstacles)
                        
                        if score > best_score:
                            best_score = score
                            best_velocity = velocity
            
            return best_velocity
            
        except Exception as e:
            self.logger.error(f"Error in dynamic window planning: {e}")
            return (0.0, 0.0, 0.0)

    def _evaluate_velocity(self, position: Tuple[float, float, float], 
                          goal: Tuple[float, float, float], 
                          velocity: Tuple[float, float, float], 
                          obstacles: List[Dict]) -> float:
        """Evaluate a velocity command"""
        try:
            # Simulate forward for prediction horizon
            sim_pos = np.array(position)
            sim_vel = np.array(velocity)
            dt = 0.1
            horizon = 2.0  # 2 seconds
            
            score = 0.0
            
            for t in np.arange(0, horizon, dt):
                sim_pos += sim_vel * dt
                
                # Goal attraction
                goal_distance = np.linalg.norm(sim_pos - np.array(goal))
                score += 1.0 / (1.0 + goal_distance)  # Higher score for closer to goal
                
                # Obstacle avoidance
                min_obs_distance = float('inf')
                for obstacle in obstacles:
                    obs_pos = np.array(obstacle['position'])
                    obs_distance = np.linalg.norm(sim_pos - obs_pos)
                    min_obs_distance = min(min_obs_distance, obs_distance)
                
                if min_obs_distance < self.safety_margin:
                    score -= 10.0  # Heavy penalty for collision
                else:
                    score += min(1.0, min_obs_distance / 5.0)  # Reward for maintaining distance
                
                # Velocity preference (penalize extreme velocities)
                vel_magnitude = np.linalg.norm(sim_vel)
                score -= 0.1 * (vel_magnitude / self.max_velocity)**2
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error evaluating velocity: {e}")
            return -float('inf')

    def _smooth_path(self, path: Path) -> Path:
        """Smooth the planned path"""
        try:
            if len(path.waypoints) < 3:
                return path
            
            smoothed_waypoints = [path.waypoints[0]]  # Keep start
            
            # Simple moving average smoothing
            for i in range(1, len(path.waypoints) - 1):
                prev_pos = np.array(path.waypoints[i-1].position)
                curr_pos = np.array(path.waypoints[i].position)
                next_pos = np.array(path.waypoints[i+1].position)
                
                # Weighted average
                smoothed_pos = 0.25 * prev_pos + 0.5 * curr_pos + 0.25 * next_pos
                
                smoothed_waypoint = Waypoint(
                    position=tuple(smoothed_pos),
                    timestamp=path.waypoints[i].timestamp,
                    velocity=path.waypoints[i].velocity,
                    acceleration=path.waypoints[i].acceleration
                )
                
                smoothed_waypoints.append(smoothed_waypoint)
            
            smoothed_waypoints.append(path.waypoints[-1])  # Keep end
            
            return Path(
                waypoints=smoothed_waypoints,
                total_distance=path.total_distance,
                total_time=path.total_time,
                safety_margin=path.safety_margin,
                cost=path.cost
            )
            
        except Exception as e:
            self.logger.error(f"Error smoothing path: {e}")
            return path

    # Helper methods
    def _update_occupancy_grid(self, obstacles: List[Dict], depth_map: Optional[np.ndarray]):
        """Update 3D occupancy grid with obstacles"""
        try:
            # Clear previous obstacles (keep some persistence)
            self.occupancy_grid *= 0.9
            
            for obstacle in obstacles:
                pos = obstacle['position']
                size = obstacle.get('size', (1.0, 1.0, 1.0))
                
                # Convert to grid coordinates
                grid_pos = self._world_to_grid(pos)
                grid_size = tuple(max(1, int(s / self.grid_resolution)) for s in size)
                
                # Mark occupied cells
                for dx in range(-grid_size[0]//2, grid_size[0]//2 + 1):
                    for dy in range(-grid_size[1]//2, grid_size[1]//2 + 1):
                        for dz in range(-grid_size[2]//2, grid_size[2]//2 + 1):
                            gx, gy, gz = grid_pos[0] + dx, grid_pos[1] + dy, grid_pos[2] + dz
                            
                            if self._is_valid_grid_pos((gx, gy, gz)):
                                self.occupancy_grid[gx, gy, gz] = 1.0
                                
        except Exception as e:
            self.logger.error(f"Error updating occupancy grid: {e}")

    def _world_to_grid(self, world_pos: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert world coordinates to grid coordinates"""
        return tuple(int((coord + self.grid_size * self.grid_resolution / 2) / self.grid_resolution) 
                    for coord in world_pos)

    def _grid_to_world(self, grid_pos: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert grid coordinates to world coordinates"""
        return tuple((coord * self.grid_resolution - self.grid_size * self.grid_resolution / 2) 
                    for coord in grid_pos)

    def _is_valid_grid_pos(self, grid_pos: Tuple[int, int, int]) -> bool:
        """Check if grid position is valid"""
        return all(0 <= coord < self.grid_size for coord in grid_pos)

    def _is_valid_position(self, grid_pos: Tuple[int, int, int]) -> bool:
        """Check if position is valid (not occupied)"""
        if not self._is_valid_grid_pos(grid_pos):
            return False
        return self.occupancy_grid[grid_pos] < 0.5

    def _get_neighbors(self, pos: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get valid neighboring grid positions"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                    if self._is_valid_grid_pos(neighbor):
                        neighbors.append(neighbor)
        return neighbors

    def _heuristic(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
        """Heuristic function for A*"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def _distance(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
        """Distance between two positions"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def _random_sample(self) -> Tuple[float, float, float]:
        """Generate random sample in planning space"""
        return (
            np.random.uniform(-20, 20),  # x range
            np.random.uniform(-20, 20),  # y range
            np.random.uniform(0, 10)     # z range (altitude)
        )

    def _find_nearest_node(self, nodes: List[Tuple[float, float, float]], 
                          sample: Tuple[float, float, float]) -> int:
        """Find nearest node to sample"""
        min_distance = float('inf')
        nearest_idx = 0
        
        for i, node in enumerate(nodes):
            distance = np.linalg.norm(np.array(node) - np.array(sample))
            if distance < min_distance:
                min_distance = distance
                nearest_idx
