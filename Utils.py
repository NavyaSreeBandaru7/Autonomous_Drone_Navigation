#!/usr/bin/env python3
"""
Utility Functions
Common utility functions for the autonomous drone system
"""

import logging
import os
import json
import time
import numpy as np
import csv
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import threading

def setup_logging(log_level: str = "INFO", 
                 log_directory: str = "logs", 
                 log_to_file: bool = True, 
                 log_to_console: bool = True) -> logging.Logger:
    """Setup logging configuration"""
    try:
        # Create logs directory
        if log_to_file:
            os.makedirs(log_directory, exist_ok=True)
        
        # Configure logging level
        level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_directory, f"drone_system_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        logger.info("Logging system initialized")
        return logger
        
    except Exception as e:
        print(f"Error setting up logging: {e}")
        return logging.getLogger()

def save_flight_data(flight_data: List[Dict], log_directory: str = "logs"):
    """Save flight data to CSV file"""
    try:
        os.makedirs(log_directory, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(log_directory, f"flight_data_{timestamp}.csv")
        
        if not flight_data:
            return
        
        # Get all unique keys from flight data
        all_keys = set()
        for entry in flight_data:
            all_keys.update(entry.keys())
        
        # Write CSV file
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted(all_keys))
            writer.writeheader()
            
            for entry in flight_data:
                # Flatten nested data
                flattened_entry = flatten_dict(entry)
                writer.writerow(flattened_entry)
        
        logging.info(f"Flight data saved to {filename}")
        
    except Exception as e:
        logging.error(f"Error saving flight data: {e}")

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            # Convert sequences to string representation
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    
    return dict(items)

def calculate_distance_3d(pos1: Tuple[float, float, float], 
                         pos2: Tuple[float, float, float]) -> float:
    """Calculate 3D Euclidean distance between two positions"""
    try:
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    except Exception as e:
        logging.error(f"Error calculating 3D distance: {e}")
        return float('inf')

def calculate_bearing(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate bearing from pos1 to pos2 in degrees"""
    try:
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        bearing = np.arctan2(dy, dx) * 180 / np.pi
        return (bearing + 360) % 360  # Normalize to 0-360 degrees
        
    except Exception as e:
        logging.error(f"Error calculating bearing: {e}")
        return 0.0

def limit_velocity(velocity: Tuple[float, float, float], 
                  max_velocity: float) -> Tuple[float, float, float]:
    """Limit velocity to maximum allowed value"""
    try:
        vel_array = np.array(velocity)
        magnitude = np.linalg.norm(vel_array)
        
        if magnitude > max_velocity:
            vel_array = vel_array * (max_velocity / magnitude)
        
        return tuple(vel_array)
        
    except Exception as e:
        logging.error(f"Error limiting velocity: {e}")
        return velocity

def smooth_trajectory(positions: List[Tuple[float, float, float]], 
                     smoothing_factor: float = 0.1) -> List[Tuple[float, float, float]]:
    """Apply smoothing to trajectory"""
    try:
        if len(positions) < 3:
            return positions
        
        smoothed = [positions[0]]  # Keep first position
        
        for i in range(1, len(positions) - 1):
            prev_pos = np.array(positions[i-1])
            curr_pos = np.array(positions[i])
            next_pos = np.array(positions[i+1])
            
            # Apply smoothing
            smoothed_pos = (1 - smoothing_factor) * curr_pos + \
                          smoothing_factor * 0.5 * (prev_pos + next_pos)
            
            smoothed.append(tuple(smoothed_pos))
        
        smoothed.append(positions[-1])  # Keep last position
        
        return smoothed
        
    except Exception as e:
        logging.error(f"Error smoothing trajectory: {e}")
        return positions

def interpolate_waypoints(start: Tuple[float, float, float], 
                         end: Tuple[float, float, float], 
                         num_points: int) -> List[Tuple[float, float, float]]:
    """Interpolate waypoints between start and end positions"""
    try:
        if num_points < 2:
            return [start, end]
        
        start_array = np.array(start)
        end_array = np.array(end)
        
        waypoints = []
        for i in range(num_points):
            t = i / (num_points - 1)
            waypoint = start_array + t * (end_array - start_array)
            waypoints.append(tuple(waypoint))
        
        return waypoints
        
    except Exception as e:
        logging.error(f"Error interpolating waypoints: {e}")
        return [start, end]

def check_collision_line_sphere(line_start: Tuple[float, float, float],
                                line_end: Tuple[float, float, float],
                                sphere_center: Tuple[float, float, float],
                                sphere_radius: float) -> bool:
    """Check if line segment collides with sphere"""
    try:
        # Convert to numpy arrays
        p1 = np.array(line_start)
        p2 = np.array(line_end)
        c = np.array(sphere_center)
        
        # Line direction and length
        line_vec = p2 - p1
        line_length = np.linalg.norm(line_vec)
        
        if line_length == 0:
            # Point to sphere distance
            return np.linalg.norm(p1 - c) <= sphere_radius
        
        line_unit = line_vec / line_length
        
        # Project sphere center onto line
        to_center = c - p1
        projection_length = np.dot(to_center, line_unit)
        
        # Clamp projection to line segment
        projection_length = max(0, min(line_length, projection_length))
        
        # Find closest point on line segment
        closest_point = p1 + projection_length * line_unit
        
        # Check distance to sphere
        distance = np.linalg.norm(closest_point - c)
        return distance <= sphere_radius
        
    except Exception as e:
        logging.error(f"Error checking line-sphere collision: {e}")
        return True  # Conservative: assume collision

def rotate_point_2d(point: Tuple[float, float], 
                   angle: float, 
                   center: Tuple[float, float] = (0, 0)) -> Tuple[float, float]:
    """Rotate point around center by angle (radians)"""
    try:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Translate to origin
        x = point[0] - center[0]
        y = point[1] - center[1]
        
        # Rotate
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        
        # Translate back
        return (x_rot + center[0], y_rot + center[1])
        
    except Exception as e:
        logging.error(f"Error rotating point: {e}")
        return point

class MovingAverage:
    """Moving average calculator"""
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.values = []
        self.lock = threading.Lock()
    
    def update(self, value: float) -> float:
        """Add new value and return current average"""
        with self.lock:
            self.values.append(value)
            
            if len(self.values) > self.window_size:
                self.values.pop(0)
            
            return sum(self.values) / len(self.values)
    
    def get_average(self) -> float:
        """Get current average"""
        with self.lock:
            if not self.values:
                return 0.0
            return sum(self.values) / len(self.values)
    
    def reset(self):
        """Reset the moving average"""
        with self.lock:
            self.values.clear()

class RateLimiter:
    """Rate limiter for function calls"""
    
    def __init__(self, max_calls_per_second: float):
        self.max_calls_per_second = max_calls_per_second
        self.min_interval = 1.0 / max_calls_per_second if max_calls_per_second > 0 else 0
        self.last_call_time = 0
        self.lock = threading.Lock()
    
    def should_allow(self) -> bool:
        """Check if call should be allowed"""
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_call_time
            
            if elapsed >= self.min_interval:
                self.last_call_time = current_time
                return True
            
            return False
    
    def wait_if_needed(self):
        """Wait if needed to maintain rate limit"""
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_call_time
            
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            
            self.last_call_time = time.time()

class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self, name: str, max_samples: int = 100):
        self.name = name
        self.max_samples = max_samples
        self.execution_times = []
        self.call_count = 0
        self.lock = threading.Lock()
    
    def start_timing(self):
        """Start timing a function call"""
        return time.time()
    
    def end_timing(self, start_time: float):
        """End timing and record execution time"""
        with self.lock:
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self.call_count += 1
            
            # Keep only recent samples
            if len(self.execution_times) > self.max_samples:
                self.execution_times.pop(0)
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        with self.lock:
            if not self.execution_times:
                return {
                    'avg_time': 0.0,
                    'min_time': 0.0,
                    'max_time': 0.0,
                    'total_calls': self.call_count,
                    'calls_per_second': 0.0
                }
            
            avg_time = np.mean(self.execution_times)
            
            return {
                'avg_time': float(avg_time),
                'min_time': float(np.min(self.execution_times)),
                'max_time': float(np.max(self.execution_times)),
                'total_calls': self.call_count,
                'calls_per_second': 1.0 / avg_time if avg_time > 0 else 0.0
            }

def convert_coordinates(lat: float, lon: float, reference_lat: float, reference_lon: float) -> Tuple[float, float]:
    """Convert GPS coordinates to local NED coordinates"""
    try:
        # Simple conversion (for small distances)
        # For accurate conversion, use proper geodetic transformations
        
        earth_radius = 6371000  # meters
        
        dlat = lat - reference_lat
        dlon = lon - reference_lon
        
        # Convert to meters (approximate)
        x = dlon * np.cos(np.radians(reference_lat)) * earth_radius * np.pi / 180
        y = dlat * earth_radius * np.pi / 180
        
        return (x, y)
        
    except Exception as e:
        logging.error(f"Error converting coordinates: {e}")
        return (0.0, 0.0)

def validate_position(position: Tuple[float, float, float], 
                     min_altitude: float = 0.0, 
                     max_altitude: float = 50.0,
                     boundary_radius: float = 100.0) -> bool:
    """Validate if position is within safe boundaries"""
    try:
        x, y, z = position
        
        # Check altitude bounds
        if z < min_altitude or z > max_altitude:
            return False
        
        # Check horizontal boundary (circular)
        horizontal_distance = np.sqrt(x**2 + y**2)
        if horizontal_distance > boundary_radius:
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating position: {e}")
        return False

def create_safety_report(flight_data: List[Dict]) -> Dict[str, Any]:
    """Create safety analysis report from flight data"""
    try:
        if not flight_data:
            return {"error": "No flight data available"}
        
        report = {
            "flight_duration": 0.0,
            "max_altitude": 0.0,
            "min_altitude": float('inf'),
            "max_velocity": 0.0,
            "avg_velocity": 0.0,
            "battery_usage": 0.0,
            "safety_violations": [],
            "obstacle_encounters": 0,
            "emergency_stops": 0
        }
        
        velocities = []
        altitudes = []
        
        start_time = flight_data[0].get('timestamp', 0)
        end_time = flight_data[-1].get('timestamp', 0)
        report["flight_duration"] = end_time - start_time
        
        start_battery = flight_data[0].get('battery_level', 100)
        end_battery = flight_data[-1].get('battery_level', 100)
        report["battery_usage"] = start_battery - end_battery
        
        for entry in flight_data:
            # Altitude analysis
            altitude = entry.get('position', [0, 0, 0])[2]
            altitudes.append(altitude)
            report["max_altitude"] = max(report["max_altitude"], altitude)
            report["min_altitude"] = min(report["min_altitude"], altitude)
            
            # Velocity analysis
            velocity = entry.get('velocity', [0, 0, 0])
            speed = np.linalg.norm(velocity)
            velocities.append(speed)
            report["max_velocity"] = max(report["max_velocity"], speed)
            
            # Safety checks
            if altitude > 50.0:
                report["safety_violations"].append(f"Altitude exceeded at {entry.get('timestamp', 0)}")
            
            if entry.get('obstacles_count', 0) > 0:
                report["obstacle_encounters"] += 1
            
            if entry.get('flight_mode') == 'emergency':
                report["emergency_stops"] += 1
        
        if velocities:
            report["avg_velocity"] = float(np.mean(velocities))
        
        if report["min_altitude"] == float('inf'):
            report["min_altitude"] = 0.0
        
        return report
        
    except Exception as e:
        logging.error(f"Error creating safety report: {e}")
        return {"error": str(e)}

def format_telemetry_display(telemetry: Dict) -> str:
    """Format telemetry data for display"""
    try:
        if not telemetry:
            return "No telemetry data available"
        
        pos = telemetry.get('position', (0, 0, 0))
        vel = telemetry.get('velocity', (0, 0, 0))
        battery = telemetry.get('battery', 0)
        armed = telemetry.get('armed', False)
        
        display = f"""
=== DRONE TELEMETRY ===
Position: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f} m
Velocity: X={vel[0]:.2f}, Y={vel[1]:.2f}, Z={vel[2]:.2f} m/s
Battery: {battery:.1f}%
Armed: {'YES' if armed else 'NO'}
Timestamp: {datetime.now().strftime('%H:%M:%S')}
========================
        """
        
        return display.strip()
        
    except Exception as e:
        logging.error(f"Error formatting telemetry: {e}")
        return "Error formatting telemetry data"

def load_mission_file(filename: str) -> Optional[List[Dict]]:
    """Load mission waypoints from JSON file"""
    try:
        with open(filename, 'r') as f:
            mission_data = json.load(f)
        
        # Validate mission data
        if not isinstance(mission_data, list):
            logging.error("Mission file must contain a list of waypoints")
            return None
        
        for i, waypoint in enumerate(mission_data):
            if not isinstance(waypoint, dict):
                logging.error(f"Waypoint {i} must be a dictionary")
                return None
            
            if 'position' not in waypoint:
                logging.error(f"Waypoint {i} missing 'position' field")
                return None
        
        logging.info(f"Loaded mission with {len(mission_data)} waypoints")
        return mission_data
        
    except Exception as e:
        logging.error(f"Error loading mission file: {e}")
        return None

def save_mission_file(waypoints: List[Dict], filename: str) -> bool:
    """Save mission waypoints to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(waypoints, f, indent=4)
        
        logging.info(f"Mission saved to {filename}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving mission file: {e}")
        return False
