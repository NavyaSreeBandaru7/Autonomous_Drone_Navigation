#!/usr/bin/env python3
"""
Configuration Module
Centralized configuration management for the autonomous drone system
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import os

@dataclass
class DroneConfig:
    """Main configuration class for the drone system"""
    
    # Connection settings
    connection_string: str = "udp:127.0.0.1:14550"
    simulation_mode: bool = True
    
    # Flight parameters
    max_velocity: float = 5.0  # m/s
    max_acceleration: float = 2.0  # m/s^2
    max_altitude: float = 50.0  # meters
    min_altitude: float = 0.5  # meters
    safety_margin: float = 2.0  # meters
    target_tolerance: float = 0.5  # meters
    
    # Battery thresholds
    low_battery_threshold: float = 30.0  # percent
    critical_battery_threshold: float = 15.0  # percent
    
    # Home position (lat, lon, alt)
    home_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Camera settings
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    camera_fps: int = 30
    camera_calibration_file: str = ""
    
    # Vision system
    use_stereo_vision: bool = False
    yolo_model_path: str = "yolov8n.pt"
    detection_confidence_threshold: float = 0.5
    
    # Obstacle detection
    min_obstacle_size: float = 0.1  # meters
    max_detection_distance: float = 30.0  # meters
    obstacle_confidence_threshold: float = 0.6
    min_obstacle_distance: float = 1.0  # meters
    use_depth_analysis: bool = True
    use_motion_detection: bool = True
    use_ml_detection: bool = True
    min_contour_area: float = 100.0
    max_contour_area: float = 10000.0
    min_dynamic_area: float = 50.0
    max_tracking_age: float = 5.0  # seconds
    max_tracking_distance: float = 2.0  # meters
    duplicate_threshold: float = 1.0  # meters
    
    # Path planning
    planner_type: str = "potential_field"
    grid_resolution: float = 0.5  # meters
    grid_size: int = 100
    planning_horizon: float = 10.0  # seconds
    replan_threshold: float = 1.0  # meters
    min_replan_interval: float = 0.5  # seconds
    emergency_stop_distance: float = 0.5  # meters
    collision_avoidance_distance: float = 2.0  # meters
    landing_speed: float = 0.5  # m/s
    
    # NLP settings
    use_wake_word: bool = False
    wake_word: str = "drone"
    language: str = "en-US"
    speech_timeout: float = 5.0
    
    # Logging
    log_level: str = "INFO"
    log_directory: str = "logs"
    log_to_file: bool = True
    log_to_console: bool = True
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration from file or defaults"""
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        else:
            # Use defaults
            pass
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update attributes from loaded data
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            logging.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            logging.info("Using default configuration")
    
    def save_to_file(self, config_file: str):
        """Save configuration to JSON file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(asdict(self), f, indent=4)
            
            logging.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        try:
            # Check numeric ranges
            assert 0 < self.max_velocity <= 20, "Max velocity must be between 0 and 20 m/s"
            assert 0 < self.max_acceleration <= 10, "Max acceleration must be between 0 and 10 m/s^2"
            assert 0 < self.max_altitude <= 120, "Max altitude must be between 0 and 120 meters"
            assert 0 <= self.min_altitude < self.max_altitude, "Min altitude must be less than max altitude"
            assert 0 < self.safety_margin <= 5, "Safety margin must be between 0 and 5 meters"
            
            # Check battery thresholds
            assert 0 < self.critical_battery_threshold < self.low_battery_threshold <= 100, \
                "Battery thresholds must be: 0 < critical < low <= 100"
            
            # Check camera settings
            assert self.camera_index >= 0, "Camera index must be non-negative"
            assert self.frame_width > 0 and self.frame_height > 0, "Frame dimensions must be positive"
            assert 1 <= self.camera_fps <= 60, "Camera FPS must be between 1 and 60"
            
            # Check detection parameters
            assert 0 < self.detection_confidence_threshold <= 1, "Detection confidence must be between 0 and 1"
            assert 0 < self.obstacle_confidence_threshold <= 1, "Obstacle confidence must be between 0 and 1"
            
            # Check planning parameters
            assert self.planner_type in ["a_star", "rrt", "potential_field", "dynamic_window"], \
                "Invalid planner type"
            assert 0 < self.grid_resolution <= 2, "Grid resolution must be between 0 and 2 meters"
            assert 10 <= self.grid_size <= 1000, "Grid size must be between 10 and 1000"
            
            logging.info("Configuration validation passed")
            return True
            
        except AssertionError as e:
            logging.error(f"Configuration validation failed: {e}")
            return False
        except Exception as e:
            logging.error(f"Error validating configuration: {e}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "connection": {
                "string": self.connection_string,
                "simulation": self.simulation_mode
            },
            "flight_limits": {
                "max_velocity": self.max_velocity,
                "max_altitude": self.max_altitude,
                "safety_margin": self.safety_margin
            },
            "camera": {
                "resolution": f"{self.frame_width}x{self.frame_height}",
                "fps": self.camera_fps
            },
            "planning": {
                "algorithm": self.planner_type,
                "grid_resolution": self.grid_resolution
            }
        }
