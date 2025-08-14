#!/usr/bin/env python3
"""
Obstacle Detector Module
Advanced obstacle detection and classification for drone navigation
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time
import math

class ObstacleType(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    UNKNOWN = "unknown"
    BUILDING = "building"
    TREE = "tree"
    VEHICLE = "vehicle"
    PERSON = "person"
    BIRD = "bird"
    POWER_LINE = "power_line"

@dataclass
class Obstacle:
    id: str
    type: ObstacleType
    position: Tuple[float, float, float]  # x, y, z in meters
    size: Tuple[float, float, float]  # width, height, depth
    velocity: Tuple[float, float, float]  # vx, vy, vz
    confidence: float
    bbox: Tuple[int, int, int, int]  # pixel coordinates
    distance: float
    threat_level: float  # 0.0 to 1.0
    timestamp: float
    last_seen: float
    tracking_points: List[Tuple[float, float]]

class ObstacleDetector:
    """Advanced obstacle detection system"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Detection parameters
        self.min_obstacle_size = config.min_obstacle_size
        self.max_detection_distance = config.max_detection_distance
        self.confidence_threshold = config.obstacle_confidence_threshold
        
        # Tracking
        self.tracked_obstacles = {}
        self.next_obstacle_id = 0
        self.max_tracking_age = config.max_tracking_age
        
        # Detection methods
        self.use_depth_analysis = config.use_depth_analysis
        self.use_motion_detection = config.use_motion_detection
        self.use_ml_detection = config.use_ml_detection
        
        # Initialize detectors
        self._init_detectors()
        
        # Performance tracking
        self.detection_times = []
        self.max_time_samples = 100
        
        self.logger.info("Obstacle detector initialized")

    def _init_detectors(self):
        """Initialize various detection methods"""
        try:
            # Initialize background subtractor for motion detection
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True, varThreshold=16, history=500
            )
            
            # Initialize optical flow tracker
            self.flow_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Initialize feature detector for corner detection
            self.corner_detector = cv2.goodFeaturesToTrack
            
            # Initialize line detector for power lines
            self.line_detector = cv2.HoughLinesP
            
            # Initialize contour detector parameters
            self.contour_params = {
                'min_area': self.config.min_contour_area,
                'max_area': self.config.max_contour_area,
                'min_aspect_ratio': 0.2,
                'max_aspect_ratio': 5.0
            }
            
            self.logger.info("Detection methods initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing detectors: {e}")

    def detect_obstacles(self, frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> List[Dict]:
        """Main obstacle detection function"""
        start_time = time.time()
        obstacles = []
        
        try:
            # Detect static obstacles
            static_obstacles = self._detect_static_obstacles(frame, depth_map)
            obstacles.extend(static_obstacles)
            
            # Detect dynamic obstacles
            if self.use_motion_detection:
                dynamic_obstacles = self._detect_dynamic_obstacles(frame, depth_map)
                obstacles.extend(dynamic_obstacles)
            
            # Detect specific object types
            if self.use_ml_detection:
                ml_obstacles = self._detect_ml_obstacles(frame, depth_map)
                obstacles.extend(ml_obstacles)
            
            # Detect power lines and wires
            power_lines = self._detect_power_lines(frame)
            obstacles.extend(power_lines)
            
            # Update tracking
            self._update_tracking(obstacles)
            
            # Filter and merge obstacles
            filtered_obstacles = self._filter_obstacles(obstacles)
            
            # Calculate threat levels
            final_obstacles = self._calculate_threat_levels(filtered_obstacles)
            
            # Performance tracking
            detection_time = time.time() - start_time
            self._update_performance_metrics(detection_time)
            
            return final_obstacles
            
        except Exception as e:
            self.logger.error(f"Error in obstacle detection: {e}")
            return []

    def _detect_static_obstacles(self, frame: np.ndarray, depth_map: Optional[np.ndarray]) -> List[Dict]:
        """Detect static obstacles using edge detection and contours"""
        obstacles = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            
            # Morphological operations to close gaps
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter by area
                area = cv2.contourArea(contour)
                if area < self.contour_params['min_area'] or area > self.contour_params['max_area']:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                if (aspect_ratio < self.contour_params['min_aspect_ratio'] or 
                    aspect_ratio > self.contour_params['max_aspect_ratio']):
                    continue
                
                # Estimate distance
                distance = self._estimate_obstacle_distance(frame, (x, y, w, h), depth_map)
                
                if distance is None or distance > self.max_detection_distance:
                    continue
                
                # Calculate 3D position
                center_x = x + w // 2
                center_y = y + h // 2
                position_3d = self._pixel_to_world(center_x, center_y, distance)
                
                # Classify obstacle type
                obstacle_type = self._classify_static_obstacle(frame[y:y+h, x:x+w], w, h)
                
                obstacle = {
                    'type': obstacle_type,
                    'position': position_3d,
                    'size': self._estimate_size(w, h, distance),
                    'distance': distance,
                    'bbox': (x, y, x + w, y + h),
                    'confidence': self._calculate_confidence(area, aspect_ratio),
                    'velocity': (0.0, 0.0, 0.0),  # Static obstacle
                    'timestamp': time.time()
                }
                
                obstacles.append(obstacle)
            
            return obstacles
            
        except Exception as e:
            self.logger.error(f"Error detecting static obstacles: {e}")
            return []

    def _detect_dynamic_obstacles(self, frame: np.ndarray, depth_map: Optional[np.ndarray]) -> List[Dict]:
        """Detect moving obstacles using background subtraction and optical flow"""
        obstacles = []
        
        try:
            # Background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours of moving objects
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < self.config.min_dynamic_area:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Estimate distance
                distance = self._estimate_obstacle_distance(frame, (x, y, w, h), depth_map)
                
                if distance is None or distance > self.max_detection_distance:
                    continue
                
                # Calculate velocity using optical flow
                velocity = self._estimate_velocity(frame, (x, y, w, h))
                
                # Calculate 3D position
                center_x = x + w // 2
                center_y = y + h // 2
                position_3d = self._pixel_to_world(center_x, center_y, distance)
                
                # Classify dynamic obstacle
                obstacle_type = self._classify_dynamic_obstacle(velocity, w, h)
                
                obstacle = {
                    'type': obstacle_type,
                    'position': position_3d,
                    'size': self._estimate_size(w, h, distance),
                    'distance': distance,
                    'bbox': (x, y, x + w, y + h),
                    'confidence': 0.7,  # Lower confidence for dynamic detection
                    'velocity': velocity,
                    'timestamp': time.time()
                }
                
                obstacles.append(obstacle)
            
            return obstacles
            
        except Exception as e:
            self.logger.error(f"Error detecting dynamic obstacles: {e}")
            return []

    def _detect_ml_obstacles(self, frame: np.ndarray, depth_map: Optional[np.ndarray]) -> List[Dict]:
        """Detect obstacles using machine learning models"""
        obstacles = []
        
        try:
            # This would integrate with YOLO or other ML models
            # For now, we'll use a simplified approach
            
            # Detect people using OpenCV's HOG detector
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Detect people
            people, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)
            
            for i, (x, y, w, h) in enumerate(people):
                if weights[i] < 0.5:  # Confidence threshold
                    continue
                
                distance = self._estimate_obstacle_distance(frame, (x, y, w, h), depth_map)
                
                if distance is None or distance > self.max_detection_distance:
                    continue
                
                center_x = x + w // 2
                center_y = y + h // 2
                position_3d = self._pixel_to_world(center_x, center_y, distance)
                
                obstacle = {
                    'type': ObstacleType.PERSON,
                    'position': position_3d,
                    'size': self._estimate_size(w, h, distance),
                    'distance': distance,
                    'bbox': (x, y, x + w, y + h),
                    'confidence': float(weights[i]),
                    'velocity': (0.0, 0.0, 0.0),  # Would need tracking for velocity
                    'timestamp': time.time()
                }
                
                obstacles.append(obstacle)
            
            return obstacles
            
        except Exception as e:
            self.logger.error(f"Error in ML obstacle detection: {e}")
            return []

    def _detect_power_lines(self, frame: np.ndarray) -> List[Dict]:
        """Detect power lines and cables"""
        obstacles = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Edge detection optimized for lines
            edges = cv2.Canny(gray, 30, 80, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLinesP(
                edges, 
                rho=1, 
                theta=np.pi/180, 
                threshold=50,
                minLineLength=100,
                maxLineGap=10
            )
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate line angle
                    angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
                    
                    # Filter for roughly horizontal lines (power lines)
                    if abs(angle) > 30 and abs(angle) < 150:
                        continue
                    
                    # Estimate distance (power lines are typically high)
                    line_y = (y1 + y2) / 2
                    distance = self._estimate_line_distance(line_y, frame.shape[0])
                    
                    # Calculate center position
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    position_3d = self._pixel_to_world(center_x, center_y, distance)
                    
                    # Calculate line length
                    line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    obstacle = {
                        'type': ObstacleType.POWER_LINE,
                        'position': position_3d,
                        'size': (line_length * distance / 500, 0.1, 0.1),  # Thin line
                        'distance': distance,
                        'bbox': (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
                        'confidence': 0.6,
                        'velocity': (0.0, 0.0, 0.0),
                        'timestamp': time.time()
                    }
                    
                    obstacles.append(obstacle)
            
            return obstacles
            
        except Exception as e:
            self.logger.error(f"Error detecting power lines: {e}")
            return []

    def _estimate_obstacle_distance(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                                  depth_map: Optional[np.ndarray]) -> Optional[float]:
        """Estimate distance to obstacle"""
        try:
            x, y, w, h = bbox
            
            if depth_map is not None:
                # Use depth map if available
                roi = depth_map[y:y+h, x:x+w]
                valid_depths = roi[roi > 0]
                
                if len(valid_depths) > 0:
                    return float(np.median(valid_depths))
            
            # Fallback to size-based estimation
            object_size = max(w, h)
            
            # Simple inverse relationship (calibrate for your camera)
            if object_size > 200:
                return 2.0
            elif object_size > 100:
                return 5.0
            elif object_size > 50:
                return 10.0
            elif object_size > 25:
                return 20.0
            else:
                return 30.0
                
        except Exception as e:
            self.logger.error(f"Error estimating distance: {e}")
            return None

    def _estimate_line_distance(self, line_y: float, frame_height: int) -> float:
        """Estimate distance to power line based on vertical position"""
        # Power lines are typically high up
        # Lines higher in the image are further away
        relative_height = line_y / frame_height
        
        # Assume power lines are 10-50 meters away
        distance = 10.0 + (1.0 - relative_height) * 40.0
        return distance

    def _pixel_to_world(self, pixel_x: float, pixel_y: float, distance: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates to world coordinates"""
        try:
            # Simple pinhole camera model
            # This should use actual camera calibration data
            focal_length = 500.0  # pixels
            sensor_width = 640.0  # pixels
            sensor_height = 480.0  # pixels
            
            # Convert to normalized coordinates
            norm_x = (pixel_x - sensor_width / 2) / focal_length
            norm_y = (pixel_y - sensor_height / 2) / focal_length
            
            # Calculate world coordinates
            world_x = norm_x * distance
            world_y = norm_y * distance
            world_z = distance
            
            return (world_x, world_y, world_z)
            
        except Exception as e:
            self.logger.error(f"Error converting pixel to world: {e}")
            return (0.0, 0.0, distance)

    def _estimate_size(self, pixel_width: int, pixel_height: int, distance: float) -> Tuple[float, float, float]:
        """Estimate real-world size of obstacle"""
        try:
            # Convert pixel size to real-world size
            focal_length = 500.0  # pixels
            
            real_width = (pixel_width * distance) / focal_length
            real_height = (pixel_height * distance) / focal_length
            
            # Assume depth is similar to width
            real_depth = real_width
            
            return (real_width, real_height, real_depth)
            
        except Exception as e:
            self.logger.error(f"Error estimating size: {e}")
            return (1.0, 1.0, 1.0)

    def _estimate_velocity(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
        """Estimate velocity of moving obstacle"""
        try:
            # This would require tracking between frames
            # For now, return zero velocity
            return (0.0, 0.0, 0.0)
            
        except Exception as e:
            self.logger.error(f"Error estimating velocity: {e}")
            return (0.0, 0.0, 0.0)

    def _classify_static_obstacle(self, roi: np.ndarray, width: int, height: int) -> ObstacleType:
        """Classify static obstacle type"""
        try:
            aspect_ratio = width / height if height > 0 else 1.0
            
            if aspect_ratio > 2.0:
                return ObstacleType.BUILDING
            elif aspect_ratio < 0.5 and height > width:
                return ObstacleType.TREE
            else:
                return ObstacleType.STATIC
                
        except Exception as e:
            self.logger.error(f"Error classifying static obstacle: {e}")
            return ObstacleType.UNKNOWN

    def _classify_dynamic_obstacle(self, velocity: Tuple[float, float, float], 
                                 width: int, height: int) -> ObstacleType:
        """Classify dynamic obstacle type"""
        try:
            speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
            aspect_ratio = width / height if height > 0 else 1.0
            
            if speed > 10.0:  # Fast moving
                if aspect_ratio > 1.5:
                    return ObstacleType.VEHICLE
                else:
                    return ObstacleType.BIRD
            elif speed > 2.0:  # Medium speed
                return ObstacleType.PERSON
            else:
                return ObstacleType.DYNAMIC
                
        except Exception as e:
            self.logger.error(f"Error classifying dynamic obstacle: {e}")
            return ObstacleType.UNKNOWN

    def _calculate_confidence(self, area: float, aspect_ratio: float) -> float:
        """Calculate detection confidence"""
        try:
            # Base confidence on area and aspect ratio
            area_score = min(1.0, area / 1000.0)
            aspect_score = 1.0 - abs(aspect_ratio - 1.0) / 2.0
            
            confidence = (area_score + aspect_score) / 2.0
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def _update_tracking(self, new_obstacles: List[Dict]):
        """Update obstacle tracking"""
        try:
            current_time = time.time()
            
            # Remove old tracks
            self._remove_old_tracks(current_time)
            
            # Match new detections with existing tracks
            for obstacle in new_obstacles:
                matched_id = self._find_matching_track(obstacle)
                
                if matched_id:
                    # Update existing track
                    self.tracked_obstacles[matched_id].update({
                        'position': obstacle['position'],
                        'last_seen': current_time,
                        'velocity': obstacle['velocity']
                    })
                else:
                    # Create new track
                    obstacle_id = f"obs_{self.next_obstacle_id}"
                    self.next_obstacle_id += 1
                    
                    obstacle.update({
                        'id': obstacle_id,
                        'last_seen': current_time
                    })
                    
                    self.tracked_obstacles[obstacle_id] = obstacle
                    
        except Exception as e:
            self.logger.error(f"Error updating tracking: {e}")

    def _find_matching_track(self, obstacle: Dict) -> Optional[str]:
        """Find matching track for new obstacle"""
        try:
            obstacle_pos = np.array(obstacle['position'])
            min_distance = float('inf')
            best_match = None
            
            for track_id, track in self.tracked_obstacles.items():
                track_pos = np.array(track['position'])
                distance = np.linalg.norm(obstacle_pos - track_pos)
                
                if distance < min_distance and distance < self.config.max_tracking_distance:
                    min_distance = distance
                    best_match = track_id
            
            return best_match
            
        except Exception as e:
            self.logger.error(f"Error finding matching track: {e}")
            return None

    def _remove_old_tracks(self, current_time: float):
        """Remove tracks that haven't been seen recently"""
        try:
            to_remove = []
            
            for track_id, track in self.tracked_obstacles.items():
                if current_time - track['last_seen'] > self.max_tracking_age:
                    to_remove.append(track_id)
            
            for track_id in to_remove:
                del self.tracked_obstacles[track_id]
                
        except Exception as e:
            self.logger.error(f"Error removing old tracks: {e}")

    def _filter_obstacles(self, obstacles: List[Dict]) -> List[Dict]:
        """Filter and merge duplicate obstacles"""
        try:
            if not obstacles:
                return []
            
            # Remove duplicates based on position
            filtered = []
            
            for obstacle in obstacles:
                is_duplicate = False
                
                for existing in filtered:
                    pos1 = np.array(obstacle['position'])
                    pos2 = np.array(existing['position'])
                    distance = np.linalg.norm(pos1 - pos2)
                    
                    if distance < self.config.duplicate_threshold:
                        # Keep the one with higher confidence
                        if obstacle['confidence'] > existing['confidence']:
                            filtered.remove(existing)
                            filtered.append(obstacle)
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered.append(obstacle)
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"Error filtering obstacles: {e}")
            return obstacles

    def _calculate_threat_levels(self, obstacles: List[Dict]) -> List[Dict]:
        """Calculate threat level for each obstacle"""
        try:
            for obstacle in obstacles:
                threat_level = 0.0
                
                # Distance factor (closer = higher threat)
                distance = obstacle['distance']
                if distance < 2.0:
                    threat_level += 0.8
                elif distance < 5.0:
                    threat_level += 0.5
                elif distance < 10.0:
                    threat_level += 0.2
                
                # Type factor
                if obstacle['type'] == ObstacleType.PERSON:
                    threat_level += 0.3
                elif obstacle['type'] == ObstacleType.VEHICLE:
                    threat_level += 0.4
                elif obstacle['type'] == ObstacleType.POWER_LINE:
                    threat_level += 0.6
                elif obstacle['type'] == ObstacleType.BUILDING:
                    threat_level += 0.5
                
                # Velocity factor (moving = higher threat)
                velocity = obstacle['velocity']
                speed = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
                if speed > 5.0:
                    threat_level += 0.3
                elif speed > 2.0:
                    threat_level += 0.1
                
                # Confidence factor
                threat_level *= obstacle['confidence']
                
                obstacle['threat_level'] = min(1.0, threat_level)
            
            return obstacles
            
        except Exception as e:
            self.logger.error(f"Error calculating threat levels: {e}")
            return obstacles

    def _update_performance_metrics(self, detection_time: float):
        """Update performance metrics"""
        try:
            self.detection_times.append(detection_time)
            
            if len(self.detection_times) > self.max_time_samples:
                self.detection_times.pop(0)
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        try:
            if not self.detection_times:
                return {}
            
            times = np.array(self.detection_times)
            
            return {
                'avg_detection_time': float(np.mean(times)),
                'max_detection_time': float(np.max(times)),
                'min_detection_time': float(np.min(times)),
                'detection_fps': 1.0 / float(np.mean(times)) if np.mean(times) > 0 else 0.0,
                'total_detections': len(self.detection_times)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance stats: {e}")
            return {}

    def get_tracked_obstacles(self) -> Dict:
        """Get currently tracked obstacles"""
        return self.tracked_obstacles.copy()

    def clear_tracking(self):
        """Clear all tracked obstacles"""
        self.tracked_obstacles.clear()
        self.next_obstacle_id = 0

    def get_status(self) -> Dict:
        """Get obstacle detector status"""
        return {
            'tracked_obstacles': len(self.tracked_obstacles),
            'detection_methods': {
                'depth_analysis': self.use_depth_analysis,
                'motion_detection': self.use_motion_detection,
                'ml_detection': self.use_ml_detection
            },
            'performance': self.get_performance_stats()
        }
