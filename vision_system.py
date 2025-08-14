#!/usr/bin/env python3
"""
Vision System Module
Computer vision processing for autonomous navigation
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional, Union
import threading
import time
from dataclasses import dataclass
import json

try:
    # Try to import advanced CV libraries
    import torch
    import torchvision.transforms as transforms
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available, using basic CV methods")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logging.warning("Open3D not available, using basic depth estimation")

@dataclass
class CameraInfo:
    width: int
    height: int
    fx: float  # focal length x
    fy: float  # focal length y
    cx: float  # optical center x
    cy: float  # optical center y
    distortion: List[float]

@dataclass
class DetectedObject:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    distance: Optional[float]
    position_3d: Optional[Tuple[float, float, float]]

class VisionSystem:
    """Computer vision system for drone navigation"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Camera setup
        self.camera_index = config.camera_index
        self.camera = None
        self.camera_info = self._load_camera_calibration()
        
        # YOLO model for object detection
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(config.yolo_model_path)
                self.logger.info("YOLO model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load YOLO model: {e}")
        
        # Stereo vision setup
        self.stereo_matcher = None
        self.use_stereo = config.use_stereo_vision
        
        if self.use_stereo:
            self._setup_stereo_vision()
        
        # Optical flow for motion detection
        self.prev_frame = None
        self.flow_tracker = cv2.goodFeaturesToTrack
        
        # Frame processing
        self.frame_buffer = []
        self.max_buffer_size = 10
        
        # Threading
        self.capture_thread = None
        self.processing_lock = threading.Lock()
        self.running = False
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        self.logger.info("Vision system initialized")

    def start(self) -> bool:
        """Start the vision system"""
        try:
            # Initialize camera
            if not self._init_camera():
                return False
            
            self.running = True
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            self.logger.info("Vision system started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting vision system: {e}")
            return False

    def stop(self):
        """Stop the vision system"""
        try:
            self.running = False
            
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
            
            if self.camera:
                self.camera.release()
                self.camera = None
            
            self.logger.info("Vision system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping vision system: {e}")

    def _init_camera(self) -> bool:
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                self.logger.error(f"Cannot open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            
            # Verify settings
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Camera initialized: {width}x{height} @ {fps}fps")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing camera: {e}")
            return False

    def _capture_loop(self):
        """Background thread for frame capture"""
        while self.running:
            try:
                ret, frame = self.camera.read()
                
                if not ret:
                    self.logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Add timestamp
                timestamp = time.time()
                
                with self.processing_lock:
                    # Add to buffer
                    self.frame_buffer.append((timestamp, frame))
                    
                    # Keep buffer size manageable
                    if len(self.frame_buffer) > self.max_buffer_size:
                        self.frame_buffer.pop(0)
                
                # Update FPS counter
                self._update_fps()
                
                time.sleep(1.0 / self.config.camera_fps)
                
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)

    def capture_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame"""
        try:
            with self.processing_lock:
                if self.frame_buffer:
                    timestamp, frame = self.frame_buffer[-1]
                    return frame.copy()
                return None
                
        except Exception as e:
            self.logger.error(f"Error capturing frame: {e}")
            return None

    def detect_objects(self, frame: np.ndarray) -> List[DetectedObject]:
        """Detect objects in frame using YOLO"""
        detected_objects = []
        
        try:
            if self.yolo_model is None:
                # Fallback to basic blob detection
                return self._detect_basic_objects(frame)
            
            # Run YOLO inference
            results = self.yolo_model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract detection data
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]
                        
                        # Filter by confidence
                        if confidence < self.config.detection_confidence_threshold:
                            continue
                        
                        # Estimate distance if depth available
                        distance = self._estimate_distance(frame, (int(x1), int(y1), int(x2), int(y2)))
                        
                        # Calculate 3D position
                        position_3d = self._pixel_to_3d(
                            ((x1 + x2) / 2, (y1 + y2) / 2), distance
                        ) if distance else None
                        
                        detected_object = DetectedObject(
                            class_name=class_name,
                            confidence=float(confidence),
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            distance=distance,
                            position_3d=position_3d
                        )
                        
                        detected_objects.append(detected_object)
            
            return detected_objects
            
        except Exception as e:
            self.logger.error(f"Error in object detection: {e}")
            return []

    def _detect_basic_objects(self, frame: np.ndarray) -> List[DetectedObject]:
        """Basic object detection using classical CV methods"""
        detected_objects = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter by area
                area = cv2.contourArea(contour)
                if area < self.config.min_object_area:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Basic size filtering
                if w < 20 or h < 20:
                    continue
                
                # Estimate distance based on size
                distance = self._estimate_distance_from_size(w, h)
                
                detected_object = DetectedObject(
                    class_name="unknown_object",
                    confidence=0.7,
                    bbox=(x, y, x + w, y + h),
                    distance=distance,
                    position_3d=self._pixel_to_3d((x + w/2, y + h/2), distance)
                )
                
                detected_objects.append(detected_object)
            
            return detected_objects
            
        except Exception as e:
            self.logger.error(f"Error in basic object detection: {e}")
            return []

    def generate_depth_map(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Generate depth map from frame"""
        try:
            if self.use_stereo and self.stereo_matcher:
                # Use stereo vision for depth
                return self._stereo_depth_estimation(frame)
            else:
                # Use monocular depth estimation
                return self._monocular_depth_estimation(frame)
                
        except Exception as e:
            self.logger.error(f"Error generating depth map: {e}")
            return None

    def _stereo_depth_estimation(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Stereo vision depth estimation"""
        try:
            # This would require a stereo camera setup
            # For now, return a placeholder
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return np.ones_like(gray) * 5.0  # Assume 5m distance
            
        except Exception as e:
            self.logger.error(f"Error in stereo depth estimation: {e}")
            return None

    def _monocular_depth_estimation(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Monocular depth estimation using various cues"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Simple depth estimation based on image features
            depth_map = np.zeros_like(gray, dtype=np.float32)
            
            # Use blur as depth cue (focused = close, blurred = far)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Use size and position cues
            for y in range(h):
                for x in range(w):
                    # Objects lower in image are typically closer
                    height_factor = (h - y) / h
                    
                    # Objects in center might be closer (assumption)
                    center_factor = 1.0 - abs(x - w/2) / (w/2)
                    
                    # Combine factors for rough depth estimate
                    depth = 2.0 + 8.0 * (1.0 - height_factor * 0.7 - center_factor * 0.3)
                    depth_map[y, x] = depth
            
            # Smooth the depth map
            depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)
            
            return depth_map
            
        except Exception as e:
            self.logger.error(f"Error in monocular depth estimation: {e}")
            return None

    def _estimate_distance(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[float]:
        """Estimate distance to object in bounding box"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Extract object region
            obj_height = y2 - y1
            obj_width = x2 - x1
            
            # Simple distance estimation based on object size
            # This is a very basic approach - in practice, you'd use
            # known object sizes or stereo vision
            
            if obj_height == 0:
                return None
            
            # Assume average object height of 1.7m (person height)
            # and use pinhole camera model
            focal_length = self.camera_info.fy if self.camera_info else 500
            object_height_real = 1.7  # meters
            
            distance = (object_height_real * focal_length) / obj_height
            
            # Clamp to reasonable range
            distance = max(0.5, min(distance, 50.0))
            
            return distance
            
        except Exception as e:
            self.logger.error(f"Error estimating distance: {e}")
            return None

    def _estimate_distance_from_size(self, width: int, height: int) -> float:
        """Estimate distance based on object size"""
        # Simple heuristic based on object size
        size = max(width, height)
        if size > 200:
            return 2.0
        elif size > 100:
            return 5.0
        elif size > 50:
            return 10.0
        else:
            return 20.0

    def _pixel_to_3d(self, pixel: Tuple[float, float], distance: float) -> Optional[Tuple[float, float, float]]:
        """Convert pixel coordinates to 3D world coordinates"""
        try:
            if not self.camera_info or distance is None:
                return None
            
            u, v = pixel
            
            # Use camera intrinsics to convert to 3D
            x = (u - self.camera_info.cx) * distance / self.camera_info.fx
            y = (v - self.camera_info.cy) * distance / self.camera_info.fy
            z = distance
            
            return (x, y, z)
            
        except Exception as e:
            self.logger.error(f"Error converting pixel to 3D: {e}")
            return None

    def detect_motion(self, frame: np.ndarray) -> List[Dict]:
        """Detect motion in frame using optical flow"""
        motion_objects = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.prev_frame is None:
                self.prev_frame = gray
                return motion_objects
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, gray, None, None,
                **dict(winSize=(15, 15), maxLevel=2,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            )
            
            if flow is not None:
                # Analyze flow vectors for significant motion
                # This is a simplified implementation
                pass
            
            self.prev_frame = gray
            return motion_objects
            
        except Exception as e:
            self.logger.error(f"Error in motion detection: {e}")
            return []

    def _setup_stereo_vision(self):
        """Setup stereo vision for depth estimation"""
        try:
            # Create stereo matcher
            self.stereo_matcher = cv2.StereoBM_create(numDisparities=64, blockSize=15)
            
            # Configure parameters
            self.stereo_matcher.setPreFilterType(cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE)
            self.stereo_matcher.setPreFilterSize(5)
            self.stereo_matcher.setPreFilterCap(61)
            self.stereo_matcher.setBlockSize(15)
            self.stereo_matcher.setMinDisparity(0)
            self.stereo_matcher.setNumDisparities(64)
            self.stereo_matcher.setTextureThreshold(507)
            self.stereo_matcher.setUniquenessRatio(0)
            self.stereo_matcher.setSpeckleWindowSize(0)
            self.stereo_matcher.setSpeckleRange(8)
            self.stereo_matcher.setDisp12MaxDiff(1)
            
            self.logger.info("Stereo vision setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up stereo vision: {e}")
            self.use_stereo = False

    def _load_camera_calibration(self) -> Optional[CameraInfo]:
        """Load camera calibration data"""
        try:
            calib_file = self.config.camera_calibration_file
            
            if not calib_file:
                # Use default values
                return CameraInfo(
                    width=640,
                    height=480,
                    fx=500.0,
                    fy=500.0,
                    cx=320.0,
                    cy=240.0,
                    distortion=[0.0, 0.0, 0.0, 0.0, 0.0]
                )
            
            with open(calib_file, 'r') as f:
                calib_data = json.load(f)
            
            return CameraInfo(**calib_data)
            
        except Exception as e:
            self.logger.warning(f"Could not load camera calibration: {e}")
            return None

    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        
        if self.fps_counter % 30 == 0:  # Update every 30 frames
            elapsed = time.time() - self.fps_start_time
            self.current_fps = 30 / elapsed
            self.fps_start_time = time.time()

    def get_fps(self) -> float:
        """Get current FPS"""
        return self.current_fps

    def calibrate_camera(self) -> bool:
        """Calibrate camera using chessboard pattern"""
        try:
            # This would implement camera calibration
            # using a chessboard pattern
            self.logger.info("Camera calibration not implemented")
            return False
            
        except Exception as e:
            self.logger.error(f"Error in camera calibration: {e}")
            return False

    def save_frame(self, frame: np.ndarray, filename: str):
        """Save frame to file"""
        try:
            cv2.imwrite(filename, frame)
            self.logger.info(f"Frame saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving frame: {e}")

    def get_status(self) -> Dict:
        """Get vision system status"""
        return {
            'running': self.running,
            'camera_connected': self.camera is not None and self.camera.isOpened(),
            'fps': self.current_fps,
            'yolo_available': YOLO_AVAILABLE,
            'stereo_enabled': self.use_stereo,
            'frame_buffer_size': len(self.frame_buffer),
            'camera_info': {
                'width': self.camera_info.width if self.camera_info else None,
                'height': self.camera_info.height if self.camera_info else None
            }
        }
