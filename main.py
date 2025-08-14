import cv2
import numpy as np
import time
import threading
import queue
import logging
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

from drone_controller import DroneController
from vision_system import VisionSystem
from obstacle_detector import ObstacleDetector
from path_planner import PathPlanner
from nlp_commander import NLPCommander
from config import DroneConfig
from utils import setup_logging, save_flight_data

class FlightMode(Enum):
    MANUAL = "manual"
    AUTONOMOUS = "autonomous"
    EMERGENCY = "emergency"
    LANDING = "landing"
    HOVERING = "hovering"

@dataclass
class DroneState:
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    orientation: Tuple[float, float, float]
    battery_level: float
    flight_mode: FlightMode
    obstacles_detected: List[Dict]
    target_position: Optional[Tuple[float, float, float]]
    is_armed: bool

class AutonomousDrone:
    """Main autonomous drone navigation system"""
    
    def __init__(self, config_path: str = "config/drone_config.json"):
        # Initialize logging
        self.logger = setup_logging()
        
        # Load configuration
        self.config = DroneConfig(config_path)
        
        # Initialize components
        self.drone_controller = DroneController(self.config)
        self.vision_system = VisionSystem(self.config)
        self.obstacle_detector = ObstacleDetector(self.config)
        self.path_planner = PathPlanner(self.config)
        self.nlp_commander = NLPCommander(self.config)
        
        # Initialize state
        self.state = DroneState(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            battery_level=100.0,
            flight_mode=FlightMode.MANUAL,
            obstacles_detected=[],
            target_position=None,
            is_armed=False
        )
        
        # Threading and communication
        self.command_queue = queue.Queue()
        self.vision_queue = queue.Queue()
        self.running = False
        self.threads = []
        
        # Flight data logging
        self.flight_data = []
        self.start_time = time.time()
        
        self.logger.info("Autonomous Drone System initialized")

    def start(self):
        """Start the autonomous drone system"""
        try:
            self.running = True
            
            # Start threads
            self.threads = [
                threading.Thread(target=self._vision_thread, daemon=True),
                threading.Thread(target=self._control_thread, daemon=True),
                threading.Thread(target=self._telemetry_thread, daemon=True),
                threading.Thread(target=self._nlp_thread, daemon=True)
            ]
            
            for thread in self.threads:
                thread.start()
            
            self.logger.info("Drone system started successfully")
            
            # Connect to drone
            if self.drone_controller.connect():
                self.logger.info("Connected to drone")
                self._main_loop()
            else:
                self.logger.error("Failed to connect to drone")
                
        except Exception as e:
            self.logger.error(f"Error starting drone system: {e}")
            self.stop()

    def stop(self):
        """Stop the autonomous drone system"""
        self.running = False
        self.logger.info("Shutting down drone system...")
        
        # Emergency landing if in flight
        if self.state.is_armed:
            self.emergency_landing()
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        # Save flight data
        self._save_flight_log()
        
        # Disconnect from drone
        self.drone_controller.disconnect()
        self.logger.info("Drone system stopped")

    def _main_loop(self):
        """Main control loop"""
        while self.running:
            try:
                # Update drone state
                self._update_state()
                
                # Process commands
                self._process_commands()
                
                # Handle current flight mode
                if self.state.flight_mode == FlightMode.AUTONOMOUS:
                    self._autonomous_flight()
                elif self.state.flight_mode == FlightMode.EMERGENCY:
                    self._emergency_behavior()
                elif self.state.flight_mode == FlightMode.LANDING:
                    self._landing_sequence()
                
                # Safety checks
                self._safety_checks()
                
                # Log flight data
                self._log_flight_data()
                
                time.sleep(0.05)  # 20Hz main loop
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                self.set_flight_mode(FlightMode.EMERGENCY)

    def _vision_thread(self):
        """Computer vision processing thread"""
        while self.running:
            try:
                # Capture frame
                frame = self.vision_system.capture_frame()
                if frame is None:
                    continue
                
                # Process frame for obstacles
                obstacles = self.obstacle_detector.detect_obstacles(frame)
                
                # Update state
                self.state.obstacles_detected = obstacles
                
                # Put processed data in queue
                vision_data = {
                    'timestamp': time.time(),
                    'frame': frame,
                    'obstacles': obstacles,
                    'depth_map': self.vision_system.generate_depth_map(frame)
                }
                
                try:
                    self.vision_queue.put_nowait(vision_data)
                except queue.Full:
                    # Remove old data if queue is full
                    try:
                        self.vision_queue.get_nowait()
                        self.vision_queue.put_nowait(vision_data)
                    except queue.Empty:
                        pass
                
            except Exception as e:
                self.logger.error(f"Error in vision thread: {e}")
                time.sleep(0.1)

    def _control_thread(self):
        """Flight control thread"""
        while self.running:
            try:
                # Get latest vision data
                vision_data = None
                try:
                    vision_data = self.vision_queue.get_nowait()
                except queue.Empty:
                    pass
                
                if vision_data and self.state.flight_mode == FlightMode.AUTONOMOUS:
                    # Plan path based on obstacles
                    target_velocity = self.path_planner.plan_path(
                        self.state.position,
                        self.state.target_position,
                        vision_data['obstacles'],
                        vision_data['depth_map']
                    )
                    
                    # Execute control commands
                    self.drone_controller.set_velocity(target_velocity)
                
                time.sleep(0.02)  # 50Hz control loop
                
            except Exception as e:
                self.logger.error(f"Error in control thread: {e}")
                time.sleep(0.1)

    def _telemetry_thread(self):
        """Telemetry and monitoring thread"""
        while self.running:
            try:
                # Get telemetry data
                telemetry = self.drone_controller.get_telemetry()
                
                if telemetry:
                    # Update state
                    self.state.position = telemetry.get('position', self.state.position)
                    self.state.velocity = telemetry.get('velocity', self.state.velocity)
                    self.state.orientation = telemetry.get('orientation', self.state.orientation)
                    self.state.battery_level = telemetry.get('battery', self.state.battery_level)
                    self.state.is_armed = telemetry.get('armed', self.state.is_armed)
                
                time.sleep(0.1)  # 10Hz telemetry update
                
            except Exception as e:
                self.logger.error(f"Error in telemetry thread: {e}")
                time.sleep(0.5)

    def _nlp_thread(self):
        """Natural Language Processing thread for voice commands"""
        while self.running:
            try:
                # Check for voice commands
                command = self.nlp_commander.listen_for_command()
                
                if command:
                    parsed_command = self.nlp_commander.parse_command(command)
                    
                    if parsed_command:
                        self.command_queue.put(parsed_command)
                        self.logger.info(f"Received NLP command: {command}")
                
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in NLP thread: {e}")
                time.sleep(1.0)

    def _process_commands(self):
        """Process commands from various sources"""
        try:
            command = self.command_queue.get_nowait()
            self._execute_command(command)
        except queue.Empty:
            pass

    def _execute_command(self, command: Dict):
        """Execute a parsed command"""
        try:
            cmd_type = command.get('type')
            
            if cmd_type == 'takeoff':
                self.takeoff()
            elif cmd_type == 'land':
                self.land()
            elif cmd_type == 'goto':
                target = command.get('target')
                self.goto_position(target)
            elif cmd_type == 'hover':
                self.hover()
            elif cmd_type == 'emergency':
                self.emergency_landing()
            elif cmd_type == 'set_mode':
                mode = command.get('mode')
                self.set_flight_mode(FlightMode(mode))
            elif cmd_type == 'return_home':
                self.return_to_home()
            
            self.logger.info(f"Executed command: {cmd_type}")
            
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")

    def _autonomous_flight(self):
        """Handle autonomous flight behavior"""
        if not self.state.target_position:
            return
        
        # Check if we've reached the target
        distance_to_target = np.linalg.norm(
            np.array(self.state.position) - np.array(self.state.target_position)
        )
        
        if distance_to_target < self.config.target_tolerance:
            self.logger.info("Target reached, switching to hover mode")
            self.set_flight_mode(FlightMode.HOVERING)

    def _emergency_behavior(self):
        """Handle emergency behavior"""
        self.logger.warning("Emergency mode active - initiating emergency landing")
        self.drone_controller.emergency_stop()
        self.set_flight_mode(FlightMode.LANDING)

    def _landing_sequence(self):
        """Handle landing sequence"""
        if self.state.position[2] > 0.5:  # If above ground
            # Gradual descent
            landing_velocity = (0, 0, -self.config.landing_speed)
            self.drone_controller.set_velocity(landing_velocity)
        else:
            # Land and disarm
            self.drone_controller.land()
            self.state.is_armed = False
            self.set_flight_mode(FlightMode.MANUAL)
            self.logger.info("Landing completed")

    def _safety_checks(self):
        """Perform safety checks"""
        # Battery check
        if self.state.battery_level < self.config.low_battery_threshold:
            self.logger.warning(f"Low battery: {self.state.battery_level}%")
            if self.state.battery_level < self.config.critical_battery_threshold:
                self.logger.critical("Critical battery level - emergency landing")
                self.set_flight_mode(FlightMode.EMERGENCY)
        
        # Altitude check
        if self.state.position[2] > self.config.max_altitude:
            self.logger.warning("Maximum altitude exceeded")
            self.set_flight_mode(FlightMode.EMERGENCY)
        
        # Obstacle proximity check
        for obstacle in self.state.obstacles_detected:
            if obstacle.get('distance', float('inf')) < self.config.min_obstacle_distance:
                self.logger.warning("Obstacle too close - emergency stop")
                self.set_flight_mode(FlightMode.EMERGENCY)

    def _update_state(self):
        """Update drone state"""
        # This would typically get real telemetry data
        # For now, we'll simulate basic updates
        pass

    def _log_flight_data(self):
        """Log flight data for analysis"""
        data_point = {
            'timestamp': time.time() - self.start_time,
            'position': self.state.position,
            'velocity': self.state.velocity,
            'orientation': self.state.orientation,
            'battery_level': self.state.battery_level,
            'flight_mode': self.state.flight_mode.value,
            'obstacles_count': len(self.state.obstacles_detected),
            'is_armed': self.state.is_armed
        }
        
        self.flight_data.append(data_point)
        
        # Keep only last 1000 data points to manage memory
        if len(self.flight_data) > 1000:
            self.flight_data = self.flight_data[-1000:]

    def _save_flight_log(self):
        """Save flight data to file"""
        try:
            save_flight_data(self.flight_data, self.config.log_directory)
            self.logger.info("Flight data saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving flight data: {e}")

    # Public API methods
    def takeoff(self, altitude: float = 2.0):
        """Takeoff to specified altitude"""
        if self.drone_controller.takeoff(altitude):
            self.set_flight_mode(FlightMode.HOVERING)
            self.logger.info(f"Taking off to {altitude}m")
            return True
        return False

    def land(self):
        """Land the drone"""
        self.set_flight_mode(FlightMode.LANDING)
        self.logger.info("Landing sequence initiated")

    def hover(self):
        """Switch to hover mode"""
        self.set_flight_mode(FlightMode.HOVERING)
        self.state.target_position = self.state.position
        self.logger.info("Switching to hover mode")

    def goto_position(self, position: Tuple[float, float, float]):
        """Go to specified position autonomously"""
        self.state.target_position = position
        self.set_flight_mode(FlightMode.AUTONOMOUS)
        self.logger.info(f"Going to position: {position}")

    def return_to_home(self):
        """Return to home position"""
        home_position = self.config.home_position
        self.goto_position(home_position)
        self.logger.info("Returning to home")

    def emergency_landing(self):
        """Initiate emergency landing"""
        self.set_flight_mode(FlightMode.EMERGENCY)
        self.logger.warning("Emergency landing initiated")

    def set_flight_mode(self, mode: FlightMode):
        """Set flight mode"""
        if mode != self.state.flight_mode:
            self.logger.info(f"Flight mode changed: {self.state.flight_mode.value} -> {mode.value}")
            self.state.flight_mode = mode

    def get_status(self) -> Dict:
        """Get current drone status"""
        return {
            'position': self.state.position,
            'velocity': self.state.velocity,
            'battery_level': self.state.battery_level,
            'flight_mode': self.state.flight_mode.value,
            'is_armed': self.state.is_armed,
            'obstacles_detected': len(self.state.obstacles_detected),
            'target_position': self.state.target_position
        }

def main():
    """Main entry point"""
    drone = AutonomousDrone()
    
    try:
        print("Starting Autonomous Drone Navigation System...")
        print("Use voice commands or keyboard input to control the drone")
        print("Commands: 'takeoff', 'land', 'hover', 'go to [x] [y] [z]', 'emergency', 'return home'")
        
        drone.start()
        
        # Simple keyboard interface for testing
        while drone.running:
            try:
                user_input = input("\nEnter command (or 'quit' to exit): ").strip().lower()
                
                if user_input == 'quit':
                    break
                elif user_input == 'status':
                    status = drone.get_status()
                    print(f"Status: {json.dumps(status, indent=2)}")
                else:
                    # Parse simple commands
                    if user_input == 'takeoff':
                        drone.takeoff()
                    elif user_input == 'land':
                        drone.land()
                    elif user_input == 'hover':
                        drone.hover()
                    elif user_input == 'emergency':
                        drone.emergency_landing()
                    elif user_input == 'home':
                        drone.return_to_home()
                    elif user_input.startswith('goto'):
                        parts = user_input.split()
                        if len(parts) >= 4:
                            try:
                                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                drone.goto_position((x, y, z))
                            except ValueError:
                                print("Invalid coordinates")
                        else:
                            print("Usage: goto <x> <y> <z>")
                    else:
                        print("Unknown command")
                        
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    finally:
        drone.stop()

if __name__ == "__main__":
    main()
