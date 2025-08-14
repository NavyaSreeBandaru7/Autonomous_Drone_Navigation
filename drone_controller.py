#!/usr/bin/env python3
"""
Drone Controller Module
Handles communication with drone hardware (MAVLink, DJI SDK, etc.)
"""

import time
import numpy as np
import logging
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import threading
import socket
import struct

try:
    # Try to import pymavlink for real drone communication
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False
    logging.warning("pymavlink not available, using simulation mode")

@dataclass
class DroneCommand:
    timestamp: float
    command_type: str
    parameters: Dict

class DroneController:
    """Interface for controlling drone hardware"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Connection parameters
        self.connection_string = config.connection_string
        self.simulation_mode = config.simulation_mode or not MAVLINK_AVAILABLE
        
        # MAVLink connection
        self.mavlink_connection = None
        self.connected = False
        
        # Simulation state (for testing without real drone)
        self.sim_position = np.array([0.0, 0.0, 0.0])
        self.sim_velocity = np.array([0.0, 0.0, 0.0])
        self.sim_orientation = np.array([0.0, 0.0, 0.0])
        self.sim_armed = False
        self.sim_battery = 100.0
        
        # Command history
        self.command_history = []
        self.max_history = 1000
        
        # Threading
        self.telemetry_thread = None
        self.running = False
        
        self.logger.info(f"DroneController initialized (simulation={self.simulation_mode})")

    def connect(self) -> bool:
        """Connect to drone"""
        try:
            if self.simulation_mode:
                self.logger.info("Running in simulation mode")
                self.connected = True
                return True
            
            if not MAVLINK_AVAILABLE:
                self.logger.error("pymavlink not available")
                return False
            
            # Connect to drone via MAVLink
            self.logger.info(f"Connecting to drone at {self.connection_string}")
            self.mavlink_connection = mavutil.mavlink_connection(self.connection_string)
            
            # Wait for heartbeat
            self.logger.info("Waiting for heartbeat...")
            self.mavlink_connection.wait_heartbeat(timeout=10)
            
            self.connected = True
            self.running = True
            
            # Start telemetry thread
            self.telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
            self.telemetry_thread.start()
            
            self.logger.info("Successfully connected to drone")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to drone: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from drone"""
        try:
            self.running = False
            
            if self.telemetry_thread and self.telemetry_thread.is_alive():
                self.telemetry_thread.join(timeout=2.0)
            
            if self.mavlink_connection:
                self.mavlink_connection.close()
                self.mavlink_connection = None
            
            self.connected = False
            self.logger.info("Disconnected from drone")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from drone: {e}")

    def _telemetry_loop(self):
        """Background thread for receiving telemetry"""
        while self.running and self.connected:
            try:
                if not self.simulation_mode and self.mavlink_connection:
                    # Receive MAVLink messages
                    msg = self.mavlink_connection.recv_match(blocking=False)
                    if msg:
                        self._process_mavlink_message(msg)
                
                time.sleep(0.02)  # 50Hz telemetry
                
            except Exception as e:
                self.logger.error(f"Error in telemetry loop: {e}")
                time.sleep(0.1)

    def _process_mavlink_message(self, msg):
        """Process received MAVLink message"""
        try:
            msg_type = msg.get_type()
            
            if msg_type == 'GLOBAL_POSITION_INT':
                # Update position from GPS
                self.sim_position = np.array([
                    msg.lat / 1e7,  # Convert to degrees
                    msg.lon / 1e7,
                    msg.alt / 1000.0  # Convert to meters
                ])
                
            elif msg_type == 'LOCAL_POSITION_NED':
                # Update local position
                self.sim_position = np.array([msg.x, msg.y, -msg.z])
                self.sim_velocity = np.array([msg.vx, msg.vy, -msg.vz])
                
            elif msg_type == 'ATTITUDE':
                # Update orientation
                self.sim_orientation = np.array([msg.roll, msg.pitch, msg.yaw])
                
            elif msg_type == 'SYS_STATUS':
                # Update battery status
                self.sim_battery = msg.battery_remaining
                
            elif msg_type == 'HEARTBEAT':
                # Update armed status
                self.sim_armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                
        except Exception as e:
            self.logger.error(f"Error processing MAVLink message: {e}")

    def takeoff(self, altitude: float = 2.0) -> bool:
        """Takeoff to specified altitude"""
        try:
            command = DroneCommand(
                timestamp=time.time(),
                command_type="takeoff",
                parameters={"altitude": altitude}
            )
            self._log_command(command)
            
            if self.simulation_mode:
                self.logger.info(f"[SIM] Taking off to {altitude}m")
                self.sim_armed = True
                # Simulate takeoff
                target_pos = self.sim_position.copy()
                target_pos[2] = altitude
                self._simulate_movement(target_pos, duration=5.0)
                return True
            
            if not self.mavlink_connection:
                return False
            
            # Send MAVLink takeoff command
            self.mavlink_connection.mav.command_long_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,  # confirmation
                0, 0, 0, 0,  # param1-4
                0, 0, altitude  # param5-7 (lat, lon, alt)
            )
            
            self.logger.info(f"Takeoff command sent (altitude: {altitude}m)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during takeoff: {e}")
            return False

    def land(self) -> bool:
        """Land the drone"""
        try:
            command = DroneCommand(
                timestamp=time.time(),
                command_type="land",
                parameters={}
            )
            self._log_command(command)
            
            if self.simulation_mode:
                self.logger.info("[SIM] Landing")
                target_pos = self.sim_position.copy()
                target_pos[2] = 0.0
                self._simulate_movement(target_pos, duration=3.0)
                self.sim_armed = False
                return True
            
            if not self.mavlink_connection:
                return False
            
            # Send MAVLink land command
            self.mavlink_connection.mav.command_long_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0,  # confirmation
                0, 0, 0, 0, 0, 0, 0  # param1-7
            )
            
            self.logger.info("Land command sent")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during landing: {e}")
            return False

    def set_position(self, position: Tuple[float, float, float]) -> bool:
        """Set target position"""
        try:
            x, y, z = position
            command = DroneCommand(
                timestamp=time.time(),
                command_type="set_position",
                parameters={"position": position}
            )
            self._log_command(command)
            
            if self.simulation_mode:
                self.logger.info(f"[SIM] Moving to position: ({x:.2f}, {y:.2f}, {z:.2f})")
                self._simulate_movement(np.array([x, y, z]), duration=2.0)
                return True
            
            if not self.mavlink_connection:
                return False
            
            # Send position setpoint
            self.mavlink_connection.mav.set_position_target_local_ned_send(
                0,  # time_boot_ms
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111111000,  # type_mask
                x, y, z,  # position
                0, 0, 0,  # velocity
                0, 0, 0,  # acceleration
                0, 0  # yaw, yaw_rate
            )
            
            self.logger.info(f"Position setpoint sent: ({x:.2f}, {y:.2f}, {z:.2f})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting position: {e}")
            return False

    def set_velocity(self, velocity: Tuple[float, float, float]) -> bool:
        """Set target velocity"""
        try:
            vx, vy, vz = velocity
            command = DroneCommand(
                timestamp=time.time(),
                command_type="set_velocity",
                parameters={"velocity": velocity}
            )
            self._log_command(command)
            
            if self.simulation_mode:
                # Update simulation velocity
                self.sim_velocity = np.array([vx, vy, vz])
                # Update position based on velocity
                dt = 0.02  # 50Hz update rate
                self.sim_position += self.sim_velocity * dt
                return True
            
            if not self.mavlink_connection:
                return False
            
            # Send velocity setpoint
            self.mavlink_connection.mav.set_position_target_local_ned_send(
                0,  # time_boot_ms
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111000111,  # type_mask (velocity control)
                0, 0, 0,  # position (ignored)
                vx, vy, vz,  # velocity
                0, 0, 0,  # acceleration (ignored)
                0, 0  # yaw, yaw_rate
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting velocity: {e}")
            return False

    def arm(self) -> bool:
        """Arm the drone"""
        try:
            command = DroneCommand(
                timestamp=time.time(),
                command_type="arm",
                parameters={}
            )
            self._log_command(command)
            
            if self.simulation_mode:
                self.logger.info("[SIM] Arming drone")
                self.sim_armed = True
                return True
            
            if not self.mavlink_connection:
                return False
            
            # Send arm command
            self.mavlink_connection.mav.command_long_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # confirmation
                1,  # arm
                0, 0, 0, 0, 0, 0  # param2-7
            )
            
            self.logger.info("Arm command sent")
            return True
            
        except Exception as e:
            self.logger.error(f"Error arming drone: {e}")
            return False

    def disarm(self) -> bool:
        """Disarm the drone"""
        try:
            command = DroneCommand(
                timestamp=time.time(),
                command_type="disarm",
                parameters={}
            )
            self._log_command(command)
            
            if self.simulation_mode:
                self.logger.info("[SIM] Disarming drone")
                self.sim_armed = False
                return True
            
            if not self.mavlink_connection:
                return False
            
            # Send disarm command
            self.mavlink_connection.mav.command_long_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # confirmation
                0,  # disarm
                0, 0, 0, 0, 0, 0  # param2-7
            )
            
            self.logger.info("Disarm command sent")
            return True
            
        except Exception as e:
            self.logger.error(f"Error disarming drone: {e}")
            return False

    def emergency_stop(self) -> bool:
        """Emergency stop - immediate hover or land"""
        try:
            command = DroneCommand(
                timestamp=time.time(),
                command_type="emergency_stop",
                parameters={}
            )
            self._log_command(command)
            
            if self.simulation_mode:
                self.logger.warning("[SIM] Emergency stop - stopping all movement")
                self.sim_velocity = np.array([0.0, 0.0, 0.0])
                return True
            
            if not self.mavlink_connection:
                return False
            
            # Send emergency stop (hover in place)
            self.set_velocity((0, 0, 0))
            
            self.logger.warning("Emergency stop command sent")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during emergency stop: {e}")
            return False

    def set_mode(self, mode: str) -> bool:
        """Set flight mode"""
        try:
            command = DroneCommand(
                timestamp=time.time(),
                command_type="set_mode",
                parameters={"mode": mode}
            )
            self._log_command(command)
            
            if self.simulation_mode:
                self.logger.info(f"[SIM] Setting mode to: {mode}")
                return True
            
            if not self.mavlink_connection:
                return False
            
            # Map mode names to MAVLink modes
            mode_mapping = {
                'MANUAL': 'MANUAL',
                'STABILIZE': 'STABILIZE',
                'GUIDED': 'GUIDED',
                'AUTO': 'AUTO',
                'LAND': 'LAND',
                'RTL': 'RTL',
                'LOITER': 'LOITER'
            }
            
            mavlink_mode = mode_mapping.get(mode.upper())
            if not mavlink_mode:
                self.logger.error(f"Unknown mode: {mode}")
                return False
            
            # Get mode number
            mode_id = self.mavlink_connection.mode_mapping().get(mavlink_mode)
            if mode_id is None:
                self.logger.error(f"Mode ID not found for: {mavlink_mode}")
                return False
            
            # Send mode change command
            self.mavlink_connection.mav.set_mode_send(
                self.mavlink_connection.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id
            )
            
            self.logger.info(f"Mode change command sent: {mode}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting mode: {e}")
            return False

    def get_telemetry(self) -> Optional[Dict]:
        """Get current telemetry data"""
        try:
            if not self.connected:
                return None
            
            telemetry = {
                'timestamp': time.time(),
                'position': tuple(self.sim_position),
                'velocity': tuple(self.sim_velocity),
                'orientation': tuple(self.sim_orientation),
                'battery': self.sim_battery,
                'armed': self.sim_armed,
                'connected': self.connected
            }
            
            return telemetry
            
        except Exception as e:
            self.logger.error(f"Error getting telemetry: {e}")
            return None

    def _simulate_movement(self, target_position: np.ndarray, duration: float):
        """Simulate smooth movement to target position"""
        if self.simulation_mode:
            start_pos = self.sim_position.copy()
            start_time = time.time()
            
            def movement_thread():
                while time.time() - start_time < duration:
                    progress = (time.time() - start_time) / duration
                    progress = min(1.0, progress)
                    
                    # Smooth interpolation
                    t = 0.5 * (1 - np.cos(np.pi * progress))
                    self.sim_position = start_pos + t * (target_position - start_pos)
                    
                    # Update velocity
                    remaining_distance = target_position - self.sim_position
                    remaining_time = duration - (time.time() - start_time)
                    if remaining_time > 0:
                        self.sim_velocity = remaining_distance / remaining_time
                    else:
                        self.sim_velocity = np.array([0.0, 0.0, 0.0])
                    
                    time.sleep(0.02)
                
                self.sim_position = target_position
                self.sim_velocity = np.array([0.0, 0.0, 0.0])
            
            thread = threading.Thread(target=movement_thread, daemon=True)
            thread.start()

    def _log_command(self, command: DroneCommand):
        """Log command to history"""
        self.command_history.append(command)
        
        # Keep history size manageable
        if len(self.command_history) > self.max_history:
            self.command_history = self.command_history[-self.max_history:]

    def get_command_history(self) -> List[DroneCommand]:
        """Get command history"""
        return self.command_history.copy()

    def is_connected(self) -> bool:
        """Check if connected to drone"""
        return self.connected

    def get_status(self) -> Dict:
        """Get controller status"""
        return {
            'connected': self.connected,
            'simulation_mode': self.simulation_mode,
            'mavlink_available': MAVLINK_AVAILABLE,
            'command_count': len(self.command_history)
        }

    def calibrate_sensors(self) -> bool:
        """Calibrate drone sensors"""
        try:
            if self.simulation_mode:
                self.logger.info("[SIM] Calibrating sensors")
                return True
            
            if not self.mavlink_connection:
                return False
            
            # Start gyro calibration
            self.mavlink_connection.mav.command_long_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_CMD_PREFLIGHT_CALIBRATION,
                0,  # confirmation
                1,  # gyro calibration
                0, 0, 0, 0, 0, 0  # other params
            )
            
            self.logger.info("Sensor calibration started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during calibration: {e}")
            return False

    def set_home_position(self, position: Tuple[float, float, float]) -> bool:
        """Set home position"""
        try:
            lat, lon, alt = position
            
            if self.simulation_mode:
                self.logger.info(f"[SIM] Setting home position: {position}")
                return True
            
            if not self.mavlink_connection:
                return False
            
            # Set home position
            self.mavlink_connection.mav.command_long_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_HOME,
                0,  # confirmation
                0,  # use current position
                0, 0, 0,  # reserved
                lat, lon, alt  # lat, lon, alt
            )
            
            self.logger.info(f"Home position set: {position}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting home position: {e}")
            return False

    def return_to_launch(self) -> bool:
        """Return to launch position"""
        try:
            command = DroneCommand(
                timestamp=time.time(),
                command_type="return_to_launch",
                parameters={}
            )
            self._log_command(command)
            
            if self.simulation_mode:
                self.logger.info("[SIM] Returning to launch")
                # Simulate return to origin
                self._simulate_movement(np.array([0.0, 0.0, 2.0]), duration=5.0)
                return True
            
            if not self.mavlink_connection:
                return False
            
            # Send RTL command
            self.mavlink_connection.mav.command_long_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
                0,  # confirmation
                0, 0, 0, 0, 0, 0, 0  # param1-7
            )
            
            self.logger.info("Return to launch command sent")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during RTL: {e}")
            return False
