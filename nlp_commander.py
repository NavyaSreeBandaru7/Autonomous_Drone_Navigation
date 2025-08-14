#!/usr/bin/env python3
"""
NLP Commander Module
Natural Language Processing for voice commands and intelligent interaction
"""

import logging
import re
import time
from typing import Dict, List, Optional, Tuple
import threading
import queue
from dataclasses import dataclass
from enum import Enum

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    logging.warning("speech_recognition not available")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning("pyttsx3 not available")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available, using basic NLP")

class CommandType(Enum):
    TAKEOFF = "takeoff"
    LAND = "land"
    GOTO = "goto"
    HOVER = "hover"
    EMERGENCY = "emergency"
    RETURN_HOME = "return_home"
    SET_MODE = "set_mode"
    STATUS = "status"
    FOLLOW = "follow"
    PATROL = "patrol"
    UNKNOWN = "unknown"

@dataclass
class ParsedCommand:
    type: CommandType
    confidence: float
    parameters: Dict
    raw_text: str
    timestamp: float

class NLPCommander:
    """Natural Language Processing system for drone commands"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Speech recognition setup
        self.recognizer = None
        self.microphone = None
        self.listening = False
        
        # Text-to-speech setup
        self.tts_engine = None
        
        # NLP models
        self.intent_classifier = None
        self.ner_model = None
        
        # Command patterns
        self.command_patterns = self._init_command_patterns()
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Threading
        self.listen_thread = None
        self.processing_thread = None
        self.running = False
        
        # Command history
        self.command_history = []
        self.max_history = 100
        
        # Performance tracking
        self.recognition_times = []
        self.confidence_scores = []
        
        self._initialize_components()
        
        self.logger.info("NLP Commander initialized")

    def _initialize_components(self):
        """Initialize NLP components"""
        try:
            # Initialize speech recognition
            if SPEECH_AVAILABLE:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                
                # Adjust for ambient noise
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.logger.info("Speech recognition initialized")
            
            # Initialize text-to-speech
            if TTS_AVAILABLE:
                self.tts_engine = pyttsx3.init()
                self._configure_tts()
                self.logger.info("Text-to-speech initialized")
            
            # Initialize advanced NLP models
            if TRANSFORMERS_AVAILABLE:
                self._init_ml_models()
            
        except Exception as e:
            self.logger.error(f"Error initializing NLP components: {e}")

    def _configure_tts(self):
        """Configure text-to-speech settings"""
        try:
            if self.tts_engine:
                # Set voice properties
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    self.tts_engine.setProperty('voice', voices[0].id)
                
                # Set speech rate
                self.tts_engine.setProperty('rate', 150)
                
                # Set volume
                self.tts_engine.setProperty('volume', 0.8)
                
        except Exception as e:
            self.logger.error(f"Error configuring TTS: {e}")

    def _init_ml_models(self):
        """Initialize machine learning models for advanced NLP"""
        try:
            # Intent classification model
            self.intent_classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                tokenizer="microsoft/DialoGPT-medium"
            )
            
            # Named Entity Recognition
            self.ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
            
            self.logger.info("ML NLP models loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not load ML models: {e}")
            self.intent_classifier = None
            self.ner_model = None

    def _init_command_patterns(self) -> Dict:
        """Initialize command recognition patterns"""
        patterns = {
            CommandType.TAKEOFF: [
                r'take\s*off',
                r'launch',
                r'lift\s*off',
                r'start\s*flying',
                r'go\s*up',
                r'ascend'
            ],
            CommandType.LAND: [
                r'land',
                r'come\s*down',
                r'descend',
                r'touch\s*down',
                r'stop\s*flying'
            ],
            CommandType.GOTO: [
                r'go\s*to\s*(-?\d+\.?\d*)\s*(-?\d+\.?\d*)\s*(-?\d+\.?\d*)',
                r'move\s*to\s*(-?\d+\.?\d*)\s*(-?\d+\.?\d*)\s*(-?\d+\.?\d*)',
                r'fly\s*to\s*(-?\d+\.?\d*)\s*(-?\d+\.?\d*)\s*(-?\d+\.?\d*)',
                r'navigate\s*to\s*(-?\d+\.?\d*)\s*(-?\d+\.?\d*)\s*(-?\d+\.?\d*)',
                r'position\s*(-?\d+\.?\d*)\s*(-?\d+\.?\d*)\s*(-?\d+\.?\d*)'
            ],
            CommandType.HOVER: [
                r'hover',
                r'stay\s*here',
                r'hold\s*position',
                r'stop',
                r'pause'
            ],
            CommandType.EMERGENCY: [
                r'emergency',
                r'stop\s*immediately',
                r'abort',
                r'help',
                r'danger'
            ],
            CommandType.RETURN_HOME: [
                r'return\s*home',
                r'go\s*home',
                r'come\s*back',
                r'return\s*to\s*base',
                r'rtl'
            ],
            CommandType.SET_MODE: [
                r'set\s*mode\s*(\w+)',
                r'change\s*mode\s*to\s*(\w+)',
                r'switch\s*to\s*(\w+)\s*mode',
                r'mode\s*(\w+)'
            ],
            CommandType.STATUS: [
                r'status',
                r'how\s*are\s*you',
                r'report',
                r'what\'s\s*your\s*status',
                r'battery',
                r'position'
            ],
            CommandType.FOLLOW: [
                r'follow\s*me',
                r'track\s*me',
                r'stay\s*with\s*me',
                r'come\s*with\s*me'
            ],
            CommandType.PATROL: [
                r'patrol',
                r'survey\s*area',
                r'search\s*pattern',
                r'reconnaissance'
            ]
        }
        
        return patterns

    def start_listening(self) -> bool:
        """Start voice command listening"""
        try:
            if not SPEECH_AVAILABLE:
                self.logger.warning("Speech recognition not available")
                return False
            
            self.running = True
            
            # Start listening thread
            self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.listen_thread.start()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            self.listening = True
            self.logger.info("Voice command listening started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting voice listening: {e}")
            return False

    def stop_listening(self):
        """Stop voice command listening"""
        try:
            self.running = False
            self.listening = False
            
            if self.listen_thread and self.listen_thread.is_alive():
                self.listen_thread.join(timeout=2.0)
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
            
            self.logger.info("Voice command listening stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping voice listening: {e}")

    def _listen_loop(self):
        """Background thread for continuous listening"""
        while self.running:
            try:
                if not self.recognizer or not self.microphone:
                    time.sleep(1.0)
                    continue
                
                # Listen for audio
                with self.microphone as source:
                    # Listen for wake word or continuous listening
                    if self.config.use_wake_word:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    else:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                
                # Add to processing queue
                self.audio_queue.put(audio)
                
            except sr.WaitTimeoutError:
                # Normal timeout, continue listening
                pass
            except Exception as e:
                self.logger.error(f"Error in listen loop: {e}")
                time.sleep(0.5)

    def _processing_loop(self):
        """Background thread for audio processing"""
        while self.running:
            try:
                # Get audio from queue
                audio = self.audio_queue.get(timeout=1.0)
                
                # Process audio
                start_time = time.time()
                text = self._speech_to_text(audio)
                processing_time = time.time() - start_time
                
                if text:
                    self.logger.info(f"Recognized: '{text}' (took {processing_time:.2f}s)")
                    
                    # Parse command
                    parsed_command = self.parse_command(text)
                    
                    if parsed_command and parsed_command.type != CommandType.UNKNOWN:
                        # Add to response queue
                        self.response_queue.put(parsed_command)
                        
                        # Provide audio feedback
                        self._provide_feedback(parsed_command)
                    
                    # Track performance
                    self.recognition_times.append(processing_time)
                    if len(self.recognition_times) > 100:
                        self.recognition_times.pop(0)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)

    def _speech_to_text(self, audio) -> Optional[str]:
        """Convert speech audio to text"""
        try:
            # Try Google Speech Recognition
            try:
                text = self.recognizer.recognize_google(audio)
                return text.lower()
            except sr.UnknownValueError:
                return None
            except sr.RequestError:
                # Fallback to offline recognition
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    return text.lower()
                except:
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error in speech to text: {e}")
            return None

    def listen_for_command(self, timeout: float = 1.0) -> Optional[str]:
        """Listen for a single voice command"""
        try:
            if not self.listening:
                return None
            
            # Check response queue
            try:
                parsed_command = self.response_queue.get(timeout=timeout)
                return parsed_command.raw_text
            except queue.Empty:
                return None
                
        except Exception as e:
            self.logger.error(f"Error listening for command: {e}")
            return None

    def parse_command(self, text: str) -> Optional[ParsedCommand]:
        """Parse text command into structured format"""
        try:
            start_time = time.time()
            
            # Clean and normalize text
            text = text.lower().strip()
            
            # Try advanced ML parsing first
            if self.intent_classifier:
                ml_result = self._parse_with_ml(text)
                if ml_result:
                    return ml_result
            
            # Fall back to pattern matching
            pattern_result = self._parse_with_patterns(text)
            
            # Add to command history
            if pattern_result:
                self.command_history.append(pattern_result)
                if len(self.command_history) > self.max_history:
                    self.command_history.pop(0)
            
            return pattern_result
            
        except Exception as e:
            self.logger.error(f"Error parsing command: {e}")
            return None

    def _parse_with_ml(self, text: str) -> Optional[ParsedCommand]:
        """Parse command using machine learning models"""
        try:
            # Intent classification
            intent_result = self.intent_classifier(text)
            
            # Extract entities
            entities = self.ner_model(text) if self.ner_model else []
            
            # Map to command types (simplified)
            confidence = intent_result[0]['score'] if intent_result else 0.5
            
            # This would be more sophisticated in a real implementation
            command_type = self._map_intent_to_command(text, entities)
            parameters = self._extract_parameters_ml(text, entities)
            
            if command_type != CommandType.UNKNOWN:
                return ParsedCommand(
                    type=command_type,
                    confidence=confidence,
                    parameters=parameters,
                    raw_text=text,
                    timestamp=time.time()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in ML parsing: {e}")
            return None

    def _parse_with_patterns(self, text: str) -> Optional[ParsedCommand]:
        """Parse command using regex patterns"""
        try:
            best_match = None
            best_confidence = 0.0
            best_type = CommandType.UNKNOWN
            best_params = {}
            
            for command_type, patterns in self.command_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        # Calculate confidence based on match quality
                        confidence = len(match.group(0)) / len(text)
                        confidence = min(1.0, confidence * 1.5)  # Boost confidence
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_type = command_type
                            best_match = match
                            
                            # Extract parameters
                            best_params = self._extract_parameters_pattern(command_type, match, text)
            
            if best_confidence > 0.3:  # Minimum confidence threshold
                return ParsedCommand(
                    type=best_type,
                    confidence=best_confidence,
                    parameters=best_params,
                    raw_text=text,
                    timestamp=time.time()
                )
            
            return ParsedCommand(
                type=CommandType.UNKNOWN,
                confidence=0.0,
                parameters={},
                raw_text=text,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error in pattern parsing: {e}")
            return None

    def _map_intent_to_command(self, text: str, entities: List) -> CommandType:
        """Map ML intent to command type"""
        # Simplified mapping - would be more sophisticated in practice
        keywords = {
            'takeoff': CommandType.TAKEOFF,
            'land': CommandType.LAND,
            'go': CommandType.GOTO,
            'move': CommandType.GOTO,
            'hover': CommandType.HOVER,
            'stop': CommandType.HOVER,
            'emergency': CommandType.EMERGENCY,
            'home': CommandType.RETURN_HOME,
            'return': CommandType.RETURN_HOME,
            'status': CommandType.STATUS,
            'follow': CommandType.FOLLOW,
            'patrol': CommandType.PATROL
        }
        
        for keyword, cmd_type in keywords.items():
            if keyword in text:
                return cmd_type
        
        return CommandType.UNKNOWN

    def _extract_parameters_ml(self, text: str, entities: List) -> Dict:
        """Extract parameters using ML entities"""
        parameters = {}
        
        try:
            # Extract numbers for coordinates
            numbers = re.findall(r'-?\d+\.?\d*', text)
            
            if len(numbers) >= 3:
                parameters['target'] = (float(numbers[0]), float(numbers[1]), float(numbers[2]))
            elif len(numbers) == 2:
                parameters['target'] = (float(numbers[0]), float(numbers[1]), 0.0)
            
            # Extract mode names
            mode_match = re.search(r'mode\s+(\w+)', text)
            if mode_match:
                parameters['mode'] = mode_match.group(1)
            
            return parameters
            
        except Exception as e:
            self.logger.error(f"Error extracting ML parameters: {e}")
            return {}

    def _extract_parameters_pattern(self, command_type: CommandType, match, text: str) -> Dict:
        """Extract parameters from regex match"""
        parameters = {}
        
        try:
            if command_type == CommandType.GOTO:
                groups = match.groups()
                if len(groups) >= 3:
                    parameters['target'] = (float(groups[0]), float(groups[1]), float(groups[2]))
                elif len(groups) == 2:
                    parameters['target'] = (float(groups[0]), float(groups[1]), 2.0)  # Default altitude
            
            elif command_type == CommandType.SET_MODE:
                groups = match.groups()
                if groups:
                    parameters['mode'] = groups[0]
            
            elif command_type == CommandType.TAKEOFF:
                # Look for altitude specification
                altitude_match = re.search(r'(\d+\.?\d*)\s*(?:meter|metre|m|foot|feet|ft)', text)
                if altitude_match:
                    altitude = float(altitude_match.group(1))
                    # Convert feet to meters if needed
                    if 'foot' in altitude_match.group(0) or 'feet' in altitude_match.group(0) or 'ft' in altitude_match.group(0):
                        altitude *= 0.3048
                    parameters['altitude'] = altitude
                else:
                    parameters['altitude'] = 2.0  # Default altitude
            
            return parameters
            
        except Exception as e:
            self.logger.error(f"Error extracting pattern parameters: {e}")
            return {}

    def _provide_feedback(self, command: ParsedCommand):
        """Provide audio feedback for recognized command"""
        try:
            if not TTS_AVAILABLE or not self.tts_engine:
                return
            
            # Generate appropriate response
            if command.type == CommandType.TAKEOFF:
                response = "Taking off"
            elif command.type == CommandType.LAND:
                response = "Landing"
            elif command.type == CommandType.GOTO:
                if 'target' in command.parameters:
                    x, y, z = command.parameters['target']
                    response = f"Moving to position {x:.1f}, {y:.1f}, {z:.1f}"
                else:
                    response = "Moving to target position"
            elif command.type == CommandType.HOVER:
                response = "Hovering in place"
            elif command.type == CommandType.EMERGENCY:
                response = "Emergency stop activated"
            elif command.type == CommandType.RETURN_HOME:
                response = "Returning home"
            elif command.type == CommandType.STATUS:
                response = "Checking status"
            elif command.type == CommandType.FOLLOW:
                response = "Follow mode activated"
            elif command.type == CommandType.PATROL:
                response = "Patrol mode activated"
            else:
                response = "Command understood"
            
            # Speak the response
            self.tts_engine.say(response)
            self.tts_engine.runAndWait()
            
        except Exception as e:
            self.logger.error(f"Error providing feedback: {e}")

    def speak(self, text: str):
        """Speak text using TTS"""
        try:
            if TTS_AVAILABLE and self.tts_engine:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            else:
                self.logger.info(f"TTS: {text}")
                
        except Exception as e:
            self.logger.error(f"Error speaking text: {e}")

    def get_command_history(self) -> List[ParsedCommand]:
        """Get command history"""
        return self.command_history.copy()

    def get_performance_stats(self) -> Dict:
        """Get NLP performance statistics"""
        try:
            if not self.recognition_times:
                return {}
            
            return {
                'avg_recognition_time': float(sum(self.recognition_times) / len(self.recognition_times)),
                'max_recognition_time': float(max(self.recognition_times)),
                'min_recognition_time': float(min(self.recognition_times)),
                'total_commands': len(self.command_history),
                'recognition_success_rate': len([c for c in self.command_history if c.type != CommandType.UNKNOWN]) / max(1, len(self.command_history))
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance stats: {e}")
            return {}

    def clear_history(self):
        """Clear command history"""
        self.command_history.clear()
        self.logger.info("Command history cleared")

    def add_custom_pattern(self, command_type: CommandType, pattern: str):
        """Add custom command pattern"""
        try:
            if command_type not in self.command_patterns:
                self.command_patterns[command_type] = []
            
            self.command_patterns[command_type].append(pattern)
            self.logger.info(f"Added custom pattern for {command_type.value}: {pattern}")
            
        except Exception as e:
            self.logger.error(f"Error adding custom pattern: {e}")

    def test_command(self, text: str) -> Optional[ParsedCommand]:
        """Test command parsing without executing"""
        return self.parse_command(text)

    def get_status(self) -> Dict:
        """Get NLP commander status"""
        return {
            'listening': self.listening,
            'speech_available': SPEECH_AVAILABLE,
            'tts_available': TTS_AVAILABLE,
            'ml_models_available': TRANSFORMERS_AVAILABLE,
            'command_patterns': len(self.command_patterns),
            'command_history_size': len(self.command_history),
            'performance': self.get_performance_stats()
        }
