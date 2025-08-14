#!/bin/bash

# Autonomous Drone Navigation System Installation Script
# This script automates the installation process for the drone navigation system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="autonomous-drone-navigation"
PYTHON_MIN_VERSION="3.8"
VENV_NAME="drone_env"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to compare version numbers
version_ge() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command_exists apt-get; then
            echo "ubuntu"
        elif command_exists yum; then
            echo "centos"
        elif command_exists pacman; then
            echo "arch"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to install system dependencies
install_system_deps() {
    local os=$(detect_os)
    print_status "Installing system dependencies for $os..."
    
    case $os in
        "ubuntu")
            sudo apt-get update
            sudo apt-get install -y \
                python3 python3-pip python3-venv python3-dev \
                build-essential cmake pkg-config \
                libjpeg-dev libtiff5-dev libpng-dev \
                libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
                libxvidcore-dev libx264-dev libfontconfig1-dev \
                libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev \
                libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran \
                libhdf5-dev libhdf5-serial-dev libhdf5-103 \
                libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5 \
                alsa-utils portaudio19-dev espeak espeak-data \
                libespeak1 libespeak-dev ffmpeg git curl wget
            ;;
        "centos")
            sudo yum update -y
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                python3 python3-pip python3-devel \
                cmake pkgconfig libjpeg-turbo-devel libpng-devel \
                libtiff-devel libv4l-devel ffmpeg-devel \
                portaudio-devel espeak espeak-devel \
                git curl wget
            ;;
        "macos")
            if ! command_exists brew; then
                print_status "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew update
            brew install python@3.9 cmake pkg-config jpeg libpng libtiff \
                         ffmpeg portaudio espeak git curl wget
            ;;
        "windows")
            print_warning "Windows detected. Please manually install:"
            print_warning "1. Python 3.8+ from python.org"
            print_warning "2. Visual Studio Build Tools"
            print_warning "3. Git for Windows"
            print_warning "4. PortAudio and other audio libraries"
            ;;
        *)
            print_warning "Unknown OS. Please install dependencies manually."
            ;;
    esac
}

# Function to check Python version
check_python() {
    print_status "Checking Python installation..."
    
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python $PYTHON_MIN_VERSION or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version | cut -d' ' -f2)
    print_status "Found Python $PYTHON_VERSION"
    
    if ! version_ge "$PYTHON_VERSION" "$PYTHON_MIN_VERSION"; then
        print_error "Python $PYTHON_MIN_VERSION or higher is required. Found $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Python version check passed"
}

# Function to create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ -d "$VENV_NAME" ]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf "$VENV_NAME"
    fi
    
    $PYTHON_CMD -m venv "$VENV_NAME"
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate" 2>/dev/null || source "$VENV_NAME/Scripts/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    print_success "Virtual environment created and activated"
}

# Function to install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Install dependencies from requirements.txt
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        print_error "requirements.txt not found"
        exit 1
    fi
    
    print_success "Python dependencies installed"
}

# Function to download models
download_models() {
    print_status "Setting up models directory..."
    
    mkdir -p models
    
    # Download YOLO model if not exists
    if [ ! -f "models/yolov8n.pt" ]; then
        print_status "Downloading YOLOv8 model..."
        python -c "
from ultralytics import YOLO
import os
os.makedirs('models', exist_ok=True)
model = YOLO('yolov8n.pt')
model.save('models/yolov8n.pt')
print('YOLOv8 model downloaded successfully')
" 2>/dev/null || print_warning "Could not download YOLO model automatically"
    fi
    
    print_success "Models setup completed"
}

# Function to setup configuration
setup_config() {
    print_status "Setting up configuration..."
    
    mkdir -p config logs missions data
    
    # Copy template config if config doesn't exist
    if [ ! -f "config/drone_config.json" ]; then
        if [ -f "config/drone_config_template.json" ]; then
            cp config/drone_config_template.json config/drone_config.json
            print_success "Configuration template copied"
        else
            print_warning "Configuration template not found"
        fi
    fi
    
    # Set permissions
    chmod +x scripts/*.py 2>/dev/null || true
    
    print_success "Configuration setup completed"
}

# Function to run tests
run_tests() {
    print_status "Running basic tests..."
    
    # Test imports
    python -c "
import sys
import cv2
import numpy as np
import logging

print(f'Python: {sys.version}')
print(f'OpenCV: {cv2.__version__}')
print(f'NumPy: {np.__version__}')

# Test camera
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print('Camera: Available')
        cap.release()
    else:
        print('Camera: Not available (this is OK for headless systems)')
except:
    print('Camera: Error testing camera')

print('Basic dependency test passed!')
"
    
    if [ $? -eq 0 ]; then
        print_success "Basic tests passed"
    else
        print_warning "Some tests failed, but installation may still work"
    fi
}

# Function to display final instructions
show_final_instructions() {
    print_success "Installation completed successfully!"
    echo
    echo "=============================================="
    echo "  Autonomous Drone Navigation System Ready  "
    echo "=============================================="
    echo
    print_status "To get started:"
    echo "1. Activate the virtual environment:"
    echo "   source $VENV_NAME/bin/activate  # On Linux/Mac"
    echo "   $VENV_NAME\\Scripts\\activate      # On Windows"
    echo
    echo "2. Configure your drone connection in:"
    echo "   config/drone_config.json"
    echo
    echo "3. Run the system:"
    echo "   python main.py"
    echo
    echo "4. For simulation mode (no real drone needed):"
    echo "   Set 'simulation_mode': true in config"
    echo
    print_status "Available commands after starting:"
    echo "- Voice: 'take off', 'land', 'go to 5 3 2', 'hover', 'emergency'"
    echo "- Keyboard: takeoff, land, goto <x> <y> <z>, hover, emergency, status"
    echo
    print_status "Documentation:"
    echo "- README.md for detailed instructions"
    echo "- GitHub: https://github.com/yourusername/$PROJECT_NAME"
    echo
    print_warning "Safety reminders:"
    echo "- Always follow local aviation regulations"
    echo "- Test in simulation mode first"
    echo "- Maintain visual line of sight with your drone"
    echo "- Have a manual override ready"
    echo
}

# Main installation function
main() {
    echo "=============================================="
    echo "  Autonomous Drone Navigation Installation   "
    echo "=============================================="
    echo
    
    # Check if running as root (not recommended)
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root is not recommended"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Parse command line arguments
    INSTALL_SYSTEM_DEPS=true
    SKIP_TESTS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-system-deps)
                INSTALL_SYSTEM_DEPS=false
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --no-system-deps  Skip system dependency installation"
                echo "  --skip-tests      Skip running tests"
                echo "  --help           Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Installation steps
    if [ "$INSTALL_SYSTEM_DEPS" = true ]; then
        install_system_deps
    fi
    
    check_python
    create_venv
    install_python_deps
    download_models
    setup_config
    
    if [ "$SKIP_TESTS" = false ]; then
        run_tests
    fi
    
    show_final_instructions
}

# Handle interruption
trap 'print_error "Installation interrupted"; exit 1' INT TERM

# Run main function
main "$@"
