#!/usr/bin/env python3
"""
Setup script for Autonomous Drone Navigation System
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="autonomous-drone-navigation",
    version="1.0.0",
    author="AI Assistant",
    author_email="ai@dronenavigation.com",
    description="Advanced computer vision-based autonomous drone navigation system",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/autonomous-drone-navigation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: System :: Hardware :: Hardware Drivers",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "sphinx>=4.0.0",
        ],
        "gpu": [
            "torch[cuda]>=1.9.0",
            "torchvision[cuda]>=0.10.0",
        ],
        "ros": [
            "rospy",
            "geometry_msgs",
            "sensor_msgs",
        ],
        "advanced": [
            "tensorflow>=2.8.0",
            "keras>=2.8.0",
            "open3d>=0.15.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "drone-nav=main:main",
            "drone-sim=scripts.simulation:main",
            "drone-calibrate=scripts.calibrate_camera:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
        "config": ["*.json"],
        "models": ["*.pt", "*.onnx"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/autonomous-drone-navigation/issues",
        "Source": "https://github.com/yourusername/autonomous-drone-navigation",
        "Documentation": "https://github.com/yourusername/autonomous-drone-navigation/wiki",
    },
)
