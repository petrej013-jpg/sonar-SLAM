"""
Configuration settings for the SLAM application.

This module contains all configurable parameters for the application,
organized by category and with appropriate type hints.
"""
import os
from typing import Dict, Any
from pathlib import Path

# Paths and file locations
# ------------------------
DATA_FILEPATH = '/app/slam_data'
SONAR_FILEPATH = '/app/sonar_data'
SONAR_FILE = 'sonar_full_scan2.h5'

# Network configuration
# --------------------
DOCKER_HOST = 'host.docker.internal'
VEHICLE_IP = '192.168.2.2'
UDP_PORT = f"{VEHICLE_IP}:9092"

# Device configuration
# -------------------
PING_BRIDGE = 'UDP 9092'  # UDP bridge for Ping360
PING_DEVICE = '/dev/ttyUSB0'  # Serial device for Ping360
VIDEO_STREAM = 'udp://192.168.2.1:5600'  # Video stream URL
VIDEO_PATH = '/dev/video2'  # Video device path

# Application behavior
# -------------------
LIVE_SONAR = False  # Whether to use live sonar or recordings

# Physical properties
# ------------------
WATER_SOS = 1481  # Speed of sound in water (m/s)

# Sonar configuration
# ------------------


class SonarConfig:
    """Configuration for the Ping360 sonar."""
    TRANSMIT_DURATION = 25  # Transmit duration in μs
    SAMPLE_PERIOD = 480  # Sample period in ns/25
    TRANSMIT_FREQUENCY = 750  # Transmit frequency in kHz
    MIN_RANGE = 0.75  # Minimum operating range in meters

# CFAR detector configuration
# --------------------------


class CFARConfig:
    """Configuration for the CFAR detector."""
    Ntc = 40  # Number of training cells
    Ngc = 10  # Number of guard cells
    Pfa = 0.01  # Probability of false alarm
    ALGORITHM = "GOCA"  # CFAR algorithm type (CA, SOCA, GOCA, OS)
    THRESHOLD = 30  # Amplitude threshold

# Video configuration
# ------------------


class VideoConfig:
    """Configuration for video processing."""
    FRAME_WIDTH = 1920
    FRAME_HEIGHT = 1080
    FPS = 30
    FOCAL_LENGTH = 1188
    PRINCIPAL_POINT = (960.0, 540.0)

# Load environment-specific settings
# ---------------------------------


def load_env_settings() -> Dict[str, Any]:
    """
    Load environment-specific settings.

    Returns:
        Dictionary of settings from environment variables
    """
    env_settings = {}

    # Override settings from environment variables if present
    if os.environ.get('VEHICLE_IP'):
        env_settings['VEHICLE_IP'] = os.environ.get('VEHICLE_IP')

    if os.environ.get('LIVE_SONAR'):
        env_settings['LIVE_SONAR'] = os.environ.get(
            'LIVE_SONAR').lower() in ('true', '1', 'yes')

    if os.environ.get('DATA_PATH'):
        env_settings['DATA_FILEPATH'] = os.environ.get('DATA_PATH')

    return env_settings


# Apply environment settings
env_vars = load_env_settings()
for key, value in env_vars.items():
    if key in globals():
        globals()[key] = value

# Ensure data directories exist
for path in [DATA_FILEPATH, SONAR_FILEPATH]:
    Path(path).mkdir(parents=True, exist_ok=True)
