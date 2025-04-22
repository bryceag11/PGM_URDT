#!/bin/bash
# ~/ros2_ws/src/ur_digital_twin/setup_dependencies.sh

# Make this script executable
# chmod +x ~/ros2_ws/src/ur_digital_twin/setup_dependencies.sh

echo "Setting up dependencies for UR Robot Digital Twin"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install filterpy matplotlib scipy numpy pgmpy pomegranate tabulate psutil

# Install ROS2 dependencies
echo "Installing ROS2 dependencies..."
sudo apt update
sudo apt install -y \
    ros-humble-ur-robot-driver \
    ros-humble-ur-description \
    ros-humble-ur-msgs \
    ros-humble-controller-manager \
    ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher \
    ros-humble-rviz2 \
    ros-humble-xacro \
    python3-websockets \ 
    python3-aiohttp      

# Create necessary directories
echo "Creating package structure..."

# Create main package directory
mkdir -p ~/ros2_ws/src/ur_digital_twin/ur_digital_twin

# Create web assets directory
mkdir -p ~/ros2_ws/src/ur_digital_twin/ur_digital_twin/web

# Create launch directory
mkdir -p ~/ros2_ws/src/ur_digital_twin/launch

# Create config directory for RViz
mkdir -p ~/ros2_ws/src/ur_digital_twin/config

# Create resource directory and package file
mkdir -p ~/ros2_ws/src/ur_digital_twin/resource
echo "ur_digital_twin" > ~/ros2_ws/src/ur_digital_twin/resource/ur_digital_twin

# Create an __init__.py file
touch ~/ros2_ws/src/ur_digital_twin/ur_digital_twin/__init__.py

# Create required message directories
mkdir -p ~/ros2_ws/src/ur_digital_twin/msg

echo "Setup complete! You can now build the package with:"
echo "cd ~/ros2_ws && colcon build --packages-select ur_digital_twin"