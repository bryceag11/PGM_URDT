# Probabilistic Graph Models for Universal Robots Digital Twins
A ROS2 package implementing a digital twin for Universal Robots (UR) manipulators using Probabilistic Graphical Models (PGMs). This digital twin continuously estimates robot parameters, detects potential faults, and provides real-time monitoring through a web dashboard.

## Features

- Real-time parameter estimation using Dynamic Bayesian Networks and Unscented Kalman Filter
- Continuous monitoring of:
  - Joint friction coefficients
  - Kinematic parameters
  - Mass parameters
  - Damping coefficients
  - Health indicators
- Predictive maintenance through fault detection and forecasting
- Web-based dashboard for real-time monitoring
- Integration with ROS2 and UR Robot Driver

## Dependencies

### ROS2 Dependencies
- ROS2 Humble
- UR Robot Driver for ROS2
- Controller Manager
- Robot State Publisher
- RViz2
- Joint State Publisher

### Python Dependencies
- filterpy
- matplotlib
- scipy
- numpy
- pgmpy
- pomegranate
- tabulate
- psutil

## Installation

1. Create a ROS2 workspace (if you don't have one):
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

2. Clone this repository:
```bash
git clone https://github.com/your-username/ur_digital_twin.git
```

3. Make the setup script executable and run it:
```bash
cd ur_digital_twin
chmod +x setup_dependencies.sh
./setup_dependencies.sh
```

4. Build the package:
```bash
cd ~/ros2_ws
colcon build --packages-select ur_digital_twin
```

5. Source the workspace:
```bash
source ~/ros2_ws/install/setup.bash
```

## Package Structure
```
ur_digital_twin/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── setup_dependencies.sh
├── launch/
│   └── digital_twin.launch.py
├── msg/
│   ├── FaultDetection.msg
│   ├── HealthStatus.msg
│   └── ParameterState.msg
└── ur_digital_twin/
    ├── dbn_model.py
    ├── digital_twin_base.py
    ├── digital_twin_node.py
    ├── inference_algorithms.py
    ├── parameter_evaluation.py
    └── web_dashboard.py
```

## Usage

### With a Real UR Robot

1. Launch the digital twin with your robot's IP address:
```bash
ros2 launch ur_digital_twin digital_twin.launch.py robot_ip:=192.168.1.100 ur_type:=ur5e
```

2. Start the web dashboard:
```bash
ros2 run ur_digital_twin web_dashboard
```

3. Open a web browser and navigate to:
```
http://localhost:8080
```

### Testing Without a Robot

1. Launch the digital twin in simulation mode:
```bash
ros2 launch ur_digital_twin digital_twin.launch.py robot_ip:=0.0.0.0 ur_type:=ur5e
```

2. Publish mock joint states (in a new terminal):
```bash
ros2 topic pub /joint_states sensor_msgs/msg/JointState '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ""}, name: ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], velocity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], effort: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}' -r 10
```

## Key Components

### Digital Twin Base (`digital_twin_base.py`)
Implements the core digital twin functionality including parameter estimation and fault detection.

### DBN Model (`dbn_model.py`)
Implements the Dynamic Bayesian Network structure for parameter evolution and relationships.

### Parameter Evaluation (`parameter_evaluation.py`)
Tools for evaluating the digital twin's parameter estimation performance.

### Web Dashboard (`web_dashboard.py`)
Real-time web interface for monitoring robot parameters and health status.

### Inference Algorithms (`inference_algorithms.py`)
Implementation of various inference methods (UKF, EKF, PF, VI) for parameter estimation.

## Custom Messages

### ParameterState.msg
Contains current parameter estimates and their uncertainties.

### HealthStatus.msg
Reports robot health indicators and predictions.

### FaultDetection.msg
Reports detected faults and their severity.

## Configuration

The digital twin can be configured through various launch file parameters:

```bash
ros2 launch ur_digital_twin digital_twin.launch.py \
    robot_ip:=192.168.1.100 \
    ur_type:=ur5e \
    update_rate:=10.0 \
    launch_rviz:=true \
    auto_save:=true \
    save_dir:=/path/to/save
```

## Web Dashboard Features

- Real-time parameter visualization
- Health status monitoring
- Fault detection alerts
- Parameter trend analysis
- Predictive maintenance indicators



```
