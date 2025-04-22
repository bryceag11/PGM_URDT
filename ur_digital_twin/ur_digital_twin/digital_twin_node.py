# ~/ros2_ws/src/ur_digital_twin/ur_digital_twin/digital_twin_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np
import os
import threading
import time
from datetime import datetime

# Import custom messages
from ur_digital_twin.msg import ParameterState, HealthStatus, FaultDetection

# Import digital twin base class
from ur_digital_twin.digital_twin_base import URRobotDigitalTwin

class URDigitalTwinNode(Node):
    """
    ROS2 Node for UR Robot Digital Twin
    """
    def __init__(self):
        super().__init__('ur_digital_twin')
        
        # Log startup
        self.get_logger().info('Initializing UR Digital Twin Node')
        
        # Parameters
        self.declare_parameter('update_rate', 10.0)  # Hz
        self.declare_parameter('model_type', 'dbn')
        self.declare_parameter('save_dir', '/tmp/ur_digital_twin')
        self.declare_parameter('auto_save', True)
        self.declare_parameter('load_from_file', '')
        
        # Get parameters
        self.update_rate = self.get_parameter('update_rate').value
        self.model_type = self.get_parameter('model_type').value
        self.save_dir = self.get_parameter('save_dir').value
        self.auto_save = self.get_parameter('auto_save').value
        self.load_file = self.get_parameter('load_from_file').value
        
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Initialize the digital twin
        self.digital_twin = URRobotDigitalTwin(model_type=self.model_type)
        
        # Load from file if specified
        if self.load_file and os.path.exists(self.load_file):
            self.get_logger().info(f'Loading digital twin state from {self.load_file}')
            self.digital_twin.load_state(self.load_file)
        
        # Joint state mapping
        self.joint_name_map = {
            'shoulder_pan_joint': 0,
            'shoulder_lift_joint': 1,
            'elbow_joint': 2,
            'wrist_1_joint': 3,
            'wrist_2_joint': 4,
            'wrist_3_joint': 5
        }
        
        # Initialize observation storage
        self.latest_observations = {
            'joint_pos': np.zeros(6),
            'joint_vel': np.zeros(6),
            'joint_curr': np.zeros(6),
            'ext_force': np.zeros(6)
        }
        
        # Initialize update lock
        self.update_lock = threading.Lock()
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)
            
        self.ft_sensor_sub = self.create_subscription(
            WrenchStamped,
            '/force_torque_sensor_controller/ft_sensor_readings',
            self.ft_sensor_callback,
            10)
        
        # Use a separate topic for joint currents as they might come from a different source
        # In a real UR setup, these would come from robot_state
        self.joint_current_sub = self.create_subscription(
            JointState,
            '/joint_currents',  # This topic might be different based on your setup
            self.joint_current_callback,
            10)
        
        # Publishers
        self.parameter_pub = self.create_publisher(
            ParameterState,
            '~/parameters',
            10)
            
        self.health_pub = self.create_publisher(
            HealthStatus,
            '~/health_status',
            10)
            
        self.fault_pub = self.create_publisher(
            FaultDetection,
            '~/fault_detection',
            10)
            
        self.visualization_pub = self.create_publisher(
            MarkerArray,
            '~/visualization',
            10)
        
        # Update timer
        self.update_timer = self.create_timer(
            1.0 / self.update_rate,
            self.update_callback)
            
        # Auto-save timer (every 5 minutes)
        if self.auto_save:
            self.save_timer = self.create_timer(
                300.0,
                self.auto_save_callback)
        
        self.get_logger().info('UR Digital Twin Node initialized and ready')
    
    def joint_state_callback(self, msg):
        """
        Process joint state messages (positions and velocities)
        """
        with self.update_lock:
            # Map joint names to indices
            for i, name in enumerate(msg.name):
                if name in self.joint_name_map:
                    idx = self.joint_name_map[name]
                    if i < len(msg.position):
                        self.latest_observations['joint_pos'][idx] = msg.position[i]
                    if i < len(msg.velocity):
                        self.latest_observations['joint_vel'][idx] = msg.velocity[i]
    
    def joint_current_callback(self, msg):
        """
        Process joint current messages
        """
        with self.update_lock:
            # Map joint names to indices
            for i, name in enumerate(msg.name):
                if name in self.joint_name_map:
                    idx = self.joint_name_map[name]
                    if i < len(msg.effort):  # Using effort field for currents
                        self.latest_observations['joint_curr'][idx] = msg.effort[i]
    
    def ft_sensor_callback(self, msg):
        """
        Process force/torque sensor messages
        """
        with self.update_lock:
            # Map wrench to a 6D vector [fx, fy, fz, tx, ty, tz]
            self.latest_observations['ext_force'] = np.array([
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z
            ])
    
    def update_callback(self):
        """
        Periodically update the digital twin
        """
        with self.update_lock:
            # Copy current observations
            observations = {k: v.copy() for k, v in self.latest_observations.items()}
        
        # Update digital twin
        parameters, fault_status = self.digital_twin.update(observations)
        
        # Publish updated information
        self.publish_parameters(parameters)
        self.publish_health_status()
        self.publish_fault_detection(fault_status)
        self.publish_visualization(parameters, fault_status)
    
    def publish_parameters(self, parameters):
        """
        Publish current parameter estimates
        """
        msg = ParameterState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Flatten parameters for publishing
        param_names = []
        means = []
        variances = []
        dimensions = []
        
        for param_name, (mean, cov) in parameters.items():
            param_names.append(param_name)
            means.extend(mean.tolist())
            
            # Extract diagonal of covariance as variance
            var = np.diag(cov).tolist()
            variances.extend(var)
            
            # Store dimension of this parameter
            dimensions.append(len(mean))
        
        msg.parameter_names = param_names
        msg.means = means
        msg.variances = variances
        msg.dimensions = dimensions
        msg.confidence = 0.95  # 95% confidence interval
        
        self.parameter_pub.publish(msg)
    
    def publish_health_status(self):
        """
        Publish health status information
        """
        health_info = self.digital_twin.get_health_status()
        
        msg = HealthStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.health_indicators = health_info['health_indicators'].tolist()
        
        # Add warning and critical thresholds for friction
        warning_thresholds = []
        critical_thresholds = []
        
        # For friction parameters
        warning_thresholds.extend(self.digital_twin.thresholds['f']['warning'].tolist())
        critical_thresholds.extend(self.digital_twin.thresholds['f']['critical'].tolist())
        
        # For health parameters
        warning_thresholds.extend(self.digital_twin.thresholds['z']['warning'].tolist())
        critical_thresholds.extend(self.digital_twin.thresholds['z']['critical'].tolist())
        
        msg.warning_thresholds = warning_thresholds
        msg.critical_thresholds = critical_thresholds
        
        # Prediction times (if available)
        if self.digital_twin.fault_status['detected']:
            # Time to warning/critical is already predicted in the digital twin
            predicted_times = [self.digital_twin.fault_status['time_to_failure']] * len(warning_thresholds)
            msg.predicted_times_to_warning = predicted_times
            msg.predicted_times_to_critical = predicted_times
        else:
            # Make predictions based on parameter trends
            # This is simplified - in practice would use proper forecasting
            msg.predicted_times_to_warning = [float('inf')] * len(warning_thresholds)
            msg.predicted_times_to_critical = [float('inf')] * len(critical_thresholds)
        
        # Get joint names
        msg.joint_names = list(self.joint_name_map.keys())
        
        self.health_pub.publish(msg)
    
    def publish_fault_detection(self, fault_status):
        """
        Publish fault detection information
        """
        msg = FaultDetection()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        msg.fault_detected = fault_status['detected']
        
        if fault_status['detected']:
            msg.fault_type = fault_status['type']
            msg.affected_joint = fault_status['joint']
            msg.severity = fault_status['severity']
            msg.confidence = fault_status['confidence']
            msg.estimated_time_to_failure = fault_status['time_to_failure']
        else:
            msg.fault_type = 'none'
            msg.affected_joint = -1
            msg.severity = 0.0
            msg.confidence = 0.0
            msg.estimated_time_to_failure = float('inf')
        
        self.fault_pub.publish(msg)
    
    def publish_visualization(self, parameters, fault_status):
        """
        Publish visualization markers
        """
        marker_array = MarkerArray()
        
        # Create markers for each joint's health
        for i in range(6):
            marker = Marker()
            marker.header.frame_id = "base_link"  # Adjust to your robot's base frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "joint_health"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Position markers along z-axis for visibility
            # In a real implementation, would position at actual joint locations
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.1 * (i + 1)
            
            marker.pose.orientation.w = 1.0
            
            # Size based on parameter variance
            variance = np.diag(parameters['f'][1])[i]
            marker.scale.x = 0.05 + variance * 10
            marker.scale.y = 0.05 + variance * 10
            marker.scale.z = 0.05 + variance * 10
            
            # Color based on parameter value relative to thresholds
            color = ColorRGBA()
            
            # Normalize friction value between min and critical threshold
            friction_value = parameters['f'][0][i]
            min_value = self.digital_twin.parameter_ranges['f']['min'][i]
            warning_value = self.digital_twin.thresholds['f']['warning'][i]
            critical_value = self.digital_twin.thresholds['f']['critical'][i]
            
            # Interpolate color from green to yellow to red
            if friction_value < warning_value:
                # Green to yellow
                t = (friction_value - min_value) / (warning_value - min_value)
                color.r = t
                color.g = 1.0
                color.b = 0.0
            else:
                # Yellow to red
                t = min(1.0, (friction_value - warning_value) / (critical_value - warning_value))
                color.r = 1.0
                color.g = 1.0 - t
                color.b = 0.0
            
            color.a = 0.8
            marker.color = color
            
            marker_array.markers.append(marker)
        
        # Add text marker for fault information
        if fault_status['detected']:
            text_marker = Marker()
            text_marker.header.frame_id = "base_link"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "fault_info"
            text_marker.id = 0
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = 0.0
            text_marker.pose.position.y = 0.0
            text_marker.pose.position.z = 0.7
            
            text_marker.pose.orientation.w = 1.0
            
            text_marker.scale.z = 0.05  # Text height
            
            text_marker.color.r = 1.0
            text_marker.color.g = 0.0
            text_marker.color.b = 0.0
            text_marker.color.a = 1.0
            
            fault_joint = fault_status['joint']
            joint_name = list(self.joint_name_map.keys())[fault_joint] if fault_joint < 6 else "health"
            
            text_marker.text = f"FAULT: {fault_status['type']} in {joint_name}\nSeverity: {fault_status['severity']:.2f}\nTime to failure: {fault_status['time_to_failure']:.1f} hours"
            
            marker_array.markers.append(text_marker)
        
        self.visualization_pub.publish(marker_array)
    
    def auto_save_callback(self):
        """
        Periodically save digital twin state
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_dir, f"digital_twin_state_{timestamp}.pkl")
        
        self.get_logger().info(f"Auto-saving digital twin state to {filename}")
        self.digital_twin.save_state(filename)
    
    def save_on_shutdown(self):
        """
        Save state when node is shutdown
        """
        if self.auto_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_dir, f"digital_twin_state_shutdown_{timestamp}.pkl")
            
            self.get_logger().info(f"Saving digital twin state on shutdown to {filename}")
            self.digital_twin.save_state(filename)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = URDigitalTwinNode()
        
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            # Save state on shutdown
            node.save_on_shutdown()
            node.destroy_node()
    except Exception as e:
        print(f"Error during node execution: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()