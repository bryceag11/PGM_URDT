# ~/ros2_ws/src/ur_digital_twin/ur_digital_twin/parameter_evaluation.py

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os
from datetime import datetime
from tabulate import tabulate
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from ur_digital_twin.digital_twin_base import URRobotDigitalTwin
from ur_digital_twin.inference_algorithms import create_estimator

class ParameterEvaluator(Node):
    """
    Evaluates parameter estimation performance using real robot data
    """
    def __init__(self, save_dir='/tmp/ur_digital_twin_eval'):
        super().__init__('parameter_evaluator')
        
        # Create save directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        
        # Initialize data storage
        self.observations = []
        self.parameter_history = []
        self.estimator_results = {
            'ukf': {'rmse': {}, 'convergence_time': {}, 'compute_time': None},
            'ekf': {'rmse': {}, 'convergence_time': {}, 'compute_time': None},
            'pf': {'rmse': {}, 'convergence_time': {}, 'compute_time': None}
        }
        
        # Create digital twin instance
        self.digital_twin = URRobotDigitalTwin()
        
        # Create estimators
        self.estimators = {
            'ukf': create_estimator('ukf', self.digital_twin.parameters),
            'ekf': create_estimator('ekf', self.digital_twin.parameters),
            'pf': create_estimator('pf', self.digital_twin.parameters, num_particles=1000)
        }
        
        # ROS2 subscribers
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        self.wrench_sub = self.create_subscription(
            WrenchStamped,
            '/force_torque_sensor_controller/ft_sensor_readings',
            self.wrench_callback,
            10
        )
        
        # Data collection
        self.joint_data = {'pos': None, 'vel': None, 'curr': None}
        self.force_data = None
        
        self.get_logger().info('Parameter evaluator initialized')
    
    def joint_callback(self, msg):
        """Handle joint state messages"""
        self.joint_data['pos'] = np.array(msg.position)
        self.joint_data['vel'] = np.array(msg.velocity)
        self.joint_data['curr'] = np.array(msg.effort)  # Using effort field for currents
        self.process_data()
    
    def wrench_callback(self, msg):
        """Handle force/torque messages"""
        self.force_data = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ])
        self.process_data()
    
    def process_data(self):
        """Process data when all measurements are available"""
        if (self.joint_data['pos'] is not None and 
            self.joint_data['vel'] is not None and
            self.joint_data['curr'] is not None and
            self.force_data is not None):
            
            # Create observation
            obs = {
                'joint_pos': self.joint_data['pos'],
                'joint_vel': self.joint_data['vel'],
                'joint_curr': self.joint_data['curr'],
                'ext_force': self.force_data
            }
            
            # Store observation
            self.observations.append(obs)
            
            # Update digital twin (ground truth)
            params, _ = self.digital_twin.update(obs)
            self.parameter_history.append(params)
            
            # Update all estimators
            for name, estimator in self.estimators.items():
                start_time = time.time()
                estimated_params = estimator.update(obs)
                compute_time = time.time() - start_time
                
                # Update compute time metric
                if self.estimator_results[name]['compute_time'] is None:
                    self.estimator_results[name]['compute_time'] = compute_time
                else:
                    self.estimator_results[name]['compute_time'] = (
                        0.95 * self.estimator_results[name]['compute_time'] +
                        0.05 * compute_time
                    )
    
    def calculate_metrics(self):
        """Calculate performance metrics for each estimator"""
        if len(self.parameter_history) < 2:
            self.get_logger().warn('Not enough data for metric calculation')
            return
        
        # Get ground truth from digital twin
        true_params = self.parameter_history[-1]
        
        # Calculate metrics for each estimator
        for name, estimator in self.estimators.items():
            estimated_params = estimator.update(self.observations[-1])
            
            # Calculate RMSE for each parameter type
            for param_name in true_params:
                error = np.mean((true_params[param_name][0] - estimated_params[param_name][0])**2)
                rmse = np.sqrt(error)
                self.estimator_results[name]['rmse'][param_name] = rmse
    
    def plot_parameter_evolution(self, param_name='f', joint_idx=0):
        """Plot parameter evolution for different estimators"""
        if len(self.parameter_history) < 2:
            self.get_logger().warn('Not enough data for plotting')
            return
        
        time_steps = range(len(self.parameter_history))
        
        # True parameter values
        true_values = [p[param_name][0][joint_idx] for p in self.parameter_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, true_values, 'k-', label='Digital Twin Estimate')
        
        # Plot estimates from each estimator
        colors = {'ukf': 'b', 'ekf': 'g', 'pf': 'r'}
        for name, estimator in self.estimators.items():
            estimated_values = []
            for obs in self.observations:
                params = estimator.update(obs)
                estimated_values.append(params[param_name][0][joint_idx])
            plt.plot(time_steps, estimated_values, f'{colors[name]}-', label=name.upper())
        
        plt.title(f'Parameter Evolution: {param_name}[{joint_idx}]')
        plt.xlabel('Time Steps')
        plt.ylabel('Parameter Value')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(self.save_dir, f'parameter_evolution_{param_name}_{joint_idx}.png'))
        plt.close()
    
    def print_results(self):
        """Print evaluation results"""
        # Parameter estimation accuracy
        print("\nParameter Estimation RMSE:")
        headers = ['Parameter'] + [name.upper() for name in self.estimators.keys()]
        rows = []
        for param in ['k', 'm', 'f', 'alpha', 'beta', 'z']:
            row = [param]
            for name in self.estimators.keys():
                rmse = self.estimator_results[name]['rmse'].get(param, 0.0)
                row.append(f'{rmse:.4f}')
            rows.append(row)
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        
        # Computational performance
        print("\nComputational Performance:")
        headers = ['Estimator', 'Compute Time (ms)']
        rows = []
        for name in self.estimators.keys():
            compute_time = self.estimator_results[name]['compute_time']
            rows.append([name.upper(), f'{compute_time*1000:.2f}'])
        print(tabulate(rows, headers=headers, tablefmt='grid'))

def main(args=None):
    rclpy.init(args=args)
    evaluator = ParameterEvaluator()
    
    try:
        rclpy.spin(evaluator)
    except KeyboardInterrupt:
        pass
    finally:
        evaluator.print_results()
        evaluator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()