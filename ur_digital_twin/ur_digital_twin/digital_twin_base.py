# ~/ros2_ws/src/ur_digital_twin/ur_digital_twin/digital_twin_base.py
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pickle
import os
from datetime import datetime
from ur_digital_twin.dbn_model import URRobotDBN
class URRobotDigitalTwin:
    """
    Digital Twin for UR Robot using Probabilistic Graphical Models
    """
    def __init__(self, model_type='dbn', n_timeslices=3):
        """
        Initialize the UR Robot Digital Twin

        Parameters:
        -----------
        model_type : str
            Type of PGM to use ('dbn', 'hmm', 'factor_graph')
        n_timeslices : int
            Number of time slices for dynamic models
        """
        self.model_type = model_type
        self.n_timeslices = n_timeslices
        
        # Initialize parameter vector with prior distributions
        # Format: [mean, covariance] for each parameter group
        self.parameters = {
            'k': [np.zeros(6), np.eye(6) * 0.01],  # Kinematic parameters
            'm': [np.zeros(6), np.eye(6) * 0.1],   # Mass parameters
            'f': [np.zeros(6), np.eye(6) * 0.05],  # Friction coefficients
            'alpha': [np.zeros(6), np.eye(6) * 0.01],  # Damping alpha
            'beta': [np.zeros(6), np.eye(6) * 0.01],   # Damping beta
            'z': [np.zeros(3), np.eye(3) * 0.1]     # Health parameters
        }
        
        # Define parameter ranges
        self.parameter_ranges = {
            'k': {'min': np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1]), 
                  'max': np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])},
            'm': {'min': np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), 
                  'max': np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])},
            'f': {'min': np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]), 
                  'max': np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])},
            'alpha': {'min': np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]), 
                      'max': np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])},
            'beta': {'min': np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]), 
                     'max': np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])},
            'z': {'min': np.array([0.0, 0.0, 0.0]), 
                  'max': np.array([1.0, 1.0, 1.0])}
        }
        
        # Define warning and critical thresholds
        self.thresholds = {
            'f': {'warning': np.array([0.18, 0.18, 0.18, 0.18, 0.18, 0.18]),
                  'critical': np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25])},
            'z': {'warning': np.array([0.7, 0.7, 0.7]),
                  'critical': np.array([0.5, 0.5, 0.5])}
        }
        
        # Create DBN model
        self.dbn = URRobotDBN(n_timeslices)
        
        # Initialize UKF for continuous state estimation
        self.ukf = self._setup_ukf()
        
        # Initialize UKF for parameter estimation
        self.ukf = self._setup_ukf()
        
        # Storage for data history
        self.data_history = []
        self.parameter_history = []
        
        # Fault detection
        self.fault_status = {
            'detected': False,
            'type': None,
            'joint': None,
            'severity': 0.0,
            'confidence': 0.0,
            'time_to_failure': float('inf')
        }
        
        # Pre-allocation for joint names
        self.joint_names = [f"joint_{i}" for i in range(6)]
    
    def _combine_estimates(self, ukf_params, dbn_params):
        """
        Combine parameter estimates from UKF and DBN
        """
        combined_params = ukf_params.copy()
        
        # For parameters estimated by both models, use weighted average
        # based on confidence/uncertainty
        for param in dbn_params:
            if param in combined_params:
                # Simple weighted average - could be more sophisticated
                ukf_weight = 0.7  # Give more weight to UKF for continuous estimation
                dbn_weight = 0.3  # Use DBN more for discrete state/fault detection
                
                combined_params[param][0] = (
                    ukf_weight * combined_params[param][0] +
                    dbn_weight * dbn_params[param]
                )
        
        return combined_params
    
    def _setup_ukf(self):
        """
        Set up the Unscented Kalman Filter for parameter estimation
        """
        # Calculate total parameter dimension
        param_dim = sum(self.parameters[p][0].shape[0] for p in self.parameters)
        
        # Observation dimension: 6 joints × (position, velocity, current) + 6 force/torque
        obs_dim = 6 * 3 + 6
        
        # Create sigma point parameters
        points = MerweScaledSigmaPoints(n=param_dim, alpha=0.1, beta=2.0, kappa=1.0)
        
        # Create UKF
        ukf = UnscentedKalmanFilter(
            dim_x=param_dim,
            dim_z=obs_dim,
            dt=0.1,  # Sampling time
            fx=self._process_model,
            hx=self._measurement_model,
            points=points
        )
        
        # Initialize state and covariance
        ukf.x = self._flatten_parameters()
        ukf.P = self._create_covariance_matrix()
        
        # Set process and measurement noise
        ukf.Q = np.eye(param_dim) * 0.001  # Process noise
        ukf.R = np.eye(obs_dim) * 0.01     # Measurement noise
        
        return ukf
    
    def _flatten_parameters(self):
        """
        Flatten parameter dictionary into a single vector
        """
        return np.concatenate([self.parameters[p][0] for p in sorted(self.parameters.keys())])
    
    def _unflatten_parameters(self, flat_params):
        """
        Convert flat parameter vector back to dictionary format
        """
        result = {}
        idx = 0
        for p in sorted(self.parameters.keys()):
            dim = self.parameters[p][0].shape[0]
            result[p] = [flat_params[idx:idx+dim], self.parameters[p][1]]
            idx += dim
        return result
    
    def _create_covariance_matrix(self):
        """
        Create the full covariance matrix from parameter covariances
        """
        # Get dimensions
        dims = [self.parameters[p][0].shape[0] for p in sorted(self.parameters.keys())]
        total_dim = sum(dims)
        
        # Create block diagonal matrix
        P = np.zeros((total_dim, total_dim))
        idx = 0
        for p in sorted(self.parameters.keys()):
            dim = self.parameters[p][0].shape[0]
            P[idx:idx+dim, idx:idx+dim] = self.parameters[p][1]
            idx += dim
        
        return P
    

    def _process_model(self, x, dt):
        """
        Process model for parameter evolution that incorporates:
        - Parameter dynamics from DBN structure
        - Wear effects
        - Temperature effects
        - Cross-parameter influences
        
        Parameters:
        -----------
        x : np.ndarray 
            Current parameter state vector
        dt : float
            Time step
            
        Returns:
        --------
        np.ndarray : Predicted next state
        """
        # Unpack parameters
        params = self._unflatten_parameters(x)
        k = params['k'][0]  # Kinematic parameters
        m = params['m'][0]  # Mass parameters
        f = params['f'][0]  # Friction coefficients
        alpha = params['alpha'][0]  # Damping alpha
        beta = params['beta'][0]  # Damping beta
        z = params['z'][0]  # Health parameters
        
        # Parameter evolution based on physics-informed models
        
        # Kinematic parameters
        k_next = k + np.random.normal(0, 0.001, k.shape)
        
        # Mass parameters
        m_next = m + np.random.normal(0, 0.005, m.shape)
        
        # Friction evolution with wear effects
        wear_rate = 0.001 * (1 - z[0])  # Higher wear rate with lower health
        f_next = f + wear_rate * dt + np.random.normal(0, 0.01, f.shape)
        
        # Damping parameters evolution
        alpha_next = alpha + 0.1 * (f - self.parameters['f'][0]) + \
                    0.05 * (m - self.parameters['m'][0]) + \
                    np.random.normal(0, 0.01, alpha.shape)
        
        # Beta affected by mass
        beta_next = beta + 0.05 * (m - self.parameters['m'][0]) + \
                    np.random.normal(0, 0.01, beta.shape)
        
        # Health parameter evolution
        stress_factor = np.mean(np.abs(f - self.parameters['f'][0]))
        z_next = z - 0.01 * stress_factor * dt + np.random.normal(0, 0.01, z.shape)
        
        # Enforce parameter bounds
        k_next = np.clip(k_next, self.parameter_ranges['k']['min'], self.parameter_ranges['k']['max'])
        m_next = np.clip(m_next, self.parameter_ranges['m']['min'], self.parameter_ranges['m']['max'])
        f_next = np.clip(f_next, self.parameter_ranges['f']['min'], self.parameter_ranges['f']['max'])
        alpha_next = np.clip(alpha_next, self.parameter_ranges['alpha']['min'], self.parameter_ranges['alpha']['max'])
        beta_next = np.clip(beta_next, self.parameter_ranges['beta']['min'], self.parameter_ranges['beta']['max'])
        z_next = np.clip(z_next, self.parameter_ranges['z']['min'], self.parameter_ranges['z']['max'])
        
        # Combine and return
        return np.concatenate([k_next, m_next, f_next, alpha_next, beta_next, z_next])

    def _measurement_model(self, x):
        """
        Measurement model that predicts observations from parameters
        using robot dynamics and sensor models
        
        Parameters:
        -----------
        x : np.ndarray
            Current parameter state vector
            
        Returns:
        --------
        np.ndarray : Predicted observations [joint_pos, joint_vel, joint_curr, ext_force]
        """
        # Unpack parameters
        params = self._unflatten_parameters(x)
        k = params['k'][0]  # Kinematic parameters
        m = params['m'][0]  # Mass parameters
        f = params['f'][0]  # Friction coefficients
        alpha = params['alpha'][0]  # Damping alpha
        beta = params['beta'][0]  # Damping beta
        z = params['z'][0]  # Health parameters
        
        # Get current joint state from history (if available)
        if len(self.data_history) > 0:
            prev_obs = self.data_history[-1]
            q = prev_obs['joint_pos']
            qdot = prev_obs['joint_vel']
        else:
            # Default to zero if no history
            q = np.zeros(6)
            qdot = np.zeros(6)
        
        # Forward kinematics adjustment based on k
        joint_pos = q + k
        
        # Velocity observation with mass effects
        joint_vel = qdot + 0.1 * (1.0 / m) * qdot
        
        # Current draw based on friction, damping, and health
        joint_curr = f * joint_vel + alpha + 0.1 * (1 - z[0])
        
        # External force reading affected by mass and damping
        ext_force = 0.5 * m * qdot + beta * qdot
        
        # Add sensor noise
        joint_pos += np.random.normal(0, 0.01, 6)
        joint_vel += np.random.normal(0, 0.05, 6)
        joint_curr += np.random.normal(0, 0.1, 6)
        ext_force += np.random.normal(0, 0.1, 6)
        
        # Combine all observations
        return np.concatenate([joint_pos, joint_vel, joint_curr, ext_force])

    def update(self, observations):
        """
        Update the digital twin with new observations using both DBN and UKF
        """
        # Update DBN for discrete inference
        dbn_params = self.dbn.update(observations)
        
        # Convert observations to flat vector for UKF
        obs_vector = np.concatenate([
            observations['joint_pos'],
            observations['joint_vel'],
            observations['joint_curr'],
            observations['ext_force']
        ])
        
        # Update UKF
        self.ukf.predict()
        self.ukf.update(obs_vector)
        
        # Combine DBN and UKF estimates
        flat_params = self.ukf.x
        param_cov = self.ukf.P
        
        # Update parameters using both DBN and UKF estimates
        self.parameters = self._combine_estimates(
            self._unflatten_parameters(flat_params),
            dbn_params
        )
        
        # Store observations and parameters for history
        self.data_history.append(observations)
        self.parameter_history.append({k: v[0].copy() for k, v in self.parameters.items()})
        
        # Run fault detection
        self._detect_faults()
        
        return self.parameters, self.fault_status
    
    def _detect_faults(self):
        """
        Detect potential faults based on parameter evolution
        """
        # Need sufficient history for trend detection
        if len(self.parameter_history) < 10:
            return
        
        # Check friction parameters for unusual increases
        for joint in range(6):
            # Get friction history for this joint
            f_history = [p['f'][joint] for p in self.parameter_history[-20:]]
            
            # Calculate trend
            slope = (f_history[-1] - f_history[0]) / len(f_history)
            
            # Check if friction is increasing rapidly
            if slope > 0.002:
                # Check if approaching warning threshold
                current_value = f_history[-1]
                warning_threshold = self.thresholds['f']['warning'][joint]
                critical_threshold = self.thresholds['f']['critical'][joint]
                
                if current_value > warning_threshold:
                    self.fault_status['detected'] = True
                    self.fault_status['type'] = 'friction_increase'
                    self.fault_status['joint'] = joint
                    self.fault_status['severity'] = (current_value - warning_threshold) / (critical_threshold - warning_threshold)
                    self.fault_status['confidence'] = min(0.9, max(0.5, slope / 0.005))
                    
                    # Estimate time to failure
                    if slope > 0:
                        time_to_critical = (critical_threshold - current_value) / slope
                        self.fault_status['time_to_failure'] = time_to_critical
        
        # Check health parameters for unusual decreases
        for i in range(3):
            # Get health history for this parameter
            z_history = [p['z'][i] for p in self.parameter_history[-20:]]
            
            # Calculate trend
            slope = (z_history[-1] - z_history[0]) / len(z_history)
            
            # Check if health is decreasing rapidly
            if slope < -0.001:  # Threshold for rapid decrease
                # Check if approaching warning threshold
                current_value = z_history[-1]
                warning_threshold = self.thresholds['z']['warning'][i]
                critical_threshold = self.thresholds['z']['critical'][i]
                
                if current_value < warning_threshold:
                    self.fault_status['detected'] = True
                    self.fault_status['type'] = 'health_degradation'
                    self.fault_status['joint'] = i
                    self.fault_status['severity'] = (warning_threshold - current_value) / (warning_threshold - critical_threshold)
                    self.fault_status['confidence'] = min(0.9, max(0.5, abs(slope) / 0.002))
                    
                    # Estimate time to failure
                    if slope < 0:
                        time_to_critical = (current_value - critical_threshold) / abs(slope)
                        self.fault_status['time_to_failure'] = time_to_critical
    
    def predict_parameter_evolution(self, steps_ahead=100):
        """
        Predict future parameter values
        
        Parameters:
        -----------
        steps_ahead : int
            Number of time steps to predict ahead
            
        Returns:
        --------
        dict: Predicted parameter values and uncertainties
        """
        # Current state and covariance
        x = self.ukf.x
        P = self.ukf.P
        
        # Predicted values
        predictions = {'time_steps': list(range(steps_ahead))}
        
        for param in self.parameters:
            predictions[param] = {'mean': [], 'upper': [], 'lower': []}
        
        # Predict future values using process model
        for i in range(steps_ahead):
            # Predict next state
            x = self._process_model(x, 0.1)
            
            # Assume process noise accumulates
            param_dict = self._unflatten_parameters(x)
            
            # Extract means and compute confidence intervals
            for param, (mean, cov) in param_dict.items():
                predictions[param]['mean'].append(mean.copy())
                
                # Assuming diagonal covariance for simplicity
                std_dev = np.sqrt(np.diag(cov))
                predictions[param]['upper'].append(mean + 2 * std_dev)
                predictions[param]['lower'].append(mean - 2 * std_dev)
        
        return predictions
    
    def get_health_status(self):
        """
        Get current health status of the robot
        
        Returns:
        --------
        dict: Health status information
        """
        # Extract current parameters
        if len(self.parameter_history) == 0:
            return {
                'health_indicators': np.ones(3),
                'warnings': [],
                'status': 'Unknown'
            }
        
        current_params = self.parameter_history[-1]
        
        # Calculate health indicators
        friction_health = np.mean([1.0 - min(1.0, max(0.0, 
                         (current_params['f'][i] - self.parameter_ranges['f']['min'][i]) / 
                         (self.thresholds['f']['critical'][i] - self.parameter_ranges['f']['min'][i])))
                         for i in range(6)])
        
        # Overall health is the minimum of component health
        health_indicators = np.array([
            friction_health,
            current_params['z'][0],
            current_params['z'][1]
        ])
        
        # Generate warnings
        warnings = []
        
        # Check friction
        for i in range(6):
            if current_params['f'][i] > self.thresholds['f']['warning'][i]:
                warnings.append(f"Joint {i+1} friction exceeds warning threshold")
        
        # Check health parameters
        for i in range(3):
            if current_params['z'][i] < self.thresholds['z']['warning'][i]:
                warnings.append(f"Health indicator {i+1} below warning threshold")
        
        # Determine overall status
        if len(warnings) == 0:
            status = 'Healthy'
        elif self.fault_status['detected'] and self.fault_status['severity'] > 0.7:
            status = 'Critical'
        else:
            status = 'Warning'
        
        return {
            'health_indicators': health_indicators,
            'warnings': warnings,
            'status': status
        }
    
    def save_state(self, filename):
        """
        Save the current state of the digital twin
        
        Parameters:
        -----------
        filename : str
            File to save state to
        """
        state = {
            'parameters': self.parameters,
            'data_history': self.data_history,
            'parameter_history': self.parameter_history,
            'fault_status': self.fault_status,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filename):
        """
        Load a saved state
        
        Parameters:
        -----------
        filename : str
            File to load state from
        """
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        self.parameters = state['parameters']
        self.data_history = state['data_history']
        self.parameter_history = state['parameter_history']
        self.fault_status = state['fault_status']
        
        # Reinitialize UKF with loaded parameters
        self.ukf = self._setup_ukf()
    
    def plot_parameter_history(self, parameter_name, joint_indices=None):
        """
        Plot parameter history
        
        Parameters:
        -----------
        parameter_name : str
            Name of parameter to plot
        joint_indices : list
            Indices of joints to plot, if None plot all
        """
        if len(self.parameter_history) == 0:
            print("No parameter history available")
            return
        
        if joint_indices is None:
            if parameter_name == 'z':
                joint_indices = list(range(3))
            else:
                joint_indices = list(range(6))
        
        plt.figure(figsize=(10, 6))
        
        for idx in joint_indices:
            values = [p[parameter_name][idx] for p in self.parameter_history]
            plt.plot(values, label=f"{parameter_name}[{idx}]")
        
        # Add thresholds
        if parameter_name in self.thresholds:
            for idx in joint_indices:
                warning = self.thresholds[parameter_name]['warning'][idx]
                critical = self.thresholds[parameter_name]['critical'][idx]
                
                plt.axhline(y=warning, color='orange', linestyle='--', 
                           label=f"Warning threshold ({idx})" if idx == joint_indices[0] else "")
                plt.axhline(y=critical, color='red', linestyle='--',
                           label=f"Critical threshold ({idx})" if idx == joint_indices[0] else "")
        
        plt.title(f"Evolution of {parameter_name} Parameter")
        plt.xlabel("Time Step")
        plt.ylabel("Parameter Value")
        plt.legend()
        plt.grid(True)
        plt.show()


