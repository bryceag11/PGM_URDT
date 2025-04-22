# ~/ros2_ws/src/ur_digital_twin/ur_digital_twin/inference_algorithms.py

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.kalman import ExtendedKalmanFilter
import time
import pickle
import os

class BaseEstimator:
    """Base class for parameter estimators"""
    def __init__(self, init_params, process_noise=0.001, measurement_noise=0.01):
        """
        Initialize estimator
        
        Parameters:
        -----------
        init_params : dict
            Initial parameter dictionary with format {param_name: [mean_vector, covariance_matrix]}
        process_noise : float
            Process noise scalar (will be expanded to matrix)
        measurement_noise : float
            Measurement noise scalar (will be expanded to matrix)
        """
        self.parameters = init_params
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Calculate dimensions
        self.param_dim = sum(self.parameters[p][0].shape[0] for p in self.parameters)
        self.obs_dim = 6 * 3 + 6  # joint_pos, joint_vel, joint_curr, force/torque
        
        # Setup internal state
        self.setup()
    
    def setup(self):
        """Setup estimator (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement setup()")
    
    def update(self, observation):
        """Update parameters with new observation"""
        raise NotImplementedError("Subclasses must implement update()")
    
    def _flatten_parameters(self):
        """Flatten parameter dictionary into a single vector"""
        return np.concatenate([self.parameters[p][0] for p in sorted(self.parameters.keys())])
    
    def _unflatten_parameters(self, flat_params):
        """Convert flat parameter vector back to dictionary format"""
        result = {}
        idx = 0
        for p in sorted(self.parameters.keys()):
            dim = self.parameters[p][0].shape[0]
            result[p] = [flat_params[idx:idx+dim], self.parameters[p][1]]
            idx += dim
        return result
    
    def _create_covariance_matrix(self):
        """Create the full covariance matrix from parameter covariances"""
        dims = [self.parameters[p][0].shape[0] for p in sorted(self.parameters.keys())]
        total_dim = sum(dims)
        
        P = np.zeros((total_dim, total_dim))
        idx = 0
        for p in sorted(self.parameters.keys()):
            dim = self.parameters[p][0].shape[0]
            P[idx:idx+dim, idx:idx+dim] = self.parameters[p][1]
            idx += dim
        
        return P
    
    def _process_model(self, x, dt):
        """
        Process model based on robot dynamics and wear patterns
        """
        # Unpack parameters
        params = self._unflatten_parameters(x)
        
        # Natural parameter evolution based on physics
        f = params['f'][0] + np.random.normal(0, 0.0001, 6)  
        
        # Mass generally stays constant
        m = params['m'][0] + np.random.normal(0, 0.00001, 6) 
        
        # Kinematic parameters drift very slowly
        k = params['k'][0] + np.random.normal(0, 0.00001, 6)
        
        # Damping parameters evolve based on friction
        alpha = params['alpha'][0] + 0.1 * (f - params['f'][0])
        beta = params['beta'][0] + np.random.normal(0, 0.0001, 6)
        
        # Health parameters degrade based on stress
        stress = np.mean(np.abs(f - params['f'][0]))
        z = params['z'][0] - stress * 0.001 + np.random.normal(0, 0.0001, 3)
        
        return np.concatenate([k, m, f, alpha, beta, z])
    
    def _measurement_model(self, x):
        """
        Measurement model based on robot dynamics
        """
        # Unpack parameters
        params = self._unflatten_parameters(x)
        k = params['k'][0]  # Kinematic parameters
        m = params['m'][0]  # Mass parameters
        f = params['f'][0]  # Friction coefficients
        alpha = params['alpha'][0]  # Damping alpha
        beta = params['beta'][0]  # Damping beta
        z = params['z'][0]  # Health parameters
        
        # Physics-based forward model
        joint_pos = k  # Position offset by kinematic parameters
        
        # Velocity affected by mass and damping
        joint_vel = np.zeros(6)
        for i in range(6):
            joint_vel[i] = 1.0/m[i]  # Base velocity scaled by inverse mass
        
        # Current draw based on friction and health
        joint_curr = np.zeros(6)
        for i in range(6):
            joint_curr[i] = f[i] * joint_vel[i] + alpha[i]
            joint_curr[i] *= (1 + 0.1 * (1 - z[0]))  # Health effect
        
        # External forces based on mass and damping
        ext_force = np.zeros(6)
        for i in range(6):
            ext_force[i] = m[i] * 9.81 + beta[i] * joint_vel[i]
        
        return np.concatenate([joint_pos, joint_vel, joint_curr, ext_force])
    
    def _format_observation(self, obs_dict):
        """Format observation dictionary into flat vector"""
        return np.concatenate([
            obs_dict['joint_pos'],
            obs_dict['joint_vel'],
            obs_dict['joint_curr'],
            obs_dict['ext_force']
        ])


class UKFEstimator(BaseEstimator):
    """Unscented Kalman Filter parameter estimator"""
    def setup(self):
        """Setup UKF"""
        # Create sigma point parameters
        points = MerweScaledSigmaPoints(n=self.param_dim, alpha=0.1, beta=2.0, kappa=1.0)
        
        # Create UKF
        self.ukf = UnscentedKalmanFilter(
            dim_x=self.param_dim,
            dim_z=self.obs_dim,
            dt=0.1,  # Sampling time
            fx=self._process_model,
            hx=self._measurement_model,
            points=points
        )
        
        # Initialize state and covariance
        self.ukf.x = self._flatten_parameters()
        self.ukf.P = self._create_covariance_matrix()
        
        # Set process and measurement noise
        self.ukf.Q = np.eye(self.param_dim) * self.process_noise
        self.ukf.R = np.eye(self.obs_dim) * self.measurement_noise
    
    def update(self, observation):
        """Update parameters with new observation"""
        # Convert observations to flat vector
        obs_vector = self._format_observation(observation)
        
        # Update UKF
        self.ukf.predict()
        self.ukf.update(obs_vector)
        
        # Update parameters
        flat_params = self.ukf.x
        self.parameters = self._unflatten_parameters(flat_params)
        
        return self.parameters


class EKFEstimator(BaseEstimator):
    """Extended Kalman Filter parameter estimator"""
    def setup(self):
        """Setup EKF"""
        # Create EKF
        self.ekf = ExtendedKalmanFilter(dim_x=self.param_dim, dim_z=self.obs_dim)
        
        # Initialize state and covariance
        self.ekf.x = self._flatten_parameters()
        self.ekf.P = self._create_covariance_matrix()
        
        # Set process and measurement noise
        self.ekf.Q = np.eye(self.param_dim) * self.process_noise
        self.ekf.R = np.eye(self.obs_dim) * self.measurement_noise
    
    def _jacobian_h(self, x):
        """Calculate Jacobian of measurement model numerically"""
        J = np.zeros((self.obs_dim, self.param_dim))
        for i in range(self.param_dim):
            x_plus = x.copy()
            x_plus[i] += self.eps
            x_minus = x.copy()
            x_minus[i] -= self.eps
            
            y_plus = self._measurement_model(x_plus)
            y_minus = self._measurement_model(x_minus)
            J[:, i] = (y_plus - y_minus) / (2 * self.eps)
        return J
    
    def update(self, observation):
        """Update parameters with new observation"""
        # Convert observations to flat vector
        obs_vector = self._format_observation(observation)
        
        # Predict state
        self.ekf.predict(u=None, F=np.eye(self.param_dim), B=None, Q=self.ekf.Q)
        
        # Update with measurement
        H = self._jacobian_h(self.ekf.x)
        predicted_measurement = self._measurement_model(self.ekf.x)
        self.ekf.update(obs_vector, HJacobian=lambda x: H, Hx=lambda x: predicted_measurement)
        
        # Update parameters
        flat_params = self.ekf.x
        self.parameters = self._unflatten_parameters(flat_params)
        
        return self.parameters


class ParticleFilterEstimator(BaseEstimator):
    """Particle Filter parameter estimator"""
    def __init__(self, init_params, num_particles=1000, process_noise=0.001, measurement_noise=0.01):
        """
        Initialize Particle Filter estimator
        
        Parameters:
        -----------
        init_params : dict
            Initial parameter dictionary
        num_particles : int
            Number of particles to use
        process_noise : float
            Process noise scalar
        measurement_noise : float
            Measurement noise scalar
        """
        self.num_particles = num_particles
        super().__init__(init_params, process_noise, measurement_noise)
    
    def setup(self):
        self.particles = np.zeros((self.num_particles, self.param_dim))
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Initialize particles around initial estimate
        init_params = self._flatten_parameters()
        for i in range(self.num_particles):
            self.particles[i] = init_params + np.random.normal(0, 0.1, self.param_dim)
    
    def _resample(self):
        """Resample particles based on weights"""
        # Systematic resampling
        indices = np.zeros(self.num_particles, dtype=np.int64)
        cumulative_sum = np.cumsum(self.weights)
        step = 1.0 / self.num_particles
        u = np.random.uniform(0, step)
        
        i, j = 0, 0
        while i < self.num_particles:
            while u > cumulative_sum[j]:
                j += 1
            indices[i] = j
            u += step
            i += 1
        
        # Create new particles and reset weights
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def update(self, observation):
        obs_vector = np.concatenate([
            observation['joint_pos'],
            observation['joint_vel'],
            observation['joint_curr'],
            observation['ext_force']
        ])
        
        # Predict
        for i in range(self.num_particles):
            self.particles[i] = self._process_model(self.particles[i], 0.1)
        
        # Update weights
        for i in range(self.num_particles):
            pred_obs = self._measurement_model(self.particles[i])
            error = obs_vector - pred_obs
            likelihood = np.exp(-0.5 * np.sum(error**2) / self.measurement_noise)
            self.weights[i] *= likelihood
        
        # Normalize weights
        self.weights /= np.sum(self.weights)
        
        # Resample if needed
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < self.num_particles / 2:
            indices = np.random.choice(
                self.num_particles,
                self.num_particles,
                p=self.weights
            )
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Estimate state
        mean_state = np.sum(self.particles.T * self.weights, axis=1)
        return self._unflatten_parameters(mean_state)


class VariationalInferenceEstimator(BaseEstimator):
    """
    Variational Inference parameter estimator
    """
    def setup(self):
        """Setup VI estimator"""
        # Initialize mean and variance parameters
        self.mean = self._flatten_parameters()
        self.variance = np.diag(self._create_covariance_matrix())
        
        # Learning rate
        self.learning_rate = 0.01
        
        # Number of VI iterations per update
        self.num_iterations = 10
    
    def update(self, observation):
        """Update parameters with new observation"""
        # Convert observations to flat vector
        obs_vector = self._format_observation(observation)
        
        # Run VI iterations
        for _ in range(self.num_iterations):
            # Generate samples from current variational distribution
            samples = np.random.normal(self.mean, np.sqrt(self.variance))
            
            # Get predicted measurement
            predicted_obs = self._measurement_model(samples)
            
            # Calculate error
            error = obs_vector - predicted_obs
            
            # Update mean and variance
            grad_mean = error * 0.1  
            self.mean += self.learning_rate * grad_mean
            
            # Apply process model
            self.mean = self._process_model(self.mean, 0.1)
            
            # Update variance
            self.variance = np.maximum(0.001, self.variance * 0.99)
        
        # Update parameters
        self.parameters = self._unflatten_parameters(self.mean)
        
        return self.parameters


def create_estimator(estimator_type, init_params, **kwargs):
    """
    Factory function to create appropriate estimator
    
    Parameters:
    -----------
    estimator_type : str
        Type of estimator ('ukf', 'ekf', 'pf', 'vi')
    init_params : dict
        Initial parameter dictionary
    **kwargs :
        Additional arguments for specific estimators
    
    Returns:
    --------
    estimator : BaseEstimator
        Parameter estimator object
    """
    if estimator_type == 'ukf':
        return UKFEstimator(init_params, **kwargs)
    elif estimator_type == 'ekf':
        return EKFEstimator(init_params, **kwargs)
    elif estimator_type == 'pf':
        return ParticleFilterEstimator(init_params, **kwargs)
    elif estimator_type == 'vi':
        return VariationalInferenceEstimator(init_params, **kwargs)
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")