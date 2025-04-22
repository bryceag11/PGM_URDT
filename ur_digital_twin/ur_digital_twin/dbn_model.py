# ~/ros2_ws/src/ur_digital_twin/ur_digital_twin/dbn_model.py

import numpy as np
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference

class URRobotDBN:
    """Dynamic Bayesian Network for UR Robot Digital Twin"""
    def __init__(self, n_time_slices=2):
        self.n_time_slices = n_time_slices
        self.model = self._create_dbn_structure()
        
        # Parameter ranges and discretization
        self.param_ranges = {
            'k': {'min': -0.1, 'max': 0.1, 'bins': 20},
            'm': {'min': 0.5, 'max': 10.0, 'bins': 20},
            'f': {'min': 0.01, 'max': 0.5, 'bins': 20},
            'alpha': {'min': 0.001, 'max': 0.2, 'bins': 20},
            'beta': {'min': 0.001, 'max': 0.2, 'bins': 20},
            'z': {'min': 0.0, 'max': 1.0, 'bins': 20}
        }
        
        # Initialize CPDs
        self._initialize_cpds()
    
    def _create_dbn_structure(self):
        """Create the DBN structure as shown in Figure 1"""
        dbn = DynamicBayesianNetwork()
        
        # Add nodes for each time slice
        nodes_t0 = ['k_t0', 'm_t0', 'f_t0', 'alpha_t0', 'beta_t0', 'z_t0',
                   'pos_t0', 'vel_t0', 'curr_t0', 'force_t0']
        nodes_t1 = ['k_t1', 'm_t1', 'f_t1', 'alpha_t1', 'beta_t1', 'z_t1',
                   'pos_t1', 'vel_t1', 'curr_t1', 'force_t1']
        
        # Add all nodes
        for node in nodes_t0 + nodes_t1:
            dbn.add_node(node)
        
        # Add intra-slice edges for t0
        intra_edges_t0 = [
            ('f_t0', 'alpha_t0'),    # Friction affects damping
            ('f_t0', 'z_t0'),        # Friction affects health
            ('m_t0', 'alpha_t0'),    # Mass affects damping
            # Parameter to observation edges
            ('k_t0', 'pos_t0'),
            ('m_t0', 'vel_t0'),
            ('f_t0', 'curr_t0'),
            ('alpha_t0', 'curr_t0'),
            ('beta_t0', 'force_t0'),
            ('z_t0', 'curr_t0')
        ]
        dbn.add_edges_from(intra_edges_t0)
        
        # Add intra-slice edges for t1 (same structure)
        intra_edges_t1 = [
            ('f_t1', 'alpha_t1'),
            ('f_t1', 'z_t1'),
            ('m_t1', 'alpha_t1'),
            ('k_t1', 'pos_t1'),
            ('m_t1', 'vel_t1'),
            ('f_t1', 'curr_t1'),
            ('alpha_t1', 'curr_t1'),
            ('beta_t1', 'force_t1'),
            ('z_t1', 'curr_t1')
        ]
        dbn.add_edges_from(intra_edges_t1)
        
        # Add inter-slice edges (temporal connections)
        inter_edges = [
            ('k_t0', 'k_t1'),
            ('m_t0', 'm_t1'),
            ('f_t0', 'f_t1'),
            ('alpha_t0', 'alpha_t1'),
            ('beta_t0', 'beta_t1'),
            ('z_t0', 'z_t1')
        ]
        dbn.add_edges_from(inter_edges)
        
        return dbn
    
    def _initialize_cpds(self):
        """Initialize Conditional Probability Distributions"""
        self.cpds = {}
        
        # Create discretized values for each parameter
        param_values = {}
        for param, range_info in self.param_ranges.items():
            param_values[param] = np.linspace(
                range_info['min'],
                range_info['max'],
                range_info['bins']
            )
        
        # Initialize priors for t0
        for param in ['k', 'm', 'f', 'alpha', 'beta', 'z']:
            values = param_values[param]
            self.cpds[f'{param}_t0'] = TabularCPD(
                variable=f'{param}_t0',
                variable_card=len(values),
                values=np.ones(len(values)) / len(values)
            )
        
        # Transition CPDs (t0 -> t1)
        for param in ['k', 'm', 'f', 'alpha', 'beta', 'z']:
            values = param_values[param]
            n_states = len(values)
            
            # Create transition matrix based on parameter type
            if param == 'f':  # Friction tends to increase
                transition_matrix = np.eye(n_states) * 0.6
                for i in range(n_states):
                    if i < n_states - 1:
                        transition_matrix[i+1, i] = 0.3  # Probability of increase
                    if i > 0:
                        transition_matrix[i-1, i] = 0.1  # Small chance of decrease
            
            elif param == 'z':  # Health tends to decrease
                transition_matrix = np.eye(n_states) * 0.6
                for i in range(n_states):
                    if i > 0:
                        transition_matrix[i-1, i] = 0.3  # Probability of decrease
                    if i < n_states - 1:
                        transition_matrix[i+1, i] = 0.1  # Small chance of increase
            
            else:  # Other parameters follow random walk
                transition_matrix = np.eye(n_states) * 0.7
                for i in range(n_states):
                    if i > 0:
                        transition_matrix[i-1, i] = 0.15
                    if i < n_states - 1:
                        transition_matrix[i+1, i] = 0.15
            
            self.cpds[f'{param}_t1'] = TabularCPD(
                variable=f'{param}_t1',
                variable_card=n_states,
                values=transition_matrix,
                evidence=[f'{param}_t0'],
                evidence_card=[n_states]
            )
        
        # Intra-slice CPDs for parameter relationships
        
        # Friction affects damping (f -> alpha)
        f_vals = param_values['f']
        alpha_vals = param_values['alpha']
        f_alpha_matrix = np.zeros((len(alpha_vals), len(f_vals)))
        for i, f in enumerate(f_vals):
            mean_idx = int(f * len(alpha_vals))  # Higher friction -> higher damping
            std = max(1, len(alpha_vals) // 10)
            for j in range(len(alpha_vals)):
                f_alpha_matrix[j, i] = np.exp(-0.5 * ((j - mean_idx) / std) ** 2)
        f_alpha_matrix /= f_alpha_matrix.sum(axis=0, keepdims=True)
        
        for t in [0, 1]:
            self.cpds[f'alpha_t{t}_given_f'] = TabularCPD(
                variable=f'alpha_t{t}',
                variable_card=len(alpha_vals),
                values=f_alpha_matrix,
                evidence=[f'f_t{t}'],
                evidence_card=[len(f_vals)]
            )
        
        # Friction affects health (f -> z)
        z_vals = param_values['z']
        f_z_matrix = np.zeros((len(z_vals), len(f_vals)))
        for i, f in enumerate(f_vals):
            mean_idx = int((1 - f) * len(z_vals))  # Higher friction -> lower health
            std = max(1, len(z_vals) // 10)
            for j in range(len(z_vals)):
                f_z_matrix[j, i] = np.exp(-0.5 * ((j - mean_idx) / std) ** 2)
        f_z_matrix /= f_z_matrix.sum(axis=0, keepdims=True)
        
        for t in [0, 1]:
            self.cpds[f'z_t{t}_given_f'] = TabularCPD(
                variable=f'z_t{t}',
                variable_card=len(z_vals),
                values=f_z_matrix,
                evidence=[f'f_t{t}'],
                evidence_card=[len(f_vals)]
            )
        
        # Mass affects damping (m -> alpha)
        m_vals = param_values['m']
        m_alpha_matrix = np.zeros((len(alpha_vals), len(m_vals)))
        for i, m in enumerate(m_vals):
            mean_idx = int(m * len(alpha_vals) / m_vals[-1])  # Higher mass -> higher damping
            std = max(1, len(alpha_vals) // 10)
            for j in range(len(alpha_vals)):
                m_alpha_matrix[j, i] = np.exp(-0.5 * ((j - mean_idx) / std) ** 2)
        m_alpha_matrix /= m_alpha_matrix.sum(axis=0, keepdims=True)
        
        for t in [0, 1]:
            self.cpds[f'alpha_t{t}_given_m'] = TabularCPD(
                variable=f'alpha_t{t}',
                variable_card=len(alpha_vals),
                values=m_alpha_matrix,
                evidence=[f'm_t{t}'],
                evidence_card=[len(m_vals)]
            )
        
        # Observation CPDs
        # For each observation type, create CPD based on parameter relationships
        obs_states = 20  # Number of discretized observation states
        
        # Position observations depend on kinematic parameters
        for t in [0, 1]:
            pos_matrix = self._create_observation_cpd(
                param_values['k'],
                obs_states,
                increasing=True
            )
            self.cpds[f'pos_t{t}'] = TabularCPD(
                variable=f'pos_t{t}',
                variable_card=obs_states,
                values=pos_matrix,
                evidence=[f'k_t{t}'],
                evidence_card=[len(param_values['k'])]
            )
        
        # Current observations depend on friction, alpha, and z
        for t in [0, 1]:
            curr_matrix = self._create_joint_observation_cpd(
                [param_values['f'], param_values['alpha'], param_values['z']],
                obs_states
            )
            self.cpds[f'curr_t{t}'] = TabularCPD(
                variable=f'curr_t{t}',
                variable_card=obs_states,
                values=curr_matrix,
                evidence=[f'f_t{t}', f'alpha_t{t}', f'z_t{t}'],
                evidence_card=[len(param_values['f']), 
                             len(param_values['alpha']),
                             len(param_values['z'])]
            )
        
        # Add all CPDs to model
        for cpd in self.cpds.values():
            self.model.add_cpds(cpd)
            
    def _create_observation_cpd(self, param_values, obs_states, increasing=True):
        """Helper to create observation CPDs"""
        matrix = np.zeros((obs_states, len(param_values)))
        for i, val in enumerate(param_values):
            mean_idx = int(val * obs_states) if increasing else int((1-val) * obs_states)
            std = max(1, obs_states // 10)
            for j in range(obs_states):
                matrix[j, i] = np.exp(-0.5 * ((j - mean_idx) / std) ** 2)
        return matrix / matrix.sum(axis=0, keepdims=True)
    
    def _create_joint_observation_cpd(self, param_values_list, obs_states):
        """Helper to create joint observation CPDs"""
        n_params = len(param_values_list)
        shape = [obs_states] + [len(pv) for pv in param_values_list]
        matrix = np.zeros(shape)
        
        # Fill the matrix based on combined effects
        indices = np.indices(shape[1:])
        for idx in np.ndindex(*shape[1:]):
            param_vals = [param_values_list[i][idx[i]] for i in range(n_params)]
            mean_obs = np.mean(param_vals) 
            mean_idx = int(mean_obs * obs_states)
            std = max(1, obs_states // 10)
            for j in range(obs_states):
                matrix[(j,) + idx] = np.exp(-0.5 * ((j - mean_idx) / std) ** 2)
        
        # Normalize
        matrix /= matrix.sum(axis=0, keepdims=True)
        
        # Reshape for pgmpy format
        return matrix.reshape(obs_states, -1)
    
    def update(self, observations):
        """Update DBN with new observations"""
        inference = DBNInference(self.model)
        
        # Convert observations to evidence format
        evidence = self._format_observations(observations)
        
        # Perform inference
        query_results = inference.forward_inference(
            ['f_t1', 'z_t1'],  # Variables we want to query
            evidence
        )
        
        # Convert results back to parameter estimates
        parameters = self._process_query_results(query_results)
        
        return parameters
    
    def _format_observations(self, observations):
        """Format observations for DBN inference"""
        # Convert continuous observations to discrete evidence
        evidence = {}
        for obs_type, value in observations.items():
            discretized_value = self._discretize_observation(obs_type, value)
            evidence[f"{obs_type}_t1"] = discretized_value
        return evidence
    
    def _discretize_observation(self, obs_type, value):
        """Discretize continuous observation values"""
        if obs_type == 'joint_curr':
            return int(min(19, max(0, value * 20)))
        return 0  # Default discretization
    
    def _process_query_results(self, query_results):
        """Convert query results to parameter estimates"""
        # Convert discrete probability distributions back to continuous estimates
        parameters = {}
        
        if 'f_t1' in query_results:
            f_dist = query_results['f_t1']
            f_values = np.linspace(
                self.param_ranges['f']['min'],
                self.param_ranges['f']['max'],
                self.param_ranges['f']['bins']
            )
            # Expected value
            parameters['f'] = np.sum(f_values * f_dist)
        
        return parameters