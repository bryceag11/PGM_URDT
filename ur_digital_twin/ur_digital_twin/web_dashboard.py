# ~/ros2_ws/src/ur_digital_twin/ur_digital_twin/web_dashboard.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ur_digital_twin.msg import ParameterState, HealthStatus, FaultDetection
import json
import threading
import time
import datetime
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import webbrowser
import numpy as np

# Define the path to the static web content
STATIC_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'web')

class DashboardData:
    """Class to store the latest data from the digital twin"""
    def __init__(self):
        self.parameters = {}
        self.health_status = {}
        self.fault_detection = {}
        self.last_update = datetime.datetime.now().isoformat()
        self.lock = threading.Lock()
    
    def update_parameters(self, msg):
        with self.lock:
            params = {}
            idx = 0
            
            # Parse parameters from flattened message
            for i, param_name in enumerate(msg.parameter_names):
                dim = msg.dimensions[i]
                means = msg.means[idx:idx+dim]
                variances = msg.variances[idx:idx+dim]
                
                params[param_name] = {
                    'means': means,
                    'variances': variances,
                    'dimension': dim
                }
                
                idx += dim
            
            self.parameters = params
            self.last_update = datetime.datetime.now().isoformat()
    
    def update_health(self, msg):
        with self.lock:
            health = {
                'health_indicators': list(msg.health_indicators),
                'warning_thresholds': list(msg.warning_thresholds),
                'critical_thresholds': list(msg.critical_thresholds),
                'joint_names': list(msg.joint_names)
            }
            
            self.health_status = health
            self.last_update = datetime.datetime.now().isoformat()
    
    def update_fault(self, msg):
        with self.lock:
            fault = {
                'detected': msg.fault_detected,
                'type': msg.fault_type,
                'joint': msg.affected_joint,
                'severity': msg.severity,
                'confidence': msg.confidence,
                'time_to_failure': msg.estimated_time_to_failure
            }
            
            self.fault_detection = fault
            self.last_update = datetime.datetime.now().isoformat()
    
    def get_data(self):
        with self.lock:
            return {
                'parameters': self.parameters,
                'health_status': self.health_status,
                'fault_detection': self.fault_detection,
                'last_update': self.last_update
            }

class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for the dashboard"""
    def __init__(self, dashboard_data, *args, **kwargs):
        self.dashboard_data = dashboard_data
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        # Serve static files or API data
        if self.path == '/':
            self.path = '/index.html'
        
        if self.path == '/api/data':
            # API endpoint for dashboard data
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            data = self.dashboard_data.get_data()
            self.wfile.write(json.dumps(data).encode())
            return
        
        # Try to serve the static file
        try:
            file_path = STATIC_PATH + self.path
            with open(file_path, 'rb') as file:
                self.send_response(200)
                
                # Set content type based on file extension
                if self.path.endswith('.html'):
                    self.send_header('Content-type', 'text/html')
                elif self.path.endswith('.js'):
                    self.send_header('Content-type', 'application/javascript')
                elif self.path.endswith('.css'):
                    self.send_header('Content-type', 'text/css')
                
                self.end_headers()
                self.wfile.write(file.read())
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'File not found')

class URDigitalTwinDashboard(Node):
    """ROS2 Node for the UR Digital Twin Web Dashboard"""
    def __init__(self):
        super().__init__('ur_digital_twin_dashboard')
        
        # Declare parameters
        self.declare_parameter('web_port', 8080)
        self.declare_parameter('open_browser', True)
        
        # Get parameters
        self.web_port = self.get_parameter('web_port').value
        self.open_browser = self.get_parameter('open_browser').value
        
        # Create dashboard data object
        self.dashboard_data = DashboardData()
        
        # Create subscribers
        self.parameter_sub = self.create_subscription(
            ParameterState,
            '/ur_digital_twin/parameters',
            self.parameter_callback,
            10)
            
        self.health_sub = self.create_subscription(
            HealthStatus,
            '/ur_digital_twin/health_status',
            self.health_callback,
            10)
            
        self.fault_sub = self.create_subscription(
            FaultDetection,
            '/ur_digital_twin/fault_detection',
            self.fault_callback,
            10)
        
        # Create HTTP server
        self.start_web_server()
        
        self.get_logger().info(f'UR Digital Twin Dashboard started on http://localhost:{self.web_port}')
        
        # Open web browser if requested
        if self.open_browser:
            webbrowser.open(f'http://localhost:{self.web_port}')
    
    def parameter_callback(self, msg):
        """Handle parameter updates"""
        self.dashboard_data.update_parameters(msg)
    
    def health_callback(self, msg):
        """Handle health status updates"""
        self.dashboard_data.update_health(msg)
    
    def fault_callback(self, msg):
        """Handle fault detection updates"""
        self.dashboard_data.update_fault(msg)
    
    def start_web_server(self):
        """Start the web server in a separate thread"""
        # Create a custom HTTP server with access to dashboard data
        class DashboardHTTPServer(HTTPServer):
            def __init__(self, server_address, RequestHandlerClass, dashboard_data):
                self.dashboard_data = dashboard_data
                super().__init__(server_address, RequestHandlerClass)
        
        # Create handler class with dashboard data
        handler = lambda *args: DashboardHandler(self.dashboard_data, *args)
        
        # Create and start the server
        self.server = DashboardHTTPServer(('', self.web_port), handler, self.dashboard_data)
        
        # Start server in a thread
        server_thread = threading.Thread(target=self.server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
    
    def shutdown(self):
        """Shutdown the web server"""
        if hasattr(self, 'server'):
            self.server.shutdown()

def main(args=None):
    # Create static web content directory
    os.makedirs(STATIC_PATH, exist_ok=True)
    
    # Create dashboard HTML file
    with open(os.path.join(STATIC_PATH, 'index.html'), 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UR Robot Digital Twin Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background-color: #333;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
        }
        .card-header {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        .parameter-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .parameter-name {
            font-weight: bold;
        }
        .parameter-value {
            text-align: right;
        }
        .gauge-container {
            text-align: center;
            margin-bottom: 20px;
        }
        .gauge-label {
            font-weight: bold;
            margin-top: 5px;
        }
        .alert {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        }
        .alert.warning {
            background-color: #fff8e1;
            border-left-color: #ffc107;
        }
        .progress-bar {
            width: 100%;
            background-color: #e0e0e0;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .progress-bar-fill {
            height: 10px;
            border-radius: 4px;
        }
        .progress-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.8em;
            color: #666;
        }
        .last-update {
            text-align: right;
            color: #666;
            font-size: 0.8em;
            margin-top: 30px;
        }
        .health-status {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .health-indicator {
            width: 15px;
            height: 15px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .chart-container {
            height: 300px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>UR Robot Digital Twin Dashboard</h1>
    </div>
    
    <div class="container">
        <div id="alerts"></div>
        
        <div class="dashboard">
            <div class="card">
                <div class="card-header">Health Status</div>
                <div id="health-status-container"></div>
            </div>
            
            <div class="card">
                <div class="card-header">Friction Parameters</div>
                <div id="friction-container"></div>
            </div>
            
            <div class="card">
                <div class="card-header">Health Parameters</div>
                <div id="health-params-container"></div>
            </div>
            
            <div class="card">
                <div class="card-header">Parameter Chart</div>
                <div id="chart-container" class="chart-container"></div>
            </div>
        </div>
        
        <div class="last-update" id="last-update"></div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // Store historical data for charts
        const historyLength = 50;
        const paramHistory = {
            timestamps: [],
            f: Array(6).fill().map(() => []),
            z: Array(3).fill().map(() => [])
        };
        
        // Chart objects
        let paramChart = null;
        
        // Function to fetch data from the API
        async function fetchData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }
        
        // Function to update the dashboard with new data
        function updateDashboard(data) {
            updateAlerts(data);
            updateHealthStatus(data);
            updateFrictionParams(data);
            updateHealthParams(data);
            updateChart(data);
            
            // Update last update time
            const lastUpdate = new Date(data.last_update);
            document.getElementById('last-update').textContent = `Last updated: ${lastUpdate.toLocaleString()}`;
        }
        
        // Function to update alerts
        function updateAlerts(data) {
            const alertsContainer = document.getElementById('alerts');
            alertsContainer.innerHTML = '';
            
            if (data.fault_detection && data.fault_detection.detected) {
                const fault = data.fault_detection;
                const alertClass = fault.severity > 0.7 ? 'alert' : 'alert warning';
                
                let jointName = 'unknown';
                if (data.health_status && data.health_status.joint_names && fault.joint >= 0 && fault.joint < data.health_status.joint_names.length) {
                    jointName = data.health_status.joint_names[fault.joint];
                }
                
                const timeToFailure = fault.time_to_failure < 1000 ? 
                    `Estimated time to failure: ${fault.time_to_failure.toFixed(1)} hours` : 
                    'No immediate failure risk predicted';
                
                alertsContainer.innerHTML = `
                    <div class="${alertClass}">
                        <strong>Fault Detected:</strong> ${fault.type} in ${jointName}
                        <br>Severity: ${(fault.severity * 100).toFixed(1)}%
                        <br>Confidence: ${(fault.confidence * 100).toFixed(1)}%
                        <br>${timeToFailure}
                    </div>
                `;
            }
        }
        
        // Function to update health status
        function updateHealthStatus(data) {
            const container = document.getElementById('health-status-container');
            container.innerHTML = '';
            
            if (data.health_status && data.health_status.health_indicators) {
                const indicators = data.health_status.health_indicators;
                
                // Overall health - average of all indicators
                const overallHealth = indicators.reduce((sum, val) => sum + val, 0) / indicators.length;
                
                // Create overall health gauge
                container.innerHTML += `
                    <div class="gauge-container">
                        <div class="progress-bar">
                            <div class="progress-bar-fill" style="width: ${overallHealth * 100}%; background-color: ${getHealthColor(overallHealth)};"></div>
                        </div>
                        <div class="progress-label">
                            <span>0%</span>
                            <span>Overall Health: ${(overallHealth * 100).toFixed(1)}%</span>
                            <span>100%</span>
                        </div>
                    </div>
                `;
                
                // Individual health indicators
                indicators.forEach((value, index) => {
                    container.innerHTML += `
                        <div class="health-status">
                            <div class="health-indicator" style="background-color: ${getHealthColor(value)};"></div>
                            <div>Health Indicator ${index + 1}: ${(value * 100).toFixed(1)}%</div>
                        </div>
                    `;
                });
            } else {
                container.innerHTML = '<p>No health data available</p>';
            }
        }
        
        // Function to update friction parameters
        function updateFrictionParams(data) {
            const container = document.getElementById('friction-container');
            container.innerHTML = '';
            
            if (data.parameters && data.parameters.f) {
                const friction = data.parameters.f;
                const means = friction.means;
                
                // Warning and critical thresholds
                const warningThresholds = data.health_status && data.health_status.warning_thresholds ? 
                    data.health_status.warning_thresholds.slice(0, 6) : Array(6).fill(0.18);
                    
                const criticalThresholds = data.health_status && data.health_status.critical_thresholds ? 
                    data.health_status.critical_thresholds.slice(0, 6) : Array(6).fill(0.25);
                
                means.forEach((value, index) => {
                    // Calculate percentage of critical
                    const percentage = (value / criticalThresholds[index]) * 100;
                    
                    container.innerHTML += `
                        <div class="parameter-row">
                            <div class="parameter-name">Joint ${index + 1}</div>
                            <div class="parameter-value">${value.toFixed(4)}</div>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-bar-fill" style="width: ${Math.min(100, percentage)}%; 
                                background-color: ${getFrictionColor(value, warningThresholds[index], criticalThresholds[index])};"></div>
                        </div>
                    `;
                });
            } else {
                container.innerHTML = '<p>No friction data available</p>';
            }
        }
        
        // Function to update health parameters
        function updateHealthParams(data) {
            const container = document.getElementById('health-params-container');
            container.innerHTML = '';
            
            if (data.parameters && data.parameters.z) {
                const health = data.parameters.z;
                const means = health.means;
                
                // Thresholds
                const warningThresholds = data.health_status && data.health_status.warning_thresholds ? 
                    data.health_status.warning_thresholds.slice(6, 9) : Array(3).fill(0.7);
                    
                const criticalThresholds = data.health_status && data.health_status.critical_thresholds ? 
                    data.health_status.critical_thresholds.slice(6, 9) : Array(3).fill(0.5);
                
                means.forEach((value, index) => {
                    // Calculate percentage of perfect health (1.0)
                    const percentage = value * 100;
                    
                    container.innerHTML += `
                        <div class="parameter-row">
                            <div class="parameter-name">Parameter ${index + 1}</div>
                            <div class="parameter-value">${value.toFixed(4)}</div>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-bar-fill" style="width: ${percentage}%; 
                                background-color: ${getHealthColor(value)};"></div>
                        </div>
                    `;
                });
            } else {
                container.innerHTML = '<p>No health parameter data available</p>';
            }
        }
        
        // Function to update the parameter chart
        function updateChart(data) {
            // Update history
            if (data.parameters) {
                // Add timestamp
                const now = new Date();
                paramHistory.timestamps.push(now.toLocaleTimeString());
                if (paramHistory.timestamps.length > historyLength) {
                    paramHistory.timestamps.shift();
                }
                
                // Add friction data
                if (data.parameters.f) {
                    const frictionValues = data.parameters.f.means;
                    frictionValues.forEach((value, index) => {
                        paramHistory.f[index].push(value);
                        if (paramHistory.f[index].length > historyLength) {
                            paramHistory.f[index].shift();
                        }
                    });
                }
                
                // Add health data
                if (data.parameters.z) {
                    const healthValues = data.parameters.z.means;
                    healthValues.forEach((value, index) => {
                        paramHistory.z[index].push(value);
                        if (paramHistory.z[index].length > historyLength) {
                            paramHistory.z[index].shift();
                        }
                    });
                }
            }
            
            // Create or update chart
            const ctx = document.getElementById('chart-container');
            
            // Prepare datasets
            const datasets = [];
            
            // Friction datasets
            paramHistory.f.forEach((values, index) => {
                datasets.push({
                    label: `Friction ${index + 1}`,
                    data: values,
                    borderColor: `rgba(255, 99, ${index * 20 + 132}, 1)`,
                    backgroundColor: `rgba(255, 99, ${index * 20 + 132}, 0.2)`,
                    borderWidth: 2,
                    tension: 0.1
                });
            });
            
            // Health datasets
            paramHistory.z.forEach((values, index) => {
                datasets.push({
                    label: `Health ${index + 1}`,
                    data: values,
                    borderColor: `rgba(54, 162, ${index * 40 + 235}, 1)`,
                    backgroundColor: `rgba(54, 162, ${index * 40 + 235}, 0.2)`,
                    borderWidth: 2,
                    tension: 0.1
                });
            });
            
            if (paramChart === null) {
                // Create new chart
                paramChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: paramHistory.timestamps,
                        datasets: datasets
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
            } else {
                // Update existing chart
                paramChart.data.labels = paramHistory.timestamps;
                paramChart.data.datasets = datasets;
                paramChart.update();
            }
        }
        
        // Helper function to get color based on health value
        function getHealthColor(value) {
            if (value < 0.5) return '#f44336'; // Red
            if (value < 0.7) return '#ff9800'; // Orange
            if (value < 0.9) return '#ffc107'; // Amber
            return '#4caf50'; // Green
        }
        
        // Helper function to get color based on friction value
        function getFrictionColor(value, warning, critical) {
            if (value >= critical) return '#f44336'; // Red
            if (value >= warning) return '#ff9800'; // Orange
            return '#4caf50'; // Green
        }
        
        // Fetch data every 1 second
        fetchData();
        setInterval(fetchData, 1000);
    </script>
</body>
</html>
""")
    
    # Initialize ROS
    rclpy.init(args=args)
    
    try:
        node = URDigitalTwinDashboard()
        
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.shutdown()
            node.destroy_node()
    except Exception as e:
        print(f"Error during node execution: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()