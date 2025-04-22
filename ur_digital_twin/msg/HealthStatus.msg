# ~/ros2_ws/src/ur_digital_twin/msg/HealthStatus.msg
std_msgs/Header header
float64[] health_indicators
float64[] warning_thresholds
float64[] critical_thresholds
float64[] predicted_times_to_warning
float64[] predicted_times_to_critical
string[] joint_names