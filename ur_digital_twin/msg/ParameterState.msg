# ~/ros2_ws/src/ur_digital_twin/msg/ParameterState.msg
std_msgs/Header header
string[] parameter_names
float64[] means
float64[] variances
int32[] dimensions
float64 confidence
