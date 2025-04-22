# ~/ros2_ws/src/ur_digital_twin/msg/FaultDetection.msg
std_msgs/Header header
bool fault_detected
string fault_type
int32 affected_joint
float64 severity
float64 confidence
float64 estimated_time_to_failure