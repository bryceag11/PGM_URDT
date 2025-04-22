# ~/ros2_ws/src/ur_digital_twin/CMakeLists.txt
cmake_minimum_required(VERSION 3.8)
project(ur_digital_twin)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom message interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ParameterState.msg"
  "msg/HealthStatus.msg"
  "msg/FaultDetection.msg"
  DEPENDENCIES std_msgs
)

ament_package()