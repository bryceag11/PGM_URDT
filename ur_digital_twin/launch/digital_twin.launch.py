# ~/ros2_ws/src/ur_digital_twin/launch/digital_twin.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # Declare arguments
    declared_arguments = []
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.56.101",
            description="IP address of the robot",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur5e",
            description="Type of UR robot (ur3, ur3e, ur5, ur5e, ur10, ur10e, ur16e)",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz with the configuration for visualization",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "update_rate",
            default_value="10.0",
            description="Update rate for the digital twin (Hz)",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "auto_save",
            default_value="true",
            description="Automatically save digital twin state periodically",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "save_dir",
            default_value="/tmp/ur_digital_twin",
            description="Directory to save digital twin state",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "load_from_file",
            default_value="",
            description="Load digital twin state from file",
        )
    )
    
    # Initialize Arguments
    robot_ip = LaunchConfiguration("robot_ip")
    ur_type = LaunchConfiguration("ur_type")
    launch_rviz = LaunchConfiguration("launch_rviz")
    update_rate = LaunchConfiguration("update_rate")
    auto_save = LaunchConfiguration("auto_save")
    save_dir = LaunchConfiguration("save_dir")
    load_from_file = LaunchConfiguration("load_from_file")
    
    # Get the launch directory
    ur_robot_driver_dir = FindPackageShare("ur_robot_driver")
    ur_digital_twin_dir = FindPackageShare("ur_digital_twin")
    
    # Include the UR driver launch file
    ur_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([ur_robot_driver_dir, "launch", "ur_control.launch.py"])]
        ),
        launch_arguments={
            "robot_ip": robot_ip,
            "ur_type": ur_type,
            "launch_rviz": "false", 
        }.items(),
    )
    
    # Define RViz node with custom config
    rviz_config_file = PathJoinSubstitution(
        [ur_digital_twin_dir, "config", "digital_twin.rviz"]
    )
    
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        condition=IfCondition(launch_rviz),
    )
    
    # Define Digital Twin node
    digital_twin_node = Node(
        package="ur_digital_twin",
        executable="digital_twin_node",
        name="ur_digital_twin",
        output="screen",
        parameters=[
            {
                "update_rate": update_rate,
                "auto_save": auto_save,
                "save_dir": save_dir,
                "load_from_file": load_from_file,
            }
        ],
    )
    
    # Create the launch description
    return LaunchDescription(
        declared_arguments +
        [
            ur_control_launch,
            rviz_node,
            digital_twin_node,
        ]
    )