import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import TextSubstitution, FindExecutable, Command, LaunchConfiguration, ThisLaunchFileDir, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # Use sim time if the parameter is set to true
    use_sim_time = LaunchConfiguration('use_sim_time', default='True')
    
    # Run the Unity Jeep pose to TF node
    unity_pose_to_tf_node = Node(
        package='measurement_mcts',
        executable='unity_pose_to_tf',
        name='unity_pose_to_tf',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )
    
    # Run the Twist action to Unity node
    twist_action_to_unity_node = Node(
        package='measurement_mcts',
        executable='twist_action_to_unity',
        name='twist_action_to_unity',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )
    
    # # Publish a static transform for the lidar
    lidar_static_transform_publisher_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_static_transform_publisher',
        # Arguments: x, y, z, yaw, pitch, roll, parent_frame, child_frame (copied from unity)
        arguments=['1.754', '0', '1.589', '-1.57', '0', '0', 'base_footprint', 'velodyne_link'],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )
    
    # Launch RVIZ with profile from config folder
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        # arguments=['-d', os.path.join(get_package_share_directory('ur3_description'), 'rviz', 'ur3.rviz')]
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        unity_pose_to_tf_node,
        twist_action_to_unity_node,
        lidar_static_transform_publisher_node,
        rviz_node,
    ])