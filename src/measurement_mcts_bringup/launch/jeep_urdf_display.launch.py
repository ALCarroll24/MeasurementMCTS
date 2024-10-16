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
    
    # Find the URDF file
    urdf_file_name = 'jeep.urdf'
    urdf = os.path.join(
        get_package_share_directory('jeep_description'), 
        'urdf',
        urdf_file_name)
    with open(urdf, 'r') as infp:
        robot_desc = infp.read()
    
    # Places URDF on the robot_description parameter, so that it is viewable in rviz
    robot_state_publisher_node = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': robot_desc,
                'use_sim_time': use_sim_time,
            }],
            output='both'
        )
    
    # Joint state publisher, used to publish the default locations of the wheels (otherwise they are stuck at 0)
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{
            'rate': 30,
            'use_sim_time': use_sim_time,
        }],
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
        robot_state_publisher_node,
        joint_state_publisher_node,
        rviz_node,
    ])