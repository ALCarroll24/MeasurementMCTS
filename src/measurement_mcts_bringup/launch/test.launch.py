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
    # use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    # Find the URDF file
    urdf_file_name = 'jeep.urdf'
    urdf = os.path.join(
        get_package_share_directory('jeep_description'), 
        'urdf',
        urdf_file_name)
    with open(urdf, 'r') as infp:
        robot_desc = infp.read()
    
    # Joint state publisher, initially used for joint commands
    joint_state_publisher_gui__node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        # remappings=[ # This now publishes commands rather than direct joint states
        #     ('joint_states', 'joint_commands'),
        # ],
        # parameters=[{'publish_default_positions': False}],
    )
    
    # Places URDF on the robot_description parameter, takes in joint_states (from pybullet) to do this
    robot_state_publisher_node = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': robot_desc,
            }],
            output='both'
        )
    
    # Launch RVIZ with profile from config folder
    # rviz_node = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     name='rviz2',
    #     output='screen',
    #     arguments=['-d', os.path.join(get_package_share_directory('ur3_description'), 'rviz', 'ur3.rviz')]
    # )

    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_gui__node,
        # rviz_node,
    ])