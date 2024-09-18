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
    
    # Run PointCloud to Polygons node
    pointcloud_to_polygons_node = Node(
        package='pointcloud_to_polygons',
        executable='pointcloud_clustering_node',
        name='pointcloud_to_polygons_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time,
                     'vehicle_footprint_x_pos': 1.2,
                     'vehicle_footprint_x_neg': -1.2,
                     'vehicle_footprint_y_pos': 2.2,
                     'vehicle_footprint_y_neg': -2.8,
                    }],
    )
    
    # Run the Polygons to Corner Points node
    polygons_to_corner_points_node = Node(
        package='corner_detection',
        executable='corner_detection_node',
        name='corner_detection_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        pointcloud_to_polygons_node,
        polygons_to_corner_points_node,
    ])