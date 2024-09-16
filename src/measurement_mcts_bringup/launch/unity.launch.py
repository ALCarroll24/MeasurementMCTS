from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package share directory
    package_share_dir = get_package_share_directory('measurement_mcts_bringup')

    # Launch file paths
    jeep_urdf_display_launch = os.path.join(package_share_dir, 'launch', 'jeep_urdf_display.launch.py')
    polygon_detection_launch = os.path.join(package_share_dir, 'launch', 'polygon_detection.launch.py')
    unity_bringup_launch = os.path.join(package_share_dir, 'launch', 'unity_bringup.launch.py')

    # Include the three launch files
    jeep_urdf_display = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(jeep_urdf_display_launch)
    )

    polygon_detection = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(polygon_detection_launch)
    )

    unity_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(unity_bringup_launch)
    )

    return LaunchDescription([
        jeep_urdf_display,
        polygon_detection,
        unity_bringup
    ])
