import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from pointcloud_array_msgs.msg import PointCloud2Array
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Point
import numpy as np
import pyransac3d as pyrsc
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
import pickle
from ament_index_python.packages import get_package_share_directory
import os

class ObjectPointsSubscriber(Node):
    def __init__(self):
        super().__init__('object_points_subscriber')
        
        # Create a subscriber to /object_points_array topic
        self.subscription = self.create_subscription(
            PointCloud2Array,
            '/object_clouds',
            self.listener_callback,
            10  # QoS history depth
        )

        # Publisher to visualize the detected lines
        self.line_publisher = self.create_publisher(MarkerArray, '/detected_lines_array', 10)
        
        # Set up a folder to save the pickle files (inside the package directory)
        package_share_directory = get_package_share_directory('corner_detection')  # Replace with your actual package name
        self.save_directory = os.path.join(package_share_directory, 'data')  # 'data' directory inside the package
        os.makedirs(self.save_directory, exist_ok=True)

    def listener_callback(self, msg: PointCloud2Array):
        # Iterate through each cloud in the PointCloud2Array
        for i, cloud in enumerate(msg.clouds):
            # Convert points into (N,3) NumPy array
            points_iter = point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)
            
            # Convert to a numpy array
            points = np.array([point for point in points_iter])
            
            # Save points to a pickle file in the specified directory
            pickle_file_path = os.path.join(self.save_directory, f"3d_points_{i}.pkl")
            self.get_logger().info(f"Saving points to {pickle_file_path}")
            with open(pickle_file_path, "wb") as f:
                pickle.dump(points, f)
            
        #     if marker.type == Marker.POINTS:
        #         # self.get_logger().info(f"Received marker with {len(marker.points)} points.")
                
        #         # Convert the points to a NumPy array
        #         points = np.array([[p.x, p.y] for p in marker.points])
                
        #         # Save points to a pickle file in the specified directory
        #         pickle_file_path = os.path.join(self.save_directory, f"points_{i}.pkl")
        #         self.get_logger().info(f"Saving points to {pickle_file_path}")
        #         with open(pickle_file_path, "wb") as f:
        #             pickle.dump(points, f)

        #         if len(points) < 2:
        #             self.get_logger().warn(f"Not enough points for line fitting in marker {i}.")
        #             continue

        #         # Perform RANSAC line fitting with multiple lines
        #         lines = self.detect_multiple_lines(points, num_lines=2)

        #         # Visualize the detected lines as markers
        #         line_marker = self.create_line_marker(lines, i, marker.header.frame_id)
        #         line_marker_array.markers.append(line_marker)

        # # Publish the line markers
        # self.line_publisher.publish(line_marker_array)


def main(args=None):
    rclpy.init(args=args)

    # Create the subscriber node
    object_points_subscriber = ObjectPointsSubscriber()

    # Keep the node running until manually interrupted
    rclpy.spin(object_points_subscriber)

    # Shutdown and cleanup when done
    object_points_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
