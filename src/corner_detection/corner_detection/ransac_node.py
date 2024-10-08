import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import numpy as np
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
            MarkerArray,
            '/object_points_array',
            self.listener_callback,
            10  # QoS history depth
        )

        # Publisher to visualize the detected lines
        self.line_publisher = self.create_publisher(MarkerArray, '/detected_lines_array', 10)
        
        # Set up a folder to save the pickle files (inside the package directory)
        package_share_directory = get_package_share_directory('corner_detection')  # Replace with your actual package name
        self.save_directory = os.path.join(package_share_directory, 'data')  # 'data' directory inside the package
        os.makedirs(self.save_directory, exist_ok=True)

    def listener_callback(self, msg: MarkerArray):
        line_marker_array = MarkerArray()

        # Iterate through each marker in the MarkerArray
        for i, marker in enumerate(msg.markers):
            if marker.type == Marker.POINTS:
                # self.get_logger().info(f"Received marker with {len(marker.points)} points.")
                
                # Convert the points to a NumPy array
                points = np.array([[p.x, p.y] for p in marker.points])
                
                # Save points to a pickle file in the specified directory
                pickle_file_path = os.path.join(self.save_directory, f"points_{i}.pkl")
                self.get_logger().info(f"Saving points to {pickle_file_path}")
                with open(pickle_file_path, "wb") as f:
                    pickle.dump(points, f)

                if len(points) < 2:
                    self.get_logger().warn(f"Not enough points for line fitting in marker {i}.")
                    continue

                # Perform RANSAC line fitting with multiple lines
                lines = self.detect_multiple_lines(points, num_lines=2)

                # Visualize the detected lines as markers
                line_marker = self.create_line_marker(lines, i, marker.header.frame_id)
                line_marker_array.markers.append(line_marker)

        # Publish the line markers
        self.line_publisher.publish(line_marker_array)

    def detect_multiple_lines(self, points, num_lines=3):
        """Detect multiple lines in a cluster using RANSAC."""
        lines = []
        remaining_points = points.copy()

        for _ in range(num_lines):
            if len(remaining_points) < 2:
                break

            # Perform RANSAC in 2D (fit line y = ax + b)
            X = remaining_points[:, 0].reshape(-1, 1)  # Extract X coordinates
            Y = remaining_points[:, 1]  # Extract Y coordinates

            ransac = RANSACRegressor()
            ransac.fit(X, Y)

            # Get the line parameters: slope (a) and intercept (b)
            a = ransac.estimator_.coef_[0]
            b = ransac.estimator_.intercept_

            # Define the line segment endpoints for visualization
            x_min, x_max = np.min(X), np.max(X)
            y_min, y_max = a * x_min + b, a * x_max + b
            lines.append(((x_min, y_min), (x_max, y_max)))

            # Get the inliers (points close to the line)
            inlier_mask = ransac.inlier_mask_
            inlier_points = remaining_points[inlier_mask]

            # Remove inlier points from the remaining points for the next iteration
            remaining_points = remaining_points[~inlier_mask]

        return lines

    def create_line_marker(self, lines, marker_id, frame_id):
        """Create a Marker for the detected lines."""
        line_marker = Marker()
        line_marker.header.frame_id = frame_id
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = "detected_lines"
        line_marker.id = marker_id
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.05  # Line width

        # Set color (green for lines)
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0

        # Add the line points to the marker
        for line in lines:
            p1, p2 = line
            start_point = Point(x=p1[0], y=p1[1], z=0.0)
            end_point = Point(x=p2[0], y=p2[1], z=0.0)
            line_marker.points.append(start_point)
            line_marker.points.append(end_point)

        return line_marker

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
