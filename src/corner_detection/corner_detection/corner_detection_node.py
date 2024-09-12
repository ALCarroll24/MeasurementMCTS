import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import numpy as np
from shapely.geometry import MultiPoint
from sklearn.decomposition import PCA

class CornerDetectionNode(Node):
    def __init__(self):
        super().__init__('corner_detection_node')

        # Subscriber to the polygon_marker_array topic
        self.subscription = self.create_subscription(
            MarkerArray,
            '/polygon_marker_array',
            self.polygon_callback,
            10
        )

        # Publisher for the corner points as a Sphere list
        self.publisher = self.create_publisher(MarkerArray, '/corner_marker_array', 10)
        
        self.get_logger().info('Corner Detection Node initialized and running.')

    def polygon_callback(self, msg: MarkerArray):
        corner_marker_array = MarkerArray()
        
        # Clear the previous markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        corner_marker_array.markers.append(clear_marker)

        # Loop over each marker in the received MarkerArray
        for i, marker in enumerate(msg.markers):
            if marker.type == Marker.LINE_STRIP:
                # Extract points from the marker
                points = [(p.x, p.y) for p in marker.points]

                # Perform convex hull and corner extraction
                corners = self.extract_corners(points)

                # Create a new marker for the detected corners
                corner_marker = Marker()
                corner_marker.header.frame_id = marker.header.frame_id
                corner_marker.type = Marker.SPHERE_LIST
                corner_marker.action = Marker.ADD
                corner_marker.id = i
                corner_marker.scale.x = 0.1  # Sphere radius
                corner_marker.scale.y = 0.1
                corner_marker.scale.z = 0.1
                corner_marker.color.r = 0.0  # Red color
                corner_marker.color.g = 0.0
                corner_marker.color.b = 1.0
                corner_marker.color.a = 1.0  # Fully opaque
                corner_marker.pose.orientation.w = 1.0

                # Add the corner points as spheres
                for corner in corners:
                    corner_point = Point()
                    corner_point.x = corner[0]
                    corner_point.y = corner[1]
                    corner_point.z = 0.0  # Assuming 2D, z can be set to 0 or the height of the object
                    corner_marker.points.append(corner_point)

                # Add the corner marker to the marker array
                corner_marker_array.markers.append(corner_marker)

        # Publish the detected corners
        self.publisher.publish(corner_marker_array)

    def extract_corners(self, points):
        # Convert to numpy array
        points_np = np.array(points)

        # Step 1: Compute the convex hull using Shapely
        hull = MultiPoint(points_np).convex_hull
        hull_points = np.array(hull.exterior.coords)

        # Step 2: Apply PCA to align the points
        pca = PCA(n_components=2)
        transformed_points = pca.fit_transform(hull_points)

        # Step 3: Find extreme points in the PCA-aligned space
        min_x_idx = np.argmin(transformed_points[:, 0])
        max_x_idx = np.argmax(transformed_points[:, 0])
        min_y_idx = np.argmin(transformed_points[:, 1])
        max_y_idx = np.argmax(transformed_points[:, 1])

        # Step 4: Collect the extreme points
        extreme_points_pca = [
            transformed_points[min_x_idx],
            transformed_points[max_x_idx],
            transformed_points[min_y_idx],
            transformed_points[max_y_idx]
        ]

        # Step 5: Transform the extreme points back to the original space
        inverse_transformed_points = pca.inverse_transform(extreme_points_pca)

        return inverse_transformed_points

def main(args=None):
    rclpy.init(args=args)
    node = CornerDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
