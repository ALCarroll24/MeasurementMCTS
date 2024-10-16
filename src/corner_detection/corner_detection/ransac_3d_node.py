import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from pointcloud_array_msgs.msg import PointCloud2Array
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Point
import open3d as o3d
import numpy as np
import pyransac3d as pyrsc
from scipy.spatial import distance
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
        self.edge_publisher = self.create_publisher(MarkerArray, '/object_edges', 10)
        
        # Set up a folder to save the pickle files (inside the package directory)
        package_share_directory = get_package_share_directory('corner_detection')  # Replace with your actual package name
        self.save_directory = os.path.join(package_share_directory, 'data')  # 'data' directory inside the package
        os.makedirs(self.save_directory, exist_ok=True)

    def listener_callback(self, msg: PointCloud2Array):
        # Create output marker array
        edge_marker_array = MarkerArray()
        
        # Iterate through each cloud in the PointCloud2Array
        for i, cloud in enumerate(msg.clouds):
            # Read points into structured NumPy array (rows are named tuples)
            points_iter = point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)
            
            # Extract each field as a separate array and stack them along the last axis to get a (N,3) array
            points = np.stack((points_iter['x'], points_iter['y'], points_iter['z']), axis=-1)

            # # Save points to a pickle file in the specified directory
            # pickle_file_path = os.path.join(self.save_directory, f"3d_points_{i}.pkl")
            # self.get_logger().info(f"Saving points to {pickle_file_path}")
            # with open(pickle_file_path, "wb") as f:
            #     pickle.dump(points, f)

            # Remove the top plane from the points
            edge_points = self.remove_top_plane(points, percent=80, draw=False)
            
            # Find the corner points
            corner_points = self.find_corner_points(edge_points, thresh=0.2, min_points=100, max_iter=100, plot=False, plot_planes=False)
            
            # Publish a line marker for the detected corner points
            edge_marker = Marker()
            edge_marker.header = cloud.header
            edge_marker.type = Marker.LINE_STRIP
            edge_marker.action = Marker.ADD
            edge_marker.id = i
            edge_marker.ns = "corner_points"
            edge_marker.scale.x = 0.2
            edge_marker.color.r = 1.0
            edge_marker.color.a = 1.0
            if len(corner_points) == 3:
                edge_marker.points.append(Point(x=float(corner_points[1,0]), y=float(corner_points[1,1]), z=float(corner_points[1,2])))
                edge_marker.points.append(Point(x=float(corner_points[0,0]), y=float(corner_points[0,1]), z=float(corner_points[0,2])))
                edge_marker.points.append(Point(x=float(corner_points[2,0]), y=float(corner_points[2,1]), z=float(corner_points[2,2])))
            elif len(corner_points) == 2:
                edge_marker.points.append(Point(x=float(corner_points[1,0]), y=float(corner_points[1,1]), z=float(corner_points[1,2])))
                edge_marker.points.append(Point(x=float(corner_points[0,0]), y=float(corner_points[0,1]), z=float(corner_points[0,2])))
            
            edge_marker_array.markers.append(edge_marker)
            
        self.edge_publisher.publish(edge_marker_array)

    def remove_top_plane(self, points, percent=80, draw=False):
        # Find the upper height which is the height of the upper percentile of points in z direction
        upper_height = np.percentile(points[:,2], percent)
        
        inlier_points = points[np.where(points[:,2] < upper_height)]
        outlier_points = points[np.where(points[:,2] >= upper_height)]

        if draw:
            inlier_pcd = o3d.geometry.PointCloud()
            inlier_pcd.points = o3d.utility.Vector3dVector(inlier_points)
            outlier_pcd = o3d.geometry.PointCloud()
            outlier_pcd.points = o3d.utility.Vector3dVector(outlier_points)

            inlier_pcd.paint_uniform_color([1, 0, 0])
            outlier_pcd.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw_geometries([inlier_pcd, outlier_pcd])

        return inlier_points

    def find_corner_points(self, points, thresh=0.1, min_pts_2planes=50, min_points=100, max_iter=1000, plot=False, plot_planes=False):
        # Calculate the mean z value of all points
        mean_z = np.mean(points[:,2])
        
        # Fit a plane to the points
        plane1 = pyrsc.Plane()
        best_eq1, best_inliers1 = plane1.fit(points, thresh=thresh, minPoints=min_points, maxIteration=max_iter)
        
        
        if plot_planes:
            self.plot_plane_fit(points, best_eq1, best_inliers1)
        
        # If there are enough points left, fit a second plane to the outliers or take the outermost points in first plane
        outliers = np.delete(points, best_inliers1, axis=0)
        # print(f"Number of outliers for second plane: {len(outliers)}")
        if len(outliers) < min_pts_2planes:
            # print("Not enough points for a second plane.")
            # Find the farthest two outermost points of the inliers on the plain
            dists = distance.cdist(points[best_inliers1], points[best_inliers1], 'euclidean') # pairwise distances
            
            # Get the indices of the two farthest points
            i, j = np.unravel_index(np.argmax(dists), dists.shape)
            point1, point2 = points[best_inliers1][i], points[best_inliers1][j]
            
            corner_points = np.vstack((point1, point2))
            
            corner_points = self.project_points_onto_plane(corner_points, best_eq1, mean_z)
            
            if plot:
                self.plot_segmented_points(points, best_inliers1, [], corner_points)
            
            return corner_points
            
        else:
            # Fit a second plane to the outliers and get 3 corner points
            plane2 = pyrsc.Plane()
            best_eq2, best_inliers2 = plane2.fit(outliers, thresh=thresh, minPoints=min_points)
            
            if plot_planes:
                self.plot_plane_fit(outliers, best_eq2, best_inliers2)
            
            # Find the intersection line of the two planes
            intersection_point, direction = self.find_intersection_line(best_eq1, best_eq2)
            
            # Find the point on the line at the mean z value of all points
            t = (mean_z - intersection_point[2]) / direction[2]
            corner_point = intersection_point + t * direction
            
            # Find the farthest point from the corner point projected to the plane for each plane
            z_value = corner_point[2]
            edge_point1 = self.get_edge_corner_point(points[best_inliers1], best_eq1, corner_point, z_value)
            edge_point2 = self.get_edge_corner_point(outliers[best_inliers2], best_eq2, corner_point, z_value)
            
            corner_points = np.vstack((corner_point, edge_point1, edge_point2))
            if plot:
                self.plot_segmented_points(points, best_inliers1, best_inliers2, corner_points)
                
            return corner_points
        
    def project_points_onto_plane(self, points, plane_params, z_value):
        A, B, C, D = plane_params
        normal = np.array([A, B, C])
        normal_squared = np.dot(normal, normal)
        
        # Calculate the distance of each point from the plane
        distances = (np.dot(points, normal) + D) / normal_squared
        
        # Project each point onto the plane
        projections = points - np.outer(distances, normal)
        
        # Place the z value of the projected points at the z_value
        projections[:,2] = z_value
        
        return projections
    
    def get_edge_corner_point(self, points, eq, corner_point, z_value):
        # Find the point on the plane that is farthest from the corner point
        A, B, C, D = eq
        normal = np.array([A, B, C])
        point = corner_point - D * normal / np.dot(normal, normal)
        distances = np.linalg.norm(points - point, axis=1)
        edge_point = points[np.argmax(distances)]
        
        # Place the z value of the edge point at the mean z value of all points
        edge_point[2] = z_value
        return edge_point

    def find_intersection_line(self, plane1_params, plane2_params):
        A1, B1, C1, D1 = plane1_params
        A2, B2, C2, D2 = plane2_params

        # Normal vectors and constants
        A = np.array([[A1, B1, C1],
                    [A2, B2, C2]])
        b = np.array([-D1, -D2])

        # Use least-squares to find the point that minimizes the distance to both planes
        point_on_line, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # Direction vector for the line of intersection
        n1 = np.array([A1, B1, C1])
        n2 = np.array([A2, B2, C2])
        direction = np.cross(n1, n2)

        # Check if the direction is near zero (planes are parallel)
        if np.allclose(direction, 0):
            print("The planes are parallel and do not intersect.")
            return None, None

        return point_on_line, direction

    def plot_segmented_points(self, points, inliers1, inliers2, corner_points):
        # Create point cloud of inliers for first plane fit
        inlier_pcd1 = o3d.geometry.PointCloud()
        inlier_pcd1.points = o3d.utility.Vector3dVector(points[inliers1])
        inlier_pcd1.paint_uniform_color([0, 1, 0])
        
        # Create point cloud of inliers for second plane fit (outliers of first plane fit)
        outliers = np.delete(points, inliers1, axis=0)
        inlier_pcd2 = o3d.geometry.PointCloud()
        inlier_pcd2.points = o3d.utility.Vector3dVector(outliers[inliers2])
        inlier_pcd2.paint_uniform_color([0, 1, 1])
        
        # Create point cloud of outliers for second plane fit (outliers of second plane fit)
        outlier_pcd = o3d.geometry.PointCloud()
        outlier_points = np.delete(outliers, inliers2, axis=0)
        outlier_pcd.points = o3d.utility.Vector3dVector(outlier_points)
        outlier_pcd.paint_uniform_color([1, 0, 0])
        
        # Create point cloud for final corner points
        corner_pcd = o3d.geometry.PointCloud()
        corner_pcd.points = o3d.utility.Vector3dVector(corner_points)
        corner_pcd.paint_uniform_color([1, 0, 1])
        
        o3d.visualization.draw_geometries([inlier_pcd1, inlier_pcd2, outlier_pcd, corner_pcd])

    def plot_plane_fit(self, points, eq, inliers):
        inlier_pcd = o3d.geometry.PointCloud()
        inlier_pcd.points = o3d.utility.Vector3dVector(points[inliers])
        inlier_pcd.paint_uniform_color([0, 1, 0])
        
        outlier_pcd = o3d.geometry.PointCloud()
        outlier_points = np.delete(points, inliers, axis=0)
        outlier_pcd.points = o3d.utility.Vector3dVector(outlier_points)
        outlier_pcd.paint_uniform_color([1, 0, 0])
        
        plane_mesh = self.create_plane_mesh(eq)
        
        o3d.visualization.draw_geometries([inlier_pcd, outlier_pcd, plane_mesh])
        
    def create_plane_mesh(self, plane_params, size=1, color=[0, 1, 0]):
        A, B, C, D = plane_params
        # Normal of the plane
        normal = np.array([A, B, C])
        # Create three points on the plane to define it
        u = np.cross(normal, [1, 0, 0])
        if np.linalg.norm(u) == 0:
            u = np.cross(normal, [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        u, v = u * size, v * size
        # Create four corners of the plane
        origin = -D * normal / np.dot(normal, normal)
        corners = [
            origin + u + v,
            origin + u - v,
            origin - u - v,
            origin - u + v
        ]
        # Create a mesh for the plane
        plane = o3d.geometry.TriangleMesh()
        plane.vertices = o3d.utility.Vector3dVector(corners)
        plane.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [2, 3, 0]])
        plane.paint_uniform_color(color)
        return plane
    
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
