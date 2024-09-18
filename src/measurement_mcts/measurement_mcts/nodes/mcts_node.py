import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import timeit
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from measurement_mcts.mcts.mcts import mcts_search, get_best_trajectory
from measurement_mcts.environment.measurement_control_env import MeasurementControlEnvironment
from measurement_mcts.environment.object_manager import ObjectTuple, get_empty_df
from measurement_mcts.utils.utils import get_ellipse_scaling

class MCTSNode(Node):
    def __init__(self):
        super().__init__('mcts_node')

        # Get timestep length parameter
        self.declare_parameter('timestep_length', 0.1)
        self.timestep_length = self.get_parameter('timestep_length').value
        
        # Get sim or real parameter
        self.declare_parameter('sim', True)
        self.sim = self.get_parameter('sim').value
        
        # Get the number of learning iterations parameter
        self.declare_parameter('learning_iterations', 100)
        self.learning_iterations = self.get_parameter('learning_iterations').value

        # Create the environment (don't reset since we want to use real objects)
        self.env = MeasurementControlEnvironment(init_reset=False)

        if self.sim:
            # Subscribe to the location of the Jeep in Unity
            self.create_subscription(
                PoseStamped,
                '/jeep_pose',
                self.pose_callback,
                10)
            
        # Subscribe to the detected corner locations
        self.create_subscription(
            MarkerArray,
            '/corner_marker_array',
            self.corner_callback,
            10)
            
        # Publisher to move the Jeep in Unity
        self.pose_publisher = self.create_publisher(PoseStamped, '/jeep_pose_cmd', 10)
        
        # Visualization publishers
        self.object_markers_publisher = self.create_publisher(MarkerArray, '/object_markers', 10)
        
        # Save the most recent car state within this node
        self.car_state = np.zeros(6)
        self.got_initial_pose = False
        
        # Maintain the object dataframe for the environment
        self.object_df = get_empty_df()
        
        # Maintain the exploration grid for the environment
        self.grid = self.env.explore_grid.reset()

        # Create a timer to run the control loop at the desired rate
        self.create_timer(self.timestep_length, self.control_loop)
        
    def mcts_search_action(self) -> np.ndarray:
        """Run the MCTS search and get the next action to take"""
        start_time = timeit.default_timer()
        # Run the MCTS search
        best_action, root = mcts_search(self.env, None, self.env.get_state(), learning_iterations=self.learning_iterations,
                                    explore_factor=0.3, discount_factor=1.0)
        print(f'MCTS search took {timeit.default_timer() - start_time} seconds.')
        # Return the best action
        return best_action
    
    def control_loop(self):
        """Control loop to run the MCTS search and send the action to Jeep"""
        # If we haven't received the initial pose yet, skip this iteration
        if not self.got_initial_pose:
            self.get_logger().warn('No pose data received yet. Skipping command.', throttle_duration_sec=1)
            return
        
        # Set the environment's state to the newest maintained by this class (real world state)
        self.env.set_state([self.car_state, self.object_df, self.grid])
        
        # Run MCTS search to get the next action
        action = self.mcts_search_action()
        
        # Update the car state with the new action
        self.car_state = self.env.car.update(self.timestep_length, action, starting_state=self.car_state)
        
        # Publish the new pose
        new_pose_msg = PoseStamped()
        new_pose_msg.header.stamp = self.get_clock().now().to_msg()
        new_pose_msg.header.frame_id = 'map'

        # Set the new position
        new_pose_msg.pose.position.x = self.car_state[0]
        new_pose_msg.pose.position.y = self.car_state[1]
        
        # Set the new orientation
        quaternion = quaternion_from_euler(0, 0, self.car_state[3])
        new_pose_msg.pose.orientation.x = quaternion[0]
        new_pose_msg.pose.orientation.y = quaternion[1]
        new_pose_msg.pose.orientation.z = quaternion[2]
        new_pose_msg.pose.orientation.w = quaternion[3]
        
        # Publish the new pose
        self.pose_publisher.publish(new_pose_msg)
        
    def corner_callback(self, msg: MarkerArray):
        """Callback for receiving the detected corner locations"""
        # Initialize a numpy array to store the corner locations
        corner_list = []
        
        # Iterate over each marker in the MarkerArray (one OOI per marker)
        for i in range(len(msg.markers)):
            # Skip markers that are not of type 0 (Add or Modify)
            if msg.markers[i].action != 0:
                continue
            
            # Warn if the marker does not have 4 corners and skip it
            if len(msg.markers[i].points) != 4:
                self.get_logger().warn(f'OOI {i} does not have 4 corners. Skipping.', throttle_duration_sec=1)
                continue
            
            # Extract the points from the marker
            points = [(p.x, p.y) for p in msg.markers[i].points]
            
            # Save the points to the corners for this OOI
            corner_list.append(np.array(points))
        
        # Stack the corner list to get the corners in the correct format
        corners = np.stack(corner_list)
        
        # Use the corner locations and apply data association to get observation dictionary (new objects are added to df and removed from corners)
        obs_dict, new_object_df, new_corners = self.env.corner_data_association(corners, self.object_df)
        
        # Apply the observations to the environment and maintain new object state dataframe
        new_df, trace_delta_sum = self.env.apply_observation(obs_dict, new_object_df, self.car_state, new_corners)
        
        # Save the new object dataframe
        self.object_df = new_df
        
        # Publish the marker visualization of the new object locations
        self.publish_object_markers(self.object_df)
        
    def publish_object_markers(self, object_df: pd.DataFrame):
        """Publish the object markers to visualize the object locations"""
        # Create a MarkerArray to hold the object markers
        marker_array = MarkerArray()
        
        # Add a clear all marker to remove previous markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        
        # Iterate over each object in the dataframe
        for tuple in object_df.itertuples():
            points = tuple.points
            cov = tuple.covariances
            
            # Iterate over each corner for the object
            for i in range(4):
                center_marker = Marker()
                center_marker.header.frame_id = 'map'
                center_marker.header.stamp = self.get_clock().now().to_msg()
                center_marker.id = tuple.ooi_id * 8 + 2*i
                center_marker.type = Marker.SPHERE
                center_marker.action = Marker.ADD
                center_marker.pose.position.x = points[i][0]
                center_marker.pose.position.y = points[i][1]
                center_marker.pose.position.z = 2.0
                center_marker.scale.x = 0.3
                center_marker.scale.y = 0.3
                center_marker.scale.z = 0.1
                center_marker.color.a = 1.0
                center_marker.color.r = 0.0
                center_marker.color.g = 0.0
                center_marker.color.b = 1.0
                marker_array.markers.append(center_marker)
                
                cov_marker = Marker()
                cov_marker.header.frame_id = 'map'
                cov_marker.header.stamp = self.get_clock().now().to_msg()
                cov_marker.id = tuple.ooi_id * 8 + 2*i + 1
                cov_marker.type = Marker.SPHERE
                eigvals, eigvec1_angle = get_ellipse_scaling(cov[i])
                cov_marker.pose.position.x = points[i][0]
                cov_marker.pose.position.y = points[i][1]
                cov_marker.pose.position.z = 2.0
                cov_marker.scale.x = eigvals[0]
                cov_marker.scale.y = eigvals[1]
                cov_marker.scale.z = 0.1
                cov_quat = quaternion_from_euler(0, 0, eigvec1_angle)
                cov_marker.pose.orientation.x = cov_quat[0]
                cov_marker.pose.orientation.y = cov_quat[1]
                cov_marker.pose.orientation.z = cov_quat[2]
                cov_marker.pose.orientation.w = cov_quat[3]
                cov_marker.color.a = 0.5
                cov_marker.color.r = 0.0
                cov_marker.color.g = 1.0
                cov_marker.color.b = 0.0
                marker_array.markers.append(cov_marker)
                
        # Publish the marker array
        self.object_markers_publisher.publish(marker_array)
        
    def pose_callback(self, msg: PoseStamped):
        """Callback for receiving the Jeep's pose from Unity"""
        # Get the Jeep's current position and yaw
        position = np.array([msg.pose.position.x, msg.pose.position.y])
        
        # Use tf transformations to convert the quaternion to a yaw
        quaternion = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        yaw = euler_from_quaternion(quaternion)[2]
        
        # Update the car state and flag that we have received the initial pose
        self.car_state[[0, 1, 3]] = np.array([position[0], position[1], yaw])
        
        # Update the grid state with the car state
        self.grid, num_explored = self.env.explore_grid.update(self.grid, self.car_state, object_df=self.object_df)
        
        # If this is the first time we have received the pose, set the car's state within the environment to initialize it
        if not self.got_initial_pose:
            self.env.car.set_state(self.car_state)
            self.got_initial_pose = True
            
def main(args=None):
    rclpy.init(args=args)

    mcts_node = MCTSNode()

    rclpy.spin(mcts_node)

    mcts_node.destroy_node()
    rclpy.shutdown()