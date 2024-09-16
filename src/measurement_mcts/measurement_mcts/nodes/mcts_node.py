import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
import tf_transformations
from measurement_mcts.mcts.mcts import mcts_search, get_best_trajectory
from measurement_mcts.environment.measurement_control_env import MeasurementControlEnvironment

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
            
        # Save the most recent car state within this node
        self.car_state = np.zeros(6)
        self.got_initial_pose = False
        
        # Maintain the object dataframe for the environment
        self.object_df = pd.DataFrame()

        # Publisher to move the Jeep in Unity
        self.pose_publisher = self.create_publisher(PoseStamped, '/jeep_pose_cmd', 10)

        # Create a timer to run the control loop at the desired rate
        self.create_timer(self.timestep_length, self.control_loop)
        
    def mcts_search_action(self) -> np.ndarray:
        """Run the MCTS search and get the next action to take"""
        # Run the MCTS search
        best_action, root = mcts_search(self.env, None, self.env.get_state(), learning_iterations=self.learning_iterations,
                                    explore_factor=0.3, discount_factor=1.0)
        
        # Return the best action
        return best_action
    
    def control_loop(self):
        """Control loop to run the MCTS search and send the action to Jeep"""
        # If we haven't received the initial pose yet, skip this iteration
        if not self.got_initial_pose:
            self.get_logger().warn('No pose data received yet. Skipping command.', throttle_duration_sec=1)
            return
        
        # Set the environment's car and object state before searching
        self.env.car.set_state(self.car_state)
        self.env.object_manager.set_df(self.object_df)
        
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
        quaternion = tf_transformations.quaternion_from_euler(0, 0, self.car_state[3])
        new_pose_msg.pose.orientation.x = quaternion[0]
        new_pose_msg.pose.orientation.y = quaternion[1]
        new_pose_msg.pose.orientation.z = quaternion[2]
        new_pose_msg.pose.orientation.w = quaternion[3]
        
        # Publish the new pose
        self.pose_publisher.publish(new_pose_msg)
        
    def corner_callback(self, msg: MarkerArray):
        """Callback for receiving the detected corner locations"""
        # Initialize a numpy array to store the corner locations
        corners = np.zeros((len(msg.markers), 4, 2))
        
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
            corners[i] = np.array(points)
        
        # Use the corner locations and apply data association to get observation dictionary
        obs_dict = self.env.corner_data_association(corners, self.object_df)
        
        # Apply the observations to the environment and maintain new object state dataframe
        new_df, trace_delta_sum = self.env.apply_observation(obs_dict, self.object_df, self.car_state)
        
        # Save the new object dataframe
        self.object_df = new_df
        
    def pose_callback(self, msg: PoseStamped):
        """Callback for receiving the Jeep's pose from Unity"""
        # Get the Jeep's current position and yaw
        position = np.array([msg.pose.position.x, msg.pose.position.y])
        
        # Use tf transformations to convert the quaternion to a yaw
        quaternion = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        yaw = tf_transformations.euler_from_quaternion(quaternion)[2]
        
        # Update the car state and flag that we have received the initial pose
        self.car_state[[0, 1, 3]] = np.array([position[0], position[1], yaw])
        
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