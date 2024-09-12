import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
import tf_transformations
from measurement_mcts.environment.car import Car

class TwistActionToUnity(Node):
    def __init__(self):
        super().__init__('twist_action_to_unity')

        # Get timestep length parameter
        self.declare_parameter('timestep_length', 0.1)
        self.timestep_length = self.get_parameter('timestep_length').value

        # Subscribers
        self.cmd_subscriber = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_callback,
            10)
        
        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            '/jeep_pose',
            self.pose_callback,
            10)

        # Publisher
        self.pose_publisher = self.create_publisher(PoseStamped, '/jeep_pose_cmd', 10)
        
        # Create a car object to use the model (matching sensing parameters as normal although it is not used)
        self.car_model = Car(60, np.radians(60), np.array([10., 90.]), np.array([-np.pi, np.pi]), state=np.zeros(6))

        # Store the latest state for the next iteration
        self.state = np.zeros(6)
        self.state[[0, 1, 3]] = np.nan # Initialize pose information (x, y, yaw) to NaN as we haven't received it yet
        
        # Store the most recent action vector
        self.action_vec = np.zeros(2)
        
        # Create a timer to run the control loop at the desired rate
        self.create_timer(self.timestep_length, self.control_loop)

    def control_loop(self):
        if np.any(np.isnan(self.state)):
            self.get_logger().warn('No pose data received yet. Skipping command.', throttle_duration_sec=1)
            return
        
        # Call the car model to compute the new pose
        self.state = self.car_model.update(self.timestep_length, self.action_vec, starting_state=self.state)
        
        self.get_logger().info(f'State: {np.round(self.state, 2)}', throttle_duration_sec=0.5)

        # Publish the new pose
        new_pose_msg = PoseStamped()
        new_pose_msg.header.stamp = self.get_clock().now().to_msg()
        new_pose_msg.header.frame_id = 'map'

        # Set the new position
        new_pose_msg.pose.position.x = self.state[0]
        new_pose_msg.pose.position.y = self.state[1]
        new_pose_msg.pose.position.z = 0.0  # Assuming ground vehicle

        # Convert yaw to quaternion using tf_transformations
        new_quat = tf_transformations.quaternion_from_euler(0.0, 0.0, self.state[3])
        new_pose_msg.pose.orientation.x = new_quat[0]
        new_pose_msg.pose.orientation.y = new_quat[1]
        new_pose_msg.pose.orientation.z = new_quat[2]
        new_pose_msg.pose.orientation.w = new_quat[3]

        # Publish the updated pose
        self.pose_publisher.publish(new_pose_msg)

    def cmd_callback(self, msg: Twist):
        # Extract linear and angular commands as an action vector
        self.action_vec = np.array([msg.linear.x, msg.angular.z])
        
    def pose_callback(self, msg: PoseStamped):
        # Extract x, y, and yaw from PoseStamped
        x = msg.pose.position.x
        y = msg.pose.position.y

        # Convert quaternion to yaw using tf_transformations
        orientation_q = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        )
        _, _, yaw = tf_transformations.euler_from_quaternion(orientation_q)
        
        # Update the pose portions of the state
        self.state[0] = x
        self.state[1] = y
        self.state[3] = yaw

def main(args=None):
    rclpy.init(args=args)
    node = TwistActionToUnity()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
