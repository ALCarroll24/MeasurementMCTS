import rclpy
from measurement_mcts.environment.measurement_control_env import MeasurementControlEnvironment

def main():
    # Initialize the ROS node
    rclpy.init()

    # Create a ROS 2 node
    node = rclpy.create_node('your_node_name')

    # Add your code here
    # For example, print a message
    node.get_logger().info('Hello, world!')

    # Spin the node
    rclpy.spin(node)

    # Clean up
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()