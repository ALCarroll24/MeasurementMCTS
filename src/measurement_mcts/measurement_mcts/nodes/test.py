import rclpy

def main():
    # Initialize the ROS node
    rclpy.init()

    # Create a ROS 2 node
    node = rclpy.create_node('your_node_name')

    # Add your code here

    # Spin the node
    rclpy.spin(node)

    # Clean up
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()