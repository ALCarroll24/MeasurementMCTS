import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class UnityPoseToTF(Node):
    def __init__(self):
        super().__init__('unity_pose_to_tf')

        # Get the parameter for setting the publish rate
        self.declare_parameter('publish_rate', 100)
        self.publish_rate = self.get_parameter('publish_rate').value

        # Create a TransformBroadcaster
        self.br = TransformBroadcaster(self)

        # Subscribe to PoseStamped topic
        self.subscription = self.create_subscription(
            PoseStamped,
            '/jeep_pose',
            self.pose_callback,
            10)
        
        # Initialize the output transform to none until it is first created
        self.t = None
        
        # Create a timer to publish the transform at the desired rate
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_transform)

    def publish_transform(self):
        if self.t is None:
            self.get_logger().info('Waiting for first pose message to publish transform', throttle_duration_sec=1)
            return
        
        # Publish the transform
        self.br.sendTransform(self.t)

    def pose_callback(self, msg: PoseStamped):
        # Create a TransformStamped message
        self.t = TransformStamped()

        # Set the timestamp and frame information
        self.t.header.stamp = msg.header.stamp
        self.t.header.frame_id = 'map'            # Parent unity frame
        self.t.child_frame_id = 'base_footprint'  # Child jeep frame

        # Set the translation from PoseStamped
        self.t.transform.translation.x = msg.pose.position.x
        self.t.transform.translation.y = msg.pose.position.y
        self.t.transform.translation.z = msg.pose.position.z

        # Set the rotation (quaternion) from PoseStamped
        self.t.transform.rotation.x = msg.pose.orientation.x
        self.t.transform.rotation.y = msg.pose.orientation.y
        self.t.transform.rotation.z = msg.pose.orientation.z
        self.t.transform.rotation.w = msg.pose.orientation.w

def main(args=None):
    rclpy.init(args=args)
    node = UnityPoseToTF()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
