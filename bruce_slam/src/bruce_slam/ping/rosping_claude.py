import numpy as np
import gtsam
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation

from .PingManager_c import PingManager
from .ping_slam_frontend_c import PingSLAMFrontEnd
from bruce_slam.slam import SLAM


class RosPingSLAMFrontEnd(PingSLAMFrontEnd, Node):
    """
    ROS 2 node that supplies PingSLAMFrontEnd with a real dead-reckoning
    pose source: MAVROS's local position odometry topic.

    WHY A SUBSCRIBER + CACHE, NOT A DIRECT SERVICE CALL:
    PingManager's callback (_on_profile) fires asynchronously whenever a
    new ping arrives, and needs a pose "right now" with minimal latency.
    Rather than blocking to request the current pose from MAVROS on every
    single ping, we keep a cheap rclpy subscription running in the
    background and always read whatever the latest cached message is.
    This does mean each point is timestamped with the most recent
    odometry sample rather than one taken at the exact ping instant -
    for a Ping1D's ping rate this offset is generally small, but if you
    see it mattering, this is the place to add proper timestamp
    interpolation between two odometry samples.
    """

    def __init__(self, ping_manager: PingManager, slam: SLAM, **kwargs):
        Node.__init__(self, "ping_slam_front_end")
        PingSLAMFrontEnd.__init__(self, ping_manager, slam, **kwargs)

        self._latest_odom: Odometry = None

        # /mavros/local_position/odom publishes nav_msgs/Odometry in the
        # local ENU frame - the same frame your orca4/MAVROS-based
        # position control already works in, so no extra frame
        # conversion is needed here beyond quaternion -> yaw.
        self.create_subscription(
            Odometry,
            "/mavros/local_position/odom",
            self._odom_callback,
            10,
        )

        self.get_logger().info("RosPingSLAMFrontEnd subscribed to MAVROS odometry")

    def _odom_callback(self, msg: Odometry):
        self._latest_odom = msg

    def get_current_dr_pose(self) -> gtsam.Pose2:
        """
        SLAM here is 2D (gtsam.Pose2: x, y, theta), so we take the
        planar (x, y) position and yaw only, discarding depth/roll/pitch
        - those are preserved separately in dr_pose3 below for logging,
        matching how the existing SLAM code already keeps a full 3D
        dr_pose3 alongside the 2D solved pose.
        """
        if self._latest_odom is None:
            return None

        p = self._latest_odom.pose.pose.position
        q = self._latest_odom.pose.pose.orientation
        yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]

        return gtsam.Pose2(p.x, p.y, yaw)

    def _to_pose3(self, pose2: gtsam.Pose2) -> np.ndarray:
        if self._latest_odom is None:
            return np.zeros(6, dtype=np.float32)

        p = self._latest_odom.pose.pose.position
        q = self._latest_odom.pose.pose.orientation
        roll, pitch, yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")

        return np.array([p.x, p.y, p.z, roll, pitch, yaw], dtype=np.float32)

    def _ros_time_now(self):
        return self.get_clock().now()


def main():
    rclpy.init()

    # Wire up PingManager and SLAM as covered previously, then hand both
    # to the ROS-aware front end.
    ping_manager = PingManager(device="/dev/ttyUSB0", baudrate=115200, udp=None, live=True)
    slam = SLAM()
    slam.configure()

    front_end = RosPingSLAMFrontEnd(ping_manager, slam)

    # PingManager's sonar_pinging() is an asyncio coroutine, while rclpy
    # uses its own spin loop - running both requires either rclpy's
    # asyncio-compatible executor or a separate thread running the
    # asyncio loop. A common pattern: run sonar_pinging() in its own
    # thread via asyncio.run(), since register_scan_update_callback's
    # callback (_on_profile) is plain synchronous code and safe to call
    # from that thread as long as it doesn't touch rclpy internals
    # directly (it doesn't - it only touches slam/gtsam objects and the
    # cached _latest_odom attribute).
    import asyncio
    import threading

    ping_thread = threading.Thread(
        target=lambda: asyncio.run(ping_manager.sonar_pinging()),
        daemon=True,
    )
    ping_thread.start()

    rclpy.spin(front_end)
    rclpy.shutdown()


if __name__ == "__main__":
    main()