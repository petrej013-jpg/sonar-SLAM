import numpy as np
import gtsam
import rclpy
from rclpy.node import Node

from ..ping.PingManager_c import PingManager
from ..ping.ping_slam_frontend_c import PingSLAMFrontEnd
from ..odometry.mavros_odom_cache_c import MavrosOdometryCache
from bruce_slam.slam.slam import SLAM


class RosPingSLAMFrontEnd(PingSLAMFrontEnd, Node):
    """
    ROS 2 node that supplies PingSLAMFrontEnd with a real dead-reckoning
    pose source: MAVROS's local position odometry topic, via
    MavrosOdometryCache. See that class for why a cache instead of a
    blocking per-ping request, and why staleness checking matters.
    """

    def __init__(self, ping_manager: PingManager, slam: SLAM, **kwargs):
        Node.__init__(self, "ping_slam_front_end")
        PingSLAMFrontEnd.__init__(self, ping_manager, slam, **kwargs)

        self._odom_cache = MavrosOdometryCache(self)

    def get_current_dr_pose(self) -> gtsam.Pose2:
        """
        SLAM here is 2D (gtsam.Pose2: x, y, theta), so we take the
        planar (x, y) position and yaw only - depth/roll/pitch are
        preserved separately in _to_pose3() below, matching how the
        existing SLAM code keeps a full 3D dr_pose3 alongside the 2D
        solved pose.
        """
        if self._odom_cache.is_stale():
            self.get_logger().warning(
                f"MAVROS odometry stale (age={self._odom_cache.age()}s) - skipping this ping"
            )
            return None

        odom = self._odom_cache.get_latest()
        return gtsam.Pose2(odom.x, odom.y, odom.yaw)

    def _to_pose3(self, pose2: gtsam.Pose2) -> np.ndarray:
        odom = self._odom_cache.get_latest()
        if odom is None:
            return np.zeros(6, dtype=np.float32)

        return np.array(
            [odom.x, odom.y, odom.z, odom.roll, odom.pitch, odom.yaw],
            dtype=np.float32,
        )

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
