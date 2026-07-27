import time
from dataclasses import dataclass
from typing import Optional

from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation


@dataclass
class OdometryData:
    """A single fused pose reading from MAVROS, in the local ENU frame."""
    stamp: float          # ROS message timestamp (seconds, from the message header)
    received_at: float    # wall-clock time this reading was cached (time.monotonic())
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


class MavrosOdometryCache:
    """
    Subscribes to MAVROS's fused local-position odometry and keeps the
    latest reading available for cheap, non-blocking reads - the same
    typed-getter shape as BlueOSSLAM's DataManager, but backed by a real
    rclpy subscription (DDS pub/sub) instead of REST polling. MAVROS's
    /mavros/local_position/odom already fuses IMU/barometer/etc. into one
    position+orientation estimate, so unlike DataManager there's nothing
    to fuse here - just cache the latest message and expose it cleanly.

    WHY A CACHE, NOT A DIRECT CALL PER PING:
    PingManager's callback fires asynchronously whenever a new ping
    arrives and needs a pose "right now" with minimal latency. Rather
    than blocking to request the current pose from MAVROS on every
    single ping, this keeps a cheap subscription running in the
    background and always returns whatever the latest cached message is.

    WHY is_stale() MATTERS:
    A dropped MAVROS connection doesn't raise an exception - it just
    stops delivering new messages, and get_latest() would keep silently
    returning the last pose forever, making it look like the vehicle
    stopped moving. Callers should check is_stale() before trusting a
    reading and skip that ping rather than feed SLAM a frozen pose.
    """

    def __init__(self, node, topic: str = "/mavros/local_position/odom", max_age_s: float = 0.5):
        """
        Args:
            node: the rclpy Node to attach this subscription to
            topic: MAVROS odometry topic (local ENU frame)
            max_age_s: how old a cached reading can be before is_stale()
                       reports True - tune this once you can measure your
                       actual MAVROS publish rate; 0.5s is a conservative
                       starting point, not a measured value
        """
        self._max_age_s = max_age_s
        self._latest: Optional[OdometryData] = None

        node.create_subscription(Odometry, topic, self._on_odom, 10)
        node.get_logger().info(f"MavrosOdometryCache subscribed to {topic}")

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        roll, pitch, yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")

        self._latest = OdometryData(
            stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            received_at=time.monotonic(),
            x=p.x, y=p.y, z=p.z,
            roll=roll, pitch=pitch, yaw=yaw,
        )

    def get_latest(self) -> Optional[OdometryData]:
        """Return the most recent cached reading, or None if nothing has arrived yet."""
        return self._latest

    def is_stale(self) -> bool:
        """True if we've never received a message, or the last one is older than max_age_s."""
        if self._latest is None:
            return True
        return (time.monotonic() - self._latest.received_at) > self._max_age_s

    def age(self) -> Optional[float]:
        """Seconds since the last reading was received, or None if none yet."""
        if self._latest is None:
            return None
        return time.monotonic() - self._latest.received_at