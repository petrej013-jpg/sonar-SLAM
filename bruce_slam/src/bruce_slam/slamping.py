import time
import numpy as np
import gtsam

from bruce_slam.slam import SLAM
from bruce_slam.slam_objects import Keyframe

# The Ping1D driver we wrote previously
from .ping_manager import PingManager


class PingSLAMFrontEnd:
    """
    Bridges a single-beam Ping1D sonar into bruce_slam's SLAM backend.

    WHY THIS CLASS EXISTS:
    SLAM.get_points() and Keyframe.points both assume every keyframe
    already owns a 2D point cloud, the way one completed Oculus sweep
    naturally provides. A single Ping1D ping gives exactly one range
    value along one fixed beam direction - never a cloud.

    This front end closes that gap by accumulating individual range
    returns over time. Each return is converted to an (x, y) point using
    the vehicle's own dead-reckoning pose at the exact moment of that
    ping - so as the vehicle moves and turns, successive points land in
    different places, gradually building up a synthetic scan the same
    way a rotating sonar builds a scan from many angle-tagged returns,
    except the "rotation" here comes from the vehicle's own maneuvering
    rather than a motor.

    Only once a buffer has enough points AND enough heading diversity is
    it handed off to SLAM as a keyframe's point cloud. This second gate
    matters because a vehicle holding still, or driving dead straight,
    produces a string of nearly collinear points along the beam
    direction - which ICP can slide along freely and gain almost no
    real constraint from.
    """

    def __init__(
        self,
        ping_manager: PingManager,
        slam: SLAM,
        beam_frame_offset: gtsam.Pose2 = gtsam.Pose2(0, 0, 0),
        min_accum_points: int = 50,
        min_accum_spread_deg: float = 20.0,
        max_accum_duration: float = 5.0,
        min_confidence: int = 50,
    ):
        """
        Args:
            ping_manager: A running PingManager wrapping the Ping1D device.
            slam: The SLAM instance this front end feeds keyframes into.
            beam_frame_offset: Pose2 describing where the sonar's single
                beam points relative to the vehicle body frame (e.g. a
                forward-facing mount is identity; a beam angled down/
                sideways would have a nonzero theta here). This replaces
                the "aperture"/FOV concept entirely - there's no cone,
                just one fixed direction.
            min_accum_points: Minimum returns needed before a buffer can
                become a keyframe.
            min_accum_spread_deg: Minimum heading swing (degrees) the
                vehicle must have covered during accumulation. This is
                the substitute for "did we get a real 2D shape, not just
                a line."
            max_accum_duration: Hard cap (seconds) on how long we wait
                for both gates - after this we submit whatever we have
                rather than stalling SLAM indefinitely. Downstream
                min_points/overlap checks in SLAM will naturally
                down-weight or reject a poor (line-like) cloud.
            min_confidence: Ping1D confidence (%) below which a return is
                discarded outright. Unlike a multibeam scan, a single bad
                point here has no other points in the same ping to
                outvote it, so we filter aggressively at the source.
        """
        self.ping_manager = ping_manager
        self.slam = slam
        self.beam_frame_offset = beam_frame_offset
        self.min_accum_points = min_accum_points
        self.min_accum_spread_deg = min_accum_spread_deg
        self.max_accum_duration = max_accum_duration
        self.min_confidence = min_confidence

        # Rolling buffer for the CURRENT (not-yet-submitted) accumulation
        # window. Points are stored in the frame of self._reference_pose,
        # i.e. relative to wherever the vehicle was when this window
        # started - matching how Keyframe.points is expected to be in a
        # keyframe-local frame (see get_points() / transf_points in slam.py).
        self._buffer_points = []
        self._buffer_headings = []
        self._accum_start_time = None
        self._reference_pose = None

        # PingManager calls this every time sonar_pinging() produces a
        # cleaned profile - see the sonar_pinging loop in ping_manager.py
        self.ping_manager.register_scan_update_callback(self._on_profile)

    def _on_profile(self, profile: np.ndarray):
        """
        Called once per ping. Converts the ping's distance estimate into
        a single point and folds it into the rolling accumulation buffer.

        We use get_distance_estimate() (the device's own best-guess
        distance+confidence) rather than trying to pick a peak out of the
        raw profile array ourselves - the profile is still available via
        self.ping_manager.get_data() if you later want a CFAR-based peak
        pick instead (SonarFeatureExtraction already supports this), but
        the built-in estimate is a reasonable starting point.
        """
        distance_info = self.ping_manager.get_distance_estimate()
        if distance_info is None:
            return

        range_m = distance_info["distance_m"]
        confidence = distance_info["confidence"]

        if confidence < self.min_confidence:
            return

        dr_pose = self.get_current_dr_pose()
        if dr_pose is None:
            return

        if self._reference_pose is None:
            self._reference_pose = dr_pose
            self._accum_start_time = time.time()

        # Where is the beam pointing right now, in the global dead-
        # reckoning frame? Compose the fixed sensor-to-body offset onto
        # the vehicle's current pose, then express that relative to the
        # window's reference pose so the resulting point lands in the
        # keyframe-local frame that SLAM expects.
        beam_pose_global = dr_pose.compose(self.beam_frame_offset)
        beam_pose_local = self._reference_pose.between(beam_pose_global)

        # The measured point sits `range_m` straight out along the
        # beam's own local x-axis.
        point_local = beam_pose_local.transformFrom(gtsam.Point2(range_m, 0.0))

        self._buffer_points.append([point_local[0], point_local[1]])
        self._buffer_headings.append(np.degrees(dr_pose.theta()))

        self._maybe_submit_keyframe(dr_pose)

    def _maybe_submit_keyframe(self, latest_dr_pose: gtsam.Pose2):
        """Check accumulation gates and, if met, hand a Keyframe to SLAM."""
        n_points = len(self._buffer_points)
        elapsed = time.time() - self._accum_start_time
        heading_spread = (
            max(self._buffer_headings) - min(self._buffer_headings)
            if self._buffer_headings else 0.0
        )

        enough_points = n_points >= self.min_accum_points
        enough_spread = heading_spread >= self.min_accum_spread_deg
        timed_out = elapsed >= self.max_accum_duration

        if (enough_points and enough_spread) or (timed_out and n_points > 0):
            self._submit_keyframe(latest_dr_pose)

    def _submit_keyframe(self, latest_dr_pose: gtsam.Pose2):
        """
        Package the accumulated buffer as a Keyframe and feed it through
        the same pipeline an Oculus-derived keyframe would go through:
        prior (if first), then sequential + non-sequential scan matching,
        then commit via update_factor_graph. This part of SLAM's public
        interface (is_keyframe / add_prior / add_sequential_scan_matching
        / add_nonsequential_scan_matching / update_factor_graph) doesn't
        need to change at all - it only cares that keyframe.points exists,
        not how it was produced.
        """
        points = np.array(self._buffer_points, dtype=np.float32)

        keyframe = Keyframe(
            time=self._ros_time_now(),
            dr_pose=latest_dr_pose,
            dr_pose3=self._to_pose3(latest_dr_pose),
            points=points,
        )
        keyframe.pose = latest_dr_pose

        if self.slam.is_keyframe(keyframe):
            if not self.slam.keyframes:
                self.slam.add_prior(keyframe)
                self.slam.keyframes.append(keyframe)
            else:
                self.slam.add_sequential_scan_matching(keyframe)
                self.slam.add_nonsequential_scan_matching()

            self.slam.update_factor_graph(keyframe)

        # Reset the window, anchored at the vehicle's current pose so the
        # next buffer starts fresh from here.
        self._buffer_points = []
        self._buffer_headings = []
        self._reference_pose = latest_dr_pose
        self._accum_start_time = time.time()

    def get_current_dr_pose(self) -> gtsam.Pose2:
        """
        Placeholder: return the vehicle's current dead-reckoning Pose2.

        This front end deliberately does not compute odometry itself -
        it needs SOME external source of (x, y, yaw) at ping time. Wire
        this to your actual navigation stack, e.g. a ROS 2 subscriber
        caching the latest MAVROS/EKF odometry message, matching the
        MAVROS-based control path already in use for MOAT.
        """
        raise NotImplementedError

    def _to_pose3(self, pose2: gtsam.Pose2):
        """
        Placeholder: build the 3D dead-reckoning pose (x, y, z, roll,
        pitch, yaw) used for Keyframe.dr_pose3 logging/plotting. Wire
        this to the same odometry source as get_current_dr_pose(),
        including depth (z) and roll/pitch, which the 2D SLAM pose
        itself discards but dr_pose3 keeps around for reference.
        """
        raise NotImplementedError

    def _ros_time_now(self):
        """Placeholder for your ROS 2 clock/timestamp source."""
        raise NotImplementedError