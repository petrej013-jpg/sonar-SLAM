from typing import Any, Optional
from enum import Enum

import gtsam
import numpy as np


from ..utils.conversions import *
from ..utils.visualization import *
from ..utils.io import *


class STATUS(Enum):
    """A class for the status of a ICP call"""

    NOT_ENOUGH_POINTS = "Not enough points"
    LARGE_TRANSFORMATION = "Large transformation"
    NOT_ENOUGH_OVERLAP = "Not enough overlap"
    NOT_CONVERGED = "Not converged"
    INITIALIZATION_FAILURE = "Initialization failure"
    SUCCESS = "Success"

    def __init__(self, *args, **kwargs):
        """Class constructor"""
        Enum.__init__(*args, **kwargs)
        self.description = None

    def __bool__(self) -> bool:
        """boolean override

        Returns:
            bool: returns true if the status is SUCCESS else returns false
        """
        return self == STATUS.SUCCESS

    def __nonzero__(self) -> bool:
        """_summary_

        Returns:
            bool: returns true if the status is SUCCESS else returns false
        """
        return self == STATUS.SUCCESS

    def __str__(self) -> str:
        """Convert this class to a string

        Returns:
            str: printable version of this class
        """
        if self.description:
            return self.value + ": " + self.description
        else:
            return self.value


class Keyframe(object):
    """Keyframe object for SLAM. In this SLAM solution we utilze a factor graph
    and a list of keyframes. The keyframe stores everything from updated pose to
    points observed at that timestep.
    """

    def __init__(
        self,
        status: bool = True,
        time: Any = None,
        dr_pose3: gtsam.Pose3 = None,
        points: np.array = np.zeros((0, 2), np.float32),
        cov: np.array = None,
        dr_pose: Optional[gtsam.Pose2] = None,
        source_pose=None, 
        between_pose=None, 
        index=None, 
        vin=None, 
        index_kf=None
    ):
        """Class constructor for a keyframe

        Args:
            status (bool, optional): is this frame a keyframe? Defaults
                to True, since a Keyframe is generally only constructed
                once the caller has already decided to submit one (see
                PingSLAMFrontEnd._submit_keyframe) - pass False
                explicitly if you need a non-keyframe placeholder.
            time (Any, optional): timestamp from incoming message. Not
                typed to a specific ROS client library's time type -
                the live pipeline (node/rosping_claude.py) hands this
                an rclpy.time.Time from self.get_clock().now(), but
                this class itself stays ROS-version-agnostic.
            dr_pose3 (gtsam.Pose3): dead reckoning 3D pose.
            points (np.array, optional): point cloud array. Defaults to np.zeros((0, 2), np.float32).
            cov (np.array, optional): covariance matrix. Defaults to None.
            dr_pose (gtsam.Pose2, optional): dead reckoning 2D pose, if
                the caller already has it (e.g. PingSLAMFrontEnd, which
                derives it straight from MAVROS odometry via
                MavrosOdometryCache). Defaults to None, in which case
                it's derived from dr_pose3 as before.
        """

        self.status = status  # used to mark keyframe
        self.time = time  # time

        # dr_pose3 is normally a gtsam.Pose3, but the MAVROS-backed
        # front end (node/rosping_claude.py's _to_pose3()) hands us a
        # raw [x, y, z, roll, pitch, yaw] array straight from
        # MavrosOdometryCache instead - accept either so this class
        # doesn't depend on which dead-reckoning source produced it.
        if dr_pose3 is not None and not isinstance(dr_pose3, gtsam.Pose3):
            dr_pose3 = n2g(dr_pose3, "Pose3")

        self.dr_pose3 = dr_pose3  # dead reckoning 3d pose
        self.dr_pose = (
            dr_pose if dr_pose is not None else pose322(dr_pose3)
        )  # dead reckoning 2d pose

        self.pose3 = dr_pose3  # estimated 3d pose (will be updated later)
        self.pose = (
            dr_pose if dr_pose is not None else pose322(dr_pose3)
        )  # estimated 2d pose

        self.cov = cov  # cov in local frame (always 2d)
        self.transf_cov = None  # cov in global frame

        self.points = points.astype(np.float32)  # points in local frame (always 2d)
        self.transf_points = None  # transformed points in global frame based on pose

        self.points3D = points.astype(
            np.float32
        )  # 3D point cloud from orthoganal sensor fusion
        self.transf_points3D = (
            None  # transformed 3D point cloud in global frame based on pose
        )

        self.constraints = (
            []
        )  # Non-sequential constraints (key, odom) aka loop closures

        self.twist = None  # twist message for publishing odom

        self.image = None  # image with this keyframe
        self.vertical_images = []
        self.horizontal_images = []

        self.poseTrue = None  # record the true pose from gazebo, simulation only

        self.sub_frames = []

        self.submap = None # multi-robot slam data
        self.ring_key = None
        self.context = None
        self.redo_submap = False
        self.source_pose = source_pose
        self.between_pose = between_pose
        self.index = index
        self.guess_pose = None
        self.vin = vin
        self.index_kf = index_kf
        self.scan_match_prediction = None
        self.scan_match_prediction_status = False
        self.scan_match_eig_max = None
        self.bits = None

    def update(self, new_pose: gtsam.Pose2, new_cov: np.array = None) -> None:
        """Update a keyframe following a SLAM update, pass in the new pose and covariance

        Args:
            new_pose (gtsam.Pose2): The new pose from the SLAM optimization
            new_cov (np.array, optional): The new covariance matrix. Defaults to None.
        """

        # push the pose in 2D and 3D
        self.pose = new_pose
        self.pose3 = n2g(
            (
                new_pose.x(),
                new_pose.y(),
                self.dr_pose3.z(),
                self.dr_pose3.rotation().roll(),
                self.dr_pose3.rotation().pitch(),
                new_pose.theta(),
            ),
            "Pose3",
        )

        # transform the points based on the new pose, 2D and 3D
        self.transf_points = Keyframe.transform_points(self.points, self.pose)
        self.transf_points3D = Keyframe.transform_points_3D(
            self.points3D, self.pose, self.pose3
        )

        # update the new covariance if we have one
        if new_cov is not None:
            self.cov = new_cov

        # transform the covariance to the global frame
        if self.cov is not None:
            c, s = np.cos(self.pose.theta()), np.sin(self.pose.theta())
            R = np.array([[c, -s], [s, c]])
            self.transf_cov = np.array(self.cov)
            self.transf_cov[:2, :2] = R.dot(self.transf_cov[:2, :2]).dot(R.T)
            self.transf_cov[:2, 2] = R.dot(self.transf_cov[:2, 2])
            self.transf_cov[2, :2] = self.transf_cov[2, :2].dot(R.T)

    @staticmethod
    def transform_points(points: np.array, pose: gtsam.Pose2) -> np.array:
        """transform a set of 2D points given a pose

        Args:
            points (np.array): point cloud to be transformed
            pose (gtsam.Pose2): transformation to be applied

        Returns:
            np.array: transformed point cloud
        """

        # check if there are actually any points
        if len(points) == 0:
            return np.empty_like(points, np.float32)

        # convert the pose to matrix format
        T = pose.matrix().astype(np.float32)

        # rotate and translate to the global frame
        return points.dot(T[:2, :2].T) + T[:2, 2]

    @staticmethod
    def transform_points_3D(
        points: np.array, pose: gtsam.Pose2, pose3: gtsam.Pose3
    ) -> np.array:
        """transform a set of 3D points to a given pose

        Args:
            points (np.array): points to be transformed
            pose (gtsam.Pose2): 2D pose to be removed
            pose3 (gtsam.Pose3): 3D transform to be applied

        Returns:
            np.array: the transformed point cloud
        """

        # check if there are actually any points
        if len(points) == 0 or points.shape[1] != 3:
            return np.empty_like(points, np.float32)

        # convert the pose to matrix format
        H = pose3.matrix().astype(np.float32)

        # rotate and translate to the global frame
        return points.dot(H[:3, :3].T) + H[:3, 3]


class InitializationResult(object):
    """Stores everything needed to attempt global ICP"""

    def __init__(self):
        """Class constructor"""

        # all points are in local frame
        self.source_points = np.zeros((0, 2))
        self.target_points = np.zeros((0, 2))
        self.source_key = None
        self.target_key = None
        self.source_pose = None
        self.target_pose = None
        # Cov for sampling
        self.cov = None
        self.occ = None
        self.status = None
        self.estimated_source_pose = None
        self.source_pose_samples = None


class ICPResult(object):
    """Stores the results of ICP"""

    def __init__(
        self,
        init_ret: InitializationResult,
        use_samples: bool = False,
        sample_eps: float = 0.01,
    ):
        """Class constructor

        Args:
            init_ret (InitializationResult): the global ICP initialization result
            use_samples (bool, optional): _description_. Defaults to False.
            sample_eps (float, optional): _description_. Defaults to 0.01.
        """

        # all points are in local frame
        self.source_points = init_ret.source_points
        self.target_points = init_ret.target_points
        self.source_key = init_ret.source_key
        self.target_key = init_ret.target_key
        self.source_pose = init_ret.source_pose
        self.target_pose = init_ret.target_pose
        self.status = init_ret.status
        self.estimated_transform = None
        self.cov = None
        self.initial_transforms = None
        self.inserted = False
        self.sample_transforms = None

        # populate the initial transform
        if init_ret.estimated_source_pose is not None:
            self.initial_transform = self.target_pose.between(
                init_ret.estimated_source_pose
            )
        else:
            self.initial_transform = self.target_pose.between(self.source_pose)

        # if we are using sampling to derive the covariance matrix
        if use_samples and init_ret.source_pose_samples is not None:
            idx = np.argsort(init_ret.source_pose_samples[:, -1])
            transforms = [
                self.target_pose.between(n2g(g, "Pose2"))
                for g in init_ret.source_pose_samples[idx, :3]
            ]
            filtered = [transforms[0]]
            for b in transforms[1:]:
                d = np.linalg.norm(g2n(filtered[-1].between(b)))
                if d < sample_eps:
                    continue
                else:
                    filtered.append(b)
            self.initial_transforms = filtered


class SMParams(object):
    """Stores scan-matching-parameters"""

    def __init__(self):
        """Constructor"""

        # Use occupancy probability map matching to initialize ICP
        self.initialization = None
        # Global search params
        self.initialization_params = None
        # Minimum number of points
        self.min_points = None
        # Max deviation from initial guess
        self.max_translation = None
        self.max_rotation = None

        # Min separation between source key and the last target frame
        self.min_st_sep = None
        # Number of source frames to build source points
        # Not used in SSM
        self.source_frames = None
        # Number of target frames to build target points
        # Not used in NSSM
        self.target_frames = None

        # Number of ICP instances to run to calculate cov
        self.cov_samples = None
