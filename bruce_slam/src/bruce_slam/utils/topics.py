"""
Topics for the MOAT SLAM project.
"""

# MAVROS - fused local-position odometry from ArduSub's onboard EKF.
# mavros_odom_cache_c.py's MavrosOdometryCache should default to this
# constant rather than a bare string.
MAVROS_ODOM_TOPIC = "/mavros/local_position/odom"


# SLAM/mapping output topics - not wired up to any publisher yet, but
# kept as the intended names for when RViz visualization gets built into
# node/rosping_claude.py (see Phase 1 of the project plan).
SLAM_NS = "/moat/slam/"
SLAM_POSE_TOPIC = SLAM_NS + "slam/pose"
SLAM_ODOM_TOPIC = SLAM_NS + "slam/odom"
SLAM_TRAJ_TOPIC = SLAM_NS + "slam/traj"
SLAM_CLOUD_TOPIC = SLAM_NS + "slam/cloud"
SLAM_CONSTRAINT_TOPIC = SLAM_NS + "slam/constraint"
SLAM_ISAM2_TOPIC = SLAM_NS + "slam/isam2"
MAPPING_INTENSITY_TOPIC = SLAM_NS + "mapping/intensity"
MAPPING_OCCUPANCY_TOPIC = SLAM_NS + "mapping/occupancy"
MAPPING_GET_MAP_SERVICE = SLAM_NS + "mapping/get_map"
