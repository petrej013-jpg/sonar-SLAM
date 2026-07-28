import os
import time
import timeit
from functools import wraps
import numpy as np
from tqdm.auto import tqdm
from threading import Event

offline = False
callback_lock_event = Event()
callback_lock_event.set()

# This module is shared by both the live MAVROS/ROS2 pipeline
# (node/rosping_claude.py etc.) and offline analysis scripts, so it
# does not hard-import rospy or rclpy at module scope. Instead, an
# rclpy node hands us its own logger once at startup via
# set_ros_logger(self.get_logger()); everything below routes through
# that logger (or falls back to plain stdout) rather than assuming any
# particular ROS client library is even installed.
_ros_logger = None


def set_ros_logger(logger) -> None:
    """Register the ROS logger (e.g. an rclpy Node's self.get_logger())
    that loginfo/logdebug/logwarn/logerror should forward to.

    Call this once, early, from your node's __init__. If it's never
    called (e.g. running fully offline), the log*() functions below
    just print to stdout via tqdm.write instead.
    """
    global _ros_logger
    _ros_logger = logger


def add_lock(callback):
    """
    Lock decorator for callback functions, which is
    very helpful for running ROS offline with bag files.
    The lock forces callback functions sequentially,
    so we can show matplotlib plot, etc.

    """

    @wraps(callback)
    def lock_callback(*args, **kwargs):
        if not offline:
            callback(*args, **kwargs)
        else:
            callback_lock_event.wait()
            callback_lock_event.clear()
            callback(*args, **kwargs)
            callback_lock_event.set()

    return lock_callback


class LOGCOLORS:
    DK_GRAY = "\033[30m"
    DK_RED = "\033[31m"
    DK_GREEN = "\033[32m"
    DK_YELLOW = "\033[33m"
    DK_BLUE = "\033[34m"
    DK_PURPLE = "\033[35m"
    DK_CYAN = "\033[36m"
    DK_WHITE = "\033[37m"

    DK_BG_GRAY = "\033[40m"
    DK_BG_RED = "\033[41m"
    DK_BG_GREEN = "\033[42m"
    DK_BG_YELLOW = "\033[43m"
    DK_BG_BLUE = "\033[44m"
    DK_BG_PURPLE = "\033[45m"
    DK_BG_CYAN = "\033[46m"
    DK_BG_WHITE = "\033[47m"

    GRAY = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    BG_GRAY = "\033[100m"
    BG_RED = "\033[101m"
    BG_GREEN = "\033[102m"
    BG_YELLOW = "\033[103m"
    BG_BLUE = "\033[104m"
    BG_PURPLE = "\033[105m"
    BG_CYAN = "\033[106m"
    BG_WHITE = "\033[107m"

    END = "\033[0m"


def colorlog(color, str):
    return color + str + LOGCOLORS.END


def loginfo(msg):
    if offline or _ros_logger is None:
        tqdm.write(msg)
    else:
        _ros_logger.info(msg)


def logdebug(msg):
    if offline or _ros_logger is None:
        tqdm.write(colorlog(LOGCOLORS.BLUE, msg))
    else:
        _ros_logger.debug(msg)


def logwarn(msg):
    if offline or _ros_logger is None:
        tqdm.write(colorlog(LOGCOLORS.YELLOW, msg))
    else:
        # rclpy loggers expose .warning(), not .warn()/.logwarn() like
        # rospy did - this is the ROS2 name.
        _ros_logger.warning(msg)


def logerror(msg):
    if offline or _ros_logger is None:
        tqdm.write(colorlog(LOGCOLORS.RED, msg))
    else:
        # rospy.logerror() never actually existed (the ROS1 function is
        # rospy.logerr()); rclpy's is .error() - either way, route
        # through the registered logger's .error().
        _ros_logger.error(msg)


def common_parser(description="node"):
    import argparse

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--file", type=str, default="", help="ROS bag file")
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="start the video from START seconds (default: 0.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="duration of the video from START (default: -1)",
    )

    return parser


def read_bag(file, start=None, duration=None, progress=True):
    """Replay a recorded ROS1 .bag file offline.

    NOTE: this is a legacy ROS1 utility (the `rosbag` package it needs
    has no ROS2 equivalent import path) for replaying old recorded
    missions - it is unrelated to, and does not need to match, the
    live MAVROS/ROS2 pipeline the rest of this module talks to. Both
    `rosbag` and `rospy` are imported lazily here so that importing
    this module - and the live ROS2 nodes that depend on it - does not
    require ROS1 to be installed at all.
    """
    import rosbag
    import rospy
    from .topics import IMU_TOPIC, DVL_TOPIC, DEPTH_TOPIC, SONAR_TOPIC

    bag = rosbag.Bag(file)
    start = start if start is not None else 0
    start_time = bag.get_start_time() + start
    end_time = bag.get_end_time()
    if duration is None or duration < 0 or duration == float("inf"):
        duration = end_time - start_time
    else:
        end_time = start_time + duration

    if progress:
        pbar = tqdm(total=int(duration), unit="s")
    for topic, msg, t in bag.read_messages(
        topics=[IMU_TOPIC, DVL_TOPIC, DEPTH_TOPIC, SONAR_TOPIC],
        start_time=rospy.Time.from_sec(start_time),
        end_time=rospy.Time.from_sec(end_time),
    ):
        if progress:
            pbar.update(int(t.to_sec() - start_time) - pbar.n)
        yield topic, msg

    bag.close()


def get_log_dir():
    """ROS1-only: relies on the `roslaunch-logs` CLI tool, which has no
    ROS2 equivalent. Only used by create_log() below.
    """
    import subprocess

    return subprocess.check_output("roslaunch-logs").strip()


def create_log(suffix, timestamp=None):
    import datetime

    if timestamp is None:
        # ROS1-only path (get_log_dir() below already requires it);
        # imported lazily so the rest of this module works without
        # rospy installed.
        import rospy

        timestamp = rospy.Time.now().to_sec()

    now = datetime.datetime.fromtimestamp(timestamp)
    log_name = now.strftime("%Y-%m-%d-%H-%M-%S-") + suffix

    log_dir = get_log_dir()

    return os.path.join(log_dir, log_name)


def load_nav_data(file, start=0, duration=None, progress=True):
    import gtsam
    from .topics import IMU_TOPIC, DVL_TOPIC, DEPTH_TOPIC

    dvl, depth, imu = [], [], []
    for topic, msg in read_bag(file, start, duration, progress):
        time = msg.header.stamp.to_sec()
        if topic == DVL_TOPIC:
            dvl.append(
                (time, msg.velocity.x, msg.velocity.y, msg.velocity.z, msg.altitude)
            )
        elif topic == DEPTH_TOPIC:
            depth.append((time, msg.depth))
        elif topic == IMU_TOPIC:
            ax = msg.linear_acceleration.x
            ay = msg.linear_acceleration.y
            az = msg.linear_acceleration.z
            wx = msg.angular_velocity.x
            wy = msg.angular_velocity.y
            wz = msg.angular_velocity.z
            qx = msg.orientation.x
            qy = msg.orientation.y
            qz = msg.orientation.z
            qw = msg.orientation.w
            # IMU is -roll90
            y, p, r = (
                gtsam.Rot3.Quaternion(qw, qx, qy, qz)
                .compose(gtsam.Rot3.Roll(np.pi / 2.0))
                .ypr()
            )
            t = msg.linear_acceleration_covariance[0]
            imu.append((time, ax, ay, az, wx, wy, wz, r, p, y, t))

    dvl = np.array(dvl)
    depth = np.array(depth)
    imu = np.array(imu)
    t0 = [a[0, 0] for a in (dvl, depth, imu) if len(a)]
    if not t0:
        return None, None, None
    else:
        t0 = min(t0)

    if len(dvl):
        dvl[:, 0] -= t0
    if len(imu):
        imu[:, 0] -= t0
        imu[:, -1] -= imu[0, -1]
    if len(depth):
        depth[:, 0] -= t0
    return dvl, depth, imu


class CodeTimer(object):
    """Timer class used with `with` statement

    - Disable output by setting CodeTimer.silent = False
    - Change log_func to print/tqdm.write/rospy.loginfo/etc

    with CodeTimer("Some function"):
        some_func()

    """

    silent = False

    def __init__(self, name="Code block"):
        self.name = name

    def __enter__(self):
        """Start measuring at the start of indent"""
        if not CodeTimer.silent:
            self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        """
            Stop measuring at the end of indent. This will run even
            if the indented lines raise an exception.
        """
        if not CodeTimer.silent:
            self.took = timeit.default_timer() - self.start

            if not CodeTimer.silent:
                msg = "{} : {:.5f} s".format(self.name, float(self.took))
                logdebug(msg)
