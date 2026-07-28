from typing import Optional, Callable, Tuple, List
import asyncio
import os
import h5py
import numpy as np
from brping import Ping1D
from loguru import logger

from ..utils.SonarFeatureExtraction_Blue import SonarFeatureExtraction
from ..utils.settings_Blue import WATER_SOS, SonarConfig, CFARConfig


class PingManager:
    """
    Manages the Ping1D single-beam echosounder, processing profiles and
    feature extraction.

    Unlike the Ping360 (a mechanically scanning multibeam-style sonar that
    sweeps through ~400 angles and assembles a 2D range-vs-angle image),
    the Ping1D is a static, fixed-beam sonar. There is no motor and no
    angle parameter: every "ping" is a single, standalone 1D array of
    return-strength values along range. Because of that, this class no
    longer builds a 2D scan matrix or tracks an angle list - it just
    tracks the most recent 1D profile.

    Attributes:
        feature_extractor: Processes sonar data to extract features
        resolution: Fallback range resolution calculated from settings
                    (Ping1D reports per-ping resolution directly, see
                    get_ping_data(), so this is mainly used in replay
                    mode where no live device metadata is available)
        current_scan: The most recent single-beam range profile (1D array)
        current_distance: Most recent single best-guess distance/confidence
                           reading (from get_distance(), not the full profile)
        start_index: The starting index for valid range data
    """

    def __init__(self, device: Optional[str], baudrate: int, udp: str, live: bool = True):
        """
        Initialize the PingManager.

        Args:
            device: Serial device path for the Ping1D
            baudrate: Baud rate for serial connection
            udp: UDP connection string in format "host:port"
            live: Whether to use a live Ping1D device or recorded data
        """
        self.current_scan = None
        self.current_distance = None
        self.start_index = 0

        # Kept as a fallback resolution estimate for replay mode, where we
        # don't have a live device telling us scan_start/scan_length/
        # num_points for each profile. In live mode we prefer computing
        # resolution directly from each profile message instead (see
        # get_ping_data()), since Ping1D profiles can vary in scan length
        # and point count from ping to ping.
        self.resolution = SonarConfig.MAX_RANGE / SonarConfig.PROFILE_POINTS ####Adjust the Max Range for real working range###########

        # Initialize for live or replay mode
        if live:
            self._init_live_device(device, baudrate, udp)
        else:
            self._init_replay_mode()

        # Initialize feature extractor
        self.feature_extractor = SonarFeatureExtraction(
            Ntc=CFARConfig.Ntc, Ngc=CFARConfig.Ngc, Pfa=CFARConfig.Pfa, alg="GOCA")

        # Initialize other instance variables
        self.costmap = None
        self.X = None
        self.Y = None

        # Callback function for when current_scan is updated
        self._on_scan_updated_callback: Optional[Callable[[
            np.ndarray], None]] = None

    def _init_live_device(self, device: Optional[str], baudrate: int, udp: str):
        """Initialize the PingManager for live device mode."""
        # Same connect/initialize pattern as Ping360 - brping exposes the
        # same connect_serial / connect_udp / initialize() calls on Ping1D.
        # No motor, no scanning mode to configure here.
        self.myPing1D = Ping1D()
        self.device = device
        self.baudrate = baudrate
        self.udp = udp

        try:
            if device is not None:
                self.myPing1D.connect_serial(device, self.baudrate)
            elif udp is not None:
                host, port = udp.split(':')
                self.myPing1D.connect_udp(host, int(port))

            self.myPing1D.initialize()
            logger.info("Ping1D initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ping1D: {e}")
            raise RuntimeError("Failed to initialize Ping1D device")

    def _init_replay_mode(self):
        """Initialize the PingManager for replay mode (using recorded data)."""
        # Ping360 built a fixed 400-step angle list here because a "scan"
        # meant a full mechanical sweep. Ping1D has no angle axis at all -
        # a "scan" is just one profile in time - so there is nothing
        # equivalent to build. This method is kept only for symmetry with
        # the live-mode branch and to leave a place for future replay-only
        # setup (e.g. loading a fixed sample rate/timestamp base).
        logger.info("Initialized in replay mode")

    async def shutdown(self):
        """Safely shut down the Ping1D device."""
        # Ping360's shutdown reconnected specifically to command the motor
        # off. Ping1D has no motor, so there's no equivalent "make the
        # hardware safe" step - just release the connection if we have one.
        try:
            if hasattr(self, 'myPing1D') and self.myPing1D is not None:
                self.myPing1D.exit()
        except Exception as e:
            logger.error(f"Error during Ping1D shutdown: {e}")
        logger.info("Ping1D shut down")

    def register_scan_update_callback(self, callback: Callable[[np.ndarray], None]):
        """
        Register a callback function to be called when current_scan is updated.

        Args:
            callback: Function to call with the scan data when updated
        """
        self._on_scan_updated_callback = callback
        logger.info("Sonar callback registered")

    async def get_ping_data(self) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """
        Request and process a single profile reading from the Ping1D.

        This replaces Ping360's two-step scan() + get_ping_data() pattern.
        Ping360 needed a separate "aim at this angle and fire" command
        before waiting for the reply, because the transducer had to
        physically rotate first. Ping1D's beam is fixed, so requesting
        and receiving a reading collapses into one call: get_profile()
        both sends the request and waits for the response, returning a
        dict of the message's fields directly (no manual wait_message /
        message-type constant needed, unlike Ping360's approach).

        Returns:
            Tuple of (resolution_m_per_sample, profile_data_array),
            or (None, None) if no message received. We return the
            per-ping resolution instead of an angle, since there is no
            angle - the caller needs resolution to convert sample index
            to physical range.
        """
        data = self.myPing1D.get_profile()

        if data:
            num_points = data["num_points"]
            scan_length_m = data["scan_length"] / 1000.0  # mm -> m

            # Ping1D profiles can vary in scan_length and num_points from
            # ping to ping (unlike Ping360's fixed sample_period/number_
            # of_samples request), so resolution is recomputed per-ping
            # rather than assumed constant.
            resolution = scan_length_m / num_points if num_points else self.resolution

            profile = np.frombuffer(bytes(data["data"]), dtype=np.uint8)

            # Also stash the device's own single best-guess distance/
            # confidence estimate, since it's a useful sanity check even
            # though we mainly work with the full profile array.
            self.current_distance = {
                "distance_m": data["distance"] / 1000.0,
                "confidence": data["confidence"],
            }

            return resolution, profile

        return None, None

    async def read_recording(self, filename: str):
        """
        Read sonar data from an HDF5 file and process it.

        Args:
            filename: Path to the HDF5 file containing sonar data
        """
        logger.info(f"Reading sonar data from {filename}")

        if not os.path.exists(filename):
            logger.error(f"File {filename} does not exist")
            return None

        try:
            with h5py.File(filename, 'r') as file:
                datasets = list(file.keys())
                logger.info(f"Found {len(datasets)} recorded pings")

                while True:  # Loop through the datasets repeatedly
                    for dataset in datasets:
                        if datasets:
                            # Load and process one profile. Ping360's
                            # version also set self.current_angles here;
                            # there's no angle equivalent to track for a
                            # single-beam profile, so that line is simply
                            # gone rather than replaced.
                            self.current_scan, self.start_index = self.clean(
                                file[dataset][:])

                            logger.debug(
                                f"Processed ping: min={np.min(self.current_scan)}, max={np.max(self.current_scan)}")

                            # Extract features if needed
                            # self.costmap, self.X, self.Y = await self.feature_extractor.extract_features(
                            #     self.current_scan, self.resolution)

                            if self._on_scan_updated_callback:
                                self._on_scan_updated_callback(
                                    self.current_scan)

                        else:
                            logger.warning("No pings found in file")

                        await asyncio.sleep(15)
        except Exception as e:
            logger.error(f"Error opening or reading file {filename}: {e}")

    def get_data(self) -> np.ndarray:
        """Get the current profile (1D range) data."""
        return self.current_scan

    def get_distance_estimate(self) -> Optional[dict]:
        """
        Get the device's own single best-guess distance/confidence
        estimate from the most recent ping (not the full profile array).
        Ping360 had no equivalent - it only ever reports raw intensity
        samples, never a pre-computed best-guess range.
        """
        return self.current_distance

    def get_costmap(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the current costmap and coordinate grids.

        NOTE: A per-ping 2D costmap made sense for Ping360 because a
        single completed sweep already contained a full range x angle
        image. A single Ping1D profile has no angle axis, so it cannot
        be turned into an XY costmap on its own - this is exactly the
        front-end architecture mismatch with Bruce's SLAM pipeline.
        Building a costmap now requires accumulating profiles over the
        vehicle's own trajectory (e.g. via dead-reckoning or odometry)
        rather than doing it per-ping, so this method is left as a
        placeholder until that accumulation strategy is implemented.
        """
        return self.costmap, self.X, self.Y

    def get_cfar_polar(self) -> np.ndarray:
        """Get the CFAR-processed data."""
        return self.feature_extractor.get_cfar()

    def get_start_index(self) -> int:
        """Get the starting index for valid range data."""
        return self.start_index

    def clean(self, data: np.ndarray, resolution: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """
        Set sonar data below operating range to zero.

        This logic is generic to any range-returning sonar - it just
        zeroes out samples closer than the device's minimum reliable
        range - so it carries over from Ping360 unchanged in spirit.
        The only change is accepting a per-ping resolution, since
        Ping1D's samples-per-meter can vary between pings (unlike
        Ping360's fixed sample_period).

        Args:
            data: Raw sonar data array
            resolution: Meters per sample for this specific ping. Falls
                        back to self.resolution (e.g. in replay mode).

        Returns:
            Tuple of (cleaned_data, start_index)
        """
        res = resolution if resolution else self.resolution

        index = 0
        while index < len(data) and index * res < SonarConfig.MIN_RANGE:
            data[index] = 0
            index += 1

        return data[index:], index

    async def sonar_pinging(self, threshold: int = 80):
        """
        Continuously ping with the Ping1D and process each profile.

        This replaces sonar_scanning(). Ping360's version had to step
        through angles (start -> end), accumulate one column per angle,
        and only produce a usable 2D scan once a full sweep (start to
        end) completed - hence the step/end-of-sweep bookkeeping and the
        final transpose into a 2D matrix. Ping1D has no angle to step
        through: every single ping is already a complete, standalone
        result, so there's no "accumulate until a sweep finishes" logic
        left at all - each iteration is a full one-and-done ping/clean/
        callback cycle.

        Args:
            threshold: Minimum amplitude threshold for data
        """
        logger.info("Starting continuous pinging")

        while True:
            try:
                resolution, data = await self.get_ping_data()

                if data is None:
                    logger.warning("Ping1D message empty")
                    await asyncio.sleep(0.1)
                    continue

                cleaned_data, self.start_index = self.clean(data, resolution)
                cleaned_data[cleaned_data < threshold] = 0

                self.current_scan = cleaned_data

                if self._on_scan_updated_callback:
                    self._on_scan_updated_callback(self.current_scan)

                logger.debug("Processed ping")

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error during pinging: {e}")
                await asyncio.sleep(1)  # Longer delay on error