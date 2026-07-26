import numpy as np
from typing import List, Tuple, Optional
from loguru import logger

from .CFAR import CFAR


class SonarFeatureExtraction:
    """
    Class for extracting features from sonar data using CFAR detection.

    This class processes sonar data in polar coordinates and extracts features 
    using Constant False Alarm Rate (CFAR) detection algorithms. It can convert 
    the detected features to Cartesian coordinates for mapping and visualization.
    """

    def __init__(self, Ntc: int = 40, Ngc: int = 10, Pfa: float = 1e-2,
                 rank: Optional[int] = None, alg: str = "GOCA",
                 resolution: float = 0.5, threshold: int = 30):
        """
        Initialize the SonarFeatureExtraction with CFAR parameters.

        Args:
            Ntc: Number of training cells (must be even)
            Ngc: Number of guard cells (must be even)
            Pfa: Probability of false alarm (0-1)
            rank: Rank parameter for OS-CFAR (default: None)
            alg: CFAR algorithm type (default: "GOCA")
            resolution: Spatial resolution in meters (default: 0.5)
            threshold: Amplitude threshold for detection (default: 30)
        """
        self.Ntc = Ntc
        self.Ngc = Ngc
        self.Pfa = Pfa
        self.rank = rank
        self.alg = alg
        self.threshold = threshold
        self.resolution = resolution

        # Initialize CFAR detector
        self.detector = CFAR(self.Ntc, self.Ngc, self.Pfa)

        # Initialize other instance variables
        self.map_x = None
        self.map_y = None
        self.cfar_polar = None

    async def create_costmap_in_cartesian(self, sonar_data: np.ndarray,
                                          bearings: List[float],
                                          range_resolution: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a mesh grid of zeros in Cartesian coordinates from sonar data in polar coordinates.

        Args:
            sonar_data: Sonar data in polar coordinates
            bearings: List of bearing angles in degrees
            range_resolution: Range resolution in meters

        Returns:
            Tuple of (costmap, X_grid, Y_grid)
        """
        # Set resolution and calculate dimensions
        _res = range_resolution
        _height = len(sonar_data) * _res

        # Calculate bearing range, handling wrap-around cases
        if bearings[-1] < bearings[0]:
            bearing_range = 360 - (bearings[0] - bearings[-1])
        else:
            bearing_range = bearings[-1] - bearings[0]

        # Calculate width based on bearing range and height
        _width = np.sin(np.radians(bearing_range)) * _height

        # Create a meshgrid for x and y axes based on the range values
        x_range = np.arange(-_width/2, _width/2, _res)
        y_range = np.arange(0, _height, _res)
        X, Y = np.meshgrid(x_range, y_range)

        # Initialize empty costmap
        costmap = np.zeros_like(X, dtype=np.float32)

        return costmap, X, Y

    async def extract_features(self, sonar_data: np.ndarray,
                               bearings: List[float],
                               range_resolution: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process sonar data and extract features using CFAR.

        Args:
            sonar_data: Sonar data in polar coordinates
            bearings: List of bearing angles in degrees
            range_resolution: Range resolution in meters

        Returns:
            Tuple of (costmap, X_grid, Y_grid)
        """
        img = sonar_data

        # CFAR Detection
        peaks = self.detector.detect(img, self.alg)
        self.cfar_polar = peaks

        # Get indices of detected peaks
        peak_indices = np.argwhere(peaks)

        # Convert directly to Cartesian coordinates
        points = []

        # Handle bearings that wrap around 0/360
        adjusted_bearings = self.adjust_bearings(bearings)

        # For each peak, calculate its Cartesian coordinates
        for peak in peak_indices:
            range_idx, azimuth_idx = peak

            # Get the range in meters
            range_m = range_idx * range_resolution

            # Get the bearing angle in radians
            bearing_deg = adjusted_bearings[azimuth_idx % len(bearings)]
            if bearing_deg > 360:
                bearing_deg -= 360
            bearing_rad = np.radians(bearing_deg)

            # Convert to Cartesian
            # Note: Using convention where 0 degrees = positive y-axis
            y = range_m * np.cos(bearing_rad)
            x = range_m * np.sin(bearing_rad)

            points.append([y, x])

        # Create costmap and grids
        costmap, X, Y = await self.create_costmap_in_cartesian(
            sonar_data, bearings, range_resolution)

        # If no points found, return empty costmap
        if not points:
            return costmap, X, Y

        # Mark detected points on costmap
        for point in points:
            # Get coordinates
            x = point[1]  # X-coordinate in meters
            y = point[0]  # Y-coordinate in meters

            # Find the closest indices on the mesh grid
            x_idx = np.abs(X[0] - x).argmin()  # Find closest X index
            y_idx = np.abs(Y[:, 0] - y).argmin()  # Find closest Y index

            # Mark the detected point on the costmap
            costmap[y_idx, x_idx] = 1  # Or increment based on detection count

        return costmap, X, Y

    def adjust_bearings(self, bearings: List[float]) -> List[float]:
        """
        Adjust bearings that cross the 0/360 boundary for consistent calculations.

        Args:
            bearings: List of bearing angles in degrees

        Returns:
            List of adjusted bearing angles
        """
        adjusted_bearings = bearings.copy()

        # If bearings cross the 0/360 boundary
        if bearings[0] > bearings[-1]:
            # Adjust bearings that are less than 180 by adding 360
            for i, bearing in enumerate(bearings):
                if bearing < 180:
                    adjusted_bearings[i] = bearing + 360

        return adjusted_bearings

    def get_cfar(self) -> np.ndarray:
        """
        Get the CFAR-processed polar data.

        Returns:
            CFAR-processed data array
        """
        return self.cfar_polar

    async def update_cfar_parameters(self, Ntc: int, Ngc: int, Pfa: float,
                                     rank: Optional[int] = None,
                                     alg: Optional[str] = None,
                                     threshold: Optional[int] = None) -> bool:
        """
        Update CFAR parameters without recreating the detector.

        Args:
            Ntc: Number of training cells (must be even)
            Ngc: Number of guard cells (must be even)
            Pfa: Probability of false alarm (0-1)
            rank: Rank parameter for OS-CFAR (optional)
            alg: CFAR algorithm type (optional)
            threshold: Amplitude threshold for detection (optional)

        Returns:
            True if parameters were updated successfully
        """
        try:
            # Update instance variables
            self.Ntc = Ntc
            self.Ngc = Ngc
            self.Pfa = Pfa

            if rank is not None:
                self.rank = rank

            if alg is not None:
                self.alg = alg

            if threshold is not None:
                self.threshold = threshold

            # Update the CFAR detector
            self.detector = CFAR(self.Ntc, self.Ngc, self.Pfa, self.rank)

            logger.info(f"Updated CFAR parameters: Ntc={Ntc}, Ngc={Ngc}, Pfa={Pfa}, "
                        f"rank={rank}, alg={alg}, threshold={threshold}")

            return True

        except Exception as e:
            logger.error(f"Failed to update CFAR parameters: {e}")
            return False
