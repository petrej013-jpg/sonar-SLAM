import asyncio
import csv
import os
import requests
from datetime import datetime
from loguru import logger

from typedefs import AttitudeData, GPSData, IMUData, PressureData, TimeData, LocalizationData, MavlinkMessage
from settings import DATA_FILEPATH, VEHICLE_IP


class DataManager:
    def __init__(self) -> None:
        self.url = f"http://{VEHICLE_IP}:6040/v1/mavlink/vehicles/1/components/1/messages"
        self.recorded_data = {}
        self.is_recording = False
        self.recording_task = None

    async def start_recording(self):
        if self.is_recording:
            logger.warning("Already recording!")
            return

        self.record_data.clear()
        self.is_recording = True
        logger.info("Recording started.")

    async def stop_recording(self):
        if not self.is_recording:
            logger.warning("No recording in progress!")
            return

        self.is_recording = False
        logger.info("Recording stopped.")

        os.makedirs(DATA_FILEPATH, exist_ok=True)
        if os.path.exists(DATA_FILEPATH):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"slam_data_{timestamp}_%03d.csv"
            filepath = os.path.join(DATA_FILEPATH, filename)

            with open(filepath, 'w') as file:
                logger.info("Writing to csv.")
                writer = csv.DictWriter(
                    file, fieldnames=self.data.keys(), delimiter=' ')

                writer.writeheader()

                rows = [dict(zip(self.data.keys(), row))
                        for row in zip(*self.data.values())]
                writer.writerows(rows)

            logger.info(f"Data saved to {filename}")
        else:
            logger.warning(f"File path {DATA_FILEPATH} does not exist.")

    async def get_gps_data(self):
        path = os.path.join(self.url, MavlinkMessage.GLOBAL_POSITION_INT)
        try:
            gps_response = requests.get(path, timeout=1)
            msg = gps_response.json()['message']

            temp_pos = GPSData(
                timestamp=msg['time_boot_ms'],
                altitude=msg['alt'],
                latitude=msg['lat'],
                longitude=msg['lon']
            )

            return temp_pos

        except requests.RequestException as e:
            logger.error(
                f"Could not get {MavlinkMessage.GLOBAL_POSITION_INT} response {e}.")

    async def get_imu_data(self):
        path = os.path.join(self.url, MavlinkMessage.RAW_IMU)
        try:
            imu_response = requests.get(path, timeout=1)
            msg = imu_response.json()['message']

            temp_pos = IMUData(
                timestamp=msg['time_usec'],
                x_acc=msg['xacc'],
                x_gyro=msg['xgyro'],
                y_acc=msg['yacc'],
                y_gyro=msg['ygyro'],
                z_acc=msg['zacc'],
                z_gyro=msg['zgyro']
            )

            return temp_pos

        except requests.RequestException as e:
            logger.error(
                f"Could not get {MavlinkMessage.RAW_IMU} response {e}.")

    async def get_attitude_data(self):
        path = os.path.join(self.url, MavlinkMessage.ATTITUDE)
        try:
            att_response = requests.get(path, timeout=1)
            msg = att_response.json()['message']

            temp_pos = AttitudeData(
                timestamp=msg['time_boot_ms'],
                roll=msg['roll'],
                pitch=msg['pitch'],
                yaw=msg['yaw'],
                roll_speed=msg['rollspeed'],
                pitch_speed=msg['pitchspeed'],
                yaw_speed=msg['yawspeed']
            )

            return temp_pos

        except requests.RequestException as e:
            logger.error(
                f"Could not get {MavlinkMessage.ATTITUDE} response {e}.")

    async def get_pressure_data(self):
        path = os.path.join(self.url, MavlinkMessage.SCALED_PRESSURE)
        try:
            att_response = requests.get(path, timeout=1)
            msg = att_response.json()['message']

            temp_press = PressureData(
                timestamp=msg['time_boot_ms'],
                press_abs=msg['press_abs'],
                press_diff=msg['press_diff']
            )

            return temp_press

        except requests.RequestException as e:
            logger.error(
                f"Could not get {MavlinkMessage.SCALED_PRESSURE} response {e}.")

    async def get_time_data(self):
        path = os.path.join(self.url, MavlinkMessage.SYSTEM_TIME)
        try:
            att_response = requests.get(path, timeout=1)
            msg = att_response.json()['message']

            temp_time = TimeData(
                timestamp=msg['time_boot_ms']
            )

            return temp_time

        except requests.RequestException as e:
            logger.error(
                f"Could not get {MavlinkMessage.SYSTEM_TIME} response {e}.")

    async def get_localization_data(self):
        gps = await self.get_gps_data()
        imu = await self.get_imu_data()
        att = await self.get_attitude_data()
        press = await self.get_pressure_data()

        loc_data = LocalizationData(
            timestamp=datetime.now(),
            gps_data=gps,
            imu_data=imu,
            attitude_data=att,
            pressure_data=press
        )

        return loc_data

    async def record_data(self):
        logger.info("Recording data is running!")
        try:
            while self.is_recording:
                gps_data = await self.get_gps_data()
                imu_data = await self.get_imu_data()
                attitude_data = await self.get_attitude_data()
                pressure_data = await self.get_pressure_data()

                if not self.recorded_data:
                    self.recorded_data['timestamp'] = [datetime.now()]
                    for key, value in gps_data.dict().items():
                        self.recorded_data[key] = [value]
                    for key, value in imu_data.dict().items():
                        self.recorded_data[key] = [value]
                    for key, value in attitude_data.dict().items():
                        self.recorded_data[key] = [value]
                    for key, value in pressure_data.dict().items():
                        self.recorded_data[key] = [value]
                else:
                    self.recorded_data['timestamp'].append(datetime.now())
                    for key, value in gps_data.dict().items():
                        self.recorded_data[key].append(value)
                    for key, value in imu_data.dict().items():
                        self.recorded_data[key].append(value)
                    for key, value in attitude_data.dict().items():
                        self.recorded_data[key].append(value)
                    for key, value in pressure_data.dict().items():
                        self.recorded_data[key].append(value)

                await asyncio.sleep(0)
        except Exception as e:
            logger.error(f"Could not get response {e}.")
        finally:
            logger.info("Coroutine completed.")
