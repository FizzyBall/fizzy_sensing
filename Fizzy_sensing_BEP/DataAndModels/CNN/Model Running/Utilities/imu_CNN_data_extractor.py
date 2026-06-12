"""
IMU Data Extraction Module

Provides utility functions for extracting and processing IMU sensor data from
Fizzy data packets. Handles unit conversion, thresholding, and magnitude
calculations for quaternion orientation, angular velocity, and linear
acceleration data.
"""

import numpy as np


# =====================================================================
# Constants
# =====================================================================
GYRO_SENSITIVITY = 131.0  # LSB/°/s for ±250°/s range
GYRO_THRESHOLD = 0.05  # rad/s - threshold for filtering out noise
ACC_THRESHOLD = 0.025  # g (25 mg) - threshold for filtering noise



# =====================================================================
# Packet Data Structure Reference
# =====================================================================
# Expected IMU data packet format from Fizzy:
# Index  0: timestamp (microseconds)
# Index  1: unused
# Index  2: unused
# Index  3: qx (quaternion x)
# Index  4: qy (quaternion y)
# Index  5: qz (quaternion z)
# Index  6: qw (quaternion w)
# Index  7: acc_x (g)
# Index  8: acc_y (g)
# Index  9: acc_z (g)
# Index 10: unused
# Index 11: unused
# Index 12: unused
# Index 13: gyro_x (raw sensor value, LSB/°/s)
# Index 14: gyro_y (raw sensor value, LSB/°/s)
# Index 15: gyro_z (raw sensor value, LSB/°/s)
# =====================================================================


def extract_euler_from_packet(packet):
	"""
	Extracts Roll, Pitch, and Yaw from the IMU quaternion packet. Relative to world. 
	
	Args:
		packet: IMU data packet (list/array)
	
	Returns:
		tuple: (roll, pitch, yaw) all in radians
	"""
	# Quaternion indices from Fizzy data packet
	qx, qy, qz, qw = packet[3], packet[4], packet[5], packet[6]
	w, x, y, z = qw, qx, qy, qz

	# Quaternion to Euler angle conversion
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + y * y)
	roll = np.arctan2(t0, t1)

	t2 = +2.0 * (w * y - z * x)
	t2 = np.clip(t2, -1.0, 1.0)
	pitch = np.arcsin(t2)

	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (y * y + z * z)
	yaw = np.arctan2(t3, t4)

	return roll, pitch, yaw

def get_angles(packet):
	return extract_euler_from_packet(packet)


def extract_angular_velocity_from_packet(packet, relative_to_world=True):
	"""
	Extracts angular velocity (gyro) from the IMU packet.
	
	Converts raw gyroscope sensor values to rad/s and applies a threshold
	to filter out noise.
	
	Args:
		packet: IMU data packet (list/array)
	
	Returns:
		tuple: (gyro_x, gyro_y, gyro_z) in rad/s
	"""
	# Angular velocity indices from Fizzy data packet (raw sensor values)
	gyro_x_raw, gyro_y_raw, gyro_z_raw = packet[13], packet[14], packet[15]
	
	# Convert from raw sensor units to rad/s
	# For ±250°/s range: sensitivity = 131 LSB/°/s
	gyro_x = gyro_x_raw / GYRO_SENSITIVITY * np.pi / 180.0
	gyro_y = gyro_y_raw / GYRO_SENSITIVITY * np.pi / 180.0
	gyro_z = gyro_z_raw / GYRO_SENSITIVITY * np.pi / 180.0
	
	# Apply threshold to filter out noise
	ang_vel_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
	if ang_vel_mag < GYRO_THRESHOLD:
		gyro_x, gyro_y, gyro_z = 0, 0, 0
	
	if relative_to_world:
		roll, pitch, yaw = extract_euler_from_packet(packet)
		return make_relative_to_world(gyro_x, gyro_y, gyro_z, roll, pitch, yaw)
	

	return gyro_x, gyro_y, gyro_z


def extract_linear_acceleration_from_packet(packet, relative_to_world=True):
	"""
	Extracts linear acceleration from the IMU packet.
	
	Applies a 25 mg (0.025 g) threshold to filter out noise.
	
	Args:
		packet: IMU data packet (list/array)
	
	Returns:
		tuple: (acc_x, acc_y, acc_z) in g
	"""
	# Linear acceleration indices from Fizzy data packet
	acc_x_raw = packet[7]
	acc_y_raw = packet[8]
	acc_z_raw = packet[9]
	
	# Apply threshold to accelerations
	acc_x = 0 if abs(acc_x_raw) < ACC_THRESHOLD else acc_x_raw
	acc_y = 0 if abs(acc_y_raw) < ACC_THRESHOLD else acc_y_raw
	acc_z = 0 if abs(acc_z_raw) < ACC_THRESHOLD else acc_z_raw
	
	if relative_to_world:
		roll, pitch, yaw = extract_euler_from_packet(packet)
		return make_relative_to_world(acc_x, acc_y, acc_z, roll, pitch, yaw)
	
	return acc_x, acc_y, acc_z

def make_relative_to_world(x, y, z, roll, pitch, yaw):
	# Rotation matrix from body frame to world frame
	R = np.array([
		[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
		[np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
		[-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]
	])
	acc_world = R @ np.array([x, y, z])
	return acc_world[0], acc_world[1], acc_world[2]


def extract_acceleration_magnitude_from_packet(packet):
	"""
	Calculates the magnitude of linear acceleration from the IMU packet.
	
	Automatically applies the acceleration threshold before calculation.
	
	Args:
		packet: IMU data packet (list/array)
	
	Returns:
		float: acceleration magnitude in g
	"""
	acc_x, acc_y, acc_z = extract_linear_acceleration_from_packet(packet)
	magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
	return magnitude


def extract_gyro_magnitude_from_packet(packet):
	"""
	Calculates the magnitude of angular velocity from the IMU packet.
	
	Args:
		packet: IMU data packet (list/array)
	
	Returns:
		float: angular velocity magnitude in rad/s
	"""
	gyro_x, gyro_y, gyro_z = extract_angular_velocity_from_packet(packet)
	magnitude = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
	return magnitude


def apply_acceleration_threshold(acc_x, acc_y, acc_z, threshold=ACC_THRESHOLD):
	"""
	Applies acceleration threshold to filter out noise.
	
	Args:
		acc_x, acc_y, acc_z: acceleration components (g)
		threshold: acceleration threshold in g (default: 0.025 = 25 mg)
	
	Returns:
		tuple: (acc_x, acc_y, acc_z) thresholded acceleration values
	"""
	acc_x = 0 if abs(acc_x) < threshold else acc_x
	acc_y = 0 if abs(acc_y) < threshold else acc_y
	acc_z = 0 if abs(acc_z) < threshold else acc_z
	return acc_x, acc_y, acc_z
