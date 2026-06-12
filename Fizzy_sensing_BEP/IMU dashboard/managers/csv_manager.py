"""
CSV Manager Module

Handles all CSV file operations including:
- Loading and parsing CSV files
- Converting CSV rows to IMU packet format
- Data extraction and preprocessing
"""

import csv
from math import nan
import numpy as np
from utilities.imu_data_extractor import apply_acceleration_threshold


class CSVManager:
    """Manages CSV file operations for IMU data."""
    
    # CSV column name mappings
    COLUMN_NAMES = {
        'timestamp': 'timestamp',
        'roll': 'roll',
        'pitch': 'pitch',
        'yaw': 'yaw',
        'acc_x': 'acc_x',
        'acc_y': 'acc_y',
        'acc_z': 'acc_z',
        'gyro_x': 'gyro_x',
        'gyro_y': 'gyro_y',
        'gyro_z': 'gyro_z',
    }
    
    @staticmethod
    def parse_csv_row(row, apply_threshold=True):
        """
        Parse a single CSV row into a data dictionary.
        
        Args:
            row: Dictionary from csv.DictReader
            apply_threshold: Whether to apply acceleration threshold
            
        Returns:
            Dictionary with parsed data or None if parsing fails
        """
        try:
            # Preserve marker information. Marker-only rows will be flagged
            # so downstream code can render marker lines without contaminating
            # the numeric time-series data.
            raw_marker = row.get('markers', '')
            marker_value = str(raw_marker).strip()
            marker_name = 'nm'  # default to 'no marker'
            if marker_value != '' and marker_value.lower() != 'none' and marker_value.lower() != 'nm':
                marker_name = marker_value

            # Parse accelerations (support both old and new CSV formats)
            acc_x_raw = float(row.get('acc_x', nan)) if row.get('acc_x') and row.get('acc_x')  != '' else nan
            acc_y_raw = float(row.get('acc_y', nan)) if row.get('acc_y') and row.get('acc_y') != '' else nan
            acc_z_raw = float(row.get('acc_z', nan)) if row.get('acc_z') and row.get('acc_z') != '' else nan
            
            # Apply threshold if requested
            if apply_threshold:
                acc_x_processed, acc_y_processed, acc_z_processed = apply_acceleration_threshold(
                    acc_x_raw, acc_y_raw, acc_z_raw
                )
            else:
                acc_x_processed = acc_x_raw
                acc_y_processed = acc_y_raw
                acc_z_processed = acc_z_raw
            # Get other values with default nan if missing or empty
            data_row = {
                'timestamp': float(row.get('timestamp', nan)),
                'roll': float(row.get('roll', nan)) if row.get('roll') and row.get('roll') != '' else nan,
                'pitch': float(row.get('pitch', nan)) if row.get('pitch') and row.get('pitch') != '' else nan,
                'yaw': float(row.get('yaw', nan)) if row.get('yaw') and row.get('yaw') != '' else nan,
                'acc_x': acc_x_processed,
                'acc_y': acc_y_processed,
                'acc_z': acc_z_processed,
                'gyro_x': float(row.get('gyro_x', nan)) if row.get('gyro_x') and row.get('gyro_x') != '' else nan,
                'gyro_y': float(row.get('gyro_y', nan)) if row.get('gyro_y') and row.get('gyro_y') != '' else nan,
                'gyro_z': float(row.get('gyro_z', nan)) if row.get('gyro_z') and row.get('gyro_z') != '' else nan,
            }
            # Attach marker metadata so analysis code can pick up marker
            # timestamps without polluting graph data arrays.
            data_row['markers'] = marker_name
            data_row['motor_input'] = float(row.get('motor_input', 0.0)) if row.get('motor_input') and row.get('motor_input') != '' else 0.0
            return data_row
        except (ValueError, KeyError):
            return None
    
    @staticmethod
    def load_csv_file(filepath, apply_threshold=True):
        """
        Load CSV file and parse all rows.
        
        Args:
            filepath: Path to CSV file
            apply_threshold: Whether to apply acceleration threshold
            
        Returns:
            List of parsed data rows or empty list if file is invalid
            Type: List[Dict[str, float]]
        """
        data = []
        try:
            with open(filepath, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    parsed_row = CSVManager.parse_csv_row(row, apply_threshold=apply_threshold)
                    if parsed_row is not None:
                        data.append(parsed_row)
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return []
        
        return data
    
    @staticmethod
    def convert_csv_row_to_packet(data_row):
        """
        Convert CSV row data to IMU packet format.
        
        Packet format: [timestamp, ?, ?, qx, qy, qz, qw, acc_x, acc_y, acc_z, ?, ?, ?, gyro_x, gyro_y, gyro_z, ...]
        
        Args:
            data_row: Dictionary with parsed CSV data
            
        Returns:
            List representing IMU packet
        """
        # Convert timestamp to microseconds
        timestamp = data_row['timestamp'] * 1_000_000.0
        
        # Convert Euler angles to quaternion
        roll = np.radians(data_row['roll'])
        pitch = np.radians(data_row['pitch'])
        yaw = np.radians(data_row['yaw'])
        
        # Euler to Quaternion conversion
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        packet = [
            timestamp,              # 0: timestamp (microseconds)
            0,                     # 1: unused
            0,                     # 2: unused
            qx,                    # 3: qx
            qy,                    # 4: qy
            qz,                    # 5: qz
            qw,                    # 6: qw
            data_row['acc_x'],     # 7: acc_x
            data_row['acc_y'],     # 8: acc_y
            data_row['acc_z'],     # 9: acc_z
            0,                     # 10: unused
            0,                     # 11: unused
            0,                     # 12: unused
            data_row['gyro_x'],    # 13: gyro_x
            data_row['gyro_y'],    # 14: gyro_y
            data_row['gyro_z'],    # 15: gyro_z
        ]
        return packet
    
    @staticmethod
    def extract_graph_data(data_list):
        """
        Extract data from list of parsed CSV rows for graph plotting.
        
        Args:
            data_list: List of parsed data dictionaries
            
        Returns:
            Dictionary containing lists for each data type
        """
        graph_data = {
            'timestamps': [],
            'rolls': [],
            'pitches': [],
            'yaws': [],
            'acc_xs': [],
            'acc_ys': [],
            'acc_zs': [],
            'gyro_xs': [],
            'gyro_ys': [],
            'gyro_zs': [],
        }
        
        for entry in data_list:
            # Skip marker-only rows when building time-series arrays
            graph_data['timestamps'].append(entry['timestamp'])
            graph_data['rolls'].append(entry['roll'])
            graph_data['pitches'].append(entry['pitch'])
            graph_data['yaws'].append(entry['yaw'])
            graph_data['acc_xs'].append(entry['acc_x'])
            graph_data['acc_ys'].append(entry['acc_y'])
            graph_data['acc_zs'].append(entry['acc_z'])
            graph_data['gyro_xs'].append(entry['gyro_x'])
            graph_data['gyro_ys'].append(entry['gyro_y'])
            graph_data['gyro_zs'].append(entry['gyro_z'])
        
        return graph_data