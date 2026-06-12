"""
Recording Manager Module

Handles all IMU data recording functionality including:
- Recording session management (start, stop, cancel)
- Data collection from IMU buffer
- CSV file output
- Status reporting via signals
"""

import os
import csv
import time
import threading
import numpy as np

from utilities.imu_data_extractor import (
    extract_euler_from_packet,
    extract_angular_velocity_from_packet,
    extract_linear_acceleration_from_packet,
    extract_acceleration_magnitude_from_packet,
)
from .audio_manager import AudioManager


class RecordingManager:
    """Manages IMU data recording sessions."""
    
    def __init__(self, recording_panel, data_buffer, data_lock, main_window=None):
        """
        Initialize the RecordingManager.
        
        Args:
            recording_panel: Reference to RecordingPanel UI component
            data_buffer: Reference to shared data buffer (deque)
            data_lock: Reference to threading lock for data_buffer access
            main_window: Reference to main window (for accessing yaw_offset and calibrating flag)
        """
        self.recording_panel = recording_panel
        self.data_buffer = data_buffer
        self.data_lock = data_lock
        self.main_window = main_window
        
        # Audio manager
        self.audio_manager = AudioManager()
        
        # Recording state
        self.is_recording = False
        self.recording_data = []
        self.recording_start_time = None
        self.recording_delay_time = None
        self.recording_duration = None
        self.recording_timestamp_offset = None
        self.last_recorded_buffer_index = -1
        self.recording_duration_exceeded = False
        self.recording_mode = "Duration"  # "Duration" or "Samples"
        self.recording_sample_count = 0
        self.recording_sample_target = 0
        self.countdown_beeps_emitted = set()  # Track which countdown seconds have been beep'd
        self.recording_elapsed = 0.0
        self.last_packet_timestamp = None
        
        # Thread management
        self.record_thread = None
        self.recording_thread_running = False
        self.recording_lock = threading.Lock()
        self.last_packet_time = None  # Track when we last received a packet
        self.packet_timeout = 2.0  # Timeout in seconds to detect data loss
        # Pending markers to apply to the next recorded samples when no
        # suitable recorded row exists yet. Stored as list of marker names.
        self.pending_markers = []
    
    def start_recording(self):
        """Start a recording session with configured parameters."""
        # Check if Fizzy is connected before starting
        if not self._is_fizzy_connected():
            status_text = "Status: Error - Fizzy is not connected!"
            self.recording_panel.status_updated.emit(status_text, "color: #FF6666;", 0.0)
            self.recording_panel.set_recording_active(False)
            return
        
        self.is_recording = True
        self.recording_data = []
        self.recording_start_time = time.time()
        self.last_packet_time = time.time()  # Reset packet timer
        self.recording_delay_time = self.recording_panel.get_delay()
        self.recording_duration = self.recording_panel.get_duration()
        self.recording_timestamp_offset = None  # Reset offset for new recording
        self.last_recorded_buffer_index = -1  # Reset packet tracking
        self.recording_duration_exceeded = False  # Reset duration flag
        self.recording_mode = self.recording_panel.get_record_mode()
        self.recording_sample_count = 0
        self.recording_sample_target = self.recording_panel.get_samples()
        self.countdown_beeps_emitted = set()  # Reset countdown tracking
        self._start_beep_emitted = False  # Reset start beep flag
        self.recording_elapsed = 0.0
        self.last_packet_timestamp = None
        
        self.recording_panel.set_recording_active(True)
        
        # Start dedicated recording thread
        self.recording_thread_running = True
        self.record_thread = threading.Thread(target=self._recording_thread, daemon=True)
        self.record_thread.start()
        
        status_text = f"Status: Waiting {self.recording_delay_time}s..."
        self.recording_panel.status_updated.emit(status_text, "color: #FFAA00;", 0.0)

    def add_marker(self, marker_name):
        """Record a marker event at the current recording timestamp."""
        if not self.is_recording or self.recording_start_time is None or self.recording_delay_time is None:
            return False

        current_time = time.time()
        elapsed = current_time - self.recording_start_time

        if elapsed < self.recording_delay_time:
            self.recording_panel.status_updated.emit(
                "Status: Marker ignored until recording starts",
                "color: #FFAA00;",
                0.3
            )
            return False

        marker_timestamp = elapsed - self.recording_delay_time
        timestamp_format = self.recording_panel.get_timestamp_format()
        if timestamp_format == "Milliseconds":
            marker_timestamp *= 1000.0
        # Try to attach the marker to the closest existing recorded data row.
        with self.recording_lock:
            if self.recording_data:
                # Find closest entry by timestamp
                closest_idx = None
                closest_diff = None
                for i, entry in enumerate(self.recording_data):
                    try:
                        ts = entry.get('timestamp')
                    except Exception:
                        ts = None
                    if ts is None:
                        continue
                    diff = abs(ts - marker_timestamp)
                    if closest_diff is None or diff < closest_diff:
                        closest_diff = diff
                        closest_idx = i

                if closest_idx is not None:
                    entry = self.recording_data[closest_idx]
                    existing = entry.get('markers')
                    if existing and existing != 'nm':
                        entry['markers'] = f"{existing};{marker_name}"
                    else:
                        entry['markers'] = marker_name

                    units = "ms" if timestamp_format == "Milliseconds" else "s"
                    applied_ts = entry.get('timestamp')
                    self.recording_panel.status_updated.emit(
                        f"Status: {marker_name} applied at {applied_ts:.3f} {units}",
                        "color: #FFAA00;",
                        0.3
                    )
                    return True
            # If we reach here no existing data row found: queue marker
            self.pending_markers.append(marker_name)

        units = "ms" if timestamp_format == "Milliseconds" else "s"
        # self.recording_panel.status_updated.emit(
        #     f"Adding {marker_name} at ({marker_timestamp:.3f} {units})",
        #     "color: #FFAA00;",
        #     0.3,
        # )
        return True
    
    def stop_recording(self):
        """Stop recording and save to CSV."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.recording_thread_running = False
        
        # Play stop beeps
        self.audio_manager.play_stop_beeps()
        
        # Only join if called from main thread (not from recording thread)
        if self.record_thread and self.record_thread.is_alive() and threading.current_thread() != self.record_thread:
            self.record_thread.join(timeout=1.0)
        
        self._save_recording_to_csv()
        
        self.recording_panel.set_recording_active(False)
        self.recording_panel.status_updated.emit("Status: Recording saved!", "color: #00FF00;", 0.5)
    
    def cancel_recording(self):
        """Cancel recording without saving."""
        self.is_recording = False
        self.recording_thread_running = False
        
        # Wait for recording thread to finish
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=1.0)
        
        self.recording_data = []
        
        self.recording_panel.set_recording_active(False)
        self.recording_panel.status_updated.emit("Status: Recording cancelled", "color: #FF6666;", 0.5)
    
    def is_duration_exceeded(self):
        """Check if recording duration has been exceeded."""
        return self.recording_duration_exceeded
    
    def _is_fizzy_connected(self):
        """Check if Fizzy is currently connected."""
        try:
            if self.main_window and hasattr(self.main_window, 'fizzy'):
                return self.main_window.fizzy.is_connected
        except Exception:
            pass
        return False
    
    def get_recording_data(self):
        """Get the current recording data."""
        with self.recording_lock:
            return self.recording_data.copy()
    
    def _recording_thread(self):
        """Background thread that processes all buffered packets for recording at full rate."""
        self.last_packet_time = time.time()  # Initialize packet tracking
        
        while self.recording_thread_running and self.is_recording:
            try:
                # Exit early if recording was stopped
                if not self.is_recording:
                    break
                
                # Check for connection loss or data timeout
                current_time = time.time()
                if self.last_packet_time is not None:
                    time_since_last_packet = current_time - self.last_packet_time
                    if time_since_last_packet > self.packet_timeout:
                        # No data received for too long
                        if not self._is_fizzy_connected():
                            status_text = "Status: Fizzy disconnected (recording paused, will resume on reconnect)"
                            self.recording_panel.status_updated.emit(status_text, "color: #FF9900;", 0.5)
                            # Keep recording running - will resume collecting data when Fizzy reconnects
                            # Reset packet timer so we don't keep spamming the warning
                            self.last_packet_time = current_time
                
                with self.data_lock:
                    if len(self.data_buffer) == 0:
                        time.sleep(0.001)
                        continue
                    # Process only new packets since last recording
                    current_index = len(self.data_buffer) - 1
                
                if current_index > self.last_recorded_buffer_index:
                    with self.data_lock:
                        # Bounds check: ensure index is still valid
                        if current_index < len(self.data_buffer):
                            data = list(self.data_buffer)[current_index]
                        else:
                            continue
                    
                    if data and len(data) >= 10:
                        # Update last packet time on successful data reception
                        self.last_packet_time = time.time()
                        
                        roll, pitch, yaw = extract_euler_from_packet(data)
                        
                        # Apply yaw offset (calibration) if available
                        if self.main_window and hasattr(self.main_window, 'yaw_offset'):
                            yaw -= self.main_window.yaw_offset
                        
                        r_deg = np.degrees(roll)
                        p_deg = np.degrees(pitch)
                        y_deg = np.degrees(yaw)
                        
                        # Update recording if active
                        self._update_recording(data, r_deg, p_deg, y_deg)
                        
                        self.last_recorded_buffer_index = current_index
            except Exception as e:
                pass
            
            # Small sleep to prevent CPU spin
            time.sleep(0.001)
    
    def _update_recording(self, data, r_deg, p_deg, y_deg):
        """Update recording with new data point."""
        # Guard against None values during thread execution
        if self.recording_start_time is None or self.recording_delay_time is None:
            return
        
        current_time = time.time()
        elapsed = current_time - self.recording_start_time
        
        # Check if we're still in delay period
        if elapsed < self.recording_delay_time:
            # Emit countdown beeps every second (rounded)
            seconds_elapsed = round(elapsed)
            if seconds_elapsed > 0 and seconds_elapsed not in self.countdown_beeps_emitted:
                self.countdown_beeps_emitted.add(seconds_elapsed)
                self.audio_manager.play_countdown_beep()
            
            remaining = self.recording_delay_time - elapsed
            status_text = f"Status: Recording in {remaining:.1f}s..."
            style = "color: #FFAA00;"
            self.recording_panel.status_updated.emit(status_text, style, 0.0)
            return
        
        # Emit start beep on first data point after delay (only once)
        if not self._start_beep_emitted:
            self._start_beep_emitted = True
            self.audio_manager.play_start_beep()
        
        # Check if recording should stop based on mode
        if self.recording_mode == "Samples":
            # Samples mode: check if we've reached target sample count
            if self.recording_sample_count >= self.recording_sample_target:
                self.recording_duration_exceeded = True
                return
        else:
            # Duration mode: check if duration exceeded
            if self.recording_duration is None:
                return
            if elapsed > self.recording_delay_time + self.recording_duration:
                self.recording_duration_exceeded = True
                return
        
        # Extract angular velocity
        gyro_x, gyro_y, gyro_z = extract_angular_velocity_from_packet(data, self.recording_panel.relative_to_world)
        
        timestamp_format = self.recording_panel.get_timestamp_format()

        # Calculate timestamp from the packet timestamp, using a per-session offset.
        raw_timestamp = float(data[0]) / 1_000_000.0

        wall_clock_elapsed = elapsed - self.recording_delay_time

        if self.recording_timestamp_offset is None:
            self.recording_timestamp_offset = raw_timestamp
        elif self.last_packet_timestamp is not None and raw_timestamp < self.last_packet_timestamp:
            self.recording_timestamp_offset = raw_timestamp - wall_clock_elapsed

        recording_elapsed = raw_timestamp - self.recording_timestamp_offset
        self.recording_elapsed = recording_elapsed
        self.last_packet_timestamp = raw_timestamp
        if timestamp_format == "Milliseconds":
            timestamp = recording_elapsed * 1_000.0
        else:  # Seconds
            timestamp = recording_elapsed
        
        # Apply 25 mg (0.025 g) threshold to accelerations
        acc_x_thresholded, acc_y_thresholded, acc_z_thresholded = extract_linear_acceleration_from_packet(data, self.recording_panel.relative_to_world)
        
        # Record selected data
        motor_input = getattr(self.main_window, 'latest_motor_input', 0.0)
        record_entry = {
            'timestamp': timestamp,
            'roll': r_deg if self.recording_panel.is_data_type_selected('Roll') else None,
            'pitch': p_deg if self.recording_panel.is_data_type_selected('Pitch') else None,
            'yaw': y_deg if self.recording_panel.is_data_type_selected('Yaw') else None,
            'acc_x': acc_x_thresholded if self.recording_panel.is_data_type_selected('Acc X') else None,
            'acc_y': acc_y_thresholded if self.recording_panel.is_data_type_selected('Acc Y') else None,
            'acc_z': acc_z_thresholded if self.recording_panel.is_data_type_selected('Acc Z') else None,
            'gyro_x': gyro_x if self.recording_panel.is_data_type_selected('Gyro X') else None,
            'gyro_y': gyro_y if self.recording_panel.is_data_type_selected('Gyro Y') else None,
            'gyro_z': gyro_z if self.recording_panel.is_data_type_selected('Gyro Z') else None,
            'motor_input': motor_input if self.recording_panel.is_data_type_selected('Motor Input') else None,
            'markers': 'nm', # no marker
        }

        with self.recording_lock:
            # Apply any pending markers to this new recorded sample
            if self.pending_markers:
                # Join multiple pending markers with semicolon
                record_entry['markers'] = ';'.join(self.pending_markers)
                self.pending_markers = []

            self.recording_data.append(record_entry)
            self.recording_sample_count += 1
        
        # Update status using signal (thread-safe)
        with self.recording_lock:
            data_count = self.recording_sample_count
        
        if self.recording_mode == "Samples":
            # Samples mode status
            remaining_samples = self.recording_sample_target - data_count
            status_text = f"Status: Recording ({data_count} / {self.recording_sample_target} samples) - {remaining_samples} left"
        else:
            # Duration mode status
            recording_elapsed = elapsed - self.recording_delay_time
            remaining_time = self.recording_duration - recording_elapsed
            status_text = f"Status: Recording ({data_count} points) - {remaining_time:.1f}s left"
        
        style = "color: #5599FF;"
        self.recording_panel.status_updated.emit(status_text, style, 0.0)
    
    def _save_recording_to_csv(self):
        """Save recorded data to CSV file."""
        with self.recording_lock:
            if not self.recording_data:
                self.recording_panel.status_updated.emit("Status: No data to save", "color: #FF6666;", 0.0)
                return
            
            data_to_save = self.recording_data.copy()
        
        filename = self.recording_panel.get_filename()
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Get folder path and create directories if needed
        folder = self.recording_panel.get_folder()
        if folder == '.' or folder == '':
            folder = ''
        
        # Convert forward slashes to OS-specific path separators
        folder = folder.replace('/', os.sep)
        
        # Create the full file path
        filepath = os.path.join(folder, filename)
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(folder, exist_ok=True)
            
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'roll', 'pitch', 'yaw', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'motor_input', 'markers']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for entry in data_to_save:
                    # Only write selected columns
                    filtered_entry = {}
                    fieldname_mapping = {
                        'timestamp': 'timestamp',
                        'roll': 'Roll',
                        'pitch': 'Pitch',
                        'yaw': 'Yaw',
                        'acc_x': 'Acc X',
                        'acc_y': 'Acc Y',
                        'acc_z': 'Acc Z',
                        'gyro_x': 'Gyro X',
                        'gyro_y': 'Gyro Y',
                        'gyro_z': 'Gyro Z',
                        'motor_input': 'Motor Input',
                        'markers': 'markers',
                    }
                    for field in fieldnames:
                        panel_field = fieldname_mapping.get(field, field)
                        if field == 'timestamp' or field == 'markers' or (panel_field in ['Roll', 'Pitch', 'Yaw', 'Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Acc Magnitude', 'Gyro Magnitude', 'Motor Input'] and self.recording_panel.is_data_type_selected(panel_field)):
                            filtered_entry[field] = entry[field]
                    writer.writerow(filtered_entry)
            
            status_text = f"Status: Saved to {filepath} ({len(data_to_save)} rows)"
            self.recording_panel.status_updated.emit(status_text, "color: #00FF00;", 0.0)
            print(f"Recording saved to {filepath}")
            
            # Auto-increment filename if enabled
            if self.recording_panel.get_auto_increment():
                self.recording_panel.auto_increment_filename()
        except Exception as e:
            status_text = f"Status: Error saving file: {str(e)}"
            self.recording_panel.status_updated.emit(status_text, "color: #FF6666;", 0.0)
            print(f"Error saving recording: {e}")

    def on_gamepad_start_recording(self, xbox_thread=None):
        """Start recording from a gamepad callback."""
        self.start_recording()
        delay = self.recording_panel.get_delay()
        if delay > 0 and xbox_thread is not None:
            from threading import Thread

            vibration_thread = Thread(target=xbox_thread.countdown_vibration, args=(delay,), daemon=True)
            vibration_thread.start()

    def on_gamepad_stop_recording(self, xbox_thread=None):
        """Stop recording from a gamepad callback."""
        self.stop_recording()
        if xbox_thread is not None:
            from threading import Thread

            completion_thread = Thread(target=xbox_thread.completion_vibration, daemon=True)
            completion_thread.start()

    def on_gamepad_cancel_recording(self, xbox_thread=None):
        """Cancel recording from a gamepad callback."""
        self.cancel_recording()

    def on_recording_marker(self, marker_name):
        """Add a marker to the active recording."""
        return self.add_marker(marker_name)

    def on_relative_frame_toggle(self, index):
        """Store the selected reference frame mode."""
        self.recording_panel.relative_to_world = (index == 1)

    def browse_folder(self, parent=None):
        """Open a folder picker for the recording output path."""
        from utilities.dialog_utils import get_existing_directory

        if parent is None:
            parent = self.main_window

        directory = get_existing_directory(parent, "Select Directory to Save Recording", "")
        if directory:
            self.recording_panel.set_folder(directory)

    def calibrate_yaw(self):
        """Calibrate yaw offset to the latest buffered packet."""
        if not self.main_window:
            return

        with self.data_lock:
            if not self.data_buffer:
                return
            data = self.data_buffer[-1]

        if data is None or len(data) < 10:
            return

        _, _, yaw = extract_euler_from_packet(data)
        self.main_window.yaw_offset = yaw

    def reconnect_requested(self):
        """Force downlink restart on the Fizzy device."""
        if not self.main_window or not hasattr(self.main_window, "fizzy"):
            return
        self.main_window.fizzy.downlink_restart_needed = True
        self.main_window.fizzy.check_and_restart_downlink()
