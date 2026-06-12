import sys
import numpy as np
import time
import threading
from collections import deque

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from utilities.fizzy_udp import Fizzy
from panels import (
    ControlPanel, RecordingPanel, AnalysePanel, FeaturizationPanel, 
    TrainingPanel, LiveClassificationPanel, ClassificationAnalysisPanel, ActionsPanel
)


from managers import (
    GraphManager, RecordingManager, AnalyseManager, FeaturizationManager, 
    TrainingManager, LiveClassificationManager, ClassificationAnalysisManager, ActionsManager
)


from utilities.gizmo_rendering import GizmoRenderer
from utilities.imu_data_extractor import (
    extract_euler_from_packet,
    extract_linear_acceleration_from_packet,
    extract_angular_velocity_from_packet,
    extract_acceleration_magnitude_from_packet,
    extract_gyro_magnitude_from_packet,
)

class FizzyIMUDashboard(QtWidgets.QMainWindow):
    # Signal for updating recording status from background thread
    recording_status_updated = QtCore.pyqtSignal(str, str)  # (text, style)
    relative_to_world = True
    
    def __init__(self, fizzy=None, xbox_thread=None):
        super().__init__()
        self.setWindowTitle("Real-Time IMU 3D Orientation")
        self.resize(1200, 600)
        
        # Store xbox thread reference for gamepad control
        self.xbox_thread = xbox_thread
        
        # UI Scale factor (1.0 = default, 1.1 = 10% larger, etc.)
        self.ui_scale_factor = 1.0
        self.base_font = QtGui.QFont()
        self.base_font.setPointSize(10)
        
        # Central widget
        self.cw = QtWidgets.QWidget()
        self.setCentralWidget(self.cw)
        self.main_layout = QtWidgets.QHBoxLayout(self.cw)
        
        # ----- LEFT SIDE: PLOTS -----
        self.left_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(self.left_layout, 1)
        
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.left_layout.addWidget(self.plot_widget, stretch=1)
        
        # Initialize graph manager for all plotting functionality
        self.graph_manager = GraphManager(self.plot_widget, data_plotting=200)

        # ----- RIGHT SIDE: 3D GIZMO & LABELS -----
        self.right_layout = QtWidgets.QVBoxLayout()
        self.right_container = QtWidgets.QWidget()
        self.right_container.setLayout(self.right_layout)
        self.main_layout.addWidget(self.right_container, 1)
        
        # Add labels to display exact angles and window statistics
        self.info_label = QtWidgets.QLabel("roll: 0.00° | pitch: 0.00° | yaw: 0.00°")
        info_font = QtGui.QFont("Courier", 7)
        info_font.setStyleStrategy(QtGui.QFont.StyleStrategy.PreferAntialias)
        self.info_label.setFont(info_font)
        self.info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.info_label.setWordWrap(True)
        self.info_label.setTextFormat(QtCore.Qt.TextFormat.RichText)  # Enable HTML rendering

        
        # Connection status label
        self.connection_status_label = QtWidgets.QLabel("Fizzy: Connected ✓")
        status_font = self.connection_status_label.font()
        status_font.setPointSize(10)
        status_font.setBold(True)
        # green
        self.connection_status_label.setStyleSheet("color: green;")
        self.connection_status_label.setFont(status_font)
        self.connection_status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        self.right_layout.addWidget(self.connection_status_label)
        self.right_layout.addWidget(self.info_label)

        # Setup OpenGL Viewport
        self.view = gl.GLViewWidget()
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.right_layout.addWidget(self.view, 1)
        
        # Initialize gizmo renderer
        self.gizmo = GizmoRenderer(self.view)
        
        # ----- INITIALIZE CONTROL PANELS -----
        self.control_panel = ControlPanel()
        self.recording_panel = RecordingPanel()
        self.analyse_panel = AnalysePanel()
        self.featurization_panel = FeaturizationPanel()
        self.training_panel = TrainingPanel()
        self.live_classification_panel = LiveClassificationPanel()
        self.classification_analysis_panel = ClassificationAnalysisPanel()
        self.actions_panel = ActionsPanel()
        self.analyse_panel.set_label_options(self.graph_manager.available_labels, checked=True)
        self._featurization_options_source = None
        self._training_options_csv = None
        self.mode_panels = [
            self.recording_panel,
            self.analyse_panel,
            self.featurization_panel,
            self.training_panel,
            self.live_classification_panel,
            self.classification_analysis_panel,
        ]
        
        # Add panels to layout
        self.right_layout.addWidget(self.control_panel)
        for panel in self.mode_panels:
            self.right_layout.addWidget(panel)
        self.right_layout.addWidget(self.actions_panel)

        # Set initial visibility state (Recording mode is default, index 0)
        self._set_mode_panels_visible(0)
        self.actions_panel.setVisible(True)  # Visible in Recording mode (index 0)
        
        # Data buffer for background thread collection
        self.data_buffer = deque(maxlen=None)  # Keeps storing data until RAM is full, TODO: save in parts, then merge?
        self.data_lock = threading.Lock()
        self.data_collection_running = True

        # Latest motor command sent by the control script (updated via set_motor)
        self.latest_motor_input = 0.0
        
        # FFT computation buffers
        self.fft_data_r = deque()  # Rolling buffer for FFT computation
        self.fft_data_p = deque()
        self.fft_data_y = deque()
        self.fft_data_ax = deque()
        self.fft_data_ay = deque()
        self.fft_data_az = deque()
        self.fft_data_acc_mag = deque()
        self.fft_data_wx = deque()
        self.fft_data_wy = deque()
        self.fft_data_wz = deque()
        self.fft_data_gyro_mag = deque()
        self.fft_data_timestamps = deque()
        self.fft_lock = threading.Lock()
        self.fft_window_size_samples = 128
        self.fft_stride_samples = 64
        self.fft_samples_since_update = 0
        
        # Latest computed FFT results
        self.latest_fft_results = None
        
        # Initialize Recording Manager (before signal connections)
        self.recording_manager = RecordingManager(self.recording_panel, self.data_buffer, self.data_lock, self)
        self.analyse_manager = AnalyseManager(self)
        self.featurization_manager = FeaturizationManager(self)
        self.training_manager = TrainingManager(self)
        self.live_classification_manager = LiveClassificationManager(self)
        self.classification_analysis_manager = ClassificationAnalysisManager(self)
        self.actions_manager = ActionsManager(self)
        self.analyse_manager.init_windowed_plots()
        self.classification_analysis_manager.init_windowed_plots()
        
        # Connect control panel signals
        self.control_panel.mode_changed.connect(self._on_mode_changed)
        self.control_panel.graph_visibility_changed.connect(self._on_graph_visibility_changed)
        self.control_panel.calibrate_requested.connect(self.recording_manager.calibrate_yaw)
        self.control_panel.reconnect_requested.connect(self.recording_manager.reconnect_requested)
        self.control_panel.fft_toggled.connect(self._on_fft_toggled)
        self.control_panel.current_window_toggled.connect(self._on_current_window_toggled)
        self.control_panel.window_size_changed.connect(self._on_window_size_changed)
        self.control_panel.window_unit_changed.connect(self._on_window_unit_changed)
        self.control_panel.overlap_changed.connect(self._on_overlap_changed)
        self.control_panel.scale_increased.connect(self._on_scale_increased)
        self.control_panel.scale_decreased.connect(self._on_scale_decreased)
        self.control_panel.window_closed.connect(self.close_window)

        # Connect recording panel signals
        self.recording_panel.bind_manager(self.recording_manager)
        
        # Setup gamepad callbacks if xbox_thread is available
        if self.xbox_thread is not None:
            self.xbox_thread.on_record_start = lambda: self.recording_manager.on_gamepad_start_recording(self.xbox_thread)
            self.xbox_thread.on_record_stop = lambda: self.recording_manager.on_gamepad_stop_recording(self.xbox_thread)
            self.xbox_thread.on_record_cancel = lambda: self.recording_manager.on_gamepad_cancel_recording(self.xbox_thread)
            self.xbox_thread.on_marker_1 = lambda: self.recording_manager.on_recording_marker("marker_1")
            self.xbox_thread.on_marker_2 = lambda: self.recording_manager.on_recording_marker("marker_2")
            self.xbox_thread.on_marker_3 = lambda: self.recording_manager.on_recording_marker("marker_3")
            self.xbox_thread.on_marker_4 = lambda: self.recording_manager.on_recording_marker("marker_4")
        
        # Bind tab panels to their managers
        self.analyse_panel.bind_manager(self.analyse_manager)
        self.featurization_panel.bind_manager(self.featurization_manager)
        self.training_panel.bind_manager(self.training_manager)
        self.live_classification_panel.bind_manager(self.live_classification_manager)
        self.classification_analysis_panel.bind_manager(self.classification_analysis_manager)
        self.control_panel.window_size_changed.connect(self.live_classification_manager.on_settings_changed)
        self.control_panel.window_unit_changed.connect(self.live_classification_manager.on_settings_changed)
        self.control_panel.window_size_changed.connect(self.classification_analysis_manager.on_window_size_changed)
        self.control_panel.window_unit_changed.connect(self.classification_analysis_manager.on_window_size_changed)
        self.actions_panel.bind_manager(self.actions_manager)

        # Sync FFT cadence with the control panel's current window settings.
        self._sync_fft_window_settings()

        # Apply the default visibility state after all managers are ready.
        self._apply_graph_visibility_state(self.control_panel.get_graph_visibility_state())
        self._sync_current_window_visibility()
        
        # Initialize IMU connection
        if fizzy is not None:
            self.fizzy = fizzy
        else:
            self.fizzy = Fizzy()
        
        # Flush any stale data in the UDP buffer before starting downlink
        # This prevents old packets from previous sessions corrupting the data stream
        try:
            print("Clearing UDP buffer...")
            self.fizzy.flush_buffer()
        except Exception as e:
            print(f"Warning: Could not flush buffer: {e}")
        
        # Start downlink mode immediately to continuously receive data at ~104Hz
        # This is critical to avoid the background thread timeout when no initial downlink was started
        try:
            self.fizzy.start_downlink()
            print("Downlink mode started successfully")
        except Exception as e:
            print(f"Warning: Could not start downlink mode: {e}")
        
        self.calibrating = False
        self.yaw_offset = 0.0
        
        # Data mode flag
        self.playback_mode = False  # False = IMU, True = Playback
        
        # Start background data collection thread (maximizes data reception)
        self.data_thread = threading.Thread(target=self._data_collection_thread, daemon=True)
        self.data_thread.start()
        
        # Start high-speed timer for live updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_gizmo)
        # Prioritize speed (~60fps)
        self.timer.start(10)

    def _set_gizmo_visibility(self, visible):
        """Show or hide the 3D gizmo and its info label."""
        self.info_label.setVisible(visible)
        self.view.setVisible(visible)

    def _set_mode_panels_visible(self, active_index):
        """Show only the panel that matches the active mode index."""
        for index, panel in enumerate(self.mode_panels):
            panel.setVisible(index == active_index)

    def set_motor(self, value):
        """Store the latest motor command so the recording manager can log it."""
        self.latest_motor_input = float(value)

    def _on_mode_changed(self, index):
        """Handle mode switching between Record, Analyse, Featurization, Train, and Live Classification."""
        # Ensure any leftover analyse-mode overlays and state are cleaned up
        # when switching modes (labels, analysis window regions, gap markers).
        # Clear overlays first so graph clearing doesn't accidentally leave items.
        try:
            self.graph_manager.clear_label_overlays()
            self.graph_manager.clear_marker_lines()
        except Exception:
            pass
        self.analyse_manager.clear_windowed_marker_lines()
        self.analyse_manager.clear_windowed_gap_regions()
        try:
            self.classification_analysis_manager.clear_colored_regions()
        except Exception:
            pass
        try:
            self.graph_manager.clear_analysis_window_regions()
        except Exception:
            pass
        try:
            self.graph_manager.clear_gap_regions()
        except Exception:
            pass
        # Reset label/window bookkeeping
        try:
            self.graph_manager.labels.clear()
            self.graph_manager.window_indices.clear()
            self.graph_manager.current_window_idx = -1
        except Exception:
            pass
        # Disable window hover/click highlighting until the next tab re-registers it.
        try:
            self.graph_manager.window_click_callback = None
            self.graph_manager.clear_window_hover_highlight()
        except Exception:
            pass
        self._set_mode_panels_visible(index)
        
        mode_config = {
            0: {"show_gizmo": True, "playback_mode": False, "start_live": False},
            1: {"show_gizmo": True, "playback_mode": True, "start_live": False},
            2: {"show_gizmo": False, "playback_mode": True, "start_live": False},
            3: {"show_gizmo": False, "playback_mode": True, "start_live": False},
            4: {"show_gizmo": False, "playback_mode": False, "start_live": True},
            5: {"show_gizmo": False, "playback_mode": True, "start_live": False},
        }.get(index)

        if mode_config is None:
            return

        self.live_classification_manager.stop()
        self._set_gizmo_visibility(mode_config["show_gizmo"])
        self.right_container.setVisible(True)
        
        # Show actions panel only in Recording (0) and Live Classification (4) modes
        self.actions_panel.setVisible(index in (0, 4))

        if mode_config["playback_mode"]:
            self.playback_mode = True
            with self.data_lock:
                self.data_buffer.clear()
        else:
            self._set_data_source_to_imu()

        self._clear_graph_data()

        self._sync_current_window_visibility()

        if mode_config["start_live"]:
            self.live_classification_manager.start()

    def _apply_graph_visibility_state(self, visibility_state):
        """Apply graph visibility toggles across the main plots and windowed views."""
        self.graph_manager.set_graph_visibility_state(visibility_state)
        self._sync_current_window_visibility()

    def _sync_current_window_visibility(self):
        """Show the current-window plots only when the mode and toggle both allow it."""
        current_window_enabled = self.control_panel.current_window_checkbox.isChecked()
        visibility_state = self.control_panel.get_graph_visibility_state()
        self.analyse_manager.set_windowed_graph_visibility_state(visibility_state, current_window_enabled and self.control_panel.get_current_mode() == 1)
        self.classification_analysis_manager.set_windowed_graph_visibility_state(visibility_state, current_window_enabled and self.control_panel.get_current_mode() == 5)

    def _on_graph_visibility_changed(self, graph_key, enabled):
        """Handle a main graph visibility toggle."""
        self.graph_manager.set_graph_visibility(graph_key, enabled)
        self.analyse_manager.set_windowed_graph_visibility(graph_key, enabled)
        self.classification_analysis_manager.set_windowed_graph_visibility(graph_key, enabled)
        self._sync_current_window_visibility()

    def _on_current_window_toggled(self, enabled):
        """Handle current-window visibility toggle."""
        self._sync_current_window_visibility()
    
    def _on_fft_toggled(self, enabled):
        """Handle FFT toggle from control panel."""
        self.graph_manager.set_fft_enabled(enabled)
        self._sync_current_window_visibility()
    
    def _on_window_size_changed(self, size):
        """Handle window size change from control panel."""
        unit = self.control_panel.get_window_unit()
        self.graph_manager.set_window_size(size, unit)
        self._sync_fft_window_settings()
        self.analyse_manager.refresh_analysis_window_state()
        
    def _on_window_unit_changed(self, unit):
        """Handle window unit change from control panel."""
        size = self.control_panel.get_window_size()
        self.graph_manager.set_window_size(size, unit)
        self._sync_fft_window_settings()
        self.analyse_manager.refresh_analysis_window_state()

    def _on_overlap_changed(self, overlap_percent):
        """Handle overlap change from control panel."""
        self.analyse_manager.on_overlap_changed(overlap_percent)
        self.classification_analysis_manager.on_overlap_changed(overlap_percent)
        self._sync_fft_window_settings()
        self.live_classification_manager.on_settings_changed()

    def _sync_fft_window_settings(self):
        """Mirror the control panel window settings into the live FFT pipeline."""
        window_size = self.control_panel.get_window_size()
        window_unit = self.control_panel.get_window_unit()
        overlap_percent = self.control_panel.get_overlap_percent()

        self.graph_manager.set_window_size(window_size, window_unit)
        self.fft_samples_since_update = 0

    def _get_current_fft_window_sample_count(self, timestamps=None):
        """Resolve the current FFT window length from the buffered packet timestamps."""
        if timestamps is None:
            timestamps = self.fft_data_timestamps

        window_samples = self.graph_manager.get_fft_window_sample_count(timestamps)
        return max(1, int(window_samples))

    def _on_scale_increased(self):
        """Handle scale increase request."""
        self.ui_scale_factor = min(3.0, self.ui_scale_factor + 0.1)  # Max 3x scale
        self._apply_ui_scale()
    
    def _on_scale_decreased(self):
        """Handle scale decrease request."""
        self.ui_scale_factor = max(0.5, self.ui_scale_factor - 0.1)  # Min 0.5x scale
        self._apply_ui_scale()
    
    def _apply_ui_scale(self):
        """Apply the current scale factor to all UI elements."""
        # Scale the application font
        new_font = QtGui.QFont(self.base_font)
        new_font.setPointSize(int(10 * self.ui_scale_factor))
        instance = QtWidgets.QApplication.instance()
        if instance is not None:
            instance.setFont(new_font)  # type: ignore

        # Update layout spacing based on scale
        scale_int = int(self.ui_scale_factor * 10)
        self.main_layout.setSpacing(scale_int)
        self.main_layout.setContentsMargins(scale_int, scale_int, scale_int, scale_int)

    def _set_data_source_to_imu(self):
        """Switch data source to live IMU."""
        self.playback_mode = False
        # Clear playback data
        with self.data_lock:
            self.data_buffer.clear()
    
    def _clear_graph_data(self):
        """Clear all accumulated graph data."""
        self.graph_manager.clear_all_data()
        with self.fft_lock:
            self.fft_data_r.clear()
            self.fft_data_p.clear()
            self.fft_data_y.clear()
            self.fft_data_ax.clear()
            self.fft_data_ay.clear()
            self.fft_data_az.clear()
            self.fft_data_acc_mag.clear()
            self.fft_data_wx.clear()
            self.fft_data_wy.clear()
            self.fft_data_wz.clear()
            self.fft_data_gyro_mag.clear()
            self.fft_data_timestamps.clear()
            self.fft_samples_since_update = 0
            self.latest_fft_results = None

    def _data_collection_thread(self):
        """Background thread that continuously collects data at maximum rate and computes FFT.
        Uses non-blocking operations to prevent blocking the main UI thread."""
        consecutive_empty_reads = 0
        max_empty_reads_before_recovery = 10  # Trigger recovery after 10 consecutive timeouts
        
        while self.data_collection_running:
            try:
                # Only collect from IMU if not in playback mode
                if hasattr(self, 'playback_mode') and self.playback_mode:
                    time.sleep(0.01)
                    continue
                
                # Use downlink mode consistently with main thread (with short timeout)
                data = self.fizzy.get_data_downlink()
                
                # Skip invalid packets (-1 from timeout or other errors)
                if data == -1 or not isinstance(data, (list, tuple)) or len(data) < 7:
                    consecutive_empty_reads += 1
                    
                    # If we're getting too many timeouts, try to recover the connection
                    if consecutive_empty_reads >= max_empty_reads_before_recovery:
                        if self.fizzy.is_connected:
                            print(f"Data collection: {consecutive_empty_reads} consecutive timeouts - attempting recovery")
                            self.fizzy.downlink_restart_needed = True  # Trigger reconnection attempt
                        consecutive_empty_reads = 0  # Reset counter to avoid spam
                    
                    # Small sleep to prevent busy-waiting during disconnect
                    time.sleep(0.01)
                    continue
                
                # Successfully received data - reset the empty read counter
                consecutive_empty_reads = 0
                
                # Try to store for display with non-blocking lock
                acquired = self.data_lock.acquire(blocking=False)
                if acquired:
                    try:
                        self.data_buffer.append(data)
                    finally:
                        self.data_lock.release()
                
                # Extract data for FFT buffer with non-blocking lock
                try:
                    roll, pitch, yaw = extract_euler_from_packet(data)
                    acc_x, acc_y, acc_z = extract_linear_acceleration_from_packet(data, self.relative_to_world)
                    acc_mag = extract_acceleration_magnitude_from_packet(data)
                    gyro_x, gyro_y, gyro_z = extract_angular_velocity_from_packet(data, self.relative_to_world)
                    gyro_mag = extract_gyro_magnitude_from_packet(data)
                    
                    # Add to FFT buffers with non-blocking lock
                    acquired = self.fft_lock.acquire(blocking=False)
                    if acquired:
                        try:
                            self.fft_data_timestamps.append(data[0] / 1_000_000.0)
                            self.fft_data_r.append(np.degrees(roll))
                            self.fft_data_p.append(np.degrees(pitch))
                            self.fft_data_y.append(np.degrees(yaw))
                            self.fft_data_ax.append(acc_x)
                            self.fft_data_ay.append(acc_y)
                            self.fft_data_az.append(acc_z)
                            self.fft_data_acc_mag.append(acc_mag)
                            self.fft_data_wx.append(gyro_x)
                            self.fft_data_wy.append(gyro_y)
                            self.fft_data_wz.append(gyro_z)
                            self.fft_data_gyro_mag.append(gyro_mag)
                            
                            # Recompute FFT when the configured stride has elapsed.
                            window_samples = self._get_current_fft_window_sample_count(self.fft_data_timestamps)
                            overlap_percent = float(self.control_panel.get_overlap_percent())
                            stride_samples = max(1, int(round(window_samples * (1.0 - overlap_percent / 100.0))))
                            self.fft_samples_since_update += 1
                            if (
                                self.fft_samples_since_update >= stride_samples
                                and len(self.fft_data_r) >= window_samples
                            ):
                                self.latest_fft_results = self._compute_all_ffts(window_samples)
                                self.fft_samples_since_update = 0
                        finally:
                            self.fft_lock.release()
                except Exception as e:
                    print(f"Data extraction error: {e}")
                    
            except Exception as e:
                # Log errors but continue collection
                print(f"Data collection thread error: {e}")
                time.sleep(0.01)


    # =====================================================================
    # Visualization Helper Functions
    # =====================================================================
    # These functions support 3D visualization and plotting of IMU data.
    # (Gizmo-specific visualization functions have been moved to gizmo_rendering.py)
    # =====================================================================

    def update_gizmo(self):
        try:
            # Update connection status indicator (non-blocking)
            if self.fizzy.is_connected:
                self.connection_status_label.setText("Fizzy: Connected ✓")
                self.connection_status_label.setStyleSheet("color: green;")
                # Enable record button when connected
                self.recording_panel.start_btn.setEnabled(True)
            else:
                self.connection_status_label.setText(f"Fizzy: Disconnected ✗ (waiting...)")
                self.connection_status_label.setStyleSheet("color: red;")
                # Disable record button when disconnected
                self.recording_panel.start_btn.setEnabled(False)
            
            # Periodically check if downlink needs to be restarted after reconnection
            self.fizzy.check_and_restart_downlink()
            
            # Skip updating gizmo and graphs with IMU data if we are in playback mode (like CSV Analysis)
            if getattr(self, 'playback_mode', False):
                return
            
            # Get latest data from buffer (non-blocking with timeout)
            try:
                acquired = self.data_lock.acquire(blocking=False)
                if not acquired:
                    # Lock is busy, skip this frame to avoid blocking the UI
                    return
                try:
                    if not self.data_buffer:
                        # Buffer is empty - keep showing last status
                        return
                    data = self.data_buffer[-1]  # Get most recent packet
                finally:
                    self.data_lock.release()
            except Exception as e:
                print(f"Error acquiring data lock: {e}")
                return
            
            if data is None or data == -1 or len(data) < 10:
                # Invalid data - skip update
                return

            roll, pitch, yaw = extract_euler_from_packet(data)
            
            yaw -= self.yaw_offset

            r_deg = np.degrees(roll)
            p_deg = np.degrees(pitch)
            y_deg = np.degrees(yaw)
            
            # Accelerations are at indices 7, 8, 9
            acc_scale = 1.2  # Increased from 0.5 for larger vectors
            # Extract accelerations with threshold applied
            acc_x_raw, acc_y_raw, acc_z_raw = extract_linear_acceleration_from_packet(data, self.relative_to_world)
            acc_x = acc_x_raw * acc_scale
            acc_y = acc_y_raw * acc_scale
            acc_z = acc_z_raw * acc_scale
            
            # Update acceleration vectors via gizmo renderer
            self.gizmo.update_acceleration_vectors(acc_x, acc_y, acc_z)
            
            # Timestamp (assume first element is timestamp in microseconds)
            t_s = data[0] / 1_000_000.0
            
            # Extract angular velocity (gyro) from packet
            gyro_x, gyro_y, gyro_z = extract_angular_velocity_from_packet(data, self.relative_to_world)
            
            # Calculate magnitude values
            acc_magnitude = extract_acceleration_magnitude_from_packet(data)
            gyro_magnitude = extract_gyro_magnitude_from_packet(data)
            
            # Update graphs with all data via graph manager
            self.graph_manager.append_data(
                t_s, r_deg, p_deg, y_deg,
                acc_x_raw, acc_y_raw, acc_z_raw, acc_magnitude,
                gyro_x, gyro_y, gyro_z, gyro_magnitude,
                self.latest_motor_input
            )
            
            # Update FFT plots with pre-computed results from high-frequency thread
            if self.control_panel.is_fft_enabled() and self.latest_fft_results:
                self.graph_manager.update_fft_plots_from_results(self.latest_fft_results)
            
            # Check if recording duration was exceeded (from recording thread)
            if self.recording_manager.is_recording and self.recording_manager.is_duration_exceeded():
                self.recording_manager.stop_recording()
            
            # Update label with frequency estimate from graph manager
            self.info_label.setText(f"Roll: {r_deg:7.2f}° | Pitch: {p_deg:7.2f}° | Yaw: {y_deg:7.2f}° | Freq: {self.graph_manager.frequency_estimate:6.1f} Hz\n"
                                    f"Acc X: {data[7]:6.2f} | Acc Y: {data[8]:6.2f} | Acc Z: {data[9]:6.2f}\n"
                                    f"Gyro X: {gyro_x:6.3f} | Gyro Y: {gyro_y:6.3f} | Gyro Z: {gyro_z:6.3f}")
            
            # Apply rotation transform to gizmo via renderer
            self.gizmo.apply_rotation_transform(r_deg, p_deg, y_deg)
            
        except Exception as e:
            print(f"Error in update_gizmo: {e}")

    def _compute_all_ffts(self, window_samples):
        """Compute FFT for all data streams using the high-frequency buffer data."""
        try:
            # Use the graph_manager's FFT computation methods
            results = {}
            results['r'] = self.graph_manager.compute_fft_windowed(list(self.fft_data_r)[-window_samples:])
            results['p'] = self.graph_manager.compute_fft_windowed(list(self.fft_data_p)[-window_samples:])
            results['y'] = self.graph_manager.compute_fft_windowed(list(self.fft_data_y)[-window_samples:])
            results['ax'] = self.graph_manager.compute_fft_windowed(list(self.fft_data_ax)[-window_samples:])
            results['ay'] = self.graph_manager.compute_fft_windowed(list(self.fft_data_ay)[-window_samples:])
            results['az'] = self.graph_manager.compute_fft_windowed(list(self.fft_data_az)[-window_samples:])
            results['acc_mag'] = self.graph_manager.compute_fft_windowed(list(self.fft_data_acc_mag)[-window_samples:])
            results['wx'] = self.graph_manager.compute_fft_windowed(list(self.fft_data_wx)[-window_samples:])
            results['wy'] = self.graph_manager.compute_fft_windowed(list(self.fft_data_wy)[-window_samples:])
            results['wz'] = self.graph_manager.compute_fft_windowed(list(self.fft_data_wz)[-window_samples:])
            results['gyro_mag'] = self.graph_manager.compute_fft_windowed(list(self.fft_data_gyro_mag)[-window_samples:])
            return results
        except Exception as e:
            return None
    
    def keyPressEvent(self, a0: QtGui.QKeyEvent | None):
        """Forward keyboard shortcuts to the active mode panel."""
        if a0 is None:
            return
        event = a0
        active_mode = self.control_panel.get_current_mode()

        if 0 <= active_mode < len(self.mode_panels):
            active_panel = self.mode_panels[active_mode]
            event.setAccepted(False)
            active_panel.keyPressEvent(event)
            if event.isAccepted():
                return

        super().keyPressEvent(event)


    def close_window(self):
        """Close the application window."""
        self.close()

    def closeEvent(self, a0):
        """Clean up background threads on window close."""
        event = a0  # Normalize parameter name for clarity
        if self.recording_manager.is_recording:
            reply = QtWidgets.QMessageBox.question(
                self, 'Recording Active',
                'Recording is still active. Save before closing?',
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            )
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                self.recording_manager.stop_recording()
            else:
                self.recording_manager.cancel_recording()
        
        # Stop live classification if running
        self.live_classification_manager.stop()
        
        self.data_collection_running = False
        self.timer.stop()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    # We create the main window
    window = FizzyIMUDashboard()
    window.show()
    
    # Run the QT main loop
    sys.exit(app.exec())