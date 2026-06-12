"""
Recording Control Panel - Recording settings and controls.
"""

import re
import json
import os
from datetime import datetime
import time
from PyQt6 import QtWidgets, QtCore


class RecordingPanel(QtWidgets.QGroupBox):
    """Panel for recording configuration and control."""
    
    # Signals
    recording_started = QtCore.pyqtSignal()
    recording_stopped = QtCore.pyqtSignal()
    recording_cancelled = QtCore.pyqtSignal()
    relative_frame_toggled = QtCore.pyqtSignal(bool)
    status_updated = QtCore.pyqtSignal(str, str, float)  # (text, style, duration_seconds)
    browse_folder_requested = QtCore.pyqtSignal()
    
    relative_to_world = True  # Default to relative to World frame
    
    # Class variable to store custom templates
    custom_templates = {}
    
    # Path to save templates
    TEMPLATES_FILE = os.path.join(os.path.dirname(__file__), "recording_templates.json")
    
    def __init__(self):
        super().__init__("Recording Controls")
        self.setCheckable(True)
        self.setChecked(True)
        # Load saved templates from disk
        self._load_templates()
        # Timestamp until which status updates are ignored (locked)
        self._status_lock_until = 0.0
        # Store the last status that set the lock (optional)
        self._status_locked_by = None
        self._init_ui()
    
    def _load_templates(self):
        """Load custom templates from disk."""
        if os.path.exists(self.TEMPLATES_FILE):
            try:
                with open(self.TEMPLATES_FILE, 'r') as f:
                    self.custom_templates = json.load(f)
            except Exception as e:
                print(f"Error loading templates: {e}")
                self.custom_templates = {}
        else:
            self.custom_templates = {}
    
    def _save_templates_to_disk(self):
        """Save custom templates to disk."""
        try:
            with open(self.TEMPLATES_FILE, 'w') as f:
                json.dump(self.custom_templates, f, indent=2)
        except Exception as e:
            print(f"Error saving templates: {e}")
    
    def _init_ui(self):
        """Initialize the UI components."""
        layout = QtWidgets.QGridLayout(self)
        
        # Row 0: Template selector
        self.template_combo = QtWidgets.QComboBox()
        self.template_combo.addItems(["Default", "128 window"])
        self._update_template_list()
        self.template_combo.setCurrentIndex(0)
        self.template_combo.currentIndexChanged.connect(self._apply_template)
        self.template_combo.setToolTip("Select a recording template to quickly apply predefined settings. You can also save your current configuration as a custom template.")
        layout.addWidget(self.template_combo, 0, 0)
        
        # Row 0: Save template button
        self.save_template_btn = QtWidgets.QPushButton("Save Template")
        self.save_template_btn.clicked.connect(self._save_template)
        self.save_template_btn.setToolTip("Save the current recording configuration as a custom template for future use.")
        layout.addWidget(self.save_template_btn, 0, 1)
        
        # Row 1: File name
        self.filename_input = QtWidgets.QLineEdit("rec_" + datetime.now().strftime("%H%M%S"))
        self.filename_input.setPlaceholderText("Enter base filename (without .csv)")
        self.filename_input.setToolTip("Enter the base filename for the recording (without .csv extension).")
        layout.addWidget(self.filename_input, 1, 0)
        
        self.increment_btn = QtWidgets.QPushButton("Increment")
        self.increment_btn.clicked.connect(self._increment_filename)
        self.increment_btn.setToolTip("Increment the filename suffix.")
        layout.addWidget(self.increment_btn, 1, 1)
        
        # Row 1: Auto-increment checkbox
        self.auto_increment_checkbox = QtWidgets.QCheckBox("Auto Increment")
        self.auto_increment_checkbox.setChecked(False)
        self.auto_increment_checkbox.setToolTip("Automatically increment the filename after each recording is saved.")
        layout.addWidget(self.auto_increment_checkbox, 1, 2)
        
        # Row 2: Folder path
        self.folder_input = QtWidgets.QLineEdit("")
        self.folder_input.setPlaceholderText("Use / for subfolders, . for current folder")
        self.folder_input.setToolTip("Specify the folder to save recordings.")
        layout.addWidget(self.folder_input, 2, 0)
        
        self.folder_browse_btn = QtWidgets.QPushButton("Browse")
        self.folder_browse_btn.clicked.connect(self.browse_folder_requested.emit)
        self.folder_browse_btn.setToolTip("Browse for a folder to save the recording.")
        layout.addWidget(self.folder_browse_btn, 2, 1)
        
        # Row 2: Recording mode selector
        self.record_mode_combo = QtWidgets.QComboBox()
        self.record_mode_combo.addItems(["Seconds to record", "Samples to record"])
        self.record_mode_combo.setCurrentIndex(0)
        self.record_mode_combo.currentIndexChanged.connect(self._on_record_mode_changed)
        self.record_mode_combo.setToolTip("Select the mode for recording duration.")
        layout.addWidget(self.record_mode_combo, 3, 0)
        
        # Row 2: Recording duration input
        self.duration_input = QtWidgets.QDoubleSpinBox()
        self.duration_input.setRange(0.1, 3600)
        self.duration_input.setValue(10.0)
        self.duration_input.setSingleStep(0.1)
        self.duration_input.setDecimals(1)
        self.duration_input.setToolTip("Set the duration of the recording in seconds.")
        layout.addWidget(self.duration_input, 3, 1)
        
        # Row 2: Recording samples input (hidden by default)
        self.samples_input = QtWidgets.QSpinBox()
        self.samples_input.setRange(1, 1000000)
        self.samples_input.setValue(1000)
        self.samples_input.setSingleStep(100)
        self.samples_input.setVisible(False)
        self.samples_input.setToolTip("Set the number of samples to record.")
        layout.addWidget(self.samples_input, 3, 1)
        
        # Row 3: Recording delay slider
        self.delay_label = QtWidgets.QLabel("Countdown: 0s")
        self.delay_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.delay_slider.setRange(0, 10)
        self.delay_slider.setValue(0)
        self.delay_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.delay_slider.setTickInterval(1)
        # connect signal
        self.delay_slider.valueChanged.connect(lambda val: self.delay_label.setText(f"Countdown: {val}s"))
        layout.addWidget(self.delay_slider, 4, 1)
        layout.addWidget(self.delay_label, 4, 0)
        
        # Row 3: Timestamp format toggle
        self.timestamp_format_combo = QtWidgets.QComboBox()
        self.timestamp_format_combo.addItems(["Timestamp in s", "Timestamp in ms"])
        self.timestamp_format_combo.setCurrentIndex(0)
        self.timestamp_format_combo.setToolTip("Select the timestamp format for the recorded data: seconds or milliseconds.")
        layout.addWidget(self.timestamp_format_combo, 4, 2)
        
        # Row 3: Relative to world frame toggle
        self.relative_frame_checkbox = QtWidgets.QComboBox()
        self.relative_frame_checkbox.addItems(["Relative to Fizzy", "Relative to World"])
        self.relative_frame_checkbox.setCurrentIndex(1)  # Default to relative to world
        self.relative_frame_checkbox.currentIndexChanged.connect(self.on_relative_frame_toggle)
        self.relative_frame_checkbox.setToolTip("Select the reference frame for the recorded data. 'Relative to Fizzy' will record data relative to the device's orientation, while 'Relative to World' will record data relative to a (more or less) fixed world frame, by transforming all data based on the pitch, yaw, and roll.")
        layout.addWidget(self.relative_frame_checkbox, 0, 2)
        
        # Row 4-5: Data selection toggles (spanning 2 rows with 4 columns each)
        self.record_checks = {}
        data_types = ['Roll', 'Pitch', 'Yaw', 'Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Acc Magnitude', 'Gyro Magnitude', 'Motor Input']
        for idx, data_type in enumerate(data_types):
            checkbox = QtWidgets.QCheckBox(data_type)
            checkbox.setChecked(True)
            self.record_checks[data_type] = checkbox
            row = 5 + (idx // 4)
            col = 0 + (idx % 4)
            layout.addWidget(checkbox, row, col)
        
        # Row 7: Control buttons
        self.start_btn = QtWidgets.QPushButton("Start Recording")
        self.start_btn.clicked.connect(self.recording_started.emit)
        self.start_btn.setToolTip("Start recording with the current settings. Saves recording after the specified duration or when stop is pressed. Discards recording if cancel is pressed.")
        layout.addWidget(self.start_btn, 8, 0)
        
        self.stop_btn = QtWidgets.QPushButton("Stop Recording")
        self.stop_btn.clicked.connect(self.recording_stopped.emit)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Stop the recording. Saves the recording with the current settings.")
        layout.addWidget(self.stop_btn, 8, 1)
        
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.recording_cancelled.emit)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setToolTip("Cancel the recording. Discards any recorded data.")
        layout.addWidget(self.cancel_btn, 8, 2)
        
        # Row 8: Recording status
        self.recording_status = QtWidgets.QLabel("Status: Ready")
        self.recording_status.setStyleSheet("color: #00FF00;")
        layout.addWidget(self.recording_status, 9, 0, 1, 4)
        
        # Connect signal for status updates
        self.status_updated.connect(self._update_status_gui)
        self.toggled.connect(self._set_contents_visible)
        self._set_contents_visible(self.isChecked())

    def _set_contents_visible(self, visible: bool):
        """Show or hide the panel contents while keeping the title checkbox visible."""
        for child in self.findChildren(
            QtWidgets.QWidget,
            options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly,
        ):
            child.setVisible(visible)
        # duration_input and samples_input share the same grid cell; only one
        # should ever show. The blanket setVisible(True) above un-hides both,
        # so re-apply the record-mode visibility to keep them mutually exclusive.
        if visible:
            self._on_record_mode_changed()
    
    def _update_status_gui(self, text, style, duration: float = 0.0):
        """Update the recording status display.

        If `duration` > 0, subsequent status updates are ignored
        for `duration` seconds so the message stays visible long enough.
        """
        now = time.time()

        # If we're within a locked period, ignore any incoming updates
        if now < self._status_lock_until:
            return

        # Apply the update
        self.recording_status.setText(text)
        self.recording_status.setStyleSheet(style)

        # If a duration was provided, lock further updates for that period
        if duration and duration > 0:
            self._status_lock_until = now + float(duration)
            # Optionally record which status set the lock
            self._status_locked_by = text
            # Schedule clearing the lock after the duration
            QtCore.QTimer.singleShot(int(duration * 1000), self._clear_status_lock)

    def bind_manager(self, manager):
        """Connect this panel's signals to the recording manager."""
        self.manager = manager
        self.recording_started.connect(manager.start_recording)
        self.recording_stopped.connect(manager.stop_recording)
        self.recording_cancelled.connect(manager.cancel_recording)
        self.relative_frame_toggled.connect(manager.on_relative_frame_toggle)
        self.browse_folder_requested.connect(manager.browse_folder)

    def _clear_status_lock(self):
        """Clear the status update lock so new updates are accepted."""
        self._status_lock_until = 0.0
        self._status_locked_by = None
    
    def _on_record_mode_changed(self):
        """Handle switching between duration and samples recording modes."""
        is_samples_mode = (self.record_mode_combo.currentIndex() == 1)
        
        # Show/hide duration controls
        self.duration_input.setVisible(not is_samples_mode)
        
        # Show/hide samples controls
        self.samples_input.setVisible(is_samples_mode)

    def keyPressEvent(self, a0):
        """Handle recording-mode keyboard shortcuts."""
        manager = getattr(self, "manager", None)
        if manager is None or a0 is None:
            super().keyPressEvent(a0)
            return

        event = a0

        if event.key() == QtCore.Qt.Key.Key_Space and not event.isAutoRepeat():
            if manager.is_recording:
                self.recording_stopped.emit()
            elif self.start_btn.isEnabled():
                self.recording_started.emit()
            event.accept()
            return

        if manager.is_recording and not event.isAutoRepeat():
            key_text = event.text().lower()
            key_code = event.key()
            marker_name = None
            if key_code == QtCore.Qt.Key.Key_1.value or key_text == "1":
                marker_name = "marker_1"
            elif key_code == QtCore.Qt.Key.Key_2.value or key_text == "2":
                marker_name = "marker_2"
            elif key_code == QtCore.Qt.Key.Key_3.value or key_text == "3":
                marker_name = "marker_3"
            elif key_code == QtCore.Qt.Key.Key_4.value or key_text == "4":
                marker_name = "marker_4"

            if marker_name is not None:
                manager.on_recording_marker(marker_name)
                event.accept()
                return

        super().keyPressEvent(event)
    
    def _apply_template(self):
        """Apply the selected template to the recording settings."""
        template = self.template_combo.currentText()
        
        if template == "Default":
            # Default template
            self.filename_input.setText("rec_" + datetime.now().strftime("%H%M%S"))
            self.folder_input.setText("")
            self.auto_increment_checkbox.setChecked(False)
            self.record_mode_combo.setCurrentIndex(0)  # Duration (s)
            self.duration_input.setValue(10.0)
            self.delay_slider.setValue(0)
            self.relative_frame_checkbox.setCurrentIndex(0)  # Relative to Fizzy
            self.timestamp_format_combo.setCurrentIndex(0)  # Seconds
            for checkbox in self.record_checks.values():
                checkbox.setChecked(True)
        
        elif template in self.custom_templates:
            # Apply custom template
            self._apply_config(self.custom_templates[template])
    
    def _update_template_list(self):
        """Update the template dropdown with custom templates."""
        current_text = self.template_combo.currentText()
        self.template_combo.clear()
        self.template_combo.addItems(["Default"])
        self.template_combo.addItems(sorted(self.custom_templates.keys()))
        
        # Restore previous selection if it still exists
        idx = self.template_combo.findText(current_text)
        if idx >= 0:
            self.template_combo.setCurrentIndex(idx)
    
    def _capture_current_config(self):
        """Capture all current settings into a configuration dictionary."""
        config = {
            'filename': self.filename_input.text(),
            'folder': self.folder_input.text(),
            'auto_increment': self.auto_increment_checkbox.isChecked(),
            'record_mode': self.record_mode_combo.currentIndex(),
            'duration': self.duration_input.value(),
            'samples': self.samples_input.value(),
            'delay': self.delay_slider.value(),
            'timestamp_format': self.timestamp_format_combo.currentIndex(),
            'relative_frame': self.relative_frame_checkbox.currentIndex(),
            'data_types': {dtype: checkbox.isChecked() for dtype, checkbox in self.record_checks.items()}
        }
        return config
    
    def _apply_config(self, config):
        """Apply a configuration dictionary to the UI."""
        # Apply file and folder settings
        self.filename_input.setText(config.get('filename', 'rec_' + datetime.now().strftime("%H%M%S")))
        self.folder_input.setText(config.get('folder', ''))
        self.auto_increment_checkbox.setChecked(config.get('auto_increment', False))
        
        # Apply recording mode and duration settings
        self.record_mode_combo.setCurrentIndex(config.get('record_mode', 0))
        self.duration_input.setValue(config.get('duration', 10.0))
        self.samples_input.setValue(config.get('samples', 1000))
        self.delay_slider.setValue(config.get('delay', 0))
        self.timestamp_format_combo.setCurrentIndex(config.get('timestamp_format', 0))
        self.relative_frame_checkbox.setCurrentIndex(config.get('relative_frame', 0))
        
        # Apply data type selections
        for dtype, is_checked in config.get('data_types', {}).items():
            if dtype in self.record_checks:
                self.record_checks[dtype].setChecked(is_checked)
    
    def _save_template(self):
        """Save the current configuration as a custom template."""
        # Create a dialog to get the template name
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Save Template")
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # Template name input
        layout.addWidget(QtWidgets.QLabel("Template Name:"))
        name_input = QtWidgets.QLineEdit()
        layout.addWidget(name_input)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Save")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        # Connect buttons
        save_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        
        # Show dialog
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            template_name = name_input.text().strip()
            if template_name and template_name not in ["Default", "128 window"]:
                # Save the current configuration
                self.custom_templates[template_name] = self._capture_current_config()
                self._save_templates_to_disk()  # Persist to disk
                self._update_template_list()
                self.status_updated.emit(f"Template '{template_name}' saved!", "color: #00FF00;", 0.0)
            elif template_name in ["Default", "128 window"]:
                self.status_updated.emit("Cannot overwrite built-in templates!", "color: #FF0000;", 0.0)
            else:
                self.status_updated.emit("Template name cannot be empty!", "color: #FF0000;", 0.0)
    
    def _increment_filename(self):
        """Increment the filename: 'test' -> 'test_1' -> 'test_2', etc."""
        current_name = self.filename_input.text()
        
        # Remove .csv extension if present
        if current_name.endswith('.csv'):
            current_name = current_name[:-4]
        
        # Check if filename already has a number suffix
        match = re.match(r'^(.+?)_?(\d+)$', current_name)
        
        if match:
            # Extract base name and increment the number
            base_name = match.group(1)
            current_num = int(match.group(2))
            new_name = f"{base_name}_{current_num + 1}"
        else:
            # Add _1 suffix
            new_name = f"{current_name}_1"
        
        self.filename_input.setText(new_name)
    
    def get_filename(self):
        """Get the configured filename."""
        return self.filename_input.text()
    
    def set_filename(self, name):
        """Set the filename."""
        self.filename_input.setText(name)
    
    def get_folder(self):
        """Get the configured folder path."""
        return self.folder_input.text().strip()
    
    def set_folder(self, folder):
        """Set the folder path."""
        self.folder_input.setText(folder)
    
    def get_duration(self):
        """Get the recording duration in seconds."""
        return self.duration_input.value()
    
    def set_duration(self, seconds):
        """Set the recording duration."""
        self.duration_input.setValue(seconds)
    
    def get_record_mode(self):
        """Get the recording mode ('Duration' or 'Samples')."""
        return "Samples" if self.record_mode_combo.currentIndex() == 1 else "Duration"
    
    def get_samples(self):
        """Get the recording sample count."""
        return self.samples_input.value()
    
    def get_delay(self):
        """Get the recording delay in seconds."""
        return self.delay_slider.value()
    
    def set_delay(self, seconds):
        """Set the recording delay."""
        self.delay_slider.setValue(seconds)
        self.delay_label.setText(f"Countdown: {seconds}s")
    
    def get_timestamp_format(self):
        """Get the selected timestamp format ('Seconds' or 'Milliseconds')."""
        return self.timestamp_format_combo.currentText()
    
    def on_relative_frame_toggle(self):
        """Get the selected frame of reference ('Relative to Fizzy' or 'Relative to World')."""
        self.relative_frame_toggled.emit(self.relative_frame_checkbox.currentIndex())
        self.relative_to_world = (self.relative_frame_checkbox.currentIndex() == 1)  # True if "Relative to World" is selected
    
    def get_auto_increment(self):
        """Check if auto-increment is enabled."""
        return self.auto_increment_checkbox.isChecked()
    
    def auto_increment_filename(self):
        """Publicly callable method to auto-increment the filename."""
        self._increment_filename()
    
    def set_recording_active(self, active):
        """Set the active state (true when recording)."""
        self.start_btn.setEnabled(not active)
        self.stop_btn.setEnabled(active)
        self.cancel_btn.setEnabled(active)
        self.duration_input.setEnabled(not active)
        self.delay_slider.setEnabled(not active)
        self.filename_input.setEnabled(not active)
        self.increment_btn.setEnabled(not active)
    
    def get_selected_data_types(self):
        """Get list of selected data types to record."""
        return [dtype for dtype, checkbox in self.record_checks.items() if checkbox.isChecked()]
    
    def is_data_type_selected(self, data_type):
        """Check if a specific data type is selected for recording."""
        return self.record_checks[data_type].isChecked()
