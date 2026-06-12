"""
Playback Control Panel - CSV playback controls.
"""

from PyQt6 import QtWidgets, QtCore


class PlaybackPanel(QtWidgets.QGroupBox):
    """Panel for CSV playback configuration and control."""
    
    # Signals
    play_requested = QtCore.pyqtSignal()
    pause_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    browse_requested = QtCore.pyqtSignal()
    slider_moved = QtCore.pyqtSignal(int)  # Emits: slider value
    
    def __init__(self):
        super().__init__("Playback Controls")
        self.setCheckable(True)
        self.setChecked(True)
        self.setVisible(False)  # Hidden by default
        self._init_ui()
        self.toggled.connect(self._set_contents_visible)
        self._set_contents_visible(self.isChecked())

    def _set_contents_visible(self, visible: bool):
        """Show or hide the panel contents while keeping the title checkbox visible."""
        for child in self.findChildren(
            QtWidgets.QWidget,
            options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly,
        ):
            child.setVisible(visible)
    
    def _init_ui(self):
        """Initialize the UI components."""
        layout = QtWidgets.QGridLayout(self)
        
        # Row 0: File selector
        layout.addWidget(QtWidgets.QLabel("CSV File:"), 0, 0)
        self.playback_file_input = QtWidgets.QLineEdit()
        self.playback_file_input.setPlaceholderText("Select CSV file to play")
        layout.addWidget(self.playback_file_input, 0, 1, 1, 2)
        
        self.browse_btn = QtWidgets.QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_requested.emit)
        layout.addWidget(self.browse_btn, 0, 3)
        
        # Row 1: Playback speed
        layout.addWidget(QtWidgets.QLabel("Playback Speed:"), 1, 0)
        self.playback_speed_spin = QtWidgets.QDoubleSpinBox()
        self.playback_speed_spin.setRange(0.1, 10.0)
        self.playback_speed_spin.setValue(1.0)
        self.playback_speed_spin.setSingleStep(0.1)
        self.playback_speed_spin.setDecimals(1)
        self.playback_speed_spin.setSuffix("x")
        layout.addWidget(self.playback_speed_spin, 1, 1)
        
        # Row 2: Playback controls
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self.play_requested.emit)
        layout.addWidget(self.play_btn, 2, 0)
        
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_requested.emit)
        self.pause_btn.setEnabled(False)
        layout.addWidget(self.pause_btn, 2, 1)
        
        self.stop_playback_btn = QtWidgets.QPushButton("Stop")
        self.stop_playback_btn.clicked.connect(self.stop_requested.emit)
        self.stop_playback_btn.setEnabled(False)
        layout.addWidget(self.stop_playback_btn, 2, 2)
        
        # Row 3: Playback progress/status
        self.playback_status = QtWidgets.QLabel("Status: Ready")
        self.playback_status.setStyleSheet("color: #00FF00;")
        layout.addWidget(self.playback_status, 3, 0, 1, 4)
        
        # Row 4: Playback progress slider
        layout.addWidget(QtWidgets.QLabel("Progress:"), 4, 0)
        self.playback_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.playback_slider.setRange(0, 100)
        self.playback_slider.setValue(0)
        self.playback_slider.sliderMoved.connect(self.slider_moved.emit)
        self.playback_slider.setTracking(True)  # Enable continuous updates while dragging
        self.playback_slider_moving = False
        layout.addWidget(self.playback_slider, 4, 1, 1, 3)
    
    def get_file_path(self):
        """Get the selected CSV file path."""
        return self.playback_file_input.text().strip()
    
    def set_file_path(self, path):
        """Set the CSV file path."""
        self.playback_file_input.setText(path)
    
    def get_playback_speed(self):
        """Get the playback speed multiplier."""
        return self.playback_speed_spin.value()
    
    def set_playback_speed(self, speed):
        """Set the playback speed multiplier."""
        self.playback_speed_spin.setValue(speed)
    
    def set_status(self, text, style="color: #00FF00;"):
        """Set the status message."""
        self.playback_status.setText(text)
        self.playback_status.setStyleSheet(style)
    
    def set_playing(self, is_playing):
        """Set the UI state for playing."""
        self.play_btn.setEnabled(not is_playing)
        self.pause_btn.setEnabled(is_playing)
        self.stop_playback_btn.setEnabled(is_playing)
        self.playback_file_input.setEnabled(not is_playing)
        self.browse_btn.setEnabled(not is_playing)
    
    def set_paused(self, is_paused):
        """Set the UI state for paused."""
        self.play_btn.setEnabled(is_paused)
        self.pause_btn.setEnabled(is_paused)
        self.stop_playback_btn.setEnabled(True)
    
    def set_stopped(self):
        """Set the UI state for stopped."""
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_playback_btn.setEnabled(False)
        self.playback_file_input.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.playback_slider.setValue(0)
    
    def set_slider_range(self, max_value):
        """Set the slider range (typically length of data)."""
        self.playback_slider.setRange(0, max_value)
    
    def get_slider_value(self):
        """Get the current slider value."""
        return self.playback_slider.value()
    
    def set_slider_value(self, value):
        """Set the slider value without triggering signals."""
        self.playback_slider.blockSignals(True)
        self.playback_slider.setValue(value)
        self.playback_slider.blockSignals(False)
