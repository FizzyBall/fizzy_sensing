"""
Main Control Panel - Mode selector and Graph display mode switcher.
"""

from PyQt6 import QtWidgets, QtCore, QtGui


class ControlPanel(QtWidgets.QGroupBox):
    """Control panel for mode selection and graph display settings."""
    
    # Signals
    mode_changed = QtCore.pyqtSignal(int)  # Emits: mode index (0=Record, 1=Analyse, 2=Featurization, 3=Train)
    graph_visibility_changed = QtCore.pyqtSignal(str, bool)  # Emits: graph key and visibility state
    calibrate_requested = QtCore.pyqtSignal()  # Emits: when calibrate button pressed
    reconnect_requested = QtCore.pyqtSignal()  # Emits: when reconnect button pressed
    fft_toggled = QtCore.pyqtSignal(bool)  # Emits: FFT enabled state
    current_window_toggled = QtCore.pyqtSignal(bool)  # Emits: current-window visibility state
    window_size_changed = QtCore.pyqtSignal(float)  # Emits: window size
    window_unit_changed = QtCore.pyqtSignal(str)    # Emits: window unit ("Samples" or "Seconds")
    overlap_changed = QtCore.pyqtSignal(int)        # Emits: overlap percent
    scale_increased = QtCore.pyqtSignal()  # Emits: when scale up button pressed
    scale_decreased = QtCore.pyqtSignal()  # Emits: when scale down button pressed
    window_closed = QtCore.pyqtSignal()
    
    
    def __init__(self):
        super().__init__("Control Panel")
        self.setCheckable(True)
        self.setChecked(True)
        self._init_ui()
        self._update_mode_dependent_control_visibility(self.mode_combo.currentIndex())
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
        
        # Row 0: Mode selector (record: red, analyse: blue)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["  Record", "  Analyse", "  Featurize", "  Train", "  Live Classification", "  Classification Analysis"])
        self.mode_combo.setItemData(0, QtGui.QColor("#FF0000"), QtCore.Qt.ItemDataRole.BackgroundRole)
        self.mode_combo.setItemData(0, QtGui.QColor("#FFFFFF"), QtCore.Qt.ItemDataRole.ForegroundRole)
        self.mode_combo.setItemData(1, QtGui.QColor("#0000FF"), QtCore.Qt.ItemDataRole.BackgroundRole)
        self.mode_combo.setItemData(1, QtGui.QColor("#FFFFFF"), QtCore.Qt.ItemDataRole.ForegroundRole)
        self.mode_combo.setItemData(2, QtGui.QColor("#008000"), QtCore.Qt.ItemDataRole.BackgroundRole)
        self.mode_combo.setItemData(2, QtGui.QColor("#FFFFFF"), QtCore.Qt.ItemDataRole.ForegroundRole)
        self.mode_combo.setItemData(3, QtGui.QColor("#FF8800"), QtCore.Qt.ItemDataRole.BackgroundRole)
        self.mode_combo.setItemData(3, QtGui.QColor("#FFFFFF"), QtCore.Qt.ItemDataRole.ForegroundRole)
        self.mode_combo.setItemData(4, QtGui.QColor("#AA00AA"), QtCore.Qt.ItemDataRole.BackgroundRole)
        self.mode_combo.setItemData(4, QtGui.QColor("#FFFFFF"), QtCore.Qt.ItemDataRole.ForegroundRole)
        self.mode_combo.setItemData(5, QtGui.QColor("#BB6666"), QtCore.Qt.ItemDataRole.BackgroundRole)
        self.mode_combo.setItemData(5, QtGui.QColor("#FFFFFF"), QtCore.Qt.ItemDataRole.ForegroundRole)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._update_mode_combo_style(self.mode_combo.currentIndex())
        self.mode_combo.setToolTip("Switch between recording, analyse, featurization, training, live classification, and classification analysis modes.")
        layout.addWidget(self.mode_combo, 0, 0, 1, 1)
        self.help_btn = QtWidgets.QPushButton("App Guide")
        self.help_btn.setStyleSheet("background-color: #333333; color: white; font-weight: bold;")
        self.help_btn.setToolTip("Open a short guide to the app workflow.")
        self.help_btn.clicked.connect(self._show_app_guide)
        layout.addWidget(self.help_btn, 0, 1)
        # Row 0: Reconnect button
        self.reconnect_button = QtWidgets.QPushButton("Reconnect Fizzy")
        self.reconnect_button.setStyleSheet("background-color: #0000FF; color: white; font-weight: bold;")
        self.reconnect_button.clicked.connect(self.reconnect_requested.emit)
        self.reconnect_button.setToolTip("Attempt to reconnect to the Fizzy device.")
        layout.addWidget(self.reconnect_button, 0, 2)
        # Quit app button
        self.quit_btn = QtWidgets.QPushButton("Quit App")
        self.quit_btn.setStyleSheet("background-color: #FF0000; color: white; font-weight: bold;")
        self.quit_btn.clicked.connect(self.window_closed.emit)
        self.quit_btn.setToolTip("Quit the application. Unsaved changes will be lost.")
        layout.addWidget(self.quit_btn, 0, 3)
        # Row 1: window controls
        self.window_size_spinbox = QtWidgets.QDoubleSpinBox()
        self.window_size_spinbox.setMinimum(1)
        self.window_size_spinbox.setMaximum(5000)
        self.window_size_spinbox.setValue(128)
        self.window_size_spinbox.setSingleStep(32)
        self.window_size_spinbox.setDecimals(0)
        self.window_size_spinbox.valueChanged.connect(self.window_size_changed.emit)
        self.window_size_spinbox.setToolTip("Set the window length for FFT calculation and windowing for classification training. Unit can be switched between samples and seconds.")
        layout.addWidget(self.window_size_spinbox, 1, 0)
        
        self.window_unit_combo = QtWidgets.QComboBox()
        self.window_unit_combo.addItems(["Samples", "Seconds"])
        self.window_unit_combo.currentIndexChanged.connect(self._on_window_unit_changed)
        self.window_unit_combo.setToolTip("Set the window length for FFT calculation and windowing for classification training. Unit can be switched between samples and seconds.")
        layout.addWidget(self.window_unit_combo, 1, 1)

        self.overlap_spinbox = QtWidgets.QSpinBox()
        self.overlap_spinbox.setRange(0, 99)
        self.overlap_spinbox.setValue(50)
        self.overlap_spinbox.setSuffix(" % overlap")
        self.overlap_spinbox.valueChanged.connect(self.overlap_changed.emit)
        self.overlap_spinbox.setToolTip("Set how much consecutive windows overlap, in percent.")
        layout.addWidget(self.overlap_spinbox, 1, 2)
        # Row 1: Calibrate button
        self.cal_button = QtWidgets.QPushButton("Offset Yaw")
        self.cal_button.clicked.connect(self.calibrate_requested.emit)
        self.cal_button.setToolTip("Offset the yaw angle to zero based on the current orientation. Use this to align Fizzy with the 3D gizmo's orientation. Does not have any effect on the recorded data, only on the live display.")
        layout.addWidget(self.cal_button, 1, 3)
        
        # Row 2: Graph visibility toggles
        self.angle_xyz_checkbox = QtWidgets.QCheckBox("Angle XYZ")
        self.angle_xyz_checkbox.setChecked(True)
        self.angle_xyz_checkbox.setToolTip("Show or hide the roll, pitch, and yaw graph.")
        self.angle_xyz_checkbox.toggled.connect(lambda checked: self.graph_visibility_changed.emit("angle_xyz", checked))
        layout.addWidget(self.angle_xyz_checkbox, 2, 0)

        self.acc_xyz_checkbox = QtWidgets.QCheckBox("Acc XYZ")
        self.acc_xyz_checkbox.setChecked(True)
        self.acc_xyz_checkbox.setToolTip("Show or hide the linear acceleration graph.")
        self.acc_xyz_checkbox.toggled.connect(lambda checked: self.graph_visibility_changed.emit("acc_xyz", checked))
        layout.addWidget(self.acc_xyz_checkbox, 2, 1)

        self.acc_mag_checkbox = QtWidgets.QCheckBox("Acc Mag")
        self.acc_mag_checkbox.setChecked(False)
        self.acc_mag_checkbox.setToolTip("Show or hide the acceleration magnitude graph.")
        self.acc_mag_checkbox.toggled.connect(lambda checked: self.graph_visibility_changed.emit("acc_mag", checked))
        layout.addWidget(self.acc_mag_checkbox, 2, 2)

        self.gyro_xyz_checkbox = QtWidgets.QCheckBox("Gyro XYZ")
        self.gyro_xyz_checkbox.setChecked(True)
        self.gyro_xyz_checkbox.setToolTip("Show or hide the angular velocity graph.")
        self.gyro_xyz_checkbox.toggled.connect(lambda checked: self.graph_visibility_changed.emit("gyro_xyz", checked))
        layout.addWidget(self.gyro_xyz_checkbox, 2, 3)

        self.gyro_mag_checkbox = QtWidgets.QCheckBox("Gyro Mag")
        self.gyro_mag_checkbox.setChecked(False)
        self.gyro_mag_checkbox.setToolTip("Show or hide the angular velocity magnitude graph.")
        self.gyro_mag_checkbox.toggled.connect(lambda checked: self.graph_visibility_changed.emit("gyro_mag", checked))
        layout.addWidget(self.gyro_mag_checkbox, 3, 0)

        self.motor_input_checkbox = QtWidgets.QCheckBox("Motor Input")
        self.motor_input_checkbox.setChecked(True)
        self.motor_input_checkbox.setToolTip("Show or hide the motor input graph.")
        self.motor_input_checkbox.toggled.connect(lambda checked: self.graph_visibility_changed.emit("motor_input", checked))
        layout.addWidget(self.motor_input_checkbox, 3, 1)

        self.fft_checkbox = QtWidgets.QCheckBox("Show FFT")
        self.fft_checkbox.setChecked(False)
        self.fft_checkbox.stateChanged.connect(self._on_fft_toggle)
        self.fft_checkbox.setToolTip("Toggle to show or hide the frequency domain graphs.")
        layout.addWidget(self.fft_checkbox, 3, 2)

        self.current_window_checkbox = QtWidgets.QCheckBox("Show Current Window")
        self.current_window_checkbox.setChecked(True)
        self.current_window_checkbox.setToolTip("Show or hide the current-window plots in Analyse and Classification Analysis modes.")
        self.current_window_checkbox.toggled.connect(self.current_window_toggled.emit)
        layout.addWidget(self.current_window_checkbox, 3, 3)

        self.full_data_checkbox = QtWidgets.QCheckBox("Show Full Data")
        self.full_data_checkbox.setChecked(True)
        self.full_data_checkbox.setToolTip("Show or hide the full-data graphs. This does not affect the current-window plots or FFT plots.")
        self.full_data_checkbox.toggled.connect(lambda checked: self.graph_visibility_changed.emit("full_data", checked))
        layout.addWidget(self.full_data_checkbox, 4, 0)

        
        

    
    def _on_fft_toggle(self):
        """Handle FFT checkbox toggle."""
        self.fft_toggled.emit(self.fft_checkbox.isChecked())

    def _on_mode_changed(self, index):
        """Update combo style and forward mode change signal."""
        self._update_mode_combo_style(index)
        self._update_mode_dependent_control_visibility(index)
        self.mode_changed.emit(index)

    def _update_mode_dependent_control_visibility(self, mode_index):
        """Show only controls that are relevant for the active mode."""
        show_offset_yaw = mode_index in (0, 4)      # Record, Live Classification
        show_window_settings = mode_index in (0, 1, 4, 5)  # Analyse, Live Classification, Classification Analysis

        self.cal_button.setVisible(show_offset_yaw)
        self.window_size_spinbox.setVisible(show_window_settings)
        self.window_unit_combo.setVisible(show_window_settings)
        self.overlap_spinbox.setVisible(show_window_settings)

    def _update_mode_combo_style(self, index):
        """Apply a different selected color per mode."""
        if index == 5:
            bg_color = "#BB6666"
        elif index == 4:
            bg_color = "#AA00AA"
        elif index == 3:
            bg_color = "#FF8800"
        elif index == 2:
            bg_color = "#008000"
        elif index == 1:
            bg_color = "#0000FF"
        else:
            bg_color = "#FF0000"

        self.mode_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {bg_color};
                color: white;
                font-weight: bold;
            }}
            QComboBox QAbstractItemView {{
                selection-background-color: #333333;
                selection-color: white;
            }}
        """)
    
    def get_current_mode(self):
        """Get the currently selected mode."""
        return self.mode_combo.currentIndex()
    
    def set_mode(self, index):
        """Set the mode without triggering signals."""
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(index)
        self.mode_combo.blockSignals(False)
        self._update_mode_combo_style(index)
        self._update_mode_dependent_control_visibility(index)
    
    def _on_window_unit_changed(self):
        """Handle window unit change to adjust spinbox properties."""
        unit = self.window_unit_combo.currentText()
        self._apply_window_unit_properties(unit)
        if unit == "Seconds":
            self.window_size_spinbox.setValue(1.0)
        else:
            self.window_size_spinbox.setValue(128)
        self.window_unit_changed.emit(unit)

    def _apply_window_unit_properties(self, unit):
        """Apply the numeric constraints for the selected window unit."""
        if unit == "Seconds":
            self.window_size_spinbox.setMinimum(0.1)
            self.window_size_spinbox.setMaximum(100.0)
            self.window_size_spinbox.setSingleStep(0.1)
            self.window_size_spinbox.setDecimals(3)
        else:
            self.window_size_spinbox.setMinimum(1)
            self.window_size_spinbox.setMaximum(5000)
            self.window_size_spinbox.setSingleStep(32)
            self.window_size_spinbox.setDecimals(0)

    def set_window_unit(self, unit):
        """Set the window unit without emitting change signals."""
        self.window_unit_combo.blockSignals(True)
        index = self.window_unit_combo.findText(unit)
        if index >= 0:
            self.window_unit_combo.setCurrentIndex(index)
        self.window_unit_combo.blockSignals(False)
        self._apply_window_unit_properties(self.window_unit_combo.currentText())

    def set_window_size(self, size):
        """Set the window size without emitting change signals."""
        self.window_size_spinbox.blockSignals(True)
        self.window_size_spinbox.setValue(size)
        self.window_size_spinbox.blockSignals(False)

    def set_overlap_percent(self, overlap_percent):
        """Set the overlap percentage without emitting change signals."""
        self.overlap_spinbox.blockSignals(True)
        self.overlap_spinbox.setValue(int(round(float(overlap_percent))))
        self.overlap_spinbox.blockSignals(False)

    def set_window_settings(self, size, unit, overlap_percent):
        """Set window size, unit, and overlap without emitting change signals."""
        self.set_window_unit(unit)
        self.set_window_size(size)
        self.set_overlap_percent(overlap_percent)

    def is_fft_enabled(self):
        """Check if FFT display is enabled."""
        return self.fft_checkbox.isChecked()
    
    def get_window_size(self):
        """Get the current window size."""
        return self.window_size_spinbox.value()
        
    def get_window_unit(self):
        """Get the current window unit."""
        return self.window_unit_combo.currentText()

    def get_overlap_percent(self):
        """Get the current overlap percentage."""
        return self.overlap_spinbox.value()

    def get_graph_visibility_state(self):
        """Return the current graph visibility toggles."""
        return {
            "angle_xyz": self.angle_xyz_checkbox.isChecked(),
            "acc_xyz": self.acc_xyz_checkbox.isChecked(),
            "acc_mag": self.acc_mag_checkbox.isChecked(),
            "gyro_xyz": self.gyro_xyz_checkbox.isChecked(),
            "gyro_mag": self.gyro_mag_checkbox.isChecked(),
            "motor_input": self.motor_input_checkbox.isChecked(),
            "fft": self.fft_checkbox.isChecked(),
            "current_window": self.current_window_checkbox.isChecked(),
            "full_data": self.full_data_checkbox.isChecked(),
        }

    def set_graph_visibility_state(self, state):
        """Apply graph visibility toggles without emitting signals."""
        checkbox_map = {
            "angle_xyz": self.angle_xyz_checkbox,
            "acc_xyz": self.acc_xyz_checkbox,
            "acc_mag": self.acc_mag_checkbox,
            "gyro_xyz": self.gyro_xyz_checkbox,
            "gyro_mag": self.gyro_mag_checkbox,
            "motor_input": self.motor_input_checkbox,
            "fft": self.fft_checkbox,
            "current_window": self.current_window_checkbox,
            "full_data": self.full_data_checkbox,
        }
        for key, checkbox in checkbox_map.items():
            if key not in state:
                continue
            checkbox.blockSignals(True)
            checkbox.setChecked(bool(state[key]))
            checkbox.blockSignals(False)

    def _show_app_guide(self):
        """Show a popup that explains the full app workflow."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Fizzy App Guide")
        dialog.setModal(True)
        dialog.resize(760, 520)

        layout = QtWidgets.QVBoxLayout(dialog)

        title = QtWidgets.QLabel("How the app works")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        text = QtWidgets.QTextBrowser()
        text.setOpenExternalLinks(True)
        text.setHtml(
                """
                <h3>Workflow Overview</h3>
                <p>This app is a full pipeline for Fizzy: capture IMU data, review & label, featurize, train models, and evaluate live or offline.</p>
                
                <h3>In Short</h3>
                <ol>
                    <li><b>Record</b> - Record IMU data in the Record tab. You can place markers while recording to note important events or transitions.</li>
                    <li><b>Analyse</b> - Open the recording in the Analyse tab, inspect the data, scrub through windows, and assign labels to each window. Assign a label to the current window by pressing either the first letter of the class on your keyboard, or by pressing the index number of the class on your keyboard. Scrub through windows using the left and right arrow keys.</li>
                    <li><b>Featurize</b> - Convert the labeled windows into feature vectors in the Featurize tab so the data is ready for machine learning.</li>
                    <li><b>Train</b> - Train a model on the featurized data in the Training tab and compare training runs and settings.</li>
                    <li><b>Live Classification</b> - Test the trained model on incoming IMU data in real time and watch predictions update live.</li>
                    <li><b>Classification Analysis</b> - Load a recording and a trained model, then inspect the model's predictions against the true labels to compare models and measure accuracy.</li>
                </ol>
                <h3>Tips</h3>
                <ul>
                    <li>Hover over UI elements to see tooltips with additional information.</li>
                    <li>You can use an XBox controller to record data and place markers.</li>
                    <li>Make sure the correct window size, unit and overlap are set, and stay consistent.</li>
                    <li>You can move and zoom all the graphs.</li>
                    <li>Right-click on graphs for some graph options.</li>
                    <li>To use (Startified)GroupedKFold cross-validation, you need multiple groups of data for the same class, that have no overlap. You can achieve this by recording multiple separate takes of the same class, or by manually creating groups by removing overlapping windows. Every group should be in a seperate folder with a folder name in this format: "class_name#group_name"</li>
                    <li>When saving a labeled recording, a json file is created that can be imported in the Classification Analysis tab to compare the true labels with the model's predictions.</li>
                    <li>If you run out of screen space, try hiding some graphs and controls using the many toggle buttons.<li>
                </ul>
                """
        )
        layout.addWidget(text)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dialog.reject)
        close_button = button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Close)
        if close_button is not None:
            close_button.clicked.connect(dialog.accept)
        layout.addWidget(button_box)

        dialog.exec()