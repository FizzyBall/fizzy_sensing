"""
Analyse Control Panel - CSV analysis controls.
"""

from PyQt6 import QtWidgets, QtCore


class AnalysePanel(QtWidgets.QGroupBox):
    """Panel for CSV analysis configuration and control."""
    
    # Signals
    browse_requested = QtCore.pyqtSignal()
    load_requested = QtCore.pyqtSignal()
    slider_moved = QtCore.pyqtSignal(int)  # Emits: slider value
    save_location_browse_requested = QtCore.pyqtSignal()
    save_labeled_data_requested = QtCore.pyqtSignal()
    save_labels_file_requested = QtCore.pyqtSignal()
    label_metadata_path_changed = QtCore.pyqtSignal(str)
    clear_labels_requested = QtCore.pyqtSignal()
    marker_visibility_changed = QtCore.pyqtSignal(bool)  # Emits: show markers enabled
    gap_visibility_changed = QtCore.pyqtSignal(bool)  # Emits: show gap regions enabled
    window_stats_visibility_changed = QtCore.pyqtSignal(bool)  # Emits: show window statistics enabled
    label_visibility_changed = QtCore.pyqtSignal(bool)  # Emits: show existing label overlays enabled
    
    def __init__(self):
        super().__init__("Analyse Controls")
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
        self.analyse_file_input = QtWidgets.QLineEdit()
        self.analyse_file_input.setPlaceholderText("Select CSV file to analyse")
        layout.addWidget(self.analyse_file_input, 0, 0)
        
        self.analyse_browse_btn = QtWidgets.QPushButton("Browse")
        self.analyse_browse_btn.clicked.connect(self.browse_requested.emit)
        layout.addWidget(self.analyse_browse_btn, 0, 1)
        
        self.analyse_load_btn = QtWidgets.QPushButton("Reload")
        self.analyse_load_btn.clicked.connect(self.load_requested.emit)
        layout.addWidget(self.analyse_load_btn, 0, 2)
        
        # Row 2: Timestamp slider
        self.analyse_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.analyse_slider.setRange(0, 100)
        self.analyse_slider.setValue(0)
        # Emit our unified `slider_moved` signal for both drag and programmatic/value changes
        # so clicks on the slider track and keyboard adjustments update the analyse view.
        self.analyse_slider.sliderMoved.connect(self.slider_moved.emit)
        self.analyse_slider.valueChanged.connect(self.slider_moved.emit)
        self.analyse_slider.setEnabled(False)
        layout.addWidget(self.analyse_slider, 1, 0, 1, 2)
        
        self.analyse_timestamp_label = QtWidgets.QLabel("Time: 0.00 s")
        layout.addWidget(self.analyse_timestamp_label, 1, 2)

        # Row 3: Marker visibility toggle
        self.show_markers_checkbox = QtWidgets.QCheckBox("Show Markers")
        self.show_markers_checkbox.setToolTip("Show or hide marker lines in full and current-window plots.")
        self.show_markers_checkbox.setChecked(True)
        self.show_markers_checkbox.toggled.connect(self.marker_visibility_changed.emit)
        layout.addWidget(self.show_markers_checkbox, 3, 0)

        self.show_gaps_checkbox = QtWidgets.QCheckBox("Show Data Gaps")
        self.show_gaps_checkbox.setToolTip("Show or hide detected data gap regions in full and current-window plots.")
        self.show_gaps_checkbox.setChecked(True)
        self.show_gaps_checkbox.toggled.connect(self.gap_visibility_changed.emit)
        layout.addWidget(self.show_gaps_checkbox, 3, 1)

        self.show_window_stats_checkbox = QtWidgets.QCheckBox("Show Window Statistics")
        self.show_window_stats_checkbox.setToolTip("Show or hide the per-window time- and frequency-domain feature statistics in the info panel.")
        self.show_window_stats_checkbox.setChecked(True)
        self.show_window_stats_checkbox.toggled.connect(self.window_stats_visibility_changed.emit)
        layout.addWidget(self.show_window_stats_checkbox, 3, 2)

        self.show_labels_checkbox = QtWidgets.QCheckBox("Show Labels")
        self.show_labels_checkbox.setToolTip("Show or hide the colored overlays for windows you've already labeled. Does not affect the current window or the stored labels.")
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.toggled.connect(self.label_visibility_changed.emit)
        layout.addWidget(self.show_labels_checkbox, 3, 3)

        # Row 4: Status
        self.analyse_status = QtWidgets.QLabel("Status: Ready")
        self.analyse_status.setStyleSheet("color: #00FF00;")
        layout.addWidget(self.analyse_status, 4, 0, 1, 4)

        # Row 5: Separator
        separator1 = QtWidgets.QFrame()
        separator1.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator1.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(separator1, 5, 0, 1, 5)
        
        # Row 6: labeling explanation
        # labeling_l = QtWidgets.QLabel("Labeling: Use the following controls to label the data for training: press < or > to scrub through the data in increments defined by window settings in the control panel, and press the number keys (1-0) or the - and = keys to assign labels to the currently selected window. Saving the labeled data will split the data into subfolders per label, with each window as a separate CSV file. A JSON file will also be saved with the same name, containing the window settings and assigned labels for each window. You can load this file back in to restore the labels and settings, here in the Analyse tab, as well as in the Classification Analysis tab, to compare model predictions to your labels.")
        # labeling_l.setWordWrap(True)
        # take necessary space and push other elements down, but don't stretch if the panel is resized larger - we want the label to be compact and not take up too much vertical space
        # labeling_l.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        # layout.addWidget(labeling_l, 6, 0, 1, 5)
        
        # Row 7: Save location
        self.save_location_input = QtWidgets.QLineEdit()
        self.save_location_input.setPlaceholderText("Select directory to save labeled data")
        layout.addWidget(self.save_location_input, 7, 0, 1, 2)
        
        self.save_location_browse_btn = QtWidgets.QPushButton("Browse")
        self.save_location_browse_btn.clicked.connect(self.save_location_browse_requested.emit)
        layout.addWidget(self.save_location_browse_btn, 7, 2)

        # Row 8: Label metadata file
        self.label_metadata_input = QtWidgets.QLineEdit()
        self.label_metadata_input.setPlaceholderText("Select window-label metadata JSON")
        self.label_metadata_input.editingFinished.connect(
            lambda: self.label_metadata_path_changed.emit(self.label_metadata_input.text().strip())
        )
        layout.addWidget(self.label_metadata_input, 8, 0, 1, 2)

        self.label_metadata_browse_btn = QtWidgets.QPushButton("Browse")
        self.label_metadata_browse_btn.clicked.connect(self._browse_label_metadata_file)
        layout.addWidget(self.label_metadata_browse_btn, 8, 2)

        # Row 8: Save mode (Add vs Replace)
        self.save_mode_combo = QtWidgets.QComboBox()
        self.save_mode_combo.addItem("Append to folders")
        self.save_mode_combo.addItem("Replace folders")
        self.save_mode_combo.setCurrentIndex(0)  # default: Add
        self.save_mode_combo.setToolTip("Add = keep existing files and append new ones. Replace = remove existing files before saving.")
        layout.addWidget(self.save_mode_combo, 9, 1)
        
        # Row 10: Feature toggles label
        features_l = QtWidgets.QLabel("Include Data:")
        features_l.setToolTip("Select the data to save to the labeled dataset. Optionally exclude data that the classification model won't need.")
        layout.addWidget(features_l, 10, 0)
        
        # Row 10-11: Feature toggles (dynamic, populated when CSV is loaded)
        self.features_widget = QtWidgets.QWidget()
        self.features_layout = QtWidgets.QGridLayout(self.features_widget)
        self.features_layout.setContentsMargins(0, 0, 0, 0)
        
        self.feature_checkboxes = {}
        
        layout.addWidget(self.features_widget, 11, 0, 2, 3)
        
        # Row 12: Class toggles
        classes_l = QtWidgets.QLabel("Include Classes:")
        classes_l.setToolTip("Select which labeled windows should be saved to disk.")
        layout.addWidget(classes_l, 13, 0)

        class_controls = QtWidgets.QHBoxLayout()
        self.class_select_all_btn = QtWidgets.QPushButton("Select All")
        self.class_select_all_btn.clicked.connect(lambda: self.check_all_labels(True))
        class_controls.addWidget(self.class_select_all_btn)

        self.class_select_none_btn = QtWidgets.QPushButton("Select None")
        self.class_select_none_btn.clicked.connect(lambda: self.check_all_labels(False))
        class_controls.addWidget(self.class_select_none_btn)
        class_controls.addStretch(1)
        layout.addLayout(class_controls, 13, 1, 1, 4)

        self.class_scroll = QtWidgets.QScrollArea()
        self.class_scroll.setWidgetResizable(True)
        self.class_scroll.setMinimumHeight(30)
        self.class_container = QtWidgets.QWidget()
        self.class_layout = QtWidgets.QGridLayout(self.class_container)
        self.class_layout.setContentsMargins(4, 4, 4, 4)
        self.class_layout.setHorizontalSpacing(10)
        self.class_layout.setVerticalSpacing(4)
        self.class_scroll.setWidget(self.class_container)
        layout.addWidget(self.class_scroll, 14, 0, 1, 5)

        self.label_checkboxes = {}
        
        # Row 14: Save labeled data buttons + Clear Labels
        self.save_labeled_data_btn = QtWidgets.QPushButton("Save Labeled Data")
        self.save_labeled_data_btn.clicked.connect(self.save_labeled_data_requested.emit)
        layout.addWidget(self.save_labeled_data_btn, 15, 0)

        self.clear_labels_btn = QtWidgets.QPushButton("Clear Labels")
        self.clear_labels_btn.setToolTip("Clear all assigned labels from the loaded dataset")
        self.clear_labels_btn.clicked.connect(self.clear_labels_requested.emit)
        layout.addWidget(self.clear_labels_btn, 15, 2)
    
    def bind_manager(self, manager):
        """Connect this panel's signals to the analyse manager."""
        self.manager = manager
        self.browse_requested.connect(manager.analyse_browse_csv_file)
        self.load_requested.connect(manager.analyse_load_csv)
        self.slider_moved.connect(manager.on_slider_moved)
        self.save_location_browse_requested.connect(manager.on_save_location_browse)
        self.label_metadata_path_changed.connect(manager.on_label_metadata_path_changed)
        self.marker_visibility_changed.connect(manager.on_marker_visibility_changed)
        self.gap_visibility_changed.connect(manager.on_gap_visibility_changed)
        self.window_stats_visibility_changed.connect(manager.on_window_stats_visibility_changed)
        self.label_visibility_changed.connect(manager.on_label_visibility_changed)
        self.save_labeled_data_requested.connect(manager.on_save_labeled_data)
        self.clear_labels_requested.connect(manager.on_clear_labels)
    
    def get_file_path(self):
        """Get the selected CSV file path."""
        return self.analyse_file_input.text().strip()
    
    def set_file_path(self, path):
        """Set the CSV file path."""
        self.analyse_file_input.setText(path)
        self.clear_labels_requested.emit()  # Clear labels when a new file is loaded
        self.load_requested.emit()  # Trigger loading of the new file
    
    def set_status(self, text, style="color: #00FF00;"):
        """Set the status message."""
        self.analyse_status.setText(text)
        self.analyse_status.setStyleSheet(style)

    def get_save_location(self):
        """Get the save location path."""
        return self.save_location_input.text().strip()

    def get_save_mode(self):
        """Return the selected save mode as 'add' or 'replace'."""
        text = self.save_mode_combo.currentText().lower()
        if 'replace' in text:
            return 'replace'
        return 'add'

    def set_save_mode(self, mode: str):
        """Set the save mode; mode should be 'add' or 'replace'."""
        if mode == 'replace':
            self.save_mode_combo.setCurrentIndex(1)
        else:
            self.save_mode_combo.setCurrentIndex(0)
    
    def set_save_location(self, path):
        """Set the save location path."""
        self.save_location_input.setText(path)

    def get_label_metadata_path(self):
        """Get the selected label-metadata file path."""
        return self.label_metadata_input.text().strip()

    def set_label_metadata_path(self, path):
        """Set the label-metadata file path."""
        self.label_metadata_input.setText(path)
        self.label_metadata_path_changed.emit(path.strip())

    def _browse_label_metadata_file(self):
        from utilities.dialog_utils import get_open_file_name

        filepath, _ = get_open_file_name(
            self,
            "Select Window-Label Metadata File",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if filepath:
            self.set_label_metadata_path(filepath)

    def keyPressEvent(self, a0):
        """Handle analyse-mode keyboard shortcuts."""
        manager = getattr(self, "manager", None)
        if manager is None or a0 is None:
            super().keyPressEvent(a0)
            return

        event = a0

        key_text = event.text().lower()
        key_code = event.key()

        if key_text.isdigit():
            label_idx = int(key_text) - 1
            if label_idx == -1:  # Handle '0' key as index 9
                label_idx = 9
            available_labels = manager.main.graph_manager.available_labels
            if 0 <= label_idx < len(available_labels):
                label = available_labels[label_idx]
                manager.main.graph_manager.label_current_window(label)
                print(f"Window {manager.main.graph_manager.current_window_idx} labeled as: {label}")
                manager.jump_to_window(forward=True, unlabeled_only=True)
                event.accept()
                return

        if key_text == "-":
            label_idx = 10
            available_labels = manager.main.graph_manager.available_labels
            if 0 <= label_idx < len(available_labels):
                label = available_labels[label_idx]
                manager.main.graph_manager.label_current_window(label)
                print(f"Window {manager.main.graph_manager.current_window_idx} labeled as: {label}")
                manager.jump_to_window(forward=True, unlabeled_only=True)
                event.accept()
                return

        if key_text == "=":
            label_idx = 11
            available_labels = manager.main.graph_manager.available_labels
            if 0 <= label_idx < len(available_labels):
                label = available_labels[label_idx]
                manager.main.graph_manager.label_current_window(label)
                print(f"Window {manager.main.graph_manager.current_window_idx} labeled as: {label}")
                manager.jump_to_window(forward=True, unlabeled_only=True)
                event.accept()
                return
            
        # First letter of class labels for quick labeling (if label starts with a unique letter)
        for label in manager.main.graph_manager.available_labels:
            if label and label[0].lower() == key_text:
                manager.main.graph_manager.label_current_window(label)
                print(f"Window {manager.main.graph_manager.current_window_idx} labeled as: {label}")
                manager.jump_to_window(forward=True, unlabeled_only=True)
                event.accept()
                return

        if key_code == QtCore.Qt.Key.Key_Left or key_code == QtCore.Qt.Key.Key_Less:
            manager.jump_to_window(forward=False)
            event.accept()
            return

        if key_code == QtCore.Qt.Key.Key_Right or key_code == QtCore.Qt.Key.Key_Greater:
            manager.jump_to_window(forward=True)
            event.accept()
            return

        if key_text == "n":
            manager.jump_to_window(forward=True, unlabeled_only=True)
            event.accept()
            return

        if key_text == "m":
            manager.jump_to_window(forward=False, unlabeled_only=True)
            event.accept()
            return

        if key_text == "s":
            manager.save_labels_to_json()
            event.accept()
            return

        super().keyPressEvent(event)

    def show_markers_enabled(self):
        """Return whether marker lines should be visible in analyse plots."""
        return self.show_markers_checkbox.isChecked()

    def show_gaps_enabled(self):
        """Return whether data gap regions should be visible in analyse plots."""
        return self.show_gaps_checkbox.isChecked()

    def show_window_stats_enabled(self):
        """Return whether the per-window statistics should be shown in the info panel."""
        return self.show_window_stats_checkbox.isChecked()

    def show_labels_enabled(self):
        """Return whether existing label overlays should be visible in the plots."""
        return self.show_labels_checkbox.isChecked()
    
    def get_selected_features(self):
        """Get a list of selected features (lowercase names)."""
        return [feature for feature, checkbox in self.feature_checkboxes.items() 
                if checkbox.isChecked()]
    
    def set_label_options(self, label_names, checked=True):
        """Populate the class toggles from a list of label names."""
        while self.class_layout.count():
            item = self.class_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        self.label_checkboxes = {}
        for idx, label_name in enumerate(label_names):
            checkbox = QtWidgets.QCheckBox(label_name)
            checkbox.setChecked(checked)
            row = idx // 4
            col = idx % 4
            self.class_layout.addWidget(checkbox, row, col)
            self.label_checkboxes[label_name] = checkbox

    def get_selected_labels(self):
        """Get the selected label names."""
        return [
            label_name
            for label_name, checkbox in self.label_checkboxes.items()
            if checkbox.isChecked()
        ]

    def set_label_checked(self, label_name, checked):
        """Set whether a specific label is checked."""
        if label_name in self.label_checkboxes:
            self.label_checkboxes[label_name].setChecked(checked)

    def check_all_labels(self, checked=True):
        """Check or uncheck all label toggles."""
        for checkbox in self.label_checkboxes.values():
            checkbox.setChecked(checked)

    def set_feature_options(self, feature_names, checked=True):
        """Populate the feature toggles from a list of feature names."""
        while self.features_layout.count():
            item = self.features_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        self.feature_checkboxes = {}
        for idx, feature_name in enumerate(feature_names):
            checkbox = QtWidgets.QCheckBox(feature_name)
            checkbox.setChecked(checked)
            row = idx // 4
            col = idx % 4
            self.features_layout.addWidget(checkbox, row, col)
            self.feature_checkboxes[feature_name.lower()] = checkbox
    
    def set_feature_checked(self, feature, checked):
        """Set whether a specific feature is checked."""
        feature_lower = feature.lower()
        if feature_lower in self.feature_checkboxes:
            self.feature_checkboxes[feature_lower].setChecked(checked)
    
    def check_all_features(self, checked=True):
        """Check or uncheck all features."""
        for checkbox in self.feature_checkboxes.values():
            checkbox.setChecked(checked)

    def set_slider_range(self, max_value):
        """Set the slider range (typically length of data)."""
        self.analyse_slider.blockSignals(True)
        self.analyse_slider.setRange(0, max_value)
        self.analyse_slider.setValue(0)
        self.analyse_slider.setEnabled(True)
        self.analyse_slider.blockSignals(False)
    
    def get_slider_value(self):
        """Get the current slider value."""
        return self.analyse_slider.value()
    
    def set_slider_value(self, value):
        """Set the slider value without triggering signals."""
        self.analyse_slider.blockSignals(True)
        self.analyse_slider.setValue(value)
        self.analyse_slider.blockSignals(False)
    
    def set_timestamp_label(self, time_str):
        """Set the timestamp display label."""
        self.analyse_timestamp_label.setText(f"Time: {time_str}")
    
    def enable_slider(self, enabled):
        """Enable or disable the slider."""
        self.analyse_slider.setEnabled(enabled)
    
    def reset(self):
        """Reset the panel to its initial state."""
        self.analyse_slider.blockSignals(True)
        self.analyse_slider.setValue(0)
        self.analyse_slider.setEnabled(False)
        self.analyse_slider.blockSignals(False)
        self.analyse_timestamp_label.setText("Time: 0.00 s")
        self.label_metadata_input.setText("")
        self.analyse_status.setText("Status: Ready")
        self.analyse_status.setStyleSheet("color: #00FF00;")
