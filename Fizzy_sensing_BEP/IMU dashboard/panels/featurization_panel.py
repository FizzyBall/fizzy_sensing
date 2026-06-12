"""
Featurization Control Panel - Folder-based feature extraction controls.
"""

import os

from PyQt6 import QtWidgets, QtCore


class FeaturizationPanel(QtWidgets.QGroupBox):
    """Panel for folder-based featurization into a single CSV file."""

    source_folder_browse_requested = QtCore.pyqtSignal()
    output_location_browse_requested = QtCore.pyqtSignal()
    featurize_requested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__("Featurization")
        self.setCheckable(True)
        self.setChecked(True)
        self.setVisible(False)  # Hidden unless Featurization mode is selected
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
        layout = QtWidgets.QGridLayout(self)

        # Row 0: Source folders selector
        source_folder_label = QtWidgets.QLabel("Labeled Data Folders:")
        source_folder_label.setToolTip(
            "Select one or more source folders. You can select either: "
            "(1) a parent folder that contains one subfolder per label, or "
            "(2) a single label subfolder directly. "
            "Each label folder should contain CSV files where each file is one data window. "
            "When selecting a single label subfolder, that subfolder name is used as the label. "
            "You can use '#' in folder names to split into class and group: e.g., 'tap#group_1' -> class='tap', group='group_1'. "
            "If no '#' is present, the group will default to 'default'."
        )
        layout.addWidget(source_folder_label, 0, 0)

        self.source_folders_list = QtWidgets.QListWidget()
        self.source_folders_list.setMaximumHeight(80)
        layout.addWidget(self.source_folders_list, 0, 1, 2, 2)

        folder_btn_layout = QtWidgets.QVBoxLayout()
        self.source_folder_add_btn = QtWidgets.QPushButton("Add Folder")
        self.source_folder_add_btn.clicked.connect(self.source_folder_browse_requested.emit)
        folder_btn_layout.addWidget(self.source_folder_add_btn)

        self.source_folder_remove_btn = QtWidgets.QPushButton("Remove Selected")
        self.source_folder_remove_btn.clicked.connect(self._on_remove_source_folder)
        folder_btn_layout.addWidget(self.source_folder_remove_btn)
        
        self.source_folder_clear_btn = QtWidgets.QPushButton("Clear All")
        self.source_folder_clear_btn.clicked.connect(self._on_clear_source_folders)
        folder_btn_layout.addWidget(self.source_folder_clear_btn)
        
        folder_btn_layout.addStretch(1)
        layout.addLayout(folder_btn_layout, 0, 3, 2, 1)

        # Row 2 (was Row 1): Output location selector
        output_location_label = QtWidgets.QLabel("CSV Save Location:")
        output_location_label.setToolTip("Select where the featurized CSV file will be saved.")
        layout.addWidget(output_location_label, 2, 0)

        self.output_location_input = QtWidgets.QLineEdit()
        self.output_location_input.setPlaceholderText("Select output folder for feature CSV")
        layout.addWidget(self.output_location_input, 2, 1, 1, 2)

        self.output_location_browse_btn = QtWidgets.QPushButton("Browse")
        self.output_location_browse_btn.clicked.connect(self.output_location_browse_requested.emit)
        layout.addWidget(self.output_location_browse_btn, 2, 3)

        # Row 3 (was Row 2): Output filename
        output_filename_label = QtWidgets.QLabel("Output File Name:")
        output_filename_label.setToolTip("Base name for output CSV file (without .csv extension).")
        layout.addWidget(output_filename_label, 3, 0)

        self.output_filename_input = QtWidgets.QLineEdit("")
        self.output_filename_input.setPlaceholderText("Set output CSV file name")
        layout.addWidget(self.output_filename_input, 3, 1, 1, 2)

        # Row 4 (was Row 3): Class include/exclude toggles
        classes_label = QtWidgets.QLabel("Include Classes:")
        classes_label.setToolTip("Select which label folders should be featurized into the CSV output.")
        layout.addWidget(classes_label, 4, 0)

        class_controls = QtWidgets.QHBoxLayout()
        self.select_all_classes_btn = QtWidgets.QPushButton("Select All")
        self.select_all_classes_btn.clicked.connect(lambda: self.check_all_labels(True))
        class_controls.addWidget(self.select_all_classes_btn)

        self.select_none_classes_btn = QtWidgets.QPushButton("Select None")
        self.select_none_classes_btn.clicked.connect(lambda: self.check_all_labels(False))
        class_controls.addWidget(self.select_none_classes_btn)
        class_controls.addStretch(1)
        layout.addLayout(class_controls, 4, 1, 1, 3)

        self.classes_scroll = QtWidgets.QScrollArea()
        self.classes_scroll.setWidgetResizable(True)
        self.classes_scroll.setMinimumHeight(30)
        self.classes_container = QtWidgets.QWidget()
        self.classes_layout = QtWidgets.QGridLayout(self.classes_container)
        self.classes_layout.setContentsMargins(4, 4, 4, 4)
        self.classes_layout.setHorizontalSpacing(10)
        self.classes_layout.setVerticalSpacing(4)
        self.classes_scroll.setWidget(self.classes_container)
        layout.addWidget(self.classes_scroll, 5, 0, 1, 4)
        self.class_checkboxes = {}

        # Row 6 (was Row 5): Feature include/exclude toggles
        features_label = QtWidgets.QLabel("Include Features:")
        features_label.setToolTip("Select which feature columns should be included in the output CSV.")
        layout.addWidget(features_label, 6, 0)

        controls_row = QtWidgets.QHBoxLayout()
        self.select_all_btn = QtWidgets.QPushButton("Select All")
        self.select_all_btn.clicked.connect(lambda: self._set_all_feature_checkboxes(True))
        controls_row.addWidget(self.select_all_btn)

        self.select_none_btn = QtWidgets.QPushButton("Select None")
        self.select_none_btn.clicked.connect(lambda: self._set_all_feature_checkboxes(False))
        controls_row.addWidget(self.select_none_btn)
        controls_row.addStretch(1)
        layout.addLayout(controls_row, 6, 1, 1, 3)

        # Category toggles layout
        category_label = QtWidgets.QLabel("Feature Categories:")
        category_label.setToolTip("Quickly toggle all features in a category (e.g., IQR toggles all IQR features).")
        layout.addWidget(category_label, 7, 0)

        self.categories_scroll = QtWidgets.QScrollArea()
        self.categories_scroll.setWidgetResizable(True)
        self.categories_scroll.setMaximumHeight(50)
        self.categories_container = QtWidgets.QWidget()
        self.categories_layout = QtWidgets.QHBoxLayout(self.categories_container)
        self.categories_layout.setContentsMargins(4, 4, 4, 4)
        self.categories_layout.setSpacing(10)
        self.categories_scroll.setWidget(self.categories_container)
        layout.addWidget(self.categories_scroll, 7, 1, 1, 3)
        self.category_checkboxes = {}

        self.features_scroll = QtWidgets.QScrollArea()
        self.features_scroll.setWidgetResizable(True)
        self.features_scroll.setMinimumHeight(50)
        self.features_container = QtWidgets.QWidget()
        self.features_layout = QtWidgets.QGridLayout(self.features_container)
        self.features_layout.setContentsMargins(4, 4, 4, 4)
        self.features_layout.setHorizontalSpacing(10)
        self.features_layout.setVerticalSpacing(4)
        self.features_scroll.setWidget(self.features_container)
        layout.addWidget(self.features_scroll, 8, 0, 1, 4)
        self.feature_checkboxes = {}

        # Row 9: Action button
        self.featurize_btn = QtWidgets.QPushButton("Featurize data")
        self.featurize_btn.clicked.connect(self.featurize_requested.emit)
        layout.addWidget(self.featurize_btn, 9, 0, 1, 2)

        # Row 9: Status
        self.status_label = QtWidgets.QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #00FF00;")
        layout.addWidget(self.status_label, 9, 2, 1, 2)

    def get_source_folders(self):
        """Get all selected source folder paths as a list."""
        folders = []
        for i in range(self.source_folders_list.count()):
            item = self.source_folders_list.item(i)
            if item is not None:
                folders.append(item.text())
        return folders

    def add_source_folder(self, folder_path):
        """Add a source folder to the list (no duplicates)."""
        import os
        # Normalize the path for comparison (handles case differences, .., etc.)
        normalized_new = os.path.normpath(os.path.abspath(folder_path)).lower()
        
        # Check if already in list
        for i in range(self.source_folders_list.count()):
            item = self.source_folders_list.item(i)
            if item is not None:
                normalized_existing = os.path.normpath(os.path.abspath(item.text())).lower()
                if normalized_existing == normalized_new:
                    return  # Already in list
        
        self.source_folders_list.addItem(folder_path)

    def bind_manager(self, manager):
        """Connect this panel's signals to the featurization manager."""
        self.source_folder_browse_requested.connect(manager.on_source_browse)
        self.output_location_browse_requested.connect(manager.on_output_browse)
        self.featurize_requested.connect(manager.on_featurize_requested)

    def _on_remove_source_folder(self):
        """Remove selected folder from the list."""
        for item in self.source_folders_list.selectedItems():
            self.source_folders_list.takeItem(self.source_folders_list.row(item))

    def _on_clear_source_folders(self):
        """Clear all folders from the list."""
        self.source_folders_list.clear()

    def get_source_folder(self):
        """Get the first source folder path (for backwards compatibility)."""
        folders = self.get_source_folders()
        return folders[0] if folders else ""

    def set_source_folder(self, folder_path, set_output_default=True):
        """Set source folder (clears previous) and optionally update output location default."""
        previous_output = self.get_output_location()

        self.source_folders_list.clear()
        self.source_folders_list.addItem(folder_path)

        if set_output_default:
            if not previous_output:
                self.output_location_input.setText(folder_path)

    def get_output_location(self):
        """Get output location folder path."""
        return self.output_location_input.text().strip()

    def set_output_location(self, folder_path):
        """Set output location folder path."""
        self.output_location_input.setText(folder_path)

    def get_output_filename(self):
        """Get output CSV base file name (without extension)."""
        return self.output_filename_input.text().strip()

    def set_output_filename(self, name):
        """Set output CSV base file name."""
        self.output_filename_input.setText(name)

    def set_status(self, text, style="color: #00FF00;"):
        """Set status message with color style."""
        self.status_label.setText(text)
        self.status_label.setStyleSheet(style)

    def set_feature_options(self, feature_names, checked=True):
        """Populate feature checkboxes from feature names."""
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
            checkbox.stateChanged.connect(lambda state, name=feature_name: self._on_feature_checkbox_changed(name))
            row = idx // 4
            col = idx % 4
            self.features_layout.addWidget(checkbox, row, col)
            self.feature_checkboxes[feature_name] = checkbox

        # Extract and create category toggles
        self._update_category_toggles()


    def _set_all_feature_checkboxes(self, checked):
        """Set all feature checkboxes to a common state."""
        for checkbox in self.feature_checkboxes.values():
            checkbox.setChecked(checked)

    def get_selected_features(self):
        """Get all selected feature names."""
        return [
            feature_name
            for feature_name, checkbox in self.feature_checkboxes.items()
            if checkbox.isChecked()
        ]

    def set_label_options(self, label_names, checked=True):
        """Populate class checkboxes from label names."""
        while self.classes_layout.count():
            item = self.classes_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        self.class_checkboxes = {}
        for idx, label_name in enumerate(label_names):
            checkbox = QtWidgets.QCheckBox(label_name)
            checkbox.setChecked(checked)
            row = idx // 4
            col = idx % 4
            self.classes_layout.addWidget(checkbox, row, col)
            self.class_checkboxes[label_name] = checkbox

    def check_all_labels(self, checked=True):
        """Set all class checkboxes to a common state."""
        for checkbox in self.class_checkboxes.values():
            checkbox.setChecked(checked)

    def get_selected_labels(self):
        """Get all selected label names."""
        return [
            label_name
            for label_name, checkbox in self.class_checkboxes.items()
            if checkbox.isChecked()
        ]

    def _extract_categories(self):
        """Extract unique feature categories from feature names.
        
        Categories are identified by the last underscore-separated suffix.
        E.g., 'acc_x_IQR' -> category 'IQR', 'gyro_z_mean' -> category 'mean'
        """
        categories = {}
        for feature_name in self.feature_checkboxes.keys():
            parts = feature_name.split('_')
            if len(parts) > 1:
                category = parts[-1]
                if category not in categories:
                    categories[category] = []
                categories[category].append(feature_name)
        return categories

    def _update_category_toggles(self):
        """Update category toggle buttons based on current features."""
        # Clear existing category checkboxes
        while self.categories_layout.count():
            item = self.categories_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        self.category_checkboxes = {}
        categories = self._extract_categories()

        for category in sorted(categories.keys()):
            checkbox = QtWidgets.QCheckBox(category)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, cat=category: self._on_category_checkbox_changed(cat))
            self.categories_layout.addWidget(checkbox)
            self.category_checkboxes[category] = checkbox

        self.categories_layout.addStretch(1)

    def _on_category_checkbox_changed(self, category):
        """Handle category checkbox state changes and update associated feature checkboxes."""
        categories = self._extract_categories()
        if category in categories:
            is_checked = self.category_checkboxes[category].isChecked()
            for feature_name in categories[category]:
                if feature_name in self.feature_checkboxes:
                    self.feature_checkboxes[feature_name].blockSignals(True)
                    self.feature_checkboxes[feature_name].setChecked(is_checked)
                    self.feature_checkboxes[feature_name].blockSignals(False)

    def _on_feature_checkbox_changed(self, feature_name):
        """Handle feature checkbox state changes and update category checkbox states."""
        categories = self._extract_categories()
        
        # Find which category this feature belongs to
        for category, features in categories.items():
            if feature_name in features:
                # Check if all features in this category are checked
                all_checked = all(
                    self.feature_checkboxes[f].isChecked() for f in features
                )
                # Check if any features in this category are checked
                any_checked = any(
                    self.feature_checkboxes[f].isChecked() for f in features
                )
                
                # Update category checkbox state
                self.category_checkboxes[category].blockSignals(True)
                if all_checked:
                    self.category_checkboxes[category].setCheckState(QtCore.Qt.CheckState.Checked)
                elif any_checked:
                    self.category_checkboxes[category].setCheckState(QtCore.Qt.CheckState.PartiallyChecked)
                else:
                    self.category_checkboxes[category].setCheckState(QtCore.Qt.CheckState.Unchecked)
                self.category_checkboxes[category].blockSignals(False)
                break
