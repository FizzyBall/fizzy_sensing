"""
Classification Analysis Panel - Load recorded data and classify segments.
"""

import importlib.util
from pathlib import Path
from PyQt6 import QtWidgets, QtCore, QtGui

# CNN settings / mean-std assets now live under BEP code/CNN_final/Model Running
# (resolved by utilities.cnn_paths, which also supports the legacy "CNN" folder).
from utilities.cnn_paths import CNN_SETTINGS_FILE, MODELS_DIR, MEAN_STD_DIR

class ClassificationAnalysisPanel(QtWidgets.QGroupBox):
    """Panel for classification analysis configuration and control."""
    
    # Signals
    model_loaded = QtCore.pyqtSignal(object)  # Emits: loaded model
    browse_model_requested = QtCore.pyqtSignal()
    browse_requested = QtCore.pyqtSignal()
    load_requested = QtCore.pyqtSignal()
    label_metadata_path_changed = QtCore.pyqtSignal(str)
    slider_moved = QtCore.pyqtSignal(int)
    marker_visibility_changed = QtCore.pyqtSignal(bool)
    gap_visibility_changed = QtCore.pyqtSignal(bool)
    window_eval_changed = QtCore.pyqtSignal(str)        # 'correct' | 'wrong' | 'ignored'
    window_true_class_changed = QtCore.pyqtSignal(str)  # selected effective true class
    save_altered_labels_requested = QtCore.pyqtSignal()
    treat_ignored_correct_changed = QtCore.pyqtSignal(bool)
    
    def __init__(self):
        super().__init__("Classification Analysis Controls")
        self.setCheckable(True)
        self.setChecked(True)
        self.setVisible(False)
        self.current_model = None
        self.model_paths = []
        self._mean_std_pairs = []
        self._settings_versions = {}
        self._discover_model_paths()
        self._discover_mean_std_pairs()
        self._load_cnn_settings()
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
    
    def _discover_mean_std_pairs(self):
        ms_dir = MEAN_STD_DIR
        self._mean_std_pairs = []
        try:
            if ms_dir.exists():
                for mean_file in sorted(ms_dir.glob("imu_mean*.npy")):
                    std_file = mean_file.parent / mean_file.name.replace("mean", "std")
                    if not std_file.exists():
                        continue
                    stem = mean_file.stem
                    version_part = stem.replace("imu_mean", "").lstrip("_")
                    display = version_part or "legacy"
                    self._mean_std_pairs.append((display, mean_file, std_file))
        except Exception as e:
            print(f"Error discovering mean/std pairs: {e}")

    def _load_cnn_settings(self):
        """Read DATA_VERSION_* from CNN/settings.py so mode changes auto-select the right mean/std."""
        try:
            settings_path = CNN_SETTINGS_FILE
            if settings_path.exists():
                spec = importlib.util.spec_from_file_location("cnn_settings", settings_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self._settings_versions = {
                    "normal": str(getattr(mod, 'data_version', '7')),
                    "hand":   str(getattr(mod, 'DATA_VERSION_HAND', '')),
                    "ground": str(getattr(mod, 'DATA_VERSION_GROUND', '')),
                }
        except Exception as e:
            print(f"Could not read CNN/settings.py: {e}")

    def _discover_model_paths(self):
        try:
            self.model_paths = []
            seen = set()

            # Random-forest models (.pkl/.joblib): scan the whole working directory
            # tree, since they aren't kept in one canonical place.
            for root in (Path.cwd(),):
                if not root.exists():
                    continue
                for ext in ("*.joblib", "*.pkl"):
                    for p in root.rglob(ext):
                        # Skip virtual environments or hidden folders to save time/prevent duplicates
                        if any(part.startswith('.') or part in ('venv', 'env', 'node_modules') for part in p.parts):
                            continue
                        key = str(p.resolve())
                        if key not in seen:
                            self.model_paths.append(p)
                            seen.add(key)

            # CNN models (.pth): only the canonical DataAndModels/CNN/Models folder,
            # so superseded weights in legacy folders don't clutter the list.
            if MODELS_DIR.exists():
                for p in sorted(MODELS_DIR.glob("*.pth")):
                    key = str(p.resolve())
                    if key not in seen:
                        self.model_paths.append(p)
                        seen.add(key)

        except Exception as e:
            print(f"Error discovering models: {e}")
            self.model_paths = []

    def _model_display_name(self, model_path):
        try:
            return model_path.stem
        except Exception:
            return str(model_path)

    def _init_ui(self):
        layout = QtWidgets.QGridLayout(self)
        
        # Row 0: Model selection header and actions
        model_label = QtWidgets.QLabel("Models (check one or more):")
        model_label.setToolTip("Select one or more trained models to load together for comparison.")
        layout.addWidget(model_label, 0, 0, 1, 1)

        self.select_all_models_btn = QtWidgets.QPushButton("All")
        self.select_all_models_btn.setToolTip("Check all discovered models.")
        self.select_all_models_btn.clicked.connect(self.select_all_models)
        layout.addWidget(self.select_all_models_btn, 0, 1)

        self.clear_models_btn = QtWidgets.QPushButton("None")
        self.clear_models_btn.setToolTip("Uncheck all models.")
        self.clear_models_btn.clicked.connect(self.clear_model_selection)
        layout.addWidget(self.clear_models_btn, 0, 2)

        self.browse_model_btn = QtWidgets.QPushButton("Browse Model")
        self.browse_model_btn.clicked.connect(self.browse_model_requested.emit)
        layout.addWidget(self.browse_model_btn, 0, 3)
        # Row 1: Model list
        self.model_list = QtWidgets.QListWidget()
        self.model_list.setToolTip("Check one or more models. The Load button will compare their predictions.")
        self.model_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        # Keep the list compact so it can't expand and push the graph out of view;
        # extra models stay reachable via the list's own scrollbar.
        self.model_list.setMaximumHeight(90)
        self.model_list.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        layout.addWidget(self.model_list, 1, 0, 1, 4)

        for model_path in self.model_paths:
            self.add_model_path(model_path, checked=False)

        # Row 2: CNN Options (mode + mean/std version for .pth models)
        cnn_opts_group = QtWidgets.QGroupBox("CNN Options")
        cnn_opts_layout = QtWidgets.QHBoxLayout(cnn_opts_group)
        cnn_opts_layout.setContentsMargins(4, 2, 4, 2)

        cnn_opts_layout.addWidget(QtWidgets.QLabel("Mode:"))
        self.cnn_mode_combo = QtWidgets.QComboBox()
        self.cnn_mode_combo.setToolTip(
            "Select which CNN architecture the loaded .pth model was trained with.\n"
            "Each option maps to the exact Python model file used during training."
        )
        self.cnn_mode_combo.addItem("Normal  (CNN_fizzy.py        6ch · idle/shake/tap/drop/lift/kick)", "normal")
        self.cnn_mode_combo.addItem("Hand    (CNN_fizzy_hand.py   6ch · idle/tap/shake/drop)", "hand")
        self.cnn_mode_combo.addItem("Ground  (CNN_fizzy_ground.py 7ch · idle/lift/kick)", "ground")
        cnn_opts_layout.addWidget(self.cnn_mode_combo)

        cnn_opts_layout.addSpacing(12)
        cnn_opts_layout.addWidget(QtWidgets.QLabel("Mean/Std:"))
        self.cnn_mean_std_combo = QtWidgets.QComboBox()
        self.cnn_mean_std_combo.setToolTip(
            "Select which normalisation files to use for CNN models.\n"
            "Pick the version the model was trained on."
        )
        for display, mp, sp in self._mean_std_pairs:
            self.cnn_mean_std_combo.addItem(display, (str(mp), str(sp)))
        cnn_opts_layout.addWidget(self.cnn_mean_std_combo)
        cnn_opts_layout.addStretch()

        self.cnn_mode_combo.currentIndexChanged.connect(self._on_cnn_mode_changed)

        layout.addWidget(cnn_opts_group, 2, 0, 1, 4)


        # Row 3: File selector
        self.analyse_file_input = QtWidgets.QLineEdit()
        self.analyse_file_input.setPlaceholderText("Select CSV file to analyse")
        layout.addWidget(self.analyse_file_input, 3, 0, 1, 2)

        self.analyse_browse_btn = QtWidgets.QPushButton("Browse Recording")
        self.analyse_browse_btn.clicked.connect(self.browse_requested.emit)
        layout.addWidget(self.analyse_browse_btn, 3, 2)

        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_btn.clicked.connect(self.load_requested.emit)
        layout.addWidget(self.load_btn, 3, 3)

        # Row 4: Label metadata selector
        self.label_metadata_input = QtWidgets.QLineEdit()
        self.label_metadata_input.setPlaceholderText("Select label metadata JSON")
        self.label_metadata_input.editingFinished.connect(
            lambda: self.label_metadata_path_changed.emit(self.label_metadata_input.text().strip())
        )
        layout.addWidget(self.label_metadata_input, 4, 0, 1, 3)

        self.label_metadata_browse_btn = QtWidgets.QPushButton("Browse Labels")
        self.label_metadata_browse_btn.setToolTip("Select the JSON sidecar that contains the true window labels.")
        self.label_metadata_browse_btn.clicked.connect(self._browse_label_metadata_file)
        layout.addWidget(self.label_metadata_browse_btn, 4, 3)

        # Row 5: Model status & Analysis status
        self.model_status_label = QtWidgets.QLabel("No model loaded")
        self.model_status_label.setStyleSheet("color: #FF6600;")
        self.model_status_label.setWordWrap(True)
        layout.addWidget(self.model_status_label, 5, 0, 1, 2)

        self.analyse_status = QtWidgets.QLabel("Status: Ready")
        self.analyse_status.setStyleSheet("color: #00FF00;")
        self.analyse_status.setWordWrap(True)
        layout.addWidget(self.analyse_status, 5, 2, 1, 3)

        # Row 6: Timestamp slider
        self.analyse_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.analyse_slider.setRange(0, 100)
        self.analyse_slider.setValue(0)
        self.analyse_slider.sliderMoved.connect(self.slider_moved.emit)
        self.analyse_slider.valueChanged.connect(self.slider_moved.emit)
        self.analyse_slider.setEnabled(False)
        layout.addWidget(self.analyse_slider, 6, 0, 1, 3)

        self.analyse_timestamp_label = QtWidgets.QLabel("Time: 0.00 s")
        layout.addWidget(self.analyse_timestamp_label, 6, 3)

        # Row 7: Marker and gap visibility toggles
        self.show_markers_checkbox = QtWidgets.QCheckBox("Show Markers")
        self.show_markers_checkbox.setToolTip("Show or hide marker lines in the main and windowed plots.")
        self.show_markers_checkbox.setChecked(True)
        self.show_markers_checkbox.toggled.connect(self.marker_visibility_changed.emit)
        layout.addWidget(self.show_markers_checkbox, 7, 0)

        self.show_gaps_checkbox = QtWidgets.QCheckBox("Show Data Gaps")
        self.show_gaps_checkbox.setToolTip("Show or hide detected data gap regions in the main and windowed plots.")
        self.show_gaps_checkbox.setChecked(True)
        self.show_gaps_checkbox.toggled.connect(self.gap_visibility_changed.emit)
        layout.addWidget(self.show_gaps_checkbox, 7, 1)

        # Row 8: Window evaluation override controls
        eval_group = QtWidgets.QGroupBox("Window Evaluation")
        eval_group.setToolTip(
            "Override how the current window counts towards the evaluation metrics.\n"
            "Correct/Wrong/Ignored and the True class dropdown stay in sync and update "
            "the metrics immediately."
        )
        eval_v = QtWidgets.QVBoxLayout(eval_group)
        eval_v.setContentsMargins(4, 2, 4, 2)

        eval_row = QtWidgets.QHBoxLayout()
        eval_row.addWidget(QtWidgets.QLabel("Evaluation:"))

        self.eval_button_group = QtWidgets.QButtonGroup(self)
        self.eval_button_group.setExclusive(True)

        self.eval_correct_btn = QtWidgets.QPushButton("Correct")
        self.eval_correct_btn.setCheckable(True)
        self.eval_correct_btn.setToolTip("Count the model's prediction for this window as correct (true class = prediction).")
        self.eval_wrong_btn = QtWidgets.QPushButton("Wrong")
        self.eval_wrong_btn.setCheckable(True)
        self.eval_wrong_btn.setToolTip("Count the model's prediction for this window as incorrect.")
        self.eval_ignored_btn = QtWidgets.QPushButton("Ignored")
        self.eval_ignored_btn.setCheckable(True)
        self.eval_ignored_btn.setToolTip("Exclude this window from the evaluation metrics.")
        for btn in (self.eval_correct_btn, self.eval_wrong_btn, self.eval_ignored_btn):
            self.eval_button_group.addButton(btn)
            eval_row.addWidget(btn)
        self.eval_correct_btn.clicked.connect(lambda *_: self.window_eval_changed.emit("correct"))
        self.eval_wrong_btn.clicked.connect(lambda *_: self.window_eval_changed.emit("wrong"))
        self.eval_ignored_btn.clicked.connect(lambda *_: self.window_eval_changed.emit("ignored"))

        eval_row.addSpacing(12)
        eval_row.addWidget(QtWidgets.QLabel("True class:"))
        self.true_class_combo = QtWidgets.QComboBox()
        self.true_class_combo.setToolTip(
            "The effective true class for this window. Pick the model's prediction to mark it "
            "correct, another class to mark it wrong, 'no match' for wrong with no matching "
            "class, or 'ignored' to exclude the window."
        )
        # activated only fires on user interaction (not programmatic changes).
        self.true_class_combo.activated.connect(
            lambda *_: self.window_true_class_changed.emit(self.true_class_combo.currentText())
        )
        eval_row.addWidget(self.true_class_combo)
        eval_row.addStretch()
        eval_v.addLayout(eval_row)

        self.treat_ignored_correct_checkbox = QtWidgets.QCheckBox("Treat ignored windows as correct")
        self.treat_ignored_correct_checkbox.setToolTip(
            "When enabled, windows marked 'ignored' are counted as correct in the "
            "evaluation metrics instead of being excluded — so you don't have to mark "
            "each one correct by hand."
        )
        self.treat_ignored_correct_checkbox.toggled.connect(self.treat_ignored_correct_changed.emit)
        eval_v.addWidget(self.treat_ignored_correct_checkbox)

        bottom_row = QtWidgets.QHBoxLayout()
        self.original_true_label = QtWidgets.QLabel("")
        self.original_true_label.setStyleSheet("color: #FFAA00;")
        self.original_true_label.setWordWrap(True)
        bottom_row.addWidget(self.original_true_label, 1)

        self.save_altered_labels_btn = QtWidgets.QPushButton("Save Altered Labels")
        self.save_altered_labels_btn.setToolTip(
            "Save a new JSON labels file with the current (possibly overridden) true "
            "labels. Ignored windows are omitted; the filename includes 'altered_labels'."
        )
        self.save_altered_labels_btn.clicked.connect(self.save_altered_labels_requested.emit)
        bottom_row.addWidget(self.save_altered_labels_btn)
        eval_v.addLayout(bottom_row)

        layout.addWidget(eval_group, 8, 0, 1, 4)

        # Start disabled until a window is selected.
        self.set_window_evaluation(False, None, None, "")

        # Row 9: Detailed Info View
        info_label = QtWidgets.QLabel("Results:")
        layout.addWidget(info_label, 9, 0)
        self.info_text = QtWidgets.QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFont(QtGui.QFont("Courier", 9))
        # Cap the height so a long info dump can't push the graph out of view.
        # Excess text stays reachable via the QTextEdit's internal scrollbar.
        self.info_text.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.info_text.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        layout.addWidget(self.info_text, 10, 0, 1, 4)

    def bind_manager(self, manager):
        self.manager = manager
        self.model_loaded.connect(manager.on_model_loaded)
        self.browse_model_requested.connect(manager.on_browse_model)
        self.browse_requested.connect(manager.on_browse_csv)
        self.load_requested.connect(manager.on_load_csv)
        self.label_metadata_path_changed.connect(manager.on_label_metadata_path_changed)
        self.slider_moved.connect(manager.on_slider_moved)
        self.marker_visibility_changed.connect(manager.on_marker_visibility_changed)
        self.gap_visibility_changed.connect(manager.on_gap_visibility_changed)
        self.window_eval_changed.connect(manager.on_window_eval_toggled)
        self.window_true_class_changed.connect(manager.on_window_true_class_changed)
        self.save_altered_labels_requested.connect(manager.save_altered_labels)
        self.treat_ignored_correct_changed.connect(manager.on_treat_ignored_correct_changed)

        self.left_window_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self)
        self.left_window_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.left_window_shortcut.activated.connect(lambda: manager.jump_to_window(forward=False))

        self.right_window_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self)
        self.right_window_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.right_window_shortcut.activated.connect(lambda: manager.jump_to_window(forward=True))

    def get_file_path(self):
        return self.analyse_file_input.text()

    def set_file_path(self, path):
        self.analyse_file_input.setText(path)

    def get_label_metadata_path(self):
        return self.label_metadata_input.text().strip()

    def set_label_metadata_path(self, path):
        self.label_metadata_input.setText(path)
        self.label_metadata_path_changed.emit(path.strip())

    def _browse_label_metadata_file(self):
        from utilities.dialog_utils import get_open_file_name

        filepath, _ = get_open_file_name(
            self,
            "Select Label Metadata File",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if filepath:
            self.set_label_metadata_path(filepath)

    def set_status(self, text, style="color: #00FF00;"):
        self.analyse_status.setText(f"Status: {text}")
        self.analyse_status.setStyleSheet(style)

    def add_model_path(self, model_path, checked=True):
        display_name = self._model_display_name(Path(model_path))
        item = QtWidgets.QListWidgetItem(display_name)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, str(model_path))
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
        item.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
        self.model_list.addItem(item)

    def get_selected_model_paths(self):
        selected_paths = []
        for index in range(self.model_list.count()):
            item = self.model_list.item(index)
            if item is not None and item.checkState() == QtCore.Qt.CheckState.Checked:
                selected_paths.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
        return selected_paths

    def select_all_models(self):
        for index in range(self.model_list.count()):
            item = self.model_list.item(index)
            if item is not None:
                item.setCheckState(QtCore.Qt.CheckState.Checked)

    def clear_model_selection(self):
        for index in range(self.model_list.count()):
            item = self.model_list.item(index)
            if item is not None:
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def set_slider_range(self, max_value):
        self.analyse_slider.setRange(0, max_value)

    def get_slider_value(self):
        return self.analyse_slider.value()

    def set_slider_value(self, value):
        self.analyse_slider.blockSignals(True)
        self.analyse_slider.setValue(value)
        self.analyse_slider.blockSignals(False)

    def set_timestamp_label(self, time_str):
        self.analyse_timestamp_label.setText(time_str)

    def enable_slider(self, enabled):
        self.analyse_slider.setEnabled(enabled)

    def set_info_text(self, text):
        self.info_text.setPlainText(text)

    def set_evaluation_classes(self, class_names):
        """Populate the True class dropdown (without emitting change signals)."""
        self.true_class_combo.blockSignals(True)
        self.true_class_combo.clear()
        self.true_class_combo.addItems([str(c) for c in class_names])
        self.true_class_combo.blockSignals(False)

    def set_window_evaluation(self, enabled, state, true_class, original_text):
        """Reflect the current window's evaluation in the toggles, dropdown, and note.

        Args:
            enabled: whether a window is selected (controls are disabled otherwise)
            state: 'correct' | 'wrong' | 'ignored' | None
            true_class: text to select in the dropdown (added if missing) or None
            original_text: note describing the original true value when altered
        """
        for widget in (self.eval_correct_btn, self.eval_wrong_btn,
                       self.eval_ignored_btn, self.true_class_combo):
            widget.setEnabled(enabled)

        # Exclusive groups won't allow clearing all buttons, so toggle exclusivity off
        # while we set the checked states programmatically.
        self.eval_button_group.setExclusive(False)
        self.eval_correct_btn.setChecked(state == "correct")
        self.eval_wrong_btn.setChecked(state == "wrong")
        self.eval_ignored_btn.setChecked(state == "ignored")
        self.eval_button_group.setExclusive(True)

        if true_class is not None:
            self.true_class_combo.blockSignals(True)
            idx = self.true_class_combo.findText(str(true_class))
            if idx < 0:
                self.true_class_combo.addItem(str(true_class))
                idx = self.true_class_combo.findText(str(true_class))
            if idx >= 0:
                self.true_class_combo.setCurrentIndex(idx)
            self.true_class_combo.blockSignals(False)

        self.original_true_label.setText(original_text or "")

    def _on_cnn_mode_changed(self, _index):
        """When mode changes, auto-select the mean/std version from settings.py."""
        mode = self.cnn_mode_combo.currentData()
        if mode is None:
            return
        version = self._settings_versions.get(mode, '')
        if not version:
            return
        for i in range(self.cnn_mean_std_combo.count()):
            data = self.cnn_mean_std_combo.itemData(i)
            if data is None:
                continue
            mean_stem = Path(data[0]).stem  # e.g. imu_mean_v13
            if f'_v{version}' in mean_stem:
                self.cnn_mean_std_combo.setCurrentIndex(i)
                return

    def get_cnn_mode(self):
        """Return 'normal', 'hand', 'ground', or None for auto-detect."""
        return self.cnn_mode_combo.currentData()

    def get_cnn_mean_std(self):
        """Return (mean_path_str, std_path_str) tuple or None for auto."""
        return self.cnn_mean_std_combo.currentData()

    def show_markers_enabled(self):
        return self.show_markers_checkbox.isChecked()

    def show_gaps_enabled(self):
        return self.show_gaps_checkbox.isChecked()

    def keyPressEvent(self, a0):
        manager = getattr(self, "manager", None)
        if manager is None or a0 is None:
            super().keyPressEvent(a0)
            return

        event = a0

        if event.key() == QtCore.Qt.Key.Key_Left:
            manager.jump_to_window(forward=False)
            event.accept()
            return

        if event.key() == QtCore.Qt.Key.Key_Right:
            manager.jump_to_window(forward=True)
            event.accept()
            return

        super().keyPressEvent(event)
