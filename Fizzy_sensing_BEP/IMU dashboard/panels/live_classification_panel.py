"""
Live Classification Panel - Real-time model classification and prediction display.
Supports both Random Forest (.joblib/.pkl) and CNN (.pth) models.
"""

import importlib.util
from pathlib import Path
from PyQt6 import QtWidgets, QtCore, QtGui
from joblib import load

# Resolve the CNN code/settings/asset locations (BEP code/CNN_final/Model Running,
# with a fallback to the legacy single "CNN" folder). Importing cnn_paths also puts
# the CNN code + settings dirs on sys.path so `import cnn_wrapper` / `import settings`
# work below.
from utilities.cnn_paths import CNN_SETTINGS_FILE, MODELS_DIR, MEAN_STD_DIR, ARCHITECTURE_FILES


class LiveClassificationPanel(QtWidgets.QGroupBox):
    """Panel for live classification with Random Forest and CNN model support."""

    # Signals
    model_loaded = QtCore.pyqtSignal(object)        # RF model object
    browse_model_requested = QtCore.pyqtSignal()
    cnn_model_loaded = QtCore.pyqtSignal(object)    # CNNModelWrapper object

    def __init__(self):
        super().__init__("Live Classification")
        self.setCheckable(True)
        self.setChecked(True)
        self.current_model = None
        self.model_paths = []
        self._cnn_model_paths = []
        self._mean_std_pairs = []   # [(display_name, mean_path, std_path), ...]
        self._settings_versions = {}
        self._discover_rf_model_paths()
        self._discover_cnn_assets()
        self._load_cnn_settings()
        self._init_ui()
        self.toggled.connect(self._set_contents_visible)
        self._set_contents_visible(self.isChecked())

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def _set_contents_visible(self, visible: bool):
        for child in self.findChildren(
            QtWidgets.QWidget,
            options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly,
        ):
            child.setVisible(visible)

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

    def _discover_rf_model_paths(self):
        try:
            # Base roots to recursively search through
            search_roots = [
                Path.cwd(),
            ]
            
            self.model_paths = []
            seen = set()
            extensions = ["*.pkl", "*.joblib"]
            
            for root in search_roots:
                if not root.exists():
                    continue
                
                for ext in extensions:
                    # rglob finds files in all subfolders; sorted() keeps them organized
                    for p in sorted(root.rglob(ext)):
                        # Skip virtual environments, node modules, or hidden configuration folders
                        if any(part.startswith('.') or part in ('venv', 'env', 'node_modules') for part in p.parts):
                            continue
                            
                        key = str(p.resolve())
                        if key not in seen:
                            self.model_paths.append(p)
                            seen.add(key)
                            
        except Exception as e:
            print(f"Error discovering RF models: {e}")
            self.model_paths = []

    def _discover_cnn_assets(self):
        """Discover CNN .pth model files and paired mean/std .npy files from the
        canonical CNN folders (DataAndModels/CNN/Models and .../Mean, std) so that
        superseded weights in legacy folders don't clutter the dropdowns."""
        # --- 1. Model Files (.pth) — only the canonical Models folder ---
        self._cnn_model_paths = []
        try:
            if MODELS_DIR.exists():
                for p in sorted(MODELS_DIR.glob("*.pth")):
                    self._cnn_model_paths.append(p)
        except Exception as e:
            print(f"Error discovering CNN models in {MODELS_DIR}: {e}")

        # --- 2. Mean/Std Pairs (.npy) — only the canonical Mean, std folder ---
        self._mean_std_pairs = []
        try:
            if MEAN_STD_DIR.exists():
                for mean_file in sorted(MEAN_STD_DIR.glob("imu_mean*.npy")):
                    # Look for the matching std file in the same folder
                    std_file = mean_file.parent / mean_file.name.replace("mean", "std")
                    if not std_file.exists():
                        continue
                    stem = mean_file.stem              # e.g. imu_mean_v13
                    version_part = stem.replace("imu_mean", "").lstrip("_")  # e.g. v13 or ""
                    display = version_part or "legacy"
                    self._mean_std_pairs.append((display, mean_file, std_file))
        except Exception as e:
            print(f"Error discovering mean/std pairs in {MEAN_STD_DIR}: {e}")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _init_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setSpacing(4)

        # ---- Model-type selector ----
        type_row = QtWidgets.QHBoxLayout()
        type_row.addWidget(QtWidgets.QLabel("Type:"))
        self.rf_radio = QtWidgets.QRadioButton("Random Forest")
        self.cnn_radio = QtWidgets.QRadioButton("CNN")
        self.rf_radio.setChecked(True)
        type_row.addWidget(self.rf_radio)
        type_row.addWidget(self.cnn_radio)
        type_row.addStretch()
        outer.addLayout(type_row)

        # ---- Random Forest section ----
        self.rf_widget = QtWidgets.QWidget()
        rf_grid = QtWidgets.QGridLayout(self.rf_widget)
        rf_grid.setContentsMargins(0, 0, 0, 0)

        rf_grid.addWidget(QtWidgets.QLabel("Model:"), 0, 0)
        self.model_combo = QtWidgets.QComboBox()
        for p in self.model_paths:
            self.model_combo.addItem(p.stem, str(p))
        rf_grid.addWidget(self.model_combo, 0, 1)

        self.load_button = QtWidgets.QPushButton("Load")
        self.load_button.clicked.connect(self._load_rf_model)
        rf_grid.addWidget(self.load_button, 0, 2)

        self.browse_button = QtWidgets.QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_model_requested.emit)
        rf_grid.addWidget(self.browse_button, 0, 3)

        self.model_status_label = QtWidgets.QLabel("No model loaded")
        self.model_status_label.setStyleSheet("color: #FF6600;")
        rf_grid.addWidget(self.model_status_label, 1, 0, 1, 4)

        outer.addWidget(self.rf_widget)

        # ---- CNN section ----
        self.cnn_widget = QtWidgets.QWidget()
        cnn_grid = QtWidgets.QGridLayout(self.cnn_widget)
        cnn_grid.setContentsMargins(0, 0, 0, 0)

        # Row 0 – CNN mode
        cnn_grid.addWidget(QtWidgets.QLabel("Mode:"), 0, 0)
        self.cnn_mode_combo = QtWidgets.QComboBox()
        self.cnn_mode_combo.setToolTip(
            "Select which CNN architecture the loaded .pth weights were trained with.\n"
            "Each option maps to the exact Python model file used during training."
        )
        self.cnn_mode_combo.addItem("Normal  (CNN_fizzy.py        6ch · idle/shake/tap/drop/lift/kick)", "normal")
        self.cnn_mode_combo.addItem("Hand    (CNN_fizzy_hand.py   6ch · idle/tap/shake/drop)", "hand")
        self.cnn_mode_combo.addItem("Ground  (CNN_fizzy_ground.py 7ch · idle/lift/kick)", "ground")
        cnn_grid.addWidget(self.cnn_mode_combo, 0, 1, 1, 2)

        # Row 1 – weights file
        cnn_grid.addWidget(QtWidgets.QLabel("Weights:"), 1, 0)
        self.cnn_model_combo = QtWidgets.QComboBox()
        for p in self._cnn_model_paths:
            short = f"{p.stem}  [{p.parent.name}]"
            self.cnn_model_combo.addItem(short, str(p))
        cnn_grid.addWidget(self.cnn_model_combo, 1, 1)

        self.cnn_browse_btn = QtWidgets.QPushButton("Browse...")
        self.cnn_browse_btn.clicked.connect(self._browse_cnn_model)
        cnn_grid.addWidget(self.cnn_browse_btn, 1, 2)

        # Row 2 – mean/std
        cnn_grid.addWidget(QtWidgets.QLabel("Mean/Std:"), 2, 0)
        self.mean_std_combo = QtWidgets.QComboBox()
        for display, mp, sp in self._mean_std_pairs:
            self.mean_std_combo.addItem(display, (str(mp), str(sp)))
        cnn_grid.addWidget(self.mean_std_combo, 2, 1)

        self.mean_browse_btn = QtWidgets.QPushButton("Browse...")
        self.mean_browse_btn.clicked.connect(self._browse_mean_file)
        cnn_grid.addWidget(self.mean_browse_btn, 2, 2)

        # Row 3 – load button
        self.load_cnn_btn = QtWidgets.QPushButton("Load CNN Weights")
        self.load_cnn_btn.clicked.connect(self._load_cnn_model)
        cnn_grid.addWidget(self.load_cnn_btn, 3, 0, 1, 3)

        # Row 4 – status
        self.cnn_status_label = QtWidgets.QLabel("No CNN weights loaded")
        self.cnn_status_label.setStyleSheet("color: #FF6600;")
        self.cnn_status_label.setWordWrap(True)
        cnn_grid.addWidget(self.cnn_status_label, 4, 0, 1, 3)

        self.cnn_widget.setVisible(False)
        outer.addWidget(self.cnn_widget)

        # Connect radio buttons and mode selector
        self.rf_radio.toggled.connect(self._on_type_changed)
        self.cnn_mode_combo.currentIndexChanged.connect(self._on_cnn_mode_changed)

        # ---- Common / results section ----
        hint = QtWidgets.QLabel(
            "Window checks are driven by the shared overlap setting in the Control Panel."
        )
        hint.setWordWrap(True)
        outer.addWidget(hint)

        outer.addWidget(QtWidgets.QLabel("Live Results:"))

        self.summary_text = QtWidgets.QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFont(QtGui.QFont("Courier", 8))
        self.summary_text.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        outer.addWidget(self.summary_text)

        self.status_label = QtWidgets.QLabel("Ready")
        outer.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_type_changed(self, rf_checked: bool):
        self.rf_widget.setVisible(rf_checked)
        self.cnn_widget.setVisible(not rf_checked)

    def _on_cnn_mode_changed(self, _index):
        """When mode changes, auto-select the mean/std version from settings.py."""
        mode = self.cnn_mode_combo.currentData()
        if mode is None:
            return
        version = self._settings_versions.get(mode, '')
        if not version:
            return
        for i in range(self.mean_std_combo.count()):
            data = self.mean_std_combo.itemData(i)
            if data is None:
                continue
            mean_stem = Path(data[0]).stem  # e.g. imu_mean_v13
            if f'_v{version}' in mean_stem:
                self.mean_std_combo.setCurrentIndex(i)
                return

    def _load_rf_model(self):
        model_path = self.model_combo.currentData()
        if not model_path:
            return
        try:
            model = load(model_path)
            self.current_model = model
            self.model_status_label.setText(f"✓ Loaded: {Path(model_path).stem}")
            self.model_status_label.setStyleSheet("color: #00AA00;")
            self.model_loaded.emit(model)
        except Exception as e:
            self.model_status_label.setText(f"✗ Error: {str(e)[:60]}")
            self.model_status_label.setStyleSheet("color: #FF0000;")

    def _browse_cnn_model(self):
        from utilities.dialog_utils import get_open_file_name

        start_dir = ""
        filepath, _ = get_open_file_name(
            self, "Select CNN Weights (.pth)", start_dir,
            "PyTorch weights (*.pth);;All files (*)"
        )
        if filepath:
            p = Path(filepath)
            short = f"{p.stem}  [{p.parent.name}]"
            self.cnn_model_combo.addItem(short, filepath)
            self.cnn_model_combo.setCurrentIndex(self.cnn_model_combo.count() - 1)

    def _browse_mean_file(self):
        from utilities.dialog_utils import get_open_file_name

        start_dir = str(MEAN_STD_DIR) if MEAN_STD_DIR.exists() else ""
        filepath, _ = get_open_file_name(
            self, "Select Mean .npy file", start_dir,
            "NumPy files (*.npy);;All files (*)"
        )
        if filepath:
            mean_p = Path(filepath)
            std_p = mean_p.parent / mean_p.name.replace("mean", "std")
            display = f"{mean_p.stem} (custom)"
            self.mean_std_combo.addItem(display, (str(mean_p), str(std_p)))
            self.mean_std_combo.setCurrentIndex(self.mean_std_combo.count() - 1)

    def _load_cnn_model(self):
        model_path = self.cnn_model_combo.currentData()
        if not model_path:
            self.cnn_status_label.setText("No weights file selected")
            self.cnn_status_label.setStyleSheet("color: #FF6600;")
            return

        mode = self.cnn_mode_combo.currentData()
        mean_std_data = self.mean_std_combo.currentData()

        self.cnn_status_label.setText("Loading CNN weights…")
        self.cnn_status_label.setStyleSheet("color: yellow;")
        QtWidgets.QApplication.processEvents()

        try:
            from utilities.cnn_paths import ensure_on_path
            ensure_on_path()
            from cnn_wrapper import CNNModelWrapper
            import settings as _s

            # Reload CNN/settings.py from disk so edits (INPUT_AMOUNT, class_names,
            # data_version, OUTPUT_AMOUNT, …) apply on the next Load without restarting
            # the UI. Without this, `settings` stays cached in sys.modules and the
            # architecture files (`from settings import …`) keep the stale values.
            try:
                _s = importlib.reload(_s)
            except Exception as exc:
                print(f"Could not reload CNN settings: {exc}")
            # Refresh the mode → mean/std version map from the same fresh file.
            self._load_cnn_settings()

            # Architecture files and class names driven by settings.py — no auto-detection
            _mode_config = {
                "normal": {
                    "class_names":  list(_s.class_names),
                    "data_version": str(_s.data_version),
                    "architecture": ARCHITECTURE_FILES["normal"],
                },
                "hand": {
                    "class_names":  list(_s.CLASS_NAMES_HAND),
                    "data_version": str(_s.DATA_VERSION_HAND),
                    "architecture": ARCHITECTURE_FILES["hand"],
                },
                "ground": {
                    "class_names":  list(_s.CLASS_NAMES_GROUND),
                    "data_version": str(_s.DATA_VERSION_GROUND),
                    "architecture": ARCHITECTURE_FILES["ground"],
                },
            }
            cfg = _mode_config.get(mode, _mode_config["normal"])
            override_class_names = cfg["class_names"]
            data_version = cfg["data_version"]
            architecture_file = cfg["architecture"]

            explicit_mean = None
            explicit_std = None
            if mean_std_data is not None:
                explicit_mean, explicit_std = mean_std_data
                stem = Path(explicit_mean).stem
                if '_v' in stem:
                    data_version = stem.split('_v')[-1]

            wrapper = CNNModelWrapper.from_model_file(
                model_path=model_path,
                data_version=data_version,
                device='cpu',
                class_names=override_class_names,
                mean_path=explicit_mean,
                std_path=explicit_std,
                architecture_file=architecture_file,
            )

            model_name = Path(model_path).stem
            classes_str = '/'.join(wrapper.get_class_names())
            mode_label = mode if mode else "normal"

            self.cnn_status_label.setText(
                f"✓ {model_name} ({mode_label})\n  [{classes_str}]"
            )
            self.cnn_status_label.setStyleSheet("color: #00AA00;")
            self.cnn_model_loaded.emit(wrapper)

        except Exception as e:
            self.cnn_status_label.setText(f"✗ {str(e)[:80]}")
            self.cnn_status_label.setStyleSheet("color: #FF0000;")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Manager binding & helpers
    # ------------------------------------------------------------------

    def bind_manager(self, manager):
        self.model_loaded.connect(manager.on_model_loaded)
        self.browse_model_requested.connect(manager.on_browse_model)
        self.cnn_model_loaded.connect(manager.on_cnn_model_loaded)

    def set_summary_text(self, text: str):
        self.summary_text.setPlainText(text)

    def set_status_text(self, text: str):
        self.status_label.setText(text)

    def add_model_path(self, model_path: str):
        """Add a browsed RF model path to the combo box."""
        p = Path(model_path)
        self.model_combo.addItem(p.stem, model_path)
        self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
