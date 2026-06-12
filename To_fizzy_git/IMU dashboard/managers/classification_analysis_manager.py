"""Classification analysis workflow manager."""

from __future__ import annotations
import csv
import importlib
import json
import logging
import os 
import warnings
from math import nan
from pathlib import Path
from typing import Any
import time
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore
from joblib import load
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from .csv_manager import CSVManager
from utilities.data_windower import apply_hanning_to_window, compute_fft, extract_fft_features, extract_time_domain_features
from utilities.live_data_windower import extract_features
import utilities.live_random_forest as live_random_forest
from utilities.live_random_forest import _extract_expected_feature_names, _align_features
from utilities.fizzy_config import CLASSIFICATION_LABELS, LABEL_COLORS

logger = logging.getLogger(__name__)

# Sentinel "true label" values for per-window evaluation overrides.
EVAL_IGNORED = "ignored"      # window excluded from the metrics
EVAL_NO_MATCH = "no match"    # window is wrong, with no specific matching true class

# Add CNN support. The CNN wrapper, settings and architecture files live under
# DataAndModels/CNN/Model Running; utilities.cnn_paths resolves their location
# and puts them on sys.path on import.
from utilities.cnn_paths import ARCHITECTURE_FILES

try:
    from cnn_wrapper import CNNModelWrapper
    import settings as _cnn_settings
    CNN_AVAILABLE = True
except ImportError as e:
    CNN_AVAILABLE = False
    _cnn_settings = None
    logger.warning(f"CNN wrapper not available - CNN models cannot be loaded: {e}")

class ClassificationAnalysisManager:
    """Owns classification analysis state and behavior."""

    def __init__(self, main_window):
        self.main = main_window
        self.classification_model = None
        self.classification_models = []
        
        # State
        self.analyse_data = []
        self.analyse_current_index = 0
        self.analyse_sampling_rate = 104
        self.analyse_markers = []
        self.analyse_gap_ranges = []
        self.true_labels = {}
        self.label_metadata_path = ""
        self.model_accuracy_summary = {}
        self.model_accuracy_text = ""

        # Reuse the main plot widget so classification-analysis plots stay in the same UI box.
        self.windowed_plot_widget = self.main.plot_widget
        
        self.predictions = []  # List of dicts with window bounds and per-model predictions
        # Per-window effective true label used for evaluation. Each value is a class
        # name, EVAL_IGNORED, or EVAL_NO_MATCH. Seeded from the loaded true labels and
        # editable per window via the evaluation toggles / true-class dropdown.
        self.window_true_overrides = {}
        # Per-model prediction timings, gathered during prediction and reused when
        # metrics are recomputed against (possibly overridden) evaluations.
        self.model_prediction_times = {}
        # When True, windows marked 'ignored' are counted as correct in the metrics
        # instead of being excluded (a convenience so they need not be set by hand).
        self.treat_ignored_as_correct = False
        self.full_plot_regions = []
        # Prediction comparison text boxes, tracked as (plot, text_item, mid_x) so they
        # can be re-pinned to the top of the view whenever the y-range changes.
        self.full_plot_text_items = []
        self._label_signals_connected = False
        self.model_status_base_text = "No model loaded"
        
        # Evaluation metrics - will be expanded per model
        self.model_metrics = {}  # {model_name: {accuracy, precision, recall, f1, confusion_matrix, prediction_times, ...}}

    def init_windowed_plots(self):
        widget: Any = self.windowed_plot_widget

        self.windowed_angle_plot = widget.addPlot(title="Current Window - Angles (deg)")
        self.windowed_angle_plot.showGrid(x=True, y=True)
        self.windowed_angle_plot.addLegend()
        self.windowed_curve_r = self.windowed_angle_plot.plot(pen=(255, 100, 100), name="roll")
        self.windowed_curve_p = self.windowed_angle_plot.plot(pen=(100, 255, 100), name="pitch")
        self.windowed_curve_y = self.windowed_angle_plot.plot(pen=(100, 150, 255), name="yaw")
        self.windowed_angle_plot.setLabel("left", "angle", units="deg")
        self.windowed_angle_plot.setLabel("bottom", "Time", units="s")

        widget.nextRow()

        self.windowed_acc_plot = widget.addPlot(title="Current Window - Accelerations (g)")
        self.windowed_acc_plot.showGrid(x=True, y=True)
        self.windowed_acc_plot.addLegend()
        self.windowed_curve_ax = self.windowed_acc_plot.plot(pen=(255, 255, 0), name="acc_x")
        self.windowed_curve_ay = self.windowed_acc_plot.plot(pen=(255, 0, 255), name="acc_y")
        self.windowed_curve_az = self.windowed_acc_plot.plot(pen=(0, 255, 255), name="acc_z")
        self.windowed_acc_plot.setLabel("left", "acceleration", units="g")
        self.windowed_acc_plot.setLabel("bottom", "Time", units="s")

        widget.nextRow()

        self.windowed_acc_mag_plot = widget.addPlot(title="Current Window - Acceleration Magnitude (g)")
        self.windowed_acc_mag_plot.showGrid(x=True, y=True)
        self.windowed_acc_mag_plot.addLegend()
        self.windowed_curve_acc_mag = self.windowed_acc_mag_plot.plot(pen=(200, 200, 100), name="acc_mag")
        self.windowed_acc_mag_plot.setLabel("left", "acc_mag", units="g")
        self.windowed_acc_mag_plot.setLabel("bottom", "Time", units="s")

        widget.nextRow()

        self.windowed_gyro_plot = widget.addPlot(title="Current Window - Angular Velocity (rad/s)")
        self.windowed_gyro_plot.showGrid(x=True, y=True)
        self.windowed_gyro_plot.addLegend()
        self.windowed_curve_wx = self.windowed_gyro_plot.plot(pen=(200, 100, 255), name="gyro_x")
        self.windowed_curve_wy = self.windowed_gyro_plot.plot(pen=(100, 200, 255), name="gyro_y")
        self.windowed_curve_wz = self.windowed_gyro_plot.plot(pen=(150, 255, 200), name="gyro_z")
        self.windowed_gyro_plot.setLabel("left", "angular_velocity", units="rad/s")
        self.windowed_gyro_plot.setLabel("bottom", "Time", units="s")

        widget.nextRow()

        self.windowed_gyro_mag_plot = widget.addPlot(title="Current Window - Angular Velocity Magnitude (rad/s)")
        self.windowed_gyro_mag_plot.showGrid(x=True, y=True)
        self.windowed_gyro_mag_plot.addLegend()
        self.windowed_curve_gyro_mag = self.windowed_gyro_mag_plot.plot(pen=(150, 200, 200), name="gyro_mag")
        self.windowed_gyro_mag_plot.setLabel("left", "gyro_mag", units="rad/s")
        self.windowed_gyro_mag_plot.setLabel("bottom", "Time", units="s")

    def set_windowed_graph_visibility_state(self, visibility_state, enabled=True):
        """Show or hide the current-window plots according to the shared toggle state."""
        self.windowed_angle_plot.setVisible(enabled and bool(visibility_state.get("angle_xyz", False)))
        self.windowed_acc_plot.setVisible(enabled and bool(visibility_state.get("acc_xyz", False)))
        self.windowed_acc_mag_plot.setVisible(enabled and bool(visibility_state.get("acc_mag", False)))
        self.windowed_gyro_plot.setVisible(enabled and bool(visibility_state.get("gyro_xyz", False)))
        self.windowed_gyro_mag_plot.setVisible(enabled and bool(visibility_state.get("gyro_mag", False)))

    def set_windowed_graph_visibility(self, graph_key, visible):
        """Show or hide a single current-window graph group."""
        graph_map = {
            "angle_xyz": self.windowed_angle_plot,
            "acc_xyz": self.windowed_acc_plot,
            "acc_mag": self.windowed_acc_mag_plot,
            "gyro_xyz": self.windowed_gyro_plot,
            "gyro_mag": self.windowed_gyro_mag_plot,
        }
        plot = graph_map.get(graph_key)
        if plot is not None:
            plot.setVisible(bool(visible))

    def _load_selected_models(self, model_sources):
        loaded_models = []
        failures = []

        # Read CNN options from the panel (mode and mean/std selection)
        panel = getattr(self.main, 'classification_analysis_panel', None)
        cnn_mode = panel.get_cnn_mode() if panel else "normal"
        cnn_mean_std = panel.get_cnn_mean_std() if panel else None

        # Reload CNN/settings.py from disk so edits to INPUT_AMOUNT / OUTPUT_AMOUNT /
        # data_version / class_names take effect without restarting the UI. Without this,
        # `settings` stays cached in sys.modules and the architecture files (which do
        # `from settings import INPUT_AMOUNT`) keep reading the stale value.
        global _cnn_settings
        if _cnn_settings is not None:
            try:
                _cnn_settings = importlib.reload(_cnn_settings)
            except Exception as exc:
                logger.warning("Could not reload CNN settings: %s", exc)

        # Architecture files and settings driven directly from CNN settings.py
        _s = _cnn_settings
        _mode_config = {
            "normal": {
                "class_names":   list(_s.class_names) if _s else ['idle', 'shake', 'tap', 'drop', 'lift', 'kick'],
                "data_version":  str(_s.data_version) if _s else '10',
                "architecture":  ARCHITECTURE_FILES["normal"],
            },
            "hand": {
                "class_names":   list(_s.CLASS_NAMES_HAND) if _s else ['idle', 'tap', 'shake', 'drop'],
                "data_version":  str(_s.DATA_VERSION_HAND) if _s else '17',
                "architecture":  ARCHITECTURE_FILES["hand"],
            },
            "ground": {
                "class_names":   list(_s.CLASS_NAMES_GROUND) if _s else ['idle', 'lift', 'kick'],
                "data_version":  str(_s.DATA_VERSION_GROUND) if _s else '11',
                "architecture":  ARCHITECTURE_FILES["ground"],
            },
        }
        _active_cfg = _mode_config.get(cnn_mode, _mode_config["normal"])
        cnn_class_names   = _active_cfg["class_names"]
        cnn_data_version  = _active_cfg["data_version"]
        cnn_arch_file     = _active_cfg["architecture"]

        cnn_explicit_mean = cnn_mean_std[0] if cnn_mean_std else None
        cnn_explicit_std  = cnn_mean_std[1] if cnn_mean_std else None

        for source in model_sources:
            try:
                if isinstance(source, (str, os.PathLike)):
                    model_path = os.fspath(source)
                    model_path_obj = Path(model_path)

                    # Detect model type by file extension
                    if model_path_obj.suffix == '.pth':
                        # CNN model
                        if not CNN_AVAILABLE:
                            failures.append(f"{model_path}: CNN support not available")
                            continue

                        # Use data_version from settings.py unless user explicitly picked a mean/std file
                        data_version = cnn_data_version
                        if cnn_explicit_mean:
                            stem = Path(cnn_explicit_mean).stem  # e.g. imu_mean_v13
                            if '_v' in stem:
                                data_version = stem.split('_v')[-1]

                        try:
                            model_object = CNNModelWrapper.from_model_file(
                                model_path=model_path,
                                data_version=data_version,
                                device='cpu',
                                class_names=cnn_class_names,
                                mean_path=cnn_explicit_mean,
                                std_path=cnn_explicit_std,
                                architecture_file=cnn_arch_file,
                            )
                            model_name = f"{model_path_obj.stem} [CNN]"
                            loaded_models.append({
                                "name": model_name,
                                "path": model_path,
                                "model": model_object,
                                "type": "cnn",
                                "data_version": data_version
                            })
                        except Exception as e:
                            failures.append(f"{model_path}: {e}")
                            
                    elif model_path_obj.suffix in ['.joblib', '.pkl']:
                        # Random Forest or scikit-learn model
                        model_object = load(model_path)
                        # Predictions run one window (single row) at a time on the GUI
                        # thread. With n_jobs=-1 sklearn spawns a fresh joblib pool per
                        # call, so the spawn overhead dwarfs the work and freezes the UI.
                        # Force single-threaded: far faster for single-row predict and it
                        # also silences the sklearn.utils.parallel "delayed" warnings.
                        try:
                            final_est = model_object
                            if hasattr(final_est, "steps") and final_est.steps:
                                final_est = final_est.steps[-1][1]
                            if hasattr(final_est, "n_jobs"):
                                final_est.n_jobs = 1
                            # Model was saved with verbose>0, which prints a joblib
                            # progress block on every predict. Silence it.
                            if hasattr(final_est, "verbose"):
                                final_est.verbose = 0
                        except Exception:
                            logger.debug("Could not set n_jobs=1 on %s", model_path, exc_info=True)
                        model_name = model_path_obj.stem
                        loaded_models.append({
                            "name": model_name,
                            "path": model_path,
                            "model": model_object,
                            "type": "sklearn"
                        })
                    else:
                        failures.append(f"{model_path}: Unsupported file type {model_path_obj.suffix}")
                else:
                    # Model object passed directly
                    model_object = source
                    model_name = getattr(getattr(model_object, "__class__", None), "__name__", "Model")
                    model_type = "cnn" if isinstance(model_object, CNNModelWrapper) else "sklearn"
                    loaded_models.append({
                        "name": model_name,
                        "path": None,
                        "model": model_object,
                        "type": model_type
                    })
            except Exception as exc:
                failures.append(f"{source}: {exc}")

        self.classification_models = loaded_models
        self.classification_model = loaded_models[0]["model"] if loaded_models else None

        if loaded_models:
            # Prefix with P1/P2/... so the legend matches the comparison boxes on the graphs.
            model_names = [f"P{idx}: {entry['name']}" for idx, entry in enumerate(loaded_models, start=1)]
            preview = " | ".join(model_names[:3])
            if len(model_names) > 3:
                preview = f"{preview} | +{len(model_names) - 3} more"
            status_text = f"{len(loaded_models)} model(s) ready: {preview}"
            status_style = "color: #00FF00;"
            if failures:
                status_text = f"{status_text} | {len(failures)} failed"
                status_style = "color: #FFAA00;"
        else:
            status_text = "No models loaded"
            status_style = "color: #FF6600;"

        self.main.classification_analysis_panel.model_status_label.setText(status_text)
        self.main.classification_analysis_panel.model_status_label.setStyleSheet(status_style)
        self.model_status_base_text = status_text

        if failures:
            logger.warning("Some models failed to load: %s", " | ".join(failures))

    def on_model_loaded(self, model_sources):
        if model_sources is None:
            self.classification_models = []
            self.classification_model = None
            self.main.classification_analysis_panel.model_status_label.setText("No model loaded")
            self.main.classification_analysis_panel.model_status_label.setStyleSheet("color: #FF6600;")
            self.model_status_base_text = "No model loaded"
            return

        if isinstance(model_sources, (str, os.PathLike)):
            model_sources = [model_sources]
        self._load_selected_models(model_sources)

    def _get_window_size_samples(self):
        window_size = self.main.control_panel.get_window_size()
        window_unit = self.main.control_panel.get_window_unit().lower()
        if window_unit == "seconds":
            return max(1, int(round(window_size * self.analyse_sampling_rate)))
        return max(1, int(round(window_size)))

    def _get_overlap_percent(self):
        return self.main.control_panel.get_overlap_percent()

    def _parse_label_metadata_file(self, metadata_path):
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(metadata_path)

        with open(metadata_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            raise ValueError("Label metadata file must contain a JSON object")

        window_size = data.get("window_size", data.get("window_length"))
        window_unit = data.get("window_unit", "Samples")
        overlap_value = data.get("window_overlap", data.get("window_overlap_percent", data.get("overlap_percent", 50)))
        if window_size is None:
            raise ValueError("Label metadata file is missing window_size/window_length")

        overlap_value = float(overlap_value)
        if overlap_value <= 1.0:
            overlap_percent = int(round(overlap_value * 100.0))
        else:
            overlap_percent = int(round(overlap_value))

        labeled_windows = data.get("labeled_windows", {})
        if not isinstance(labeled_windows, dict):
            raise ValueError("labeled_windows must be a JSON object")

        labels = {}
        for window_idx_str, label_value in labeled_windows.items():
            if isinstance(label_value, dict):
                label = label_value.get("label", label_value.get("class"))
            else:
                label = label_value
            if label is None or str(label).strip() == "":
                continue
            # "Unknown" windows are not real ground truth; treat them like
            # unlabeled windows so they're excluded from accuracy/metrics.
            if str(label).strip().lower() in {"unknown", "unkown"}:
                continue
            labels[int(window_idx_str)] = str(label)

        return {
            "csv_file": data.get("csv_file", ""),
            "window_size": window_size,
            "window_unit": window_unit,
            "window_overlap": overlap_percent,
            "labels": labels,
        }

    def _apply_true_labels_to_graphs(self):
        self.main.graph_manager.labels.clear()
        self.main.graph_manager.labels.update(self.true_labels)
        self.main.graph_manager.clear_label_overlays()
        # The manual label is shown inside the combined comparison box drawn by
        # draw_colored_regions (the "C:" line), so suppress the overlay's own text
        # to avoid a second box. The colored region itself stays.
        self.main.graph_manager.label_overlay_show_text = False
        if self.true_labels:
            self.main.graph_manager.update_label_overlays()

    def _get_true_label_for_window_idx(self, window_idx):
        return self.true_labels.get(window_idx)

    def _calculate_advanced_metrics(self):
        """Calculate confusion matrix, precision, recall, F1, and average prediction time for each model."""
        for model_name, metrics_data in self.model_metrics.items():
            true_labels = metrics_data.get("true_labels", [])
            predicted_labels = metrics_data.get("predicted_labels", [])
            prediction_times = metrics_data.get("prediction_times", [])
            
            # Calculate average prediction time
            if prediction_times:
                metrics_data["avg_prediction_time_ms"] = np.mean(prediction_times)
                metrics_data["min_prediction_time_ms"] = np.min(prediction_times)
                metrics_data["max_prediction_time_ms"] = np.max(prediction_times)
            else:
                metrics_data["avg_prediction_time_ms"] = 0.0
                metrics_data["min_prediction_time_ms"] = 0.0
                metrics_data["max_prediction_time_ms"] = 0.0
            
            # Calculate confusion matrix and other metrics if we have labels
            if true_labels and predicted_labels and len(true_labels) == len(predicted_labels):
                try:
                    # Get unique labels in order they appear
                    unique_labels = sorted(list(set(true_labels) | set(predicted_labels)))
                    
                    # Calculate confusion matrix
                    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
                    metrics_data["confusion_matrix"] = cm
                    metrics_data["class_labels"] = unique_labels
                    
                    # Calculate precision with macro averaging (unweighted mean across classes)
                    try:
                        precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
                        metrics_data["precision"] = precision * 100.0  # Convert to percentage
                    except Exception as e:
                        logger.warning(f"Could not calculate precision for {model_name}: {e}")
                        metrics_data["precision"] = None
                    
                    # Calculate recall with macro averaging
                    try:
                        recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
                        metrics_data["recall"] = recall * 100.0
                    except Exception as e:
                        logger.warning(f"Could not calculate recall for {model_name}: {e}")
                        metrics_data["recall"] = None
                    
                    # Calculate F1 score with macro averaging
                    try:
                        f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
                        metrics_data["f1"] = f1 * 100.0
                    except Exception as e:
                        logger.warning(f"Could not calculate F1 for {model_name}: {e}")
                        metrics_data["f1"] = None
                        
                except Exception as e:
                    logger.error(f"Error calculating metrics for {model_name}: {e}")
                    metrics_data["confusion_matrix"] = None
                    metrics_data["precision"] = None
                    metrics_data["recall"] = None
                    metrics_data["f1"] = None
            else:
                metrics_data["confusion_matrix"] = None
                metrics_data["precision"] = None
                metrics_data["recall"] = None
                metrics_data["f1"] = None

    def _update_model_accuracy_summary(self):
        has_evaluated = any(stats.get("total", 0) for stats in self.model_accuracy_summary.values())
        if not has_evaluated:
            self.model_accuracy_text = "No evaluated windows (load true labels or mark windows)"
            self.main.classification_analysis_panel.model_status_label.setText(self.model_status_base_text)
            return

        parts = []
        best_model = None
        best_accuracy = -1.0
        for model_name, stats in sorted(self.model_accuracy_summary.items()):
            total = stats.get("total", 0)
            correct = stats.get("correct", 0)
            accuracy = (100.0 * correct / total) if total else 0.0
            
            # Build metric string with additional metrics if available
            metric_parts = [f"Acc: {accuracy:.1f}%"]
            
            metrics = self.model_metrics.get(model_name, {})
            if metrics.get("precision") is not None:
                metric_parts.append(f"Prec: {metrics['precision']:.1f}%")
            if metrics.get("recall") is not None:
                metric_parts.append(f"Rec: {metrics['recall']:.1f}%")
            if metrics.get("f1") is not None:
                metric_parts.append(f"F1: {metrics['f1']:.1f}%")
            if metrics.get("avg_prediction_time_ms"):
                metric_parts.append(f"Speed: {metrics['avg_prediction_time_ms']:.2f}ms")
            
            metric_str = " | ".join(metric_parts)
            
            # CHANGED: Find abbreviation for model_name instead of printing full name
            model_abbr = model_name
            for idx, entry in enumerate(self.classification_models, start=1):
                if entry['name'] == model_name:
                    model_abbr = f"P{idx}"
                    break
            parts.append(f"{model_abbr}: {metric_str} ({correct}/{total})")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name

        self.model_accuracy_text = " | ".join(parts)

        if best_model is not None:
            self.main.classification_analysis_panel.model_status_label.setText(
                f"{self.model_status_base_text} | Best: {best_model} ({best_accuracy:.1f}%)"
            )

    def apply_selected_label_metadata_file(self):
        metadata_path = self.label_metadata_path or self.main.classification_analysis_panel.get_label_metadata_path()
        if not metadata_path or not self.analyse_data:
            return False

        try:
            metadata = self._parse_label_metadata_file(metadata_path)
        except Exception as exc:
            self.main.classification_analysis_panel.set_status(f"Error loading label metadata: {str(exc)}", "color: #FF0000;")
            logger.exception("Failed to load label metadata from %s", metadata_path)
            return False

        self.label_metadata_path = metadata_path
        self.true_labels = metadata["labels"]
        self.main.control_panel.set_window_settings(
            metadata["window_size"],
            metadata["window_unit"],
            metadata["window_overlap"],
        )
        self.main.graph_manager.set_window_size(metadata["window_size"], metadata["window_unit"])
        self._apply_true_labels_to_graphs()
        self.recompute_predictions()

        csv_file = metadata.get("csv_file", "")
        loaded_csv = self.main.classification_analysis_panel.get_file_path()
        if csv_file and loaded_csv and os.path.basename(csv_file) != os.path.basename(loaded_csv):
            self.main.classification_analysis_panel.set_status(
                f"Loaded labels from {os.path.basename(metadata_path)} (saved for {os.path.basename(csv_file)})",
                "color: #FFAA00;",
            )
        else:
            self.main.classification_analysis_panel.set_status(
                f"Loaded labels from {os.path.basename(metadata_path)}",
                "color: #00FF00;",
            )
        return True

    def on_browse_model(self):
        from utilities.dialog_utils import get_open_file_name

        model_path, _ = get_open_file_name(
            self.main,
            "Select Model",
            "",
            "Model files (*.joblib *.pkl *.pth);;Random Forest (*.joblib *.pkl);;CNN Models (*.pth);;All files (*)",
        )
        if model_path:
            self.main.classification_analysis_panel.add_model_path(model_path)

    def on_browse_csv(self):
        from utilities.dialog_utils import get_open_file_name

        filepath, _ = get_open_file_name(
            self.main,
            "Select CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if filepath:
            self.main.classification_analysis_panel.set_file_path(filepath)

    def on_label_metadata_path_changed(self, path):
        self.label_metadata_path = path.strip()
        if self.analyse_data and self.label_metadata_path:
            self.apply_selected_label_metadata_file()
        elif self.analyse_data and not self.label_metadata_path:
            self.true_labels = {}
            self._apply_true_labels_to_graphs()
            self.recompute_predictions()

    def on_load_csv(self):
        filepath = self.main.classification_analysis_panel.get_file_path()
        if not filepath:
            self.main.classification_analysis_panel.set_status("No file selected", "color: #FF6600;")
            return

        try:
            selected_model_paths = self.main.classification_analysis_panel.get_selected_model_paths()
            if not selected_model_paths:
                self.main.classification_analysis_panel.set_status("No models selected", "color: #FF6600;")
                return

            self.main.classification_analysis_panel.model_status_label.setText("Loading models...")
            self.main.classification_analysis_panel.model_status_label.setStyleSheet("color: yellow;")
            QtWidgets.QApplication.processEvents()
            self.on_model_loaded(selected_model_paths)

            if not self.classification_models:
                self.main.classification_analysis_panel.set_status("No models could be loaded", "color: #FF0000;")
                return

            rows = CSVManager.load_csv_file(filepath, apply_threshold=False)

            if not rows:
                self.main.classification_analysis_panel.set_status("CSV is empty", "color: #FF0000;")
                return

            self.analyse_data.clear()
            self.analyse_data.extend(rows)
            self.analyse_data = [
                row for row in sorted(self.analyse_data, key=lambda item: item.get("timestamp", 0.0))
                if row.get("timestamp") is not None and not np.isnan(row.get("timestamp", np.nan))
            ]

            self.analyse_markers = [
                (row["timestamp"], row.get("markers"))
                for row in self.analyse_data
                if row.get("markers") and row.get("markers") not in {"none", "nm"}
            ]
            self.analyse_gap_ranges = []
            for idx in range(len(self.analyse_data) - 1):
                current_time = self.analyse_data[idx]["timestamp"]
                next_time = self.analyse_data[idx + 1]["timestamp"]
                if (next_time - current_time) > 0.1:
                    self.analyse_gap_ranges.append((current_time, next_time))

            if len(self.analyse_data) < 2:
                self.main.classification_analysis_panel.set_status("Not enough valid data", "color: #FF0000;")
                return

            duration = self.analyse_data[-1]["timestamp"] - self.analyse_data[0]["timestamp"]
            if duration > 0:
                self.analyse_sampling_rate = len(self.analyse_data) / duration

            self.main.classification_analysis_panel.set_slider_range(len(self.analyse_data) - 1)
            self.main.classification_analysis_panel.enable_slider(True)
            self.main.classification_analysis_panel.set_slider_value(0)
            self.main.classification_analysis_panel.set_status(f"Loaded {len(self.analyse_data)} rows", "color: #00FF00;")

            # Plot full data on main graph
            timestamps = np.array([d["timestamp"] for d in self.analyse_data])
            self.main.graph_manager.curve_r.setData(timestamps, [d["roll"] for d in self.analyse_data])
            self.main.graph_manager.curve_p.setData(timestamps, [d["pitch"] for d in self.analyse_data])
            self.main.graph_manager.curve_y.setData(timestamps, [d["yaw"] for d in self.analyse_data])
            self.main.graph_manager.curve_ax.setData(timestamps, [d["acc_x"] for d in self.analyse_data])
            self.main.graph_manager.curve_ay.setData(timestamps, [d["acc_y"] for d in self.analyse_data])
            self.main.graph_manager.curve_az.setData(timestamps, [d["acc_z"] for d in self.analyse_data])
            self.main.graph_manager.curve_wx.setData(timestamps, [d["gyro_x"] for d in self.analyse_data])
            self.main.graph_manager.curve_wy.setData(timestamps, [d["gyro_y"] for d in self.analyse_data])
            self.main.graph_manager.curve_wz.setData(timestamps, [d["gyro_z"] for d in self.analyse_data])

            self.on_marker_visibility_changed(self.main.classification_analysis_panel.show_markers_enabled())
            self.on_gap_visibility_changed(self.main.classification_analysis_panel.show_gaps_enabled())

            if not self.apply_selected_label_metadata_file():
                self.true_labels = {}
                self._apply_true_labels_to_graphs()
                self.recompute_predictions()
            self.on_slider_moved(0)

        except Exception as exc:
            self.main.classification_analysis_panel.set_status(f"Error: {exc}", "color: #FF0000;")

    def _get_detailed_metrics_display(self):
        """Format detailed metrics (confusion matrix, precision, recall, F1, speed) for display."""
        if not self.model_metrics:
            return []
        
        lines = []
        lines.append("Detailed Metrics by Model:")
        
        for model_name, metrics in sorted(self.model_metrics.items()):
            lines.append("")
            
            # CHANGED: Use abbreviation instead of full model name
            model_abbr = model_name
            for idx, entry in enumerate(self.classification_models, start=1):
                if entry['name'] == model_name:
                    model_abbr = f"P{idx}"
                    break
            lines.append(f"  {model_abbr}:")
            
            # Prediction speed metrics
            if metrics.get("avg_prediction_time_ms"):
                avg_time = metrics.get("avg_prediction_time_ms", 0)
                min_time = metrics.get("min_prediction_time_ms", 0)
                max_time = metrics.get("max_prediction_time_ms", 0)
                lines.append(f"    Speed: avg={avg_time:.2f}ms | min={min_time:.2f}ms | max={max_time:.2f}ms")
            
            # Precision, Recall, F1
            metric_display = []
            if metrics.get("precision") is not None:
                metric_display.append(f"Precision={metrics['precision']:.1f}%")
            if metrics.get("recall") is not None:
                metric_display.append(f"Recall={metrics['recall']:.1f}%")
            if metrics.get("f1") is not None:
                metric_display.append(f"F1={metrics['f1']:.1f}%")
            
            if metric_display:
                lines.append(f"    {' | '.join(metric_display)}")
            
            # Confusion Matrix
            cm = metrics.get("confusion_matrix")
            class_labels = metrics.get("class_labels", [])
            if cm is not None and len(class_labels) > 0:
                lines.append(f"    Confusion Matrix ({' vs '.join(class_labels)}):")
                # Format confusion matrix nicely. Every cell (header labels,
                # row labels and counts) uses the same column width so columns
                # line up. Each data row starts with "<label> | ", so the header
                # is padded by that same row-label field + separator width.
                indent = "    "
                max_label_len = max(len(str(label)) for label in class_labels) if class_labels else 5
                separator = " | "
                row_label_field = max_label_len + len(separator)

                # Header row (offset so column headers sit above their counts)
                header = indent + " " * row_label_field + " ".join(
                    f"{str(label):>{max_label_len}}" for label in class_labels
                )
                lines.append(header)

                # Data rows
                for i, true_label in enumerate(class_labels):
                    row_str = f"{str(true_label):<{max_label_len}}{separator}" + " ".join(
                        f"{int(cm[i, j]):>{max_label_len}}" for j in range(len(class_labels))
                    )
                    lines.append(indent + row_str)
        
        return lines

    def _on_graph_window_clicked(self, window_idx):
        """Jump the scrubber to a window clicked directly on a graph.

        The scrubber/current index tracks the window's end timestamp (matching the
        keyboard window navigation in jump_to_window).
        """
        gm = self.main.graph_manager
        if not self.analyse_data or window_idx < 0 or window_idx >= len(gm.window_indices):
            return
        gm.current_window_idx = window_idx
        _, window_end = gm.window_indices[window_idx]
        closest_idx = 0
        min_diff = float("inf")
        for data_idx, data_point in enumerate(self.analyse_data):
            diff = abs(data_point["timestamp"] - window_end)
            if diff < min_diff:
                min_diff = diff
                closest_idx = data_idx
        self.main.classification_analysis_panel.set_slider_value(closest_idx)
        self.on_slider_moved(closest_idx)

    def on_slider_moved(self, value):
        self.analyse_current_index = value
        self.main.graph_manager.window_click_callback = self._on_graph_window_clicked
        if not self.analyse_data:
            return

        ts = self.analyse_data[value]["timestamp"]
        self.main.classification_analysis_panel.set_timestamp_label(f"Time: {ts:.2f} s")
        
        window_size = self._get_window_size_samples()
        half_window = window_size // 2
        
        start_idx = max(0, value - half_window)
        end_idx = min(len(self.analyse_data), value + half_window + 1)
        
        if end_idx - start_idx < 2:
            return

        window_start_ts = self.analyse_data[start_idx]["timestamp"]
        window_end_ts = self.analyse_data[end_idx - 1]["timestamp"]

        current_window_idx = -1
        if self.main.graph_manager.window_indices:
            min_diff = float("inf")
            for idx, (_, window_end) in enumerate(self.main.graph_manager.window_indices):
                diff = abs(window_end - ts)
                if diff < min_diff:
                    min_diff = diff
                    current_window_idx = idx
            if current_window_idx != -1:
                self.main.graph_manager.current_window_idx = current_window_idx
                window_start_ts, window_end_ts = self.main.graph_manager.window_indices[current_window_idx]

        # Handle drawing the main window highlight
        try:
            self.main.graph_manager.set_analysis_window_region(window_start_ts, window_end_ts)
        except Exception:
            pass

        # Update windowed plots
        window_data = [d for d in self.analyse_data if window_start_ts <= d["timestamp"] <= window_end_ts]
        t = np.array([d["timestamp"] for d in window_data])
        
        self.windowed_curve_r.setData(t, [d["roll"] for d in window_data])
        self.windowed_curve_p.setData(t, [d["pitch"] for d in window_data])
        self.windowed_curve_y.setData(t, [d["yaw"] for d in window_data])
        
        self.windowed_curve_ax.setData(t, [d["acc_x"] for d in window_data])
        self.windowed_curve_ay.setData(t, [d["acc_y"] for d in window_data])
        self.windowed_curve_az.setData(t, [d["acc_z"] for d in window_data])
        self.windowed_curve_acc_mag.setData(t, [np.sqrt(d["acc_x"] ** 2 + d["acc_y"] ** 2 + d["acc_z"] ** 2) for d in window_data])
        
        self.windowed_curve_wx.setData(t, [d["gyro_x"] for d in window_data])
        self.windowed_curve_wy.setData(t, [d["gyro_y"] for d in window_data])
        self.windowed_curve_wz.setData(t, [d["gyro_z"] for d in window_data])
        self.windowed_curve_gyro_mag.setData(t, [np.sqrt(d["gyro_x"] ** 2 + d["gyro_y"] ** 2 + d["gyro_z"] ** 2) for d in window_data])

        self.windowed_angle_plot.enableAutoRange()
        self.windowed_acc_plot.enableAutoRange()
        self.windowed_acc_mag_plot.enableAutoRange()
        self.windowed_gyro_plot.enableAutoRange()
        self.windowed_gyro_mag_plot.enableAutoRange()

        if len(window_data) < 2:
            self.main.classification_analysis_panel.set_info_text("Current window has fewer than 2 samples.")
            return

        raw_window_dict = {
            key: np.array([d[key] for d in window_data])
            for key in window_data[0].keys()
            if key != "markers"
        }
        time_features = extract_time_domain_features([raw_window_dict])
        windowed = apply_hanning_to_window(window_data)
        freq_features = []
        if windowed:
            windowed_data = windowed[0]
            fft_window = compute_fft([windowed_data])
            freq_features = extract_fft_features(
                fft_window,
                exclude_keys={"timestamp", "label", "roll", "pitch", "yaw"},
            )

        info_lines = [
            f"Window Center Index: {value}",
            f"Window Range: {window_start_ts:.2f}s - {window_end_ts:.2f}s",
            f"Samples: {len(window_data)} | Duration: {window_end_ts - window_start_ts:.3f}s",
        ]

        if current_window_idx != -1 and self.main.graph_manager.window_indices:
            info_lines.append(
                f"Window Index: {current_window_idx + 1}/{len(self.main.graph_manager.window_indices)}"
            )

        # ADDED: Model legend near the top of the results text
        if self.classification_models:
            info_lines.append("")
            info_lines.append("Model Legend:")
            for idx, entry in enumerate(self.classification_models, start=1):
                info_lines.append(f"  P{idx}: {entry['name']}")

        effective_true = self.window_true_overrides.get(current_window_idx) if current_window_idx != -1 else None
        original_true = self._get_true_label_for_window_idx(current_window_idx) if current_window_idx != -1 else None
        if effective_true == EVAL_IGNORED:
            if self.treat_ignored_as_correct:
                info_lines.append("Evaluation: ignored → counted as correct")
            else:
                info_lines.append("Evaluation: ignored (excluded from metrics)")
        elif effective_true == EVAL_NO_MATCH:
            info_lines.append("Evaluation: wrong (true class: no match)")
        elif effective_true is not None:
            info_lines.append(f"True Class: {effective_true}")
        if original_true is not None and self._norm(effective_true) != self._norm(original_true):
            info_lines.append(f"  (original true class: {original_true})")
        elif original_true is None and effective_true not in (None, EVAL_IGNORED):
            info_lines.append("  (originally ignored — no true label provided)")

        matching_prediction = None
        for prediction in self.predictions:
            if prediction["start"] <= ts <= prediction["end"]:
                matching_prediction = prediction
                break

        info_lines.append("")
        
        if matching_prediction:
            info_lines.append("Model Comparison:")
            for model_idx, model_prediction in enumerate(matching_prediction["predictions"], start=1):
                # CHANGED: Replaced P{model_idx} ({model_prediction['model']}) with just P{model_idx}
                info_lines.append(
                    f"  P{model_idx}: {model_prediction['class']} ({model_prediction['certainty']:.1f}%)"
                )
            if matching_prediction.get("display_prediction"):
                display_prediction = matching_prediction["display_prediction"]
                
                # CHANGED: Map full name back to its P{idx} abbreviation
                model_abbr = display_prediction['model']
                for idx, entry in enumerate(self.classification_models, start=1):
                    if entry['name'] == display_prediction['model']:
                        model_abbr = f"P{idx}"
                        break
                info_lines.append("")
                info_lines.append(
                    f"Displayed Region: {model_abbr} -> {display_prediction['class']} ({display_prediction['certainty']:.1f}%)"
                )
        else:
            info_lines.append("No prediction overlap for this exact time.")

        if self.model_accuracy_text:
            info_lines.extend(["", f"Accuracy Summary: {self.model_accuracy_text}"])
        
        # Add detailed metrics if available
        detailed_metrics_lines = self._get_detailed_metrics_display()
        if detailed_metrics_lines:
            info_lines.extend([""] + detailed_metrics_lines)

        info_lines.extend([
            "",
            f"Point: roll: {self.analyse_data[value]['roll']:7.2f}° | pitch: {self.analyse_data[value]['pitch']:7.2f}° | yaw: {self.analyse_data[value]['yaw']:7.2f}°",
            f"acc_x: {self.analyse_data[value]['acc_x']:6.2f} | acc_y: {self.analyse_data[value]['acc_y']:6.2f} | acc_z: {self.analyse_data[value]['acc_z']:6.2f}",
            f"gyro_x: {self.analyse_data[value]['gyro_x']:6.3f} | gyro_y: {self.analyse_data[value]['gyro_y']:6.3f} | gyro_z: {self.analyse_data[value]['gyro_z']:6.3f}",
        ])

        if time_features and freq_features:
            time_feat = time_features[0]
            freq_feat = freq_features[0]
            sensors = ["acc", "gyro"]
            axes = ["x", "y", "z"]
            feature_types = ["mean", "var", "max", "min", "RMS", "energy"]
            info_lines.extend(["", "Time Domain:"])
            for sensor in sensors:
                info_lines.append(f"{sensor.upper()}:")
                for axis in axes:
                    key = f"{sensor}_{axis}"
                    features_for_axis = []
                    for feat_type in feature_types:
                        feat_key = f"{key}_{feat_type}"
                        if feat_key in time_feat:
                            features_for_axis.append(f"{feat_type}={time_feat[feat_key]:.2f}")
                    if features_for_axis:
                        info_lines.append(f"  {axis.upper()}: {' | '.join(features_for_axis)}")

            info_lines.append("")
            info_lines.append("Frequency Domain:")
            for sensor in sensors:
                info_lines.append(f"{sensor.upper()}:")
                for axis in axes:
                    key = f"{sensor}_{axis}"
                    features_for_axis = []
                    for feat_key in freq_feat:
                        if feat_key.startswith(key + "_"):
                            feat_type = feat_key[len(key) + 1 :]
                            features_for_axis.append(f"{feat_type}={freq_feat[feat_key]:.2f}")
                    if features_for_axis:
                        info_lines.append(f"  {axis.upper()}: {' | '.join(features_for_axis)}")

        self.main.classification_analysis_panel.set_info_text("\n".join(info_lines))
        self._update_evaluation_controls()

    def on_overlap_changed(self, overlap_percent):
        if self.analyse_data:
            self.main.classification_analysis_panel.set_status(
                "Window settings changed. Press Load to apply.",
                "color: #FF6600;",
            )

    def on_window_size_changed(self, *args):
        if self.analyse_data:
            self.main.classification_analysis_panel.set_status(
                "Window settings changed. Press Load to apply.",
                "color: #FF6600;",
            )

    def on_marker_visibility_changed(self, enabled):
        self.main.graph_manager.clear_marker_lines()
        if not enabled:
            return
        for timestamp, label in self.analyse_markers:
            try:
                self.main.graph_manager.add_marker_line(timestamp, label)
            except Exception:
                pass

    def on_gap_visibility_changed(self, enabled):
        self.main.graph_manager.clear_gap_regions()
        if not enabled:
            return
        self.main.graph_manager.highlight_data_gaps(self.analyse_data, gap_threshold=0.1)

    def jump_to_window(self, forward=True):
        if not self.main.graph_manager.window_indices:
            return

        start_idx = self.main.graph_manager.current_window_idx
        if start_idx < 0:
            start_idx = 0

        step = 1 if forward else -1
        idx = start_idx + step

        while 0 <= idx < len(self.main.graph_manager.window_indices):
            self.main.graph_manager.current_window_idx = idx
            _, next_end = self.main.graph_manager.window_indices[idx]
            closest_idx = 0
            min_diff = float("inf")
            for data_idx, data_point in enumerate(self.analyse_data):
                diff = abs(data_point["timestamp"] - next_end)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = data_idx
            self.main.classification_analysis_panel.set_slider_value(closest_idx)
            self.on_slider_moved(closest_idx)
            return

            idx += step

    def _extract_features_for_classification(self, window_data, model=None):
        """Extract features from window_data for classification.
        
        Args:
            window_data: List of sample dictionaries with IMU data
            model: Optional scikit-learn model to align features for backward compatibility
        
        Returns:
            Dictionary of extracted features, or None on error.
            If model is provided, features are aligned to the model's expected columns.
        """
        rolls = []
        pitches = []
        yaws = []
        acc_xs = []
        acc_ys = []
        acc_zs = []
        gyro_xs = []
        gyro_ys = []
        gyro_zs = []

        for d in window_data:
            rolls.append(d["roll"])
            pitches.append(d["pitch"])
            yaws.append(d["yaw"])
            acc_xs.append(d["acc_x"])
            acc_ys.append(d["acc_y"])
            acc_zs.append(d["acc_z"])
            gyro_xs.append(d["gyro_x"])
            gyro_ys.append(d["gyro_y"])
            gyro_zs.append(d["gyro_z"])

        data_dict = {
            "roll": np.array(rolls), "pitch": np.array(pitches), "yaw": np.array(yaws),
            "acc_x": np.array(acc_xs), "acc_y": np.array(acc_ys), "acc_z": np.array(acc_zs),
            "gyro_x": np.array(gyro_xs), "gyro_y": np.array(gyro_ys), "gyro_z": np.array(gyro_zs)
        }
        
        # Extract both time-domain and frequency-domain features
        try:
            features_dict = extract_features(data_dict)
            
            # If a model is provided, align features to match the model's expected columns
            # This ensures backward compatibility with older trained models that may have different features
            if model is not None:
                expected_feature_names = _extract_expected_feature_names(model)
                if expected_feature_names is not None:
                    # Use the alignment function to handle missing/extra features
                    # Missing features are filled with 0.0 (neutral value)
                    aligned_df = _align_features(features_dict, expected_feature_names, fill_value=0.0)
                    # Convert back to dict for consistency with the rest of the code
                    features_dict = aligned_df.iloc[0].to_dict()
            
            return features_dict
        except Exception as exc:
            self.main.classification_analysis_panel.set_status(f"Feature extraction error: {str(exc)[:50]}", "color: #FF6600;")
            logger.exception(f"Feature extraction failed: {exc}")
            return None
    
    def _extract_cnn_features_for_classification(self, window_data, cnn_model):
        """
        Extract features for CNN model.
        
        Args:
            window_data: Window of IMU data samples
            cnn_model: CNNModelWrapper instance (to determine input channels)
        
        Returns a numpy array of shape (input_channels, window_size) where
        input_channels is determined by the model (6, 7, 9, etc.)
        """
        acc_xs = []
        acc_ys = []
        acc_zs = []
        gyro_xs = []
        gyro_ys = []
        gyro_zs = []
        motor_inputs = []
        acc_mags = []
        gyro_mags = []

        for d in window_data:
            acc_x = d["acc_x"]
            acc_y = d["acc_y"]
            acc_z = d["acc_z"]
            gyro_x = d["gyro_x"]
            gyro_y = d["gyro_y"]
            gyro_z = d["gyro_z"]
            
            acc_xs.append(acc_x)
            acc_ys.append(acc_y)
            acc_zs.append(acc_z)
            gyro_xs.append(gyro_x)
            gyro_ys.append(gyro_y)
            gyro_zs.append(gyro_z)
            motor_inputs.append(d.get("motor_input", 0.0))
            acc_mags.append(np.sqrt(acc_x**2 + acc_y**2 + acc_z**2))
            gyro_mags.append(np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2))

        # Build feature array based on model's expected input channels
        input_channels = cnn_model.input_channels
        
        channel_names = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        feature_list = [
            acc_xs, acc_ys, acc_zs,
            gyro_xs, gyro_ys, gyro_zs,
        ]
        
        # Add optional features based on model's input channel count
        if input_channels >= 7:
            feature_list.append(motor_inputs)
            channel_names.append('motor_input')
        if input_channels >= 8:
            feature_list.append(acc_mags)
            channel_names.append('acc_magnitude')
        if input_channels >= 9:
            feature_list.append(gyro_mags)
            channel_names.append('gyro_magnitude')
        
        # Trim to exact number of channels needed
        feature_list = feature_list[:input_channels]
        channel_names = channel_names[:input_channels]
        
        # Log the channels being extracted (once per prediction run to avoid spam)
        logger.debug(f"CNN feature extraction - {input_channels} channels: {channel_names}")
        
        features = np.array(feature_list, dtype=np.float32)  # Shape: (input_channels, actual_window_size)

        # Resample to the model's expected window size if the control panel window differs.
        # The CNN was trained on a fixed window length (e.g. 64 samples); feeding a different
        # length causes a shape mismatch in the FC layer.
        target_size = cnn_model.window_size
        actual_size = features.shape[1]
        if actual_size != target_size:
            x_orig   = np.linspace(0, 1, actual_size)
            x_target = np.linspace(0, 1, target_size)
            features = np.array(
                [np.interp(x_target, x_orig, ch) for ch in features],
                dtype=np.float32,
            )

        return features

    # --- Per-window evaluation overrides ---
    @staticmethod
    def _norm(value):
        """Normalize a label for case/whitespace-insensitive comparison."""
        return str(value).strip().lower() if value is not None else None

    def _prediction_for_window_idx(self, window_idx):
        for prediction in self.predictions:
            if prediction["window_idx"] == window_idx:
                return prediction
        return None

    def _display_class_for_window(self, window_idx):
        """The class of the highest-certainty (displayed) prediction, or None."""
        prediction = self._prediction_for_window_idx(window_idx)
        if not prediction:
            return None
        display = prediction.get("display_prediction")
        return display["class"] if display else None

    def _init_window_evaluations(self):
        """Seed effective true labels from the loaded JSON (else 'ignored')."""
        overrides = {}
        for prediction in self.predictions:
            window_idx = prediction["window_idx"]
            original = self.true_labels.get(window_idx)
            overrides[window_idx] = original if original is not None else EVAL_IGNORED
        self.window_true_overrides = overrides

    def _evaluation_class_list(self):
        """Classes offered in the True class dropdown: every observed/known class
        plus the 'no match' and 'ignored' sentinels. Deduplicated case-insensitively
        (predictions win the casing) so e.g. 'Tap' and 'tap' don't both appear."""
        canonical = {}  # normalized -> display string

        def add(cls):
            if not cls or cls == "Error":
                return
            key = self._norm(cls)
            if key not in canonical:
                canonical[key] = str(cls)

        for prediction in self.predictions:
            for model_prediction in prediction.get("predictions", []):
                add(model_prediction.get("class"))
        for value in self.true_labels.values():
            add(value)
        for cls in CLASSIFICATION_LABELS:
            add(cls)

        ordered = [canonical[key] for key in sorted(canonical.keys())]
        return ordered + [EVAL_NO_MATCH, EVAL_IGNORED]

    def _eval_state(self, effective_true, predicted_class):
        """Map an effective true label + prediction to a toggle state."""
        if effective_true == EVAL_IGNORED or effective_true is None:
            return "ignored"
        if effective_true == EVAL_NO_MATCH:
            return "wrong"
        if self._norm(effective_true) == self._norm(predicted_class):
            return "correct"
        return "wrong"

    def _recompute_metrics_from_evaluations(self):
        """Rebuild accuracy/precision/recall/F1/confusion matrices from the stored
        predictions and the current per-window evaluations (no re-prediction)."""
        self.model_accuracy_summary = {entry["name"]: {"correct": 0, "total": 0} for entry in self.classification_models}
        self.model_metrics = {}
        for entry in self.classification_models:
            name = entry["name"]
            self.model_metrics[name] = {
                "correct": 0,
                "total": 0,
                "prediction_times": list(self.model_prediction_times.get(name, [])),
                "true_labels": [],
                "predicted_labels": [],
                "precision": None,
                "recall": None,
                "f1": None,
                "confusion_matrix": None,
            }

        for prediction in self.predictions:
            effective_true = self.window_true_overrides.get(prediction["window_idx"], EVAL_IGNORED)
            is_ignored = effective_true == EVAL_IGNORED or effective_true is None
            if is_ignored and not self.treat_ignored_as_correct:
                continue  # ignored windows are excluded from the metrics
            for model_prediction in prediction.get("predictions", []):
                predicted_class = model_prediction.get("class")
                if predicted_class == "Error":
                    continue
                name = model_prediction.get("model")
                metrics = self.model_metrics.get(name)
                if metrics is None:
                    continue
                pred_norm = self._norm(predicted_class)
                # Ignored windows (when counted) are treated as correct for every
                # model, so the effective true label matches each model's prediction.
                true_norm = pred_norm if is_ignored else self._norm(effective_true)
                stats = self.model_accuracy_summary.setdefault(name, {"correct": 0, "total": 0})
                stats["total"] += 1
                metrics["total"] += 1
                metrics["true_labels"].append(true_norm)
                metrics["predicted_labels"].append(pred_norm)
                if pred_norm == true_norm:
                    stats["correct"] += 1
                    metrics["correct"] += 1

        self._calculate_advanced_metrics()
        self._update_model_accuracy_summary()

    def _apply_evaluation_change(self):
        """Recompute metrics + redraw after a window's evaluation was overridden."""
        self._recompute_metrics_from_evaluations()
        self.draw_colored_regions()
        if self.analyse_data:
            self.on_slider_moved(self.analyse_current_index)

    def on_window_eval_toggled(self, state):
        """Handle a Correct/Wrong/Ignored toggle for the current window."""
        window_idx = self.main.graph_manager.current_window_idx
        if window_idx is None or window_idx < 0:
            return
        predicted_class = self._display_class_for_window(window_idx)
        if predicted_class is None:
            return
        original = self.true_labels.get(window_idx)
        if state == "ignored":
            effective = EVAL_IGNORED
        elif state == "correct":
            effective = predicted_class
        else:  # wrong: keep the original true class if it disagrees, else 'no match'
            if original is not None and self._norm(original) != self._norm(predicted_class):
                effective = original
            else:
                effective = EVAL_NO_MATCH
        self.window_true_overrides[window_idx] = effective
        self._apply_evaluation_change()

    def on_window_true_class_changed(self, class_name):
        """Handle a True class dropdown change for the current window."""
        window_idx = self.main.graph_manager.current_window_idx
        if window_idx is None or window_idx < 0:
            return
        text = str(class_name).strip()
        if text.lower() == EVAL_IGNORED:
            effective = EVAL_IGNORED
        elif text.lower() == EVAL_NO_MATCH:
            effective = EVAL_NO_MATCH
        else:
            effective = text
        self.window_true_overrides[window_idx] = effective
        self._apply_evaluation_change()

    def on_treat_ignored_correct_changed(self, enabled):
        """Toggle whether 'ignored' windows count as correct, then refresh metrics."""
        self.treat_ignored_as_correct = bool(enabled)
        if self.predictions:
            self._apply_evaluation_change()

    def _update_evaluation_controls(self):
        """Sync the toggle buttons, dropdown, and 'original true' note to the current window."""
        panel = self.main.classification_analysis_panel
        window_idx = self.main.graph_manager.current_window_idx
        predicted_class = self._display_class_for_window(window_idx) if (window_idx is not None and window_idx >= 0) else None
        if window_idx is None or window_idx < 0 or predicted_class is None:
            panel.set_window_evaluation(False, None, None, "")
            return

        effective_true = self.window_true_overrides.get(window_idx, EVAL_IGNORED)
        state = self._eval_state(effective_true, predicted_class)

        original = self.true_labels.get(window_idx)
        default_effective = original if original is not None else EVAL_IGNORED
        if self._norm(effective_true) != self._norm(default_effective):
            if original is not None:
                original_text = f"Altered — original true class: {original}"
            else:
                original_text = "Altered — originally ignored (no true label provided)"
        else:
            original_text = ""

        panel.set_window_evaluation(True, state, str(effective_true), original_text)

    def save_altered_labels(self):
        """Save a new label-metadata JSON from the current (overridden) true labels.

        Mirrors the label-metadata format so the file can be loaded again as ground
        truth. Ignored windows are omitted (they have no true label). The suggested
        filename includes 'altered_labels'.
        """
        from utilities.dialog_utils import get_save_file_name

        panel = self.main.classification_analysis_panel
        if not self.predictions:
            panel.set_status("No windows to save", "color: #FF6600;")
            return

        labeled_windows = {}
        for window_idx, effective in sorted(self.window_true_overrides.items()):
            if effective is None or effective == EVAL_IGNORED:
                continue  # ignored windows carry no true label
            labeled_windows[str(window_idx)] = effective

        if not labeled_windows:
            panel.set_status("No true labels to save (all windows ignored)", "color: #FF6600;")
            return

        csv_path = panel.get_file_path()
        base = os.path.splitext(os.path.basename(csv_path))[0] if csv_path else "labels"
        default_name = f"{base}_altered_labels.json"
        default_dir = os.path.dirname(csv_path) if csv_path else ""
        default_path = os.path.join(default_dir, default_name) if default_dir else default_name

        filepath, _ = get_save_file_name(
            self.main,
            "Save Altered Labels",
            default_path,
            "JSON Files (*.json);;All Files (*)",
        )
        if not filepath:
            return
        if not filepath.lower().endswith(".json"):
            filepath += ".json"

        data = {
            "csv_file": csv_path,
            "window_size": self.main.control_panel.get_window_size(),
            "window_unit": self.main.control_panel.get_window_unit(),
            "window_overlap": self.main.control_panel.get_overlap_percent(),
            "labeled_windows": labeled_windows,
        }

        try:
            with open(filepath, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
            panel.set_status(
                f"Saved altered labels ({len(labeled_windows)} windows) to {os.path.basename(filepath)}",
                "color: #00FF00;",
            )
        except Exception as exc:
            panel.set_status(f"Error saving altered labels: {exc}", "color: #FF0000;")
            logger.exception("Failed to save altered labels to %s", filepath)

    def recompute_predictions(self):
        self.predictions.clear()
        self.clear_colored_regions()

        if not self.analyse_data:
            return

        window_samples = self._get_window_size_samples()
        stride_samples = max(1, int(window_samples * (1.0 - (self._get_overlap_percent() / 100.0))))
        window_indices_times = []
        # Prediction timings are collected here; accuracy/precision/etc. are computed
        # afterwards in _recompute_metrics_from_evaluations against the per-window
        # evaluations (which the user can override).
        self.model_prediction_times = {entry["name"]: [] for entry in self.classification_models}

        for start_idx in range(0, len(self.analyse_data) - window_samples + 1, stride_samples):
            end_idx = start_idx + window_samples
            window_data = self.analyse_data[start_idx:end_idx]
            if len(window_data) < window_samples:
                break

            window_indices_times.append((window_data[0]["timestamp"], window_data[-1]["timestamp"]))
            window_idx = len(window_indices_times) - 1
            true_label = self._get_true_label_for_window_idx(window_idx)

            if not self.classification_models:
                continue
            
            # Features will be extracted and aligned per-model as needed
            # This ensures backward compatibility with older trained models
            
            window_predictions = []
            for model_entry in self.classification_models:
                prediction_start_time = time.time()
                try:
                    model_type = model_entry.get("type", "sklearn")
                    if model_type == "cnn":
                        # CNN model prediction
                        cnn_features = self._extract_cnn_features_for_classification(window_data, model_entry["model"])
                        if cnn_features is None:
                            window_predictions.append({
                                "model": model_entry["name"],
                                "class": "Error",
                                "certainty": 0.0,
                            })
                            continue
                        
                        cnn_model = model_entry["model"]
                        pred_labels, confidences = cnn_model.predict_with_confidence(cnn_features[np.newaxis, :, :])
                        
                        # pred_labels contains class indices or -1 for unsure
                        pred_idx = int(pred_labels[0])
                        confidence = float(confidences[0]) * 100.0

                        class_names = cnn_model.get_class_names()
                        if 0 <= pred_idx < len(class_names):
                            predicted_class = class_names[pred_idx]
                        else:
                            # Below confidence threshold (-1): mark as Unsure
                            predicted_class = "Unsure"

                        window_predictions.append({
                            "model": model_entry["name"],
                            "class": predicted_class,
                            "certainty": confidence,
                        })

                        prediction_time_ms = (time.time() - prediction_start_time) * 1000
                        self.model_prediction_times[model_entry["name"]].append(prediction_time_ms)
                    else:
                        # Random Forest or other scikit-learn model
                        # Align features to model's expected columns for backward compatibility
                        model = model_entry["model"]
                        aligned_features = self._extract_features_for_classification(window_data, model=model)
                        if aligned_features is None:
                            window_predictions.append({
                                "model": model_entry["name"],
                                "class": "Error",
                                "certainty": 0.0,
                            })
                            continue
                        
                        result = live_random_forest.predict_and_sort(model, aligned_features)
                        if result:
                            predicted_class, certainty = result[0]
                        else:
                            predicted_class, certainty = "Unsure", 0.0
                        
                        window_predictions.append({
                            "model": model_entry["name"],
                            "class": predicted_class,
                            "certainty": certainty,
                        })

                        prediction_time_ms = (time.time() - prediction_start_time) * 1000
                        self.model_prediction_times[model_entry["name"]].append(prediction_time_ms)
                except Exception as exc:
                    err_msg = str(exc)
                    logger.error("Prediction failed for %s: %s", model_entry["name"], err_msg)
                    self.main.classification_analysis_panel.set_status(
                        f"Prediction error: {err_msg[:80]}", "color: #FF4444;"
                    )
                    window_predictions.append({
                        "model": model_entry["name"],
                        "class": "Error",
                        "certainty": 0.0,
                    })

            successful_predictions = [entry for entry in window_predictions if entry["class"] not in ("Error",)]
            display_prediction = None
            if successful_predictions:
                display_prediction = max(successful_predictions, key=lambda item: item["certainty"])

            self.predictions.append({
                "window_idx": window_idx,
                "start": window_data[0]["timestamp"],
                "end": window_data[-1]["timestamp"],
                "true_class": true_label,
                "predictions": window_predictions,
                "display_prediction": display_prediction,
            })

        if window_indices_times:
            durations = [end - start for start, end in window_indices_times]
            window_width = float(sorted(durations)[len(durations) // 2])
        else:
            window_width = max(0.01, window_samples / self.analyse_sampling_rate if self.analyse_sampling_rate > 0 else float(window_samples))

        self.main.graph_manager.set_analysis_window_width(window_width)
        self.main.graph_manager.set_window_parameters(window_indices_times, window_samples, stride_samples)
        self.main.graph_manager.init_analysis_window_regions()
        self._apply_true_labels_to_graphs()

        # Seed each window's evaluation from the loaded true labels (windows with no
        # true label start ignored), refresh the True class dropdown, then compute the
        # metrics from those evaluations and draw the comparison boxes.
        self._init_window_evaluations()
        self.main.classification_analysis_panel.set_evaluation_classes(self._evaluation_class_list())
        self._recompute_metrics_from_evaluations()
        self.draw_colored_regions()

        # If no explicit error/warning occurred, set to Ready
        if self.predictions and "warning" not in self.main.classification_analysis_panel.analyse_status.text().lower() and "failed" not in self.main.classification_analysis_panel.analyse_status.text().lower():
             self.main.classification_analysis_panel.set_status("Predictions updated", "color: #00FF00;")

        if self.analyse_data:
            self.on_slider_moved(self.analyse_current_index)

    def clear_colored_regions(self):
        for region in self.full_plot_regions:
            try:
                self.main.graph_manager.acc_plot.removeItem(region)
                self.main.graph_manager.angvel_plot.removeItem(region)
            except Exception:
                pass
        self.full_plot_regions.clear()
        self.full_plot_text_items.clear()

    def _label_top_y(self, plot):
        """Y position just below the top edge of the plot's current view."""
        try:
            y_range = plot.getViewBox().viewRange()[1]
            return y_range[1] - 0.02 * (y_range[1] - y_range[0])
        except Exception:
            return 0

    def _reposition_prediction_labels(self, *args):
        """Keep the class comparison boxes pinned to the top of each graph as the
        y-range changes (pan/zoom/auto-range), instead of drifting with the data."""
        for plot, text_item, mid_x in self.full_plot_text_items:
            try:
                text_item.setPos(mid_x, self._label_top_y(plot))
            except Exception:
                pass

    def _connect_label_reposition_signals(self):
        """Connect once so y-range changes re-pin the prediction boxes to the top."""
        if self._label_signals_connected:
            return
        for plot in (self.main.graph_manager.acc_plot, self.main.graph_manager.angvel_plot):
            try:
                plot.getViewBox().sigYRangeChanged.connect(self._reposition_prediction_labels)
            except Exception:
                pass
        self._label_signals_connected = True

    def _get_class_color(self, class_name):
        if class_name in CLASSIFICATION_LABELS:
            return LABEL_COLORS.get(class_name, (200, 200, 200, 30))
        import hashlib
        h = int(hashlib.md5(str(class_name).encode('utf-8')).hexdigest(), 16)
        r = (h & 0xFF0000) >> 16
        g = (h & 0x00FF00) >> 8
        b = (h & 0x0000FF)
        return (r, g, b, 50)

    def draw_colored_regions(self):
        self.clear_colored_regions()
        for p in self.predictions:
            display_prediction = p.get("display_prediction")
            if display_prediction is None:
                continue  # skip windows where every model threw an error
            color = self._get_class_color(display_prediction["class"])
            brush = pg.mkBrush(color)
            pen = pg.mkPen(color)

            region_acc = pg.LinearRegionItem([p["start"], p["end"]], movable=False, brush=brush, pen=pen)
            self.main.graph_manager.acc_plot.addItem(region_acc)
            self.full_plot_regions.append(region_acc)

            region_gyro = pg.LinearRegionItem([p["start"], p["end"]], movable=False, brush=brush, pen=pen)
            self.main.graph_manager.angvel_plot.addItem(region_gyro)
            self.full_plot_regions.append(region_gyro)

            # Build a single comparison box per window:
            #   C:  <manual label>
            #   P1: <model 1 class> <certainty>%
            #   P2: <model 2 class> <certainty>%   (one line per loaded model)
            # Certainty is shown to one decimal for every model type (CNN and RF).
            label_lines = []
            # Use the (possibly overridden) effective true label for the "C:" line and
            # the mismatch test. Ignored windows show no C line; 'no match' and any
            # real class that disagrees with the displayed prediction are mismatches,
            # drawn red and raised in front of matching boxes where they overlap.
            effective_true = self.window_true_overrides.get(p["window_idx"], EVAL_IGNORED)
            displayed_class = display_prediction["class"]
            if effective_true == EVAL_IGNORED or effective_true is None:
                is_mismatch = False
            elif effective_true == EVAL_NO_MATCH:
                is_mismatch = True
                label_lines.append("C: no match")
            else:
                is_mismatch = self._norm(displayed_class) != self._norm(effective_true)
                label_lines.append(f"C: {effective_true}")
            for model_idx, model_prediction in enumerate(p["predictions"], start=1):
                predicted_class = model_prediction["class"]
                if predicted_class == "Error":
                    label_lines.append(f"P{model_idx}: Error")
                else:
                    label_lines.append(
                        f'P{model_idx}: {predicted_class} {model_prediction["certainty"]:.1f}%'
                    )
            label_html = "<br>".join(label_lines)

            text_color = "#FF4444" if is_mismatch else "white"
            # Mismatched boxes get a higher Z so they render in front of the
            # matching boxes wherever they overlap.
            z_value = 2000 if is_mismatch else 1000

            mid_x = (p["start"] + p["end"]) / 2.0
            for plot in (self.main.graph_manager.acc_plot, self.main.graph_manager.angvel_plot):
                text_item = pg.TextItem(
                    html=(
                        f'<div style="background:black;color:{text_color};padding:2px;'
                        f'border-radius:3px;text-align:left;">{label_html}</div>'
                    ),
                    # Top-anchored so the box hangs down from the top edge of the view.
                    anchor=(0.5, 0),
                )
                text_item.setPos(mid_x, self._label_top_y(plot))
                try:
                    text_item.setZValue(z_value)
                except Exception:
                    pass
                # ignoreBounds=True so the box near the top doesn't push auto-range up.
                plot.addItem(text_item, ignoreBounds=True)
                self.full_plot_regions.append(text_item)
                self.full_plot_text_items.append((plot, text_item, mid_x))

        # Keep the boxes pinned to the top of each graph on pan/zoom/auto-range.
        self._connect_label_reposition_signals()
        self._reposition_prediction_labels()
