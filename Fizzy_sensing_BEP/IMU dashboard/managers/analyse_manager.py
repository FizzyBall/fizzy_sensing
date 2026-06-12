"""Analyse workflow manager for the Analyse tab."""

from __future__ import annotations

import csv
import json
import os
from math import nan
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets

from .csv_manager import CSVManager
from utilities.data_windower import apply_hanning_to_window, compute_fft, extract_fft_features, extract_time_domain_features

class AnalyseManager:
    """Owns Analyse-tab state and behavior."""

    def __init__(self, main_window):
        self.main = main_window
        
        # Analyse state
        self.analyse_data = []  # All CSV data for analysis
        self.analyse_current_index = 0  # Current selected timestamp index
        self.analyse_sampling_rate: int = 104  # Default sampling rate in Hz (will be updated from data)
        self.analyse_markers = []  # Parsed (timestamp, label) marker entries in analyse CSV
        self.windowed_marker_items = []  # Marker items currently drawn in windowed plots
        self.analyse_gap_ranges = []  # Parsed (start_ts, end_ts) data gap ranges
        self.windowed_gap_regions = []  # Gap region items currently drawn in windowed plots
        self.analyse_label_metadata_path = ""
        
        # Reuse the main plot widget so analyse plots stay in the same UI box.
        self.windowed_plot_widget = self.main.plot_widget

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

        # Leave a clean row boundary for the next manager that reuses this shared widget.
        self.windowed_plot_widget.nextRow()

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

    def _window_size_to_samples(self, size, unit):
        if unit == "Seconds":
            if self.analyse_sampling_rate > 0:
                return max(1, int(round(size * self.analyse_sampling_rate)))
            return 1
        return max(1, int(round(size)))

    def _window_size_to_seconds(self, size, unit):
        if unit == "Seconds":
            return max(0.01, float(size))
        if self.analyse_sampling_rate > 0:
            return max(0.01, float(size) / self.analyse_sampling_rate)
        return max(0.01, float(size))

    def refresh_analysis_window_state(self):
        if not self.analyse_data:
            return

        size = self.main.control_panel.get_window_size()
        unit = self.main.control_panel.get_window_unit()
        overlap_ratio = self.main.control_panel.get_overlap_percent() / 100.0
        window_indices_times = []

        if unit == "Seconds":
            window_width = self._window_size_to_seconds(size, unit)
            stride_seconds = window_width * (1 - overlap_ratio)
            first_ts = self.analyse_data[0]["timestamp"]
            last_ts = self.analyse_data[-1]["timestamp"]

            start_time = first_ts
            while start_time + window_width <= last_ts:
                start_idx = next((i for i, d in enumerate(self.analyse_data) if d["timestamp"] >= start_time), None)
                if start_idx is None:
                    break

                end_time = start_time + window_width
                end_idx = None
                for j in range(start_idx, len(self.analyse_data)):
                    if self.analyse_data[j]["timestamp"] <= end_time:
                        end_idx = j
                    else:
                        break

                if end_idx is not None:
                    window_indices_times.append((self.analyse_data[start_idx]["timestamp"], self.analyse_data[end_idx]["timestamp"]))
                start_time += stride_seconds

            if window_indices_times:
                sample_counts = [int(round((end - start) * self.analyse_sampling_rate)) if self.analyse_sampling_rate > 0 else 1 for start, end in window_indices_times]
                fft_samples = int(sorted(sample_counts)[len(sample_counts) // 2])
                stride = max(1, int(round(stride_seconds * self.analyse_sampling_rate))) if self.analyse_sampling_rate > 0 else 1
            else:
                fft_samples = self._window_size_to_samples(size, unit)
                stride = max(1, int(fft_samples * (1 - overlap_ratio)))
        else:
            fft_samples = self._window_size_to_samples(size, unit)
            stride = max(1, int(fft_samples * (1 - overlap_ratio)))
            for i in range(0, len(self.analyse_data) - fft_samples + 1, stride):
                start_time = self.analyse_data[i]["timestamp"]
                end_time = self.analyse_data[i + fft_samples - 1]["timestamp"]
                window_indices_times.append((start_time, end_time))

        if unit == "Seconds":
            window_width = self._window_size_to_seconds(size, unit)
        else:
            if window_indices_times:
                durations = [end - start for start, end in window_indices_times]
                window_width = float(sorted(durations)[len(durations) // 2])
            else:
                window_width = self._window_size_to_seconds(size, unit)

        self.main.graph_manager.set_analysis_window_width(window_width)
        self.main.graph_manager.set_window_parameters(window_indices_times, fft_samples, stride)
        self.main.graph_manager.labels.clear()
        self.main.graph_manager.init_analysis_window_regions()
        self.update_analyse_display(self.analyse_current_index)

    def analyse_browse_csv_file(self):
        from utilities.dialog_utils import get_open_file_name

        filepath, _ = get_open_file_name(
            self.main,
            "Select CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if filepath:
            self.main.analyse_panel.set_file_path(filepath)

    def analyse_load_csv(self):
        filepath = self.main.analyse_panel.get_file_path()
        if not filepath:
            self.main.analyse_panel.set_status("Error: No file selected", "color: #FF6666;")
            return

        try:
            self.analyse_data = CSVManager.load_csv_file(filepath, apply_threshold=False)
            if not self.analyse_data:
                self.main.analyse_panel.set_status("Error: No valid data in CSV file", "color: #FF6666;")
                return

            self.main._clear_graph_data()
            try:
                self.main.graph_manager.clear_marker_lines()
            except Exception:
                pass
            self.clear_windowed_marker_lines()
            self.clear_windowed_gap_regions()
            self.clear_labels()

            graph_data = CSVManager.extract_graph_data(self.analyse_data)
            acc_mags = [np.sqrt(ax**2 + ay**2 + az**2) for ax, ay, az in zip(graph_data["acc_xs"], graph_data["acc_ys"], graph_data["acc_zs"])]
            gyro_mags = [np.sqrt(wx**2 + wy**2 + wz**2) for wx, wy, wz in zip(graph_data["gyro_xs"], graph_data["gyro_ys"], graph_data["gyro_zs"])]

            gm = self.main.graph_manager
            self._set_curve(gm.curve_r, graph_data["timestamps"], graph_data["rolls"])
            self._set_curve(gm.curve_p, graph_data["timestamps"], graph_data["pitches"])
            self._set_curve(gm.curve_y, graph_data["timestamps"], graph_data["yaws"])
            self._set_curve(gm.curve_ax, graph_data["timestamps"], graph_data["acc_xs"])
            self._set_curve(gm.curve_ay, graph_data["timestamps"], graph_data["acc_ys"])
            self._set_curve(gm.curve_az, graph_data["timestamps"], graph_data["acc_zs"])
            self._set_curve(gm.curve_wx, graph_data["timestamps"], graph_data["gyro_xs"])
            self._set_curve(gm.curve_wy, graph_data["timestamps"], graph_data["gyro_ys"])
            self._set_curve(gm.curve_wz, graph_data["timestamps"], graph_data["gyro_zs"])

            motor_inputs = [entry.get("motor_input", 0.0) or 0.0 for entry in self.analyse_data]
            self._set_curve(gm.curve_motor_input, graph_data["timestamps"], motor_inputs)
            self._set_curve(gm.curve_roll_sep, graph_data["timestamps"], graph_data["rolls"])
            self._set_curve(gm.curve_pitch_sep, graph_data["timestamps"], graph_data["pitches"])
            self._set_curve(gm.curve_yaw_sep, graph_data["timestamps"], graph_data["yaws"])
            self._set_curve(gm.curve_acc_x_sep, graph_data["timestamps"], graph_data["acc_xs"])
            self._set_curve(gm.curve_acc_y_sep, graph_data["timestamps"], graph_data["acc_ys"])
            self._set_curve(gm.curve_acc_z_sep, graph_data["timestamps"], graph_data["acc_zs"])
            self._set_curve(gm.curve_acc_mag_sep, graph_data["timestamps"], acc_mags)
            self._set_curve(gm.curve_gyro_x_sep, graph_data["timestamps"], graph_data["gyro_xs"])
            self._set_curve(gm.curve_gyro_y_sep, graph_data["timestamps"], graph_data["gyro_ys"])
            self._set_curve(gm.curve_gyro_z_sep, graph_data["timestamps"], graph_data["gyro_zs"])
            self._set_curve(gm.curve_gyro_mag_sep, graph_data["timestamps"], gyro_mags)

            for plot in (
                self.main.graph_manager.angle_plot,
                self.main.graph_manager.acc_plot,
                self.main.graph_manager.angvel_plot,
                self.main.graph_manager.motor_input_plot,
                self.main.graph_manager.angle_roll_plot,
                self.main.graph_manager.angle_pitch_plot,
                self.main.graph_manager.angle_yaw_plot,
                self.main.graph_manager.acc_y_plot,
                self.main.graph_manager.acc_z_plot,
                self.main.graph_manager.acc_mag_plot,
                self.main.graph_manager.gyro_x_plot,
                self.main.graph_manager.gyro_y_plot,
                self.main.graph_manager.gyro_z_plot,
                self.main.graph_manager.gyro_mag_plot,
            ):
                plot.enableAutoRange()

            self.analyse_gap_ranges = self.extract_analyse_gaps(gap_threshold=0.1)
            self.refresh_full_plot_gaps()

            if len(self.analyse_data) > 1:
                total_time = self.analyse_data[-1]["timestamp"] - self.analyse_data[0]["timestamp"]
                num_points = len(self.analyse_data)
                self.main.analyse_sampling_rate = (num_points / total_time) if total_time > 0 else 104
                if not self.apply_selected_label_metadata_file():
                    self.refresh_analysis_window_state()

            self.main.graph_manager.init_analysis_window_regions()

            try:
                self.analyse_markers = self.extract_analyse_markers()
                self.refresh_full_plot_markers()
            except Exception:
                self.analyse_markers = []

            self.main.analyse_panel.set_slider_range(len(self.analyse_data) - 1)
            
            # Populate feature checkboxes dynamically from CSV columns
            if self.analyse_data:
                csv_columns = list(self.analyse_data[0].keys())
                # Remove timestamp from features list as it's always saved
                if "timestamp" in csv_columns:
                    csv_columns.remove("timestamp")
                self.main.analyse_panel.set_feature_options(csv_columns, checked=True)
            
            self.analyse_current_index = 0
            self.main.graph_manager.window_click_callback = self._on_graph_window_clicked
            self.update_analyse_display(0)

            if hasattr(self.main, "_sync_current_window_visibility"):
                self.main._sync_current_window_visibility()

            if self.main.control_panel.is_fft_enabled() and self.analyse_data:
                size = self.main.control_panel.get_window_size()
                unit = self.main.control_panel.get_window_unit()
                fft_sample_count = self._window_size_to_samples(size, unit)
                fft_data = self.analyse_data[: min(fft_sample_count, len(self.analyse_data))]
                if len(fft_data) > 1:
                    self.main.graph_manager.update_fft_plots_from_csv(fft_data)

            status_text = f"Analysing {len(self.analyse_data)} data points from {filepath.split(os.sep)[-1]}"
            self.main.analyse_panel.set_status(status_text, "color: #00FF00;")
        except Exception as exc:
            self.main.analyse_panel.set_status(f"Error: {str(exc)}", "color: #FF6666;")
            print(f"Error loading CSV for analysis: {exc}")

    def on_slider_moved(self, value):
        self.analyse_current_index = value
        self.main.graph_manager.window_click_callback = self._on_graph_window_clicked
        self.update_analyse_display(value)

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
        for i, d in enumerate(self.analyse_data):
            diff = abs(d["timestamp"] - window_end)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        self.main.analyse_panel.set_slider_value(closest_idx)
        self.on_slider_moved(closest_idx)

    def on_save_location_browse(self):
        from utilities.dialog_utils import get_existing_directory

        directory = get_existing_directory(self.main, "Select Directory to Save Labeled Data", "")
        if directory:
            self.main.analyse_panel.set_save_location(directory)
            self.main.analyse_panel.set_status(f"Save location set: {directory}", "color: #00FF00;")

    def on_label_metadata_path_changed(self, path):
        self.analyse_label_metadata_path = path.strip()
        if self.analyse_data and self.analyse_label_metadata_path:
            self.apply_selected_label_metadata_file()

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

        if isinstance(overlap_value, str):
            overlap_value = float(overlap_value)
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
            labels[int(window_idx_str)] = str(label)

        return {
            "csv_file": data.get("csv_file", ""),
            "window_size": window_size,
            "window_unit": window_unit,
            "window_overlap": overlap_percent,
            "labels": labels,
        }

    def apply_selected_label_metadata_file(self):
        metadata_path = self.analyse_label_metadata_path or self.main.analyse_panel.get_label_metadata_path()
        if not metadata_path or not self.analyse_data:
            return False

        try:
            metadata = self._parse_label_metadata_file(metadata_path)
        except Exception as exc:
            self.main.analyse_panel.set_status(f"Error loading label metadata: {str(exc)}", "color: #FF6666;")
            print(f"Error loading label metadata from {metadata_path}: {exc}")
            return False

        self.analyse_label_metadata_path = metadata_path
        self.main.control_panel.set_window_settings(
            metadata["window_size"],
            metadata["window_unit"],
            metadata["window_overlap"],
        )
        self.main.graph_manager.set_window_size(metadata["window_size"], metadata["window_unit"])
        self.refresh_analysis_window_state()

        self.main.graph_manager.labels.clear()
        self.main.graph_manager.labels.update(metadata["labels"])
        # Labeling tab shows the manual label text on overlays.
        self.main.graph_manager.label_overlay_show_text = True
        self.main.graph_manager.update_label_overlays()
        self.update_analyse_display(self.analyse_current_index)

        csv_file = self.main.analyse_panel.get_file_path()
        metadata_csv = metadata.get("csv_file", "")
        if metadata_csv and csv_file:
            metadata_base = os.path.basename(metadata_csv)
            csv_base = os.path.basename(csv_file)
            if metadata_base != csv_base:
                self.main.analyse_panel.set_status(
                    f"Loaded label metadata from {os.path.basename(metadata_path)} (saved for {metadata_base})",
                    "color: #FFCC66;",
                )
                return True

        self.main.analyse_panel.set_status(
            f"Loaded label metadata from {os.path.basename(metadata_path)}",
            "color: #00FF00;",
        )
        return True

    def on_marker_visibility_changed(self, enabled):
        if not enabled:
            self.main.graph_manager.clear_marker_lines()
            self.clear_windowed_marker_lines()
            return
        self.refresh_full_plot_markers()
        if self.analyse_data:
            self.update_analyse_display(self.analyse_current_index)

    def on_gap_visibility_changed(self, enabled):
        if not enabled:
            self.main.graph_manager.clear_gap_regions()
            self.clear_windowed_gap_regions()
            return
        self.refresh_full_plot_gaps()
        if self.analyse_data:
            self.update_analyse_display(self.analyse_current_index)

    def on_window_stats_visibility_changed(self, enabled):
        if self.analyse_data:
            self.update_analyse_display(self.analyse_current_index)

    def on_label_visibility_changed(self, enabled):
        self.main.graph_manager.set_label_overlays_visible(enabled)

    def extract_analyse_markers(self):
        markers = []
        for entry in self.analyse_data:
            marker_label = entry.get("markers")
            if marker_label and marker_label != "none" and marker_label != "nm":
                markers.append((entry.get("timestamp", nan), marker_label))
        return markers

    def refresh_full_plot_markers(self):
        self.main.graph_manager.clear_marker_lines()
        if not self.main.analyse_panel.show_markers_enabled():
            return
        for ts, lbl in self.analyse_markers:
            try:
                self.main.graph_manager.add_marker_line(ts, lbl)
            except Exception:
                pass

    def extract_analyse_gaps(self, gap_threshold=0.1):
        gaps = []
        if not self.analyse_data or len(self.analyse_data) < 2:
            return gaps
        for i in range(len(self.analyse_data) - 1):
            current_time = self.analyse_data[i]["timestamp"]
            next_time = self.analyse_data[i + 1]["timestamp"]
            if (next_time - current_time) > gap_threshold:
                gaps.append((current_time, next_time))
        return gaps

    def refresh_full_plot_gaps(self):
        self.main.graph_manager.clear_gap_regions()
        if not self.main.analyse_panel.show_gaps_enabled():
            return
        self.main.graph_manager.highlight_data_gaps(self.analyse_data, gap_threshold=0.1)

    def _get_marker_color(self, label):
        return {
            "marker_1": (255, 0, 0),
            "marker_2": (0, 0, 255),
            "marker_3": (0, 255, 0),
            "marker_4": (255, 255, 0),
        }.get(str(label).lower(), (200, 200, 200))

    def clear_windowed_marker_lines(self):
        for plot, line, text in self.windowed_marker_items:
            try:
                if text is not None:
                    plot.removeItem(text)
            except Exception:
                pass
            try:
                plot.removeItem(line)
            except Exception:
                pass
        self.windowed_marker_items.clear()

    def update_windowed_marker_lines(self, window_start_ts, window_end_ts):
        self.clear_windowed_marker_lines()
        if not self.main.analyse_panel.show_markers_enabled():
            return

        plots_to_update = [self.windowed_angle_plot, self.windowed_acc_plot, self.windowed_gyro_plot]
        for ts, lbl in self.analyse_markers:
            if not (window_start_ts <= ts <= window_end_ts):
                continue
            pen = pg.mkPen(color=self._get_marker_color(lbl), width=2)
            for plot in plots_to_update:
                try:
                    line = pg.InfiniteLine(angle=90, movable=False, pen=pen)
                    line.setValue(ts)
                    plot.addItem(line)
                    try:
                        y_range = plot.getViewBox().viewRange()[1]
                        y_pos = y_range[1] - 0.05 * (y_range[1] - y_range[0])
                    except Exception:
                        y_pos = 0
                    txt = pg.TextItem(html=f'<div style="background:black;color:white;padding:2px;border-radius:3px;">{lbl}</div>', anchor=(0.5, 1))
                    txt.setPos(ts, y_pos)
                    try:
                        txt.setZValue(1000)
                    except Exception:
                        pass
                    plot.addItem(txt)
                    self.windowed_marker_items.append((plot, line, txt))
                except Exception:
                    pass

    def clear_windowed_gap_regions(self):
        for plot, region, text in self.windowed_gap_regions:
            try:
                if text is not None:
                    plot.removeItem(text)
            except Exception:
                pass
            try:
                plot.removeItem(region)
            except Exception:
                pass
        self.windowed_gap_regions.clear()

    def update_windowed_gap_regions(self, window_start_ts, window_end_ts):
        self.clear_windowed_gap_regions()
        if not self.main.analyse_panel.show_gaps_enabled():
            return

        plots_to_update = [self.windowed_angle_plot, self.windowed_acc_plot, self.windowed_gyro_plot]
        region_pen = pg.mkPen(color=(255, 0, 0), width=0)
        region_brush = pg.mkBrush(color=(255, 0, 0, 30))

        for gap_start, gap_end in self.analyse_gap_ranges:
            if gap_end < window_start_ts or gap_start > window_end_ts:
                continue

            gap_duration = gap_end - gap_start
            label_text = f"Gap ({gap_duration:.3f}s)"
            clipped_start = max(gap_start, window_start_ts)
            clipped_end = min(gap_end, window_end_ts)
            x_mid = 0.5 * (clipped_start + clipped_end)

            for plot in plots_to_update:
                try:
                    region = pg.LinearRegionItem(
                        values=[clipped_start, clipped_end],
                        orientation="vertical",
                        pen=region_pen,
                        brush=region_brush,
                        movable=False,
                    )
                    plot.addItem(region)
                    text_item = None
                    try:
                        y_range = plot.getViewBox().viewRange()[1]
                        y_pos = y_range[1] - 0.05 * (y_range[1] - y_range[0])
                        text_item = pg.TextItem(
                            html=f'<div style="background:black;color:#ff9999;padding:2px;border-radius:3px;">{label_text}</div>',
                            anchor=(0.5, 1),
                        )
                        text_item.setPos(x_mid, y_pos)
                        try:
                            text_item.setZValue(1000)
                        except Exception:
                            pass
                        plot.addItem(text_item)
                    except Exception:
                        text_item = None
                    self.windowed_gap_regions.append((plot, region, text_item))
                except Exception:
                    pass

    def on_overlap_changed(self, overlap_percent):
        if not self.analyse_data:
            return
        try:
            self.refresh_analysis_window_state()
            self.main.analyse_panel.set_status(
                f"Windows recalculated with overlap={overlap_percent}% ({len(self.main.graph_manager.window_indices)} windows)",
                "color: #00FF00;",
            )
        except Exception as exc:
            self.main.analyse_panel.set_status(f"Error updating overlap: {str(exc)}", "color: #FF6666;")
            print(f"Error in on_overlap_changed: {exc}")

    def clear_labels(self):
        self.main.graph_manager.labels.clear()
        self.main.graph_manager.clear_label_overlays()
        self.main.analyse_panel.set_status("All labels cleared", "color: #00FF00;")

    def save_labels_to_json(self):
        if not self.analyse_data or not self.main.graph_manager.labels:
            print("No labels to save.")
            return

        labels_data = {
            "csv_file": self.main.analyse_panel.get_file_path(),
            "window_size": self.main.control_panel.get_window_size(),
            "window_unit": self.main.control_panel.get_window_unit(),
            "window_overlap": self.main.control_panel.get_overlap_percent(),
            "labeled_windows": {},
        }

        for window_idx, label in sorted(self.main.graph_manager.labels.items()):
            labels_data["labeled_windows"][str(window_idx)] = label

        csv_path = self.main.analyse_panel.get_file_path()
        if csv_path:
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            output_file = os.path.join(os.path.dirname(csv_path), f"{base_name}_labels.json")
            with open(output_file, "w", encoding="utf-8") as handle:
                json.dump(labels_data, handle, indent=2)
            print(f"Labels saved to: {output_file}")
            self.main.analyse_panel.set_status("Labels file saved successfully", "color: #00FF00;")

    def jump_to_window(self, forward=True, unlabeled_only=False):
        if not self.main.graph_manager.window_indices:
            return
        start_idx = self.main.graph_manager.current_window_idx
        if start_idx < 0:
            start_idx = 0
        step = 1 if forward else -1
        idx = start_idx + step
        while 0 <= idx < len(self.main.graph_manager.window_indices):
            if idx not in self.main.graph_manager.labels or not unlabeled_only:
                self.main.graph_manager.current_window_idx = idx
                _, next_end = self.main.graph_manager.window_indices[idx]
                closest_idx = 0
                min_diff = float("inf")
                for i, d in enumerate(self.analyse_data):
                    diff = abs(d["timestamp"] - next_end)
                    if diff < min_diff:
                        min_diff = diff
                        closest_idx = i
                self.main.analyse_panel.set_slider_value(closest_idx)
                self.on_slider_moved(closest_idx)
                return
            idx += step

    @staticmethod
    def _finite_xy(x, y):
        """Return x, y as float arrays with non-finite (NaN/Inf) pairs removed.

        The CSV loader stores NaN for missing/empty cells, so a window that lands
        inside a data gap can be all-NaN. pyqtgraph then auto-ranges over all-NaN
        data and raises "Not setting range to [nan, nan]" from inside Qt's linked
        view repaint, which re-fires on every paint and freezes the UI. Dropping
        non-finite points avoids that; an empty result is safe (auto-range is a
        no-op when there is nothing to bound).
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.size == 0 or y_arr.size == 0:
            return x_arr[:0], y_arr[:0]
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        return x_arr[mask], y_arr[mask]

    def _set_curve(self, curve, x, y):
        """setData on a curve after stripping non-finite points (see _finite_xy)."""
        fx, fy = self._finite_xy(x, y)
        curve.setData(fx, fy)

    def update_analyse_display(self, index):
        if not self.analyse_data or index >= len(self.analyse_data):
            return

        data_entry = self.analyse_data[index]
        timestamp = data_entry["timestamp"]
        roll_deg = data_entry["roll"]
        pitch_deg = data_entry["pitch"]
        yaw_deg = data_entry["yaw"]
        acc_x = data_entry["acc_x"]
        acc_y = data_entry["acc_y"]
        acc_z = data_entry["acc_z"]
        gyro_x = data_entry["gyro_x"]
        gyro_y = data_entry["gyro_y"]
        gyro_z = data_entry["gyro_z"]
        motor_input = data_entry.get("motor_input", 0.0) or 0.0

        self.main.analyse_panel.set_timestamp_label(f"{timestamp:.3f} s")
        self.main.info_label.setText(
            f"roll: {roll_deg:7.2f}° | pitch: {pitch_deg:7.2f}° | yaw: {yaw_deg:7.2f}°\n"
            f"acc_x: {acc_x:6.2f} | acc_y: {acc_y:6.2f} | acc_z: {acc_z:6.2f}\n"
            f"gyro_x: {gyro_x:6.3f} | gyro_y: {gyro_y:6.3f} | gyro_z: {gyro_z:6.3f}\n"
            f"motor_input: {motor_input:6.2f}"
        )

        self.main.gizmo.update_acceleration_vectors(acc_x * 1.2, acc_y * 1.2, acc_z * 1.2)
        self.main.gizmo.apply_rotation_transform(roll_deg, pitch_deg, yaw_deg)

        size = self.main.control_panel.get_window_size()
        unit = self.main.control_panel.get_window_unit()
        if unit == "Samples":
            fft_sample_count = self._window_size_to_samples(size, unit)
            start_idx = max(0, index - fft_sample_count + 1)
            window_start_ts = self.analyse_data[start_idx]["timestamp"]
            window_end_ts = timestamp
        else:
            window_end_ts = timestamp
            window_start_ts = timestamp - self.main.graph_manager.analysis_window_width

        self.main.graph_manager.update_scrubber_line_position(timestamp)
        self.main.graph_manager.set_analysis_window_region(window_start_ts, window_end_ts)
        self.update_windowed_marker_lines(window_start_ts, window_end_ts)
        self.update_windowed_gap_regions(window_start_ts, window_end_ts)

        windowed_data = [d for d in self.analyse_data if window_start_ts <= d["timestamp"] <= window_end_ts]
        if windowed_data:
            window_times = [d["timestamp"] for d in windowed_data]
            self._set_curve(self.windowed_curve_r, window_times, [d["roll"] for d in windowed_data])
            self._set_curve(self.windowed_curve_p, window_times, [d["pitch"] for d in windowed_data])
            self._set_curve(self.windowed_curve_y, window_times, [d["yaw"] for d in windowed_data])
            self._set_curve(self.windowed_curve_ax, window_times, [d["acc_x"] for d in windowed_data])
            self._set_curve(self.windowed_curve_ay, window_times, [d["acc_y"] for d in windowed_data])
            self._set_curve(self.windowed_curve_az, window_times, [d["acc_z"] for d in windowed_data])
            self._set_curve(self.windowed_curve_acc_mag, window_times, [np.sqrt(d["acc_x"] ** 2 + d["acc_y"] ** 2 + d["acc_z"] ** 2) for d in windowed_data])
            self._set_curve(self.windowed_curve_wx, window_times, [d["gyro_x"] for d in windowed_data])
            self._set_curve(self.windowed_curve_wy, window_times, [d["gyro_y"] for d in windowed_data])
            self._set_curve(self.windowed_curve_wz, window_times, [d["gyro_z"] for d in windowed_data])
            self._set_curve(self.windowed_curve_gyro_mag, window_times, [np.sqrt(d["gyro_x"] ** 2 + d["gyro_y"] ** 2 + d["gyro_z"] ** 2) for d in windowed_data])
            self.windowed_angle_plot.enableAutoRange()
            self.windowed_acc_plot.enableAutoRange()
            self.windowed_acc_mag_plot.enableAutoRange()
            self.windowed_gyro_plot.enableAutoRange()
            self.windowed_gyro_mag_plot.enableAutoRange()

        min_diff = float("inf")
        closest_idx = -1
        for i, (_, w_end) in enumerate(self.main.graph_manager.window_indices):
            diff = abs(w_end - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        if closest_idx != -1:
            self.main.graph_manager.current_window_idx = closest_idx

        if self.main.control_panel.is_fft_enabled() and self.analyse_data:
            fft_sample_count = self._window_size_to_samples(size, unit)
            start_idx = max(0, index - fft_sample_count + 1)
            fft_data = self.analyse_data[start_idx : index + 1]
            if len(fft_data) > 1:
                self.main.graph_manager.update_fft_plots_from_csv(fft_data)

        if self.main.analyse_panel.show_window_stats_enabled():
            self.update_window_statistics(index)

    def update_window_statistics(self, index):
        if not self.analyse_data or index >= len(self.analyse_data):
            return

        timestamp = self.analyse_data[index]["timestamp"]
        size = self.main.control_panel.get_window_size()
        unit = self.main.control_panel.get_window_unit()
        if unit == "Samples":
            fft_sample_count = self._window_size_to_samples(size, unit)
            start_idx = max(0, index - fft_sample_count + 1)
            window_start = self.analyse_data[start_idx]["timestamp"]
            window_end = timestamp
        else:
            window_end = timestamp
            window_start = timestamp - self.main.graph_manager.analysis_window_width

        window_data = [d for d in self.analyse_data if window_start <= d["timestamp"] <= window_end]
        if len(window_data) < 2:
            return

        raw_window_dict = {key: np.array([d[key] for d in window_data]) for key in window_data[0].keys()}
        time_features = extract_time_domain_features([raw_window_dict])
        windowed = apply_hanning_to_window(window_data)
        if not windowed:
            return
        windowed_data = windowed[0]
        fft_window = compute_fft([windowed_data])
        freq_features = extract_fft_features(
            fft_window,
            exclude_keys={"timestamp", "label", "roll", "pitch", "yaw"},
        )

        if time_features and freq_features:
            time_feat = time_features[0]
            freq_feat = freq_features[0]
            acc_colors = {"x": "#FF6666", "y": "#FF8C42", "z": "#FFA500"}
            gyro_colors = {"x": "#6699FF", "y": "#00BFFF", "z": "#1E90FF"}
            header_color = "#FFEB3B"
            window_color = "#E0E0E0"
            point_data = (
                f"Point: roll: {self.analyse_data[index]['roll']:7.2f}° | "
                f"pitch: {self.analyse_data[index]['pitch']:7.2f}° | "
                f"yaw: {self.analyse_data[index]['yaw']:7.2f}°<br/>"
                f"acc_x: {self.analyse_data[index]['acc_x']:6.2f} | "
                f"acc_y: {self.analyse_data[index]['acc_y']:6.2f} | "
                f"acc_z: {self.analyse_data[index]['acc_z']:6.2f}<br/>"
                f"gyro_x: {self.analyse_data[index]['gyro_x']:6.3f} | "
                f"gyro_y: {self.analyse_data[index]['gyro_y']:6.3f} | "
                f"gyro_z: {self.analyse_data[index]['gyro_z']:6.3f}<br/>"
            )

            html_lines = [f"<span style='color: {window_color}'>Window: {len(window_data)} samples ({window_end - window_start:.3f}s)</span><br/>"]
            
            # Add label information if the current window is labeled
            current_window_idx = self.main.graph_manager.current_window_idx
            if current_window_idx in self.main.graph_manager.labels:
                label = self.main.graph_manager.labels[current_window_idx]
                label_color = "#90EE90"  # Light green for labels
                html_lines.append(f"<span style='color: {label_color}'>Label: {label}</span><br/>")
            
            sensors = ["acc", "gyro"]
            axes = ["x", "y", "z"]
            feature_types = ["mean", "var", "max", "min", "RMS", "energy"]
            for sensor in sensors:
                colors = acc_colors if sensor == "acc" else gyro_colors
                html_lines.append(f"<br/><span style='color: {header_color}'>{sensor.upper()} - Time Domain:</span><br/>")
                for axis in axes:
                    key = f"{sensor}_{axis}"
                    color = colors[axis]
                    features_for_axis = []
                    for feat_type in feature_types:
                        feat_key = f"{key}_{feat_type}"
                        if feat_key in time_feat:
                            val = time_feat[feat_key]
                            features_for_axis.append(f"<span style='color: {color}'>{feat_type}:{val:7.2f}</span>")
                    if features_for_axis:
                        html_lines.append(f"  <span style='color: {color}'>{axis.upper()}:</span> {' | '.join(features_for_axis)}<br/>")

            if freq_features and freq_feat:
                html_lines.append(f"<br/><span style='color: {header_color}'>Frequency Domain:</span><br/>")
                for sensor in sensors:
                    colors = acc_colors if sensor == "acc" else gyro_colors
                    for axis in axes:
                        key = f"{sensor}_{axis}"
                        color = colors[axis]
                        freq_feats_for_axis = []
                        for feat_key in freq_feat:
                            if feat_key.startswith(key + "_"):
                                val = freq_feat[feat_key]
                                feat_type = feat_key[len(key)+1:]
                                freq_feats_for_axis.append(f"<span style='color: {color}'>{feat_type}:{val:7.2f}</span>")
                        if freq_feats_for_axis:
                            html_lines.append(f"  <span style='color: {color}'>{key.upper()}:</span> {' | '.join(freq_feats_for_axis)}<br/>")

            self.main.info_label.setText("<br/>".join([point_data] + html_lines))

    def _save_window_to_csv(self, save_dir, label, label_counter, window_data, csv_headers):
        output_file = os.path.join(save_dir, label, f"{label}{label_counter}.csv")
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers, extrasaction="ignore")
            writer.writeheader()
            for data_point in window_data:
                point_export = data_point.copy()
                if "acc_mag" in csv_headers:
                    point_export["acc_mag"] = (data_point["acc_x"] ** 2 + data_point["acc_y"] ** 2 + data_point["acc_z"] ** 2) ** 0.5
                if "gyro_mag" in csv_headers:
                    point_export["gyro_mag"] = (data_point["gyro_x"] ** 2 + data_point["gyro_y"] ** 2 + data_point["gyro_z"] ** 2) ** 0.5
                writer.writerow(point_export)

    def _build_selected_csv_headers(self, include_class=False):
        if self.analyse_data:
            all_csv_columns = list(self.analyse_data[0].keys())
            selected_features_lower = [f.lower() for f in self.main.analyse_panel.get_selected_features()]
            csv_headers = ["timestamp"] if "timestamp" in all_csv_columns else []
            if include_class:
                csv_headers.append("class")
            filtered_columns = [
                col
                for col in all_csv_columns
                if col.lower() != "timestamp"
                and col.lower() != "class"
                and col.lower() in selected_features_lower
            ]
            csv_headers.extend(filtered_columns)
            if include_class and "class" not in csv_headers:
                csv_headers.insert(1 if csv_headers and csv_headers[0] == "timestamp" else 0, "class")
            return csv_headers

        return ["timestamp", "class"] if include_class else ["timestamp"]

    def _get_labeled_windows_to_export(self, selected_labels):
        windows = []
        for window_idx in sorted(self.main.graph_manager.labels.keys()):
            label = self.main.graph_manager.labels[window_idx]
            if label not in selected_labels:
                continue
            if window_idx >= len(self.main.graph_manager.window_indices):
                continue
            start_time, end_time = self.main.graph_manager.window_indices[window_idx]
            window_data = [d for d in self.analyse_data if start_time <= d["timestamp"] <= end_time]
            if window_data:
                windows.append((window_idx, label, start_time, end_time, window_data))
        return windows

    def on_save_labeled_data(self):
        if not self.analyse_data:
            self.main.analyse_panel.set_status("Error: No data loaded", "color: #FF6666;")
            return

        save_location = self.main.analyse_panel.get_save_location()
        if not save_location:
            self.main.analyse_panel.set_status("Error: No save location selected", "color: #FF6666;")
            return

        try:
            save_dir = save_location
            os.makedirs(save_dir, exist_ok=True)
            selected_features_lower = [f.lower() for f in self.main.analyse_panel.get_selected_features()]
            selected_labels = self.main.analyse_panel.get_selected_labels()
            if not selected_labels:
                self.main.analyse_panel.set_status("Error: No classes selected for saving", "color: #FF6666;")
                return

            label_dirs = {}
            for label in selected_labels:
                dir_path = os.path.join(save_dir, label)
                os.makedirs(dir_path, exist_ok=True)
                label_dirs[label] = dir_path

            save_mode = self.main.analyse_panel.get_save_mode()
            import glob

            label_counters = {}
            for label, dir_path in label_dirs.items():
                if save_mode == "replace":
                    for file_path in glob.glob(os.path.join(dir_path, "*.csv")):
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
                    label_counters[label] = 0
                else:
                    existing = glob.glob(os.path.join(dir_path, f"{label}*.csv"))
                    label_counters[label] = len(existing)

            # Dynamically build csv_headers from selected columns in the loaded CSV
            if self.analyse_data:
                all_csv_columns = list(self.analyse_data[0].keys())
                # Always include timestamp first
                csv_headers = ["timestamp"] if "timestamp" in all_csv_columns else []
                # Filter other columns to only include selected features
                filtered_columns = [col for col in all_csv_columns 
                                  if col.lower() != "timestamp" and col.lower() in selected_features_lower]
                csv_headers.extend(filtered_columns)
            else:
                csv_headers = ["timestamp"]

            files_saved = 0
            for window_idx, label, _, _, window_data in self._get_labeled_windows_to_export(selected_labels):
                label_counters[label] = label_counters.get(label, 0) + 1
                try:
                    self._save_window_to_csv(save_dir, label, label_counters[label], window_data, csv_headers)
                    files_saved += 1
                except Exception as exc:
                    print(f"Error saving window {window_idx}: {exc}")
                    
            # Save the label metadata file
            self.save_labels_to_json()

            summary = f"Saved {files_saved} windows to {save_dir} | "
            for label, count in sorted(label_counters.items()):
                summary += f"{label}: {count} files | "
            self.main.analyse_panel.set_status(summary.rstrip(" |"), "color: #00FF00;")
        except Exception as exc:
            self.main.analyse_panel.set_status(f"Error saving labeled data: {str(exc)}", "color: #FF6666;")
            print(exc)

    def on_save_labeled_csv(self):
        if not self.analyse_data:
            self.main.analyse_panel.set_status("Error: No data loaded", "color: #FF6666;")
            return

        save_location = self.main.analyse_panel.get_save_location()
        if not save_location:
            self.main.analyse_panel.set_status("Error: No save location selected", "color: #FF6666;")
            return

        try:
            os.makedirs(save_location, exist_ok=True)
            selected_labels = self.main.analyse_panel.get_selected_labels()
            if not selected_labels:
                self.main.analyse_panel.set_status("Error: No classes selected for saving", "color: #FF6666;")
                return

            csv_headers = self._build_selected_csv_headers(include_class=True)
            labeled_windows = self._get_labeled_windows_to_export(selected_labels)
            if not labeled_windows:
                self.main.analyse_panel.set_status("Error: No labeled windows matched the selected classes", "color: #FF6666;")
                return

            csv_path = self.main.analyse_panel.get_file_path()
            base_name = os.path.splitext(os.path.basename(csv_path))[0] if csv_path else "labeled_data"
            output_file = os.path.join(save_location, f"{base_name}_labeled.csv")

            save_mode = self.main.analyse_panel.get_save_mode()
            file_exists = os.path.exists(output_file)
            write_header = save_mode == "replace" or not file_exists or os.path.getsize(output_file) == 0
            file_mode = "w" if save_mode == "replace" or not file_exists else "a"

            rows_saved = 0
            with open(output_file, file_mode, newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_headers, extrasaction="ignore")
                if write_header:
                    writer.writeheader()

                for window_idx, label, _, _, window_data in labeled_windows:
                    for data_point in window_data:
                        row = {key: data_point.get(key, "") for key in csv_headers if key != "class"}
                        row["class"] = label
                        writer.writerow(row)
                        rows_saved += 1

            label_counts = {}
            for _, label, _, _, window_data in labeled_windows:
                label_counts[label] = label_counts.get(label, 0) + len(window_data)

            summary = f"Saved {rows_saved} rows to {output_file}"
            if label_counts:
                summary += " | " + " | ".join(f"{label}: {count} rows" for label, count in sorted(label_counts.items()))
            self.main.analyse_panel.set_status(summary, "color: #00FF00;")
        except Exception as exc:
            self.main.analyse_panel.set_status(f"Error saving labeled CSV: {str(exc)}", "color: #FF6666;")
            print(exc)

    def on_clear_labels(self):
        self.clear_labels()
