"""Featurization workflow manager for the Featurization tab."""

from __future__ import annotations

import csv
import os

import pandas as pd
from PyQt6 import QtWidgets

from utilities.data_windower import extract_features_from_single_window_file


class FeaturizationManager:
    """Owns Featurization-tab behavior."""

    def __init__(self, main_window):
        self.main = main_window

    def on_source_browse(self):
        from utilities.dialog_utils import get_existing_directory

        directory = get_existing_directory(
            self.main,
            "Select Labeled Data Parent Folder or Label Subfolder",
            "",
        )
        if directory:
            self.main.featurization_panel.add_source_folder(directory)
            self.refresh_feature_options()
            self.main.featurization_panel.set_status(f"Added folder: {directory}", "color: #00FF00;")

    def _find_folders_with_csv_files(self, folder):
        """Recursively find all folders (at any depth) that contain CSV files.
        
        Searches through nested subfolders until finding folders with CSV files.
        Returns a list of tuples: [(folder_name, folder_path), ...]
        """
        result = []
        
        # Check if this folder has CSV files
        try:
            entries = os.listdir(folder)
            has_csv_files = any(os.path.isfile(os.path.join(folder, name)) and name.lower().endswith(".csv") for name in entries)
            if has_csv_files:
                result.append((os.path.basename(os.path.normpath(folder)), folder))
        except (OSError, PermissionError):
            return result
        
        # Recursively search subdirectories
        try:
            entries = os.listdir(folder)
            for name in sorted(entries):
                subfolder = os.path.join(folder, name)
                if os.path.isdir(subfolder):
                    result.extend(self._find_folders_with_csv_files(subfolder))
        except (OSError, PermissionError):
            pass
        
        return result

    def resolve_label_folders(self, input_folder):
        if not input_folder or not os.path.isdir(input_folder):
            raise ValueError(f"Invalid input folder: {input_folder}")

        label_folders = self._find_folders_with_csv_files(input_folder)
        
        if not label_folders:
            raise ValueError(f"No CSV files found in selected folder or any of its subfolders: {input_folder}")
        
        return label_folders

    def on_output_browse(self):
        from utilities.dialog_utils import get_existing_directory

        directory = get_existing_directory(self.main, "Select Feature CSV Save Folder", "")
        if directory:
            self.main.featurization_panel.set_output_location(directory)
            self.main.featurization_panel.set_status(f"Output folder set: {directory}", "color: #00FF00;")

    def _split_label_name(self, label_name):
        """Split label name into class and group.
        
        Format: class_name or class_name#group_name
        If no '#' is present, group defaults to 'default'.
        """
        if "#" in label_name:
            parts = label_name.split("#", 1)
            return parts[0], parts[1]
        else:
            return label_name, "default"

    def on_featurize_requested(self):
        input_folders = self.main.featurization_panel.get_source_folders()
        output_folder = self.main.featurization_panel.get_output_location()
        output_filename = self.main.featurization_panel.get_output_filename()

        if not input_folders:
            self.main.featurization_panel.set_status("Error: No labeled data folders selected", "color: #FF6666;")
            return
        if not output_folder:
            self.main.featurization_panel.set_status("Error: No CSV save location selected", "color: #FF6666;")
            return
        if not output_filename:
            self.main.featurization_panel.set_status("Error: Output file name cannot be empty", "color: #FF6666;")
            return

        expected_labels = set()
        for folder in input_folders:
            try:
                label_sources = self.resolve_label_folders(folder)
                expected_labels.update(label_name for label_name, _ in label_sources)
            except Exception as exc:
                self.main.featurization_panel.set_status(f"Error reading {folder}: {str(exc)}", "color: #FF6666;")
                return

        if not expected_labels:
            self.main.featurization_panel.set_status("Error: No labels found in selected folders", "color: #FF6666;")
            return

        if self.main._featurization_options_source is None or set(self.main.featurization_panel.class_checkboxes.keys()) != expected_labels:
            self.refresh_feature_options()

        selected_features = self.main.featurization_panel.get_selected_features()
        selected_labels = self.main.featurization_panel.get_selected_labels()
        if not selected_features:
            self.main.featurization_panel.set_status("Error: No features selected", "color: #FF6666;")
            return
        if not selected_labels:
            self.main.featurization_panel.set_status("Error: No classes selected", "color: #FF6666;")
            return

        try:
            self.main.featurization_panel.set_status("Featurizing data...", "color: #FFAA00;")
            results = self.featurize_multiple_folders(input_folders, output_folder, output_filename, selected_features, selected_labels)
            status_text = f"Done: {results['total_rows']} rows from {results['processed_files']} files ({results['labels_found']} labels) -> {results['output_path']}"
            if results['skipped_files'] > 0:
                status_text += f" | Skipped: {results['skipped_files']}"
            nan_warning = self.check_csv_for_issues(results['output_path'])
            if nan_warning:
                self.main.featurization_panel.set_status(status_text + f" | ⚠ WARNING: {nan_warning}", "color: #FFAA00;")
            else:
                self.main.featurization_panel.set_status(status_text, "color: #00FF00;")
        except Exception as exc:
            self.main.featurization_panel.set_status(f"Error: {str(exc)}", "color: #FF6666;")
            print(f"Error while featurizing data: {exc}")

    def check_csv_for_issues(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            nan_count = df.isna().sum().sum()
            empty_count = 0
            for col in df.select_dtypes(include=["object"]).columns:
                empty_count += (df[col] == "").sum()
            issues = []
            if nan_count > 0:
                issues.append(f"{nan_count} NaN cells")
            if empty_count > 0:
                issues.append(f"{empty_count} empty cells")
            if issues:
                return " and ".join(issues) + " found in output CSV"
            return None
        except Exception as exc:
            print(f"Warning: Could not check CSV for issues: {exc}")
            return None

    def featurize_multiple_folders(self, input_folders, output_folder, output_filename, include_features=None, include_labels=None):
        include_feature_set = set(include_features) if include_features is not None else None
        include_label_set = set(include_labels) if include_labels is not None else None

        seen_input_folders = set()
        unique_input_folders = []
        for folder in input_folders:
            normalized = os.path.normpath(os.path.abspath(folder)).lower()
            if normalized not in seen_input_folders:
                seen_input_folders.add(normalized)
                unique_input_folders.append(folder)

        seen_label_folders = set()
        all_label_sources = []
        for input_folder in unique_input_folders:
            if not os.path.isdir(input_folder):
                continue
            try:
                label_sources = self.resolve_label_folders(input_folder)
            except Exception as exc:
                print(f"Warning: Could not get labels from {input_folder}: {exc}")
                continue

            for label_name, label_folder in label_sources:
                normalized_label_folder = os.path.normpath(os.path.abspath(label_folder)).lower()
                if normalized_label_folder not in seen_label_folders:
                    seen_label_folders.add(normalized_label_folder)
                    all_label_sources.append((label_name, label_folder))

        all_features = []
        processed_files = 0
        skipped_files = 0
        labels_found = set()

        for label_name, label_folder in all_label_sources:
            if include_label_set is not None and label_name not in include_label_set:
                continue
            labels_found.add(label_name)
            csv_files = sorted(filename for filename in os.listdir(label_folder) if filename.lower().endswith(".csv"))
            for filename in csv_files:
                filepath = os.path.join(label_folder, filename)
                try:
                    features = extract_features_from_single_window_file(filepath)
                    class_name, group_name = self._split_label_name(label_name)
                    features["class"] = class_name
                    features["group"] = group_name
                    if include_feature_set is not None:
                        features = {key: value for key, value in features.items() if key in include_feature_set or key == "class" or key == "group"}
                    all_features.append(features)
                    processed_files += 1
                except Exception as exc:
                    skipped_files += 1
                    print(f"Skipping {filepath}: {exc}")

        if not all_features:
            raise ValueError("No feature rows were created from the selected folders.")
        if all(not row for row in all_features):
            raise ValueError("No features selected. Please enable at least one feature.")

        os.makedirs(output_folder, exist_ok=True)
        output_name = output_filename[:-4] if output_filename.lower().endswith(".csv") else output_filename
        output_path = os.path.join(output_folder, f"{output_name}.csv")

        fieldnames = []
        for row in all_features:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_features)

        return {
            "output_path": output_path,
            "total_rows": len(all_features),
            "processed_files": processed_files,
            "skipped_files": skipped_files,
            "labels_found": len(labels_found),
        }

    def refresh_feature_options(self):
        input_folders = self.main.featurization_panel.get_source_folders()
        if not input_folders:
            self.main.featurization_panel.set_label_options([], checked=True)
            self.main.featurization_panel.set_feature_options([], checked=True)
            self.main._featurization_options_source = None
            return

        try:
            all_label_names = set()
            all_feature_names = set()
            for folder in input_folders:
                try:
                    label_sources = self.resolve_label_folders(folder)
                    for label_name, label_folder in label_sources:
                        all_label_names.add(label_name)
                        csv_files = sorted(filename for filename in os.listdir(label_folder) if filename.lower().endswith(".csv"))
                        for filename in csv_files:
                            filepath = os.path.join(label_folder, filename)
                            try:
                                feature_row = extract_features_from_single_window_file(filepath)
                                all_feature_names.update(feature_row.keys())
                                all_feature_names.add("class")
                                break
                            except Exception:
                                continue
                except Exception as exc:
                    print(f"Warning: Could not load options from {folder}: {exc}")
                    continue

            if not all_label_names or not all_feature_names:
                self.main.featurization_panel.set_label_options([], checked=True)
                self.main.featurization_panel.set_feature_options([], checked=True)
                self.main._featurization_options_source = None
                self.main.featurization_panel.set_status("Warning: No labels or features found in selected folders", "color: #FFAA00;")
                return

            self.main.featurization_panel.set_label_options(sorted(all_label_names), checked=True)
            self.main.featurization_panel.set_feature_options(sorted(all_feature_names), checked=True)
            self.main._featurization_options_source = tuple(sorted(input_folders))
        except Exception as exc:
            self.main.featurization_panel.set_label_options([], checked=True)
            self.main.featurization_panel.set_feature_options([], checked=True)
            self.main._featurization_options_source = None
            self.main.featurization_panel.set_status(f"Warning: {str(exc)}", "color: #FFAA00;")
