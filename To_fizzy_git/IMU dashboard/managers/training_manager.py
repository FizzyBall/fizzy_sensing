"""
Random Forest training helpers for the Train panel.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold, GroupKFold, StratifiedGroupKFold
from PyQt6 import QtWidgets, QtCore

DEFAULT_PARAM_DISTRIBUTION_SPEC = {
    "n_estimators": {"type": "randint", "low": 100, "high": 500},
    "max_depth": {"type": "randint", "low": 3, "high": 15},
    "min_samples_split": {"type": "randint", "low": 2, "high": 10},
    "min_samples_leaf": {"type": "randint", "low": 1, "high": 5},
    "criterion": {"type": "choices", "values": ["gini", "entropy"]},
    "max_features": {"type": "choices", "values": ["sqrt", "log2", None]},
}


def _build_distribution(value: Any, for_grid_search: bool = False):
    if isinstance(value, dict):
        distribution_type = value.get("type", "choices")
        if distribution_type == "randint":
            low = int(value["low"])
            high = int(value["high"])
            if for_grid_search:
                # For GridSearchCV, create a list of values
                step = max(1, (high - low) // 10)  # Create ~10 values or more
                return list(range(low, high + 1, step))
            else:
                return randint(low, high)
        if distribution_type in ("choices", "values"):
            # Explicit list of values to search over (used by GridSearchCV and as a
            # discrete sampling space for RandomizedSearchCV).
            return value.get("values", [])
        raise ValueError(f"Unsupported distribution type: {distribution_type}")

    if isinstance(value, list):
        if len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
            low = int(value[0])
            high = int(value[1])
            if for_grid_search:
                # For GridSearchCV, create a list of values
                step = max(1, (high - low) // 10)
                return list(range(low, high + 1, step))
            else:
                return randint(low, high + 1)
        return value

    return value


def parse_param_distribution_spec(spec_text: str | None, for_grid_search: bool = False):
    """Parse a JSON param-distribution specification into sklearn-compatible values."""
    if not spec_text or not spec_text.strip():
        spec = DEFAULT_PARAM_DISTRIBUTION_SPEC
    else:
        spec = json.loads(spec_text)

    if not isinstance(spec, dict):
        raise ValueError("Parameter distributions must decode to a JSON object")

    return {key: _build_distribution(value, for_grid_search) for key, value in spec.items()}


def _format_confusion_matrix(conf_matrix: np.ndarray, class_names: list[str]) -> str:
    frame = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    return frame.to_string()


def get_available_class_names_from_csv(csv_path: str):
    """Return the class names present in a featurized CSV file."""
    if not csv_path or not os.path.exists(csv_path):
        raise ValueError(f"CSV file not found: {csv_path}")

    try:
        data = pd.read_csv(csv_path, usecols=["class"])
    except ValueError as exc:
        raise ValueError("Input CSV must contain a 'class' column") from exc

    class_names = (
        data["class"]
        .dropna()
        .astype(str)
        .map(str.strip)
    )
    class_names = sorted([class_name for class_name in class_names.unique().tolist() if class_name])

    if not class_names:
        raise ValueError("No class values found in the selected CSV")

    return class_names


def _build_cv_splitter(cv_method: str, cv_n_splits: int, cv_n_repeats: int, random_state: int):
    """Build a cross-validation splitter based on the method name."""
    if cv_method == "KFold":
        return KFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)
    elif cv_method == "StratifiedKFold":
        return StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)
    elif cv_method == "RepeatedKFold":
        return RepeatedKFold(n_splits=cv_n_splits, n_repeats=cv_n_repeats, random_state=random_state)
    elif cv_method == "RepeatedStratifiedKFold":
        return RepeatedStratifiedKFold(n_splits=cv_n_splits, n_repeats=cv_n_repeats, random_state=random_state)
    elif cv_method == "GroupKFold":
        return GroupKFold(n_splits=cv_n_splits)
    elif cv_method == "StratifiedGroupKFold":
        return StratifiedGroupKFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)
    else:
        # Default to StratifiedKFold
        return StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)


def _generate_settings_abbreviation(
    test_size: float,
    n_iter: int,
    cv: int,
    criterion: str,
    max_features: str | None,
    model_type: str,
    search_method: str,
    cv_method: str,
) -> str:
    """Generate an abbreviated string representation of training settings."""
    # Create abbreviations for key settings
    test_size_abbr = f"ts{int(test_size*100)}"
    n_iter_abbr = f"ni{n_iter}"
    cv_abbr = f"cv{cv}"
    criterion_abbr = f"c{criterion[0].upper()}"  # 'g' for gini, 'e' for entropy
    max_feat_map = {"sqrt": "sq", "log2": "l2", None: "all"}
    max_feat_abbr = f"mf{max_feat_map.get(max_features, 'all')}"
    model_abbr = "BRF" if "Balanced" in model_type else "RF"
    search_abbr = "RS" if "Randomized" in search_method else "GS"
    cv_method_abbr = cv_method[:3].upper()  # KFo, Str, Rep, etc.
    
    parts = [test_size_abbr, n_iter_abbr, cv_abbr, criterion_abbr, max_feat_abbr, model_abbr, search_abbr, cv_method_abbr]
    return "_".join(parts)


def _validate_and_adjust_scoring_for_multiclass(scoring: str, n_classes: int) -> str:
    """
    Validate and adjust scoring metric for multiclass problems.
    Binary-only metrics are converted to their multiclass equivalents.
    """
    # Binary-only metrics that need adjustment
    binary_only_adjustments = {
        "roc_auc": "roc_auc_ovr",  # Use one-vs-rest for multiclass
        "average_precision": "f1_macro",  # AP is binary; use F1 for multiclass
    }
    
    if n_classes <= 2:
        # For binary classification, keep the original scoring metric
        return scoring
    
    # For multiclass, adjust if necessary
    if scoring in binary_only_adjustments:
        adjusted = binary_only_adjustments[scoring]
        print(f"Note: Adjusted scoring metric from '{scoring}' to '{adjusted}' for {n_classes}-class classification")
        return adjusted
    
    # Precision and recall need 'weighted' or 'macro' for multiclass
    if scoring == "precision":
        return "precision_macro"
    elif scoring == "recall":
        return "recall_macro"
    
    return scoring


def train_random_forest_from_csv(
    csv_path: str,
    output_folder: str,
    output_filename: str,
    test_size: float = 0.2,
    random_state: int = 42,
    param_distribution_spec: str | None = None,
    n_iter: int = 10,
    drop_unknown: bool = True,
    include_classes=None,
    scoring: str = "accuracy",
    model_type: str = "Random Forest Classifier",
    search_method: str = "RandomizedSearchCV",
    cv_method: str = "StratifiedKFold",
    cv_n_splits: int = 5,
    cv_n_repeats: int = 10,
    criterion: str = "gini",
    max_features: str | None = "sqrt",
    class_weight: str | dict | None = None,
    bootstrap: bool = True,
    max_samples: float | None = None,
):
    """Train a random forest from a featurized CSV and save artifacts."""
    if not csv_path or not os.path.exists(csv_path):
        raise ValueError(f"CSV file not found: {csv_path}")
    if not output_folder:
        raise ValueError("Output folder cannot be empty")
    if not output_filename:
        raise ValueError("Output filename cannot be empty")

    data = pd.read_csv(csv_path)
    if "class" not in data.columns:
        raise ValueError("Input CSV must contain a 'class' column")

    data = data.dropna(axis=0, how="all")

    # Extract group column if it exists, otherwise use 'default'
    if "group" in data.columns:
        groups_raw = data["group"].astype(str)
    else:
        groups_raw = pd.Series(["default"] * len(data))

    include_class_set = None
    if include_classes is not None:
        include_class_set = {
            str(class_name).strip()
            for class_name in include_classes
            if str(class_name).strip()
        }
        if not include_class_set:
            raise ValueError("No selected classes remain after filtering")
        data = data[data["class"].astype(str).isin(include_class_set)]
        groups_raw = groups_raw[data.index]
        if any(class_name.lower() == "unknown" for class_name in include_class_set):
            drop_unknown = False

    if drop_unknown:
        mask = data["class"].astype(str).str.lower() != "unknown"
        data = data[mask]
        groups_raw = groups_raw[data.index]

    if data.empty:
        raise ValueError("No training rows remain after filtering")

    X = data.drop(["class", "group"], axis=1, errors='ignore')
    y = data["class"].astype(str)

    if X.empty:
        raise ValueError("No feature columns found in the CSV")

    min_class_count = int(y.value_counts().min())
    stratify_target = y if min_class_count >= 2 else None

    if stratify_target is not None:
        max_cv = max(2, min_class_count)
        cv_n_splits = min(cv_n_splits, max_cv)

    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X,
        y,
        groups_raw,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    param_dist = parse_param_distribution_spec(param_distribution_spec, for_grid_search=(search_method == "GridSearchCV"))

    # Validate and adjust scoring metric for the number of classes
    n_classes = len(y.unique())
    scoring = _validate_and_adjust_scoring_for_multiclass(scoring, n_classes)

    # Build the CV splitter
    cv_splitter = _build_cv_splitter(cv_method, cv_n_splits, cv_n_repeats, random_state)

    # Convert group names to numeric IDs for GroupKFold/StratifiedGroupKFold
    groups_train_numeric = None
    if cv_method in ("GroupKFold", "StratifiedGroupKFold"):
        le = LabelEncoder()
        groups_train_numeric = le.fit_transform(groups_train)

    # Select the appropriate classifier based on model_type
    if model_type == "Balanced Random Forest Classifier":
        base_model = BalancedRandomForestClassifier(
            criterion=criterion,
            max_features=max_features,
            class_weight=class_weight,
            bootstrap=bootstrap,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
    else:
        base_model = RandomForestClassifier(
            criterion=criterion,
            max_features=max_features,
            class_weight=class_weight,
            bootstrap=bootstrap,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
    
    # Select the appropriate search method
    if search_method == "GridSearchCV":
        search = GridSearchCV(
            base_model,
            param_grid=param_dist,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1,
            verbose=3
        )
    else:
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1,
            random_state=random_state,
            verbose=3
        )
    
    # Fit with groups if using GroupKFold or StratifiedGroupKFold
    if groups_train_numeric is not None:
        search.fit(X_train, y_train, groups=groups_train_numeric)
    else:
        search.fit(X_train, y_train)

    # Use the best fitted model from the search
    model = search.best_estimator_

    y_pred = model.predict(X_test)
    labels = sorted(y.unique().tolist())
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

    accuracy = accuracy_score(y_test, y_pred)
    balanced = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    output_base = output_filename[:-4] if output_filename.lower().endswith(".joblib") else output_filename
    os.makedirs(output_folder, exist_ok=True)

    model_path = os.path.join(output_folder, f"{output_base}.joblib")
    x_test_path = os.path.join(output_folder, f"{output_base}_X_test.joblib")
    y_test_path = os.path.join(output_folder, f"{output_base}_y_test.joblib")
    metrics_path = os.path.join(output_folder, f"{output_base}_metrics.json")
    confusion_path = os.path.join(output_folder, f"{output_base}_confusion_matrix.csv")
    report_path = os.path.join(output_folder, f"{output_base}_classification_report.json")

    dump(model, model_path)
    dump(X_test, x_test_path)
    dump(y_test, y_test_path)

    pd.DataFrame(conf_matrix, index=labels, columns=labels).to_csv(confusion_path)
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(
            {
                "best_params": search.best_params_,
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "precision_weighted": precision,
                "recall_weighted": recall,
                "f1_weighted": f1,
                "classes": labels,
                "test_size": test_size,
                "random_state": random_state,
                "n_iter": n_iter,
                "cv_method": cv_method,
                "cv_n_splits": cv_n_splits,
                "cv_n_repeats": cv_n_repeats if "Repeated" in cv_method else None,
                "criterion": criterion,
                "max_features": max_features,
                "class_weight": class_weight,
                "bootstrap": bootstrap,
                "max_samples": max_samples,
                "model_type": model_type,
            },
            metrics_file,
            indent=2,
        )
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    feature_importances = []
    if hasattr(model, "feature_importances_"):
        importances = list(zip(X.columns.tolist(), model.feature_importances_))
        feature_importances = sorted(importances, key=lambda item: item[1], reverse=True)[:10]

    top_features_text = "\n".join([f"{name}: {score:.6f}" for name, score in feature_importances])

    # Prepare group information for results
    group_info = {
        "cv_method": cv_method,
        "groups_used": cv_method in ("GroupKFold", "StratifiedGroupKFold"),
        "unique_groups": sorted(groups_train.unique().tolist()),
        "group_counts": dict(groups_train.value_counts().sort_index()),
        "total_groups": len(groups_train.unique()),
    }

    return {
        "model_path": model_path,
        "x_test_path": x_test_path,
        "y_test_path": y_test_path,
        "metrics_path": metrics_path,
        "confusion_path": confusion_path,
        "report_path": report_path,
        "best_params": search.best_params_,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "confusion_matrix": conf_matrix,
        "confusion_matrix_text": _format_confusion_matrix(conf_matrix, labels),
        "classification_report": report,
        "top_features": feature_importances,
        "top_features_text": top_features_text,
        "feature_names": X.columns.tolist(),
        "labels": labels,
        "search_best_score": search.best_score_,
        "group_info": group_info,
    }


class TrainingWorker(QtCore.QThread):
    """Worker thread for training random forest models."""
    
    progress_update = QtCore.pyqtSignal(dict)  # Emits progress updates
    training_finished = QtCore.pyqtSignal(dict)  # Emits final results
    training_error = QtCore.pyqtSignal(str)  # Emits error messages
    
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self._start_time = None
        self._total_iterations = None
        self._stop_progress_tracking = False
    
    def run(self):
        """Run training in background thread."""
        try:
            self._start_time = time.time()
            self._stop_progress_tracking = False
            
            # Calculate total iterations for progress estimation
            search_method = self.kwargs.get("search_method", "RandomizedSearchCV")
            n_iter = self.kwargs.get("n_iter", 10)
            cv_method = self.kwargs.get("cv_method", "StratifiedKFold")
            cv_n_splits = self.kwargs.get("cv_n_splits", 5)
            cv_n_repeats = self.kwargs.get("cv_n_repeats", 10)
            
            if search_method == "GridSearchCV":
                # For grid search, we estimate based on parameter combinations
                # This is a rough estimate: we'll use n_iter as proxy
                self._total_iterations = n_iter * cv_n_splits
            else:
                # For randomized search
                self._total_iterations = n_iter * cv_n_splits
            
            # Start progress tracking timer
            self._start_progress_tracking()
            
            # Run training
            results = train_random_forest_from_csv(**self.kwargs)
            
            # Stop progress tracking
            self._stop_progress_tracking = True
            
            # Emit results
            self.training_finished.emit(results)
            
        except Exception as exc:
            self._stop_progress_tracking = True
            self.training_error.emit(str(exc))
    
    # TODO: progress estimation is crap
    def _start_progress_tracking(self):
        """Start a background timer to emit progress updates."""
        def progress_timer():
            iteration = 0
            while not self._stop_progress_tracking and self._start_time is not None:
                elapsed = time.time() - self._start_time
                
                # Estimate progress (linear approximation)
                # Each iteration takes roughly the same time
                progress_fraction = min(0.95, iteration / max(1, self._total_iterations))
                
                if self._total_iterations and elapsed > 0:
                    # Estimate time per iteration based on first few iterations
                    time_per_iteration = elapsed / max(1, iteration + 1)
                    estimated_total = time_per_iteration * self._total_iterations
                    estimated_remaining = max(0, estimated_total - elapsed)
                else:
                    estimated_remaining = None
                
                self.progress_update.emit({
                    "elapsed": elapsed,
                    "progress_fraction": progress_fraction,
                    "estimated_remaining": estimated_remaining,
                    "iteration": iteration,
                    "total_iterations": self._total_iterations,
                })
                
                iteration += 1
                time.sleep(0.5)  # Update every 500ms
        
        progress_thread = threading.Thread(target=progress_timer, daemon=True)
        progress_thread.start()


def _format_training_settings_summary(
    csv_path: str,
    test_size: float,
    random_state: int,
    n_iter: int,
    cv_n_splits: int,
    cv_n_repeats: int,
    criterion: str,
    max_features: str | None,
    class_weight: str | dict | None,
    model_type: str,
    search_method: str,
    cv_method: str,
    scoring: str,
    n_estimators_range: tuple,
    max_depth_range: tuple,
    min_samples_split_range: tuple,
    min_samples_leaf_range: tuple,
) -> str:
    """Format training settings and parameters into a human-readable summary."""
    lines = [
        "="*60,
        "TRAINING SETTINGS AND PARAMETERS",
        "="*60,
        "",
        "Dataset Configuration:",
        f"  Input CSV: {csv_path}",
        f"  Test size: {test_size*100:.1f}%",
        "",
        "Model Configuration:",
        f"  Model type: {model_type}",
        f"  Criterion: {criterion}",
        f"  Max features: {max_features if max_features else 'all'}",
        f"  Class weight: {class_weight if class_weight else 'None'}",
        "",
        "Hyperparameter Search:",
        f"  Search method: {search_method}",
        f"  Number of iterations: {n_iter}",
        f"  Scoring metric: {scoring}",
        "",
        "Cross-validation:",
        f"  CV method: {cv_method}",
        f"  n_splits: {cv_n_splits}",
    ]
    
    if "Repeated" in cv_method:
        lines.append(f"  n_repeats: {cv_n_repeats}")
    
    lines.extend([
        "",
        "Hyperparameter Ranges:",
        f"  n_estimators: {n_estimators_range[0]} - {n_estimators_range[1]}",
        f"  max_depth: {max_depth_range[0]} - {max_depth_range[1]}",
        f"  min_samples_split: {min_samples_split_range[0]} - {min_samples_split_range[1]}",
        f"  min_samples_leaf: {min_samples_leaf_range[0]} - {min_samples_leaf_range[1]}",
        "",
        "Reproducibility:",
        f"  Random state: {random_state}",
        "="*60,
    ])
    
    return "\n".join(lines)


class TrainingManager:
    """Owns Training-tab interactions."""

    def __init__(self, main_window):
        self.main = main_window
        self.training_worker = None
        self.last_training_results = None  # Store latest results for saving

    def on_csv_browse(self):
        from utilities.dialog_utils import get_open_file_names

        filepaths, _ = get_open_file_names(
            self.main,
            "Select Featurized CSV File(s)",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if filepaths:
            self.main.training_panel.add_csv_paths(filepaths)
            self.refresh_class_options(self.main.training_panel.get_csv_paths())
            self.main.training_panel.set_status(
                f"CSV files selected: {len(self.main.training_panel.get_csv_paths())} file(s)",
                "color: #00FF00;",
            )

    def on_output_browse(self):
        from utilities.dialog_utils import get_existing_directory

        directory = get_existing_directory(self.main, "Select Training Output Folder", "")
        if directory:
            self.main.training_panel.set_output_location(directory)
            self.main.training_panel.set_status(f"Output folder set: {directory}", "color: #00FF00;")

    def refresh_class_options(self, csv_paths):
        try:
            if isinstance(csv_paths, str):
                csv_paths = [csv_paths]

            all_class_names = set()
            for csv_path in csv_paths:
                if csv_path and os.path.exists(csv_path):
                    class_names = get_available_class_names_from_csv(csv_path)
                    all_class_names.update(class_names)

            sorted_class_names = sorted(list(all_class_names))
            self.main.training_panel.set_class_options(sorted_class_names, checked=True)
            self.main._training_options_csv = csv_paths[0] if csv_paths else None
        except Exception as exc:
            self.main.training_panel.set_class_options([], checked=True)
            self.main._training_options_csv = None
            self.main.training_panel.set_status(f"Warning: {str(exc)}", "color: #FFAA00;")

    def train_requested(self):
        csv_paths = self.main.training_panel.get_csv_paths()
        output_folder = self.main.training_panel.get_output_location()
        output_filename = self.main.training_panel.get_output_filename()
        param_text = self.main.training_panel.get_param_distributions_text()

        if not csv_paths:
            self.main.training_panel.set_status("Error: No featurized CSV selected", "color: #FF6666;")
            return
        if not output_folder:
            self.main.training_panel.set_status("Error: No training output folder selected", "color: #FF6666;")
            return
        if not output_filename:
            self.main.training_panel.set_status("Error: Output file name cannot be empty", "color: #FF6666;")
            return

        try:
            combined_df = None
            for csv_path in csv_paths:
                if not os.path.exists(csv_path):
                    raise ValueError(f"CSV file not found: {csv_path}")
                df = pd.read_csv(csv_path)
                combined_df = df if combined_df is None else pd.concat([combined_df, df], ignore_index=True)

            if combined_df is None or combined_df.empty:
                self.main.training_panel.set_status("Error: No data to train on", "color: #FF6666;")
                return

            temp_fd, temp_csv_path = tempfile.mkstemp(suffix=".csv")
            os.close(temp_fd)
            combined_df.to_csv(temp_csv_path, index=False)
            csv_path_for_training = temp_csv_path
            expected_classes = set(get_available_class_names_from_csv(csv_path_for_training))
        except Exception as exc:
            self.main.training_panel.set_status(f"Error: {str(exc)}", "color: #FF6666;")
            return

        try:
            loaded_csv = os.path.normpath(csv_paths[0])
            if self.main._training_options_csv is None or os.path.normpath(self.main._training_options_csv) != loaded_csv or set(self.main.training_panel.class_checkboxes.keys()) != expected_classes:
                self.refresh_class_options(csv_paths)
        except Exception:
            pass

        selected_classes = self.main.training_panel.get_selected_labels()
        if not selected_classes:
            self.main.training_panel.set_status("Error: No classes selected", "color: #FF6666;")
            return

        # Stop any existing training worker
        if self.training_worker is not None and self.training_worker.isRunning():
            self.training_worker.wait()

        # Apply filename suffixes based on checkboxes
        final_output_filename = output_filename
        
        if self.main.training_panel.get_add_date_to_filename():
            current_date = datetime.now().strftime("%Y-%m-%d")
            final_output_filename = f"{final_output_filename}_{current_date}"
        
        if self.main.training_panel.get_add_settings_to_filename():
            settings_abbr = _generate_settings_abbreviation(
                test_size=self.main.training_panel.get_test_size(),
                n_iter=self.main.training_panel.get_n_iter(),
                cv=self.main.training_panel.get_cv_n_splits(),
                criterion=self.main.training_panel.get_criterion(),
                max_features=self.main.training_panel.get_max_features(),
                model_type=self.main.training_panel.get_model_type(),
                search_method=self.main.training_panel.get_search_method(),
                cv_method=self.main.training_panel.get_cv_method(),
            )
            final_output_filename = f"{final_output_filename}_{settings_abbr}"

        drop_unknown = not any(class_name.lower() == "unknown" for class_name in selected_classes)
        
        # Create and start training worker
        self.training_worker = TrainingWorker(
            csv_path=csv_path_for_training,
            output_folder=output_folder,
            output_filename=final_output_filename,
            test_size=self.main.training_panel.get_test_size(),
            random_state=self.main.training_panel.get_random_state(),
            param_distribution_spec=param_text,
            n_iter=self.main.training_panel.get_n_iter(),
            drop_unknown=drop_unknown,
            include_classes=selected_classes,
            scoring=self.main.training_panel.get_scoring(),
            model_type=self.main.training_panel.get_model_type(),
            search_method=self.main.training_panel.get_search_method(),
            cv_method=self.main.training_panel.get_cv_method(),
            cv_n_splits=self.main.training_panel.get_cv_n_splits(),
            cv_n_repeats=self.main.training_panel.get_cv_n_repeats(),
            criterion=self.main.training_panel.get_criterion(),
            max_features=self.main.training_panel.get_max_features(),
            class_weight=self.main.training_panel.get_class_weight(),
            bootstrap=self.main.training_panel.get_bootstrap(),
            max_samples=self.main.training_panel.get_max_samples(),
        )
        
        # Connect signals
        self.training_worker.progress_update.connect(self._on_training_progress)
        self.training_worker.training_finished.connect(lambda results: self._on_training_finished(results, temp_csv_path))
        self.training_worker.training_error.connect(lambda error: self._on_training_error(error, temp_csv_path))
        
        # Clear results and set initial status
        self.main.training_panel.set_results_text("")
        self.main.training_panel.set_status("Training in progress... (0%)", "color: #FFAA00;")
        
        # Start training
        self.training_worker.start()
    
    def _on_training_progress(self, progress_data):
        """Update UI with training progress."""
        elapsed = progress_data.get("elapsed", 0)
        progress_fraction = progress_data.get("progress_fraction", 0)
        estimated_remaining = progress_data.get("estimated_remaining")
        
        # Format elapsed time
        elapsed_str = self._format_time(elapsed)
        
        # Format estimated remaining time
        if estimated_remaining is not None:
            remaining_str = self._format_time(estimated_remaining)
            finish_time = (datetime.now() + timedelta(seconds=estimated_remaining)).strftime("%H:%M:%S")
            status_text = f"Training in progress... ({progress_fraction*100:.0f}%) | Elapsed: {elapsed_str} | Est. remaining: {remaining_str} | Finish: ~{finish_time}"
        else:
            status_text = f"Training in progress... ({progress_fraction*100:.0f}%) | Elapsed: {elapsed_str}"
        
        self.main.training_panel.set_status(status_text, "color: #FFAA00;")
    
    def _on_training_finished(self, results, temp_csv_path):
        """Handle training completion."""
        try:
            # Store results for later saving
            self.last_training_results = results
            
            summary_lines = [
                f"Model saved to: {results['model_path']}",
                f"Accuracy: {results['accuracy']:.4f}",
                f"Balanced accuracy: {results['balanced_accuracy']:.4f}",
                f"Precision (weighted): {results['precision_weighted']:.4f}",
                f"Recall (weighted): {results['recall_weighted']:.4f}",
                f"F1 score (weighted): {results['f1_weighted']:.4f}",
                f"Best cross-val score: {results['search_best_score']:.4f}",
                "",
                "Best hyperparameters:",
                json.dumps(results['best_params'], indent=2),
                "",
            ]

            # Add group information if available
            if results.get('group_info'):
                group_info = results['group_info']
                summary_lines.append("Cross-validation Groups:")
                summary_lines.append(f"  CV Method: {group_info['cv_method']}")
                
                if group_info['groups_used']:
                    summary_lines.append(f"  ✓ Groups are being used for CV splits")
                    summary_lines.append(f"  Total groups: {group_info['total_groups']}")
                    summary_lines.append(f"  Groups: {', '.join(group_info['unique_groups'])}")
                    summary_lines.append(f"  Samples per group:")
                    for group, count in sorted(group_info['group_counts'].items()):
                        summary_lines.append(f"    - {group}: {count} samples")
                else:
                    summary_lines.append(f"  ℹ Groups are available but NOT being used (CV method '{group_info['cv_method']}' does not use groups)")
                    summary_lines.append(f"  To use groups, select 'GroupKFold' or 'StratifiedGroupKFold' as CV method")
                summary_lines.append("")

            summary_lines.extend([
                "Confusion matrix:",
                results['confusion_matrix_text'],
            ])

            if results.get('top_features_text'):
                summary_lines.extend(["", "Top feature importances:", results['top_features_text']])

            self.main.training_panel.set_results_text("\n".join(summary_lines))
            self.main.training_panel.set_status("Training complete", "color: #00FF00;")
            
            # Automatically save results and settings
            self._save_training_results_and_settings(results)
            
        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_csv_path):
                    os.remove(temp_csv_path)
            except Exception:
                pass
    
    def _on_training_error(self, error_msg, temp_csv_path):
        """Handle training error."""
        self.main.training_panel.set_status(f"Error: {error_msg}", "color: #FF6666;")
        print(f"Error while training model: {error_msg}")
        
        # Clean up temp file
        try:
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
        except Exception:
            pass
    
    def _save_training_results_and_settings(self, results):
        """Automatically save training results and settings to files."""
        if not results:
            return
        
        try:
            model_path = results['model_path']
            
            # Create properties file path by replacing .joblib with _properties.txt
            if model_path.lower().endswith('.joblib'):
                properties_path = model_path[:-7] + '_properties.txt'
                settings_path = model_path[:-7] + '_settings.json'
            else:
                properties_path = model_path + '_properties.txt'
                settings_path = model_path + '_settings.json'
            
            # Get training parameters from panel
            csv_paths = self.main.training_panel.get_csv_paths()
            csv_display = "Multiple CSV files" if len(csv_paths) > 1 else (csv_paths[0] if csv_paths else "Unknown")
            
            try:
                class_weight = self.main.training_panel.get_class_weight()
            except Exception:
                class_weight = None
            
            settings_summary = _format_training_settings_summary(
                csv_path=csv_display,
                test_size=self.main.training_panel.get_test_size(),
                random_state=self.main.training_panel.get_random_state(),
                n_iter=self.main.training_panel.get_n_iter(),
                cv_n_splits=self.main.training_panel.get_cv_n_splits(),
                cv_n_repeats=self.main.training_panel.get_cv_n_repeats(),
                criterion=self.main.training_panel.get_criterion(),
                max_features=self.main.training_panel.get_max_features(),
                class_weight=class_weight,
                model_type=self.main.training_panel.get_model_type(),
                search_method=self.main.training_panel.get_search_method(),
                cv_method=self.main.training_panel.get_cv_method(),
                scoring=self.main.training_panel.get_scoring(),
                n_estimators_range=(self.main.training_panel.n_estimators_min_spinbox.value(),
                                   self.main.training_panel.n_estimators_max_spinbox.value()),
                max_depth_range=(self.main.training_panel.max_depth_min_spinbox.value(),
                                self.main.training_panel.max_depth_max_spinbox.value()),
                min_samples_split_range=(self.main.training_panel.min_samples_split_min_spinbox.value(),
                                        self.main.training_panel.min_samples_split_max_spinbox.value()),
                min_samples_leaf_range=(self.main.training_panel.min_samples_leaf_min_spinbox.value(),
                                       self.main.training_panel.min_samples_leaf_max_spinbox.value()),
            )
            
            # Build full results text
            results_lines = [
                settings_summary,
                "",
                "TRAINING RESULTS",
                "="*60,
                f"Accuracy: {results['accuracy']:.6f}",
                f"Balanced accuracy: {results['balanced_accuracy']:.6f}",
                f"Precision (weighted): {results['precision_weighted']:.6f}",
                f"Recall (weighted): {results['recall_weighted']:.6f}",
                f"F1 score (weighted): {results['f1_weighted']:.6f}",
                f"Best cross-validation score: {results['search_best_score']:.6f}",
                "",
                "Best hyperparameters:",
                json.dumps(results['best_params'], indent=2),
                "",
                "Classes: " + ", ".join(results['labels']),
                "",
                "Confusion Matrix:",
                results['confusion_matrix_text'],
            ]
            
            if results.get('top_features_text'):
                results_lines.extend([
                    "",
                    "Top 10 Feature Importances:",
                    results['top_features_text'],
                ])
            
            results_lines.extend([
                "",
                "="*60,
                f"Results saved to: {properties_path}",
                f"Settings saved to: {settings_path}",
                f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ])
            
            # Write results to text file
            with open(properties_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(results_lines))
            
            print(f"Training results saved to: {properties_path}")
            
            # Save settings to JSON file
            settings_dict = self.main.training_panel.save_settings_to_dict()
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(settings_dict, f, indent=2)
            
            print(f"Training settings saved to: {settings_path}")
            
            self.main.training_panel.set_status(
                f"Results and settings saved successfully", "color: #00FF00;"
            )
            
        except Exception as exc:
            error_msg = f"Failed to save results: {str(exc)}"
            self.main.training_panel.set_status(error_msg, "color: #FF6666;")
            print(f"Error saving training results: {error_msg}")
    
    def on_load_settings(self):
        """Handle loading settings from a JSON file."""
        from utilities.dialog_utils import get_open_file_name

        filepath, _ = get_open_file_name(
            self.main,
            "Load Training Settings",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            self.main.training_panel.load_settings_from_dict(settings)
            self.refresh_class_options(self.main.training_panel.get_csv_paths())
            
        except Exception as exc:
            error_msg = f"Failed to load settings: {str(exc)}"
            self.main.training_panel.set_status(error_msg, "color: #FF6666;")
            print(f"Error loading settings: {error_msg}")
    
    @staticmethod
    def _format_time(seconds):
        """Format seconds into HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

