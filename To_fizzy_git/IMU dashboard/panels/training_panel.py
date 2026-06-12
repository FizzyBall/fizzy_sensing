"""
Training Control Panel - Random forest training controls.
"""

import os
import json

from PyQt6 import QtWidgets, QtCore


class TrainingPanel(QtWidgets.QGroupBox):
    """Panel for training a random forest model from a featurized CSV file."""

    csv_browse_requested = QtCore.pyqtSignal()
    output_location_browse_requested = QtCore.pyqtSignal()
    train_requested = QtCore.pyqtSignal()
    load_settings_requested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__("Train")
        self.setCheckable(True)
        self.setChecked(True)
        self.setVisible(False)
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
        # Re-apply conditional visibility that the blanket toggle above overrides
        if visible:
            self._update_cv_options_visibility()
            self._update_class_weight_visibility()
            self._update_search_method_visibility()

    def _init_ui(self):
        layout = QtWidgets.QGridLayout(self)

        # Row 0: CSV files label
        csv_label = QtWidgets.QLabel("Featurized CSV Files:")
        csv_label.setToolTip("Select one or more featurized CSV files created in the Featurization tab. They will be treated as one combined CSV.")
        layout.addWidget(csv_label, 0, 0)

        # Row 1: CSV files list widget
        self.csv_files_list = QtWidgets.QListWidget()
        self.csv_files_list.setMinimumHeight(50)
        self.csv_files_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self.csv_files_list, 1, 0, 1, 6)

        # Row 2: CSV file controls
        csv_controls = QtWidgets.QHBoxLayout()
        
        self.csv_add_btn = QtWidgets.QPushButton("Add CSV File(s)")
        self.csv_add_btn.clicked.connect(self.csv_browse_requested.emit)
        csv_controls.addWidget(self.csv_add_btn)

        self.csv_remove_btn = QtWidgets.QPushButton("Remove Selected")
        self.csv_remove_btn.clicked.connect(self._remove_selected_csv_files)
        csv_controls.addWidget(self.csv_remove_btn)

        self.csv_clear_btn = QtWidgets.QPushButton("Clear All")
        self.csv_clear_btn.clicked.connect(self._clear_all_csv_files)
        csv_controls.addWidget(self.csv_clear_btn)

        csv_controls.addStretch(1)
        layout.addLayout(csv_controls, 0, 1, 1, 3)

        # Row 3: Class include/exclude toggles
        classes_label = QtWidgets.QLabel("Include Classes:")
        classes_label.setToolTip("Select which classes from the featurized CSV should be used for training.")
        layout.addWidget(classes_label, 3, 0)

        class_controls = QtWidgets.QHBoxLayout()
        self.select_all_classes_btn = QtWidgets.QPushButton("Select All")
        self.select_all_classes_btn.clicked.connect(lambda: self.check_all_labels(True))
        class_controls.addWidget(self.select_all_classes_btn)

        self.select_none_classes_btn = QtWidgets.QPushButton("Select None")
        self.select_none_classes_btn.clicked.connect(lambda: self.check_all_labels(False))
        class_controls.addWidget(self.select_none_classes_btn)
        class_controls.addStretch(1)
        layout.addLayout(class_controls, 3, 1, 1, 5)

        self.classes_scroll = QtWidgets.QScrollArea()
        self.classes_scroll.setWidgetResizable(True)
        self.classes_scroll.setMinimumHeight(30)
        self.classes_container = QtWidgets.QWidget()
        self.classes_layout = QtWidgets.QGridLayout(self.classes_container)
        self.classes_layout.setContentsMargins(4, 4, 4, 4)
        self.classes_layout.setHorizontalSpacing(10)
        self.classes_layout.setVerticalSpacing(4)
        self.classes_scroll.setWidget(self.classes_container)
        layout.addWidget(self.classes_scroll, 4, 0, 1, 6)
        self.class_checkboxes = {}
        self.class_weight_inputs = {}  # Store weight inputs for custom weights

        # Row 5: Output location
        self.output_location_input = QtWidgets.QLineEdit()
        self.output_location_input.setPlaceholderText("Select output folder")
        layout.addWidget(self.output_location_input, 5, 0, 1, 2)

        self.output_location_browse_btn = QtWidgets.QPushButton("Browse")
        self.output_location_browse_btn.clicked.connect(self.output_location_browse_requested.emit)
        layout.addWidget(self.output_location_browse_btn, 5, 2)

        # Row 5.5: Filename suffix options
        
        self.add_date_checkbox = QtWidgets.QCheckBox("Add date to filename")
        self.add_date_checkbox.setToolTip("Append the current date (YYYY-MM-DD) to the filename.")
        self.add_date_checkbox.setChecked(False)
        layout.addWidget(self.add_date_checkbox, 5, 3)

        self.add_settings_checkbox = QtWidgets.QCheckBox("Add settings abbreviation")
        self.add_settings_checkbox.setToolTip("Append an abbreviated list of training settings to the filename.")
        self.add_settings_checkbox.setChecked(False)
        layout.addWidget(self.add_settings_checkbox, 6, 3)

        # Row 6: Output filename
        self.output_filename_input = QtWidgets.QLineEdit()
        self.output_filename_input.setPlaceholderText("Output file name")
        layout.addWidget(self.output_filename_input, 6, 0, 1, 2)

        # Row 7: Training settings
        settings_label = QtWidgets.QLabel("Training Settings:")
        settings_label.setToolTip("Control the train/test split and random search settings.")
        layout.addWidget(settings_label, 7, 0)

        self.test_size_spinbox = QtWidgets.QDoubleSpinBox()
        self.test_size_spinbox.setRange(0.05, 0.95)
        self.test_size_spinbox.setSingleStep(0.05)
        self.test_size_spinbox.setDecimals(2)
        self.test_size_spinbox.setValue(0.20)
        self.test_size_spinbox.setSuffix(" test size")
        self.test_size_spinbox.setToolTip("Proportion of the dataset to use as the test set. This portion will be held out from training and used to evaluate the final model's performance.")
        layout.addWidget(self.test_size_spinbox, 7, 1)

        self.random_state_spinbox = QtWidgets.QSpinBox()
        self.random_state_spinbox.setRange(0, 999999)
        self.random_state_spinbox.setValue(42)
        self.random_state_spinbox.setPrefix("random state ")
        self.random_state_spinbox.setToolTip("Random seed for reproducibility. Setting this ensures that the train/test split and random search results can be reproduced exactly in future runs.")
        layout.addWidget(self.random_state_spinbox, 7, 2)

        self.n_iter_spinbox = QtWidgets.QSpinBox()
        self.n_iter_spinbox.setRange(1, 200)
        self.n_iter_spinbox.setValue(10)
        self.n_iter_spinbox.setSuffix(" iterations")
        self.n_iter_spinbox.setToolTip("Number of parameter settings (below) that are sampled. A total of n_iter * cv iterations are performed. Example: with n_iter=10 and cv=5, 50 different models will be trained and evaluated during the random search process. More iterations provide a better chance of finding optimal hyperparameters but increase training time.")
        layout.addWidget(self.n_iter_spinbox, 7, 3)

        # Row 8: Model type
        model_type_label = QtWidgets.QLabel("Methods:")
        model_type_label.setToolTip("Select the type of random forest classifier to use.")
        layout.addWidget(model_type_label, 8, 0)

        self.model_type_combobox = QtWidgets.QComboBox()
        self.model_type_options = [
            "Random Forest",
            "Balanced RF",
        ]
        self.model_type_combobox.addItems(self.model_type_options)
        self.model_type_combobox.setCurrentText("Random Forest")
        self.model_type_combobox.setToolTip("Choose between a standard Random Forest Classifier or a Balanced Random Forest Classifier (better for imbalanced datasets).")
        layout.addWidget(self.model_type_combobox, 8, 1)

        # Row 8b: Search method

        self.search_method_combobox = QtWidgets.QComboBox()
        self.search_method_options = [
            "RandomizedSearchCV",
            "GridSearchCV",
        ]
        self.search_method_combobox.addItems(self.search_method_options)
        self.search_method_combobox.setCurrentText("RandomizedSearchCV")
        self.search_method_combobox.setToolTip("RandomizedSearchCV: Random sampling of parameters from a min/max range (faster). GridSearchCV: Exhaustive search over all combinations of the specific values you list for each hyperparameter (thorough but slower).")
        self.search_method_combobox.currentTextChanged.connect(self._update_search_method_visibility)
        layout.addWidget(self.search_method_combobox, 8, 2)

        # Row 8c: Cross-validation method

        self.cv_method_combobox = QtWidgets.QComboBox()
        self.cv_method_options = [
            "KFold",
            "StratifiedKFold",
            "RepeatedKFold",
            "RepeatedStratifiedKFold",
            "GroupKFold",
            "StratifiedGroupKFold",
        ]
        self.cv_method_combobox.addItems(self.cv_method_options)
        self.cv_method_combobox.setCurrentText("StratifiedKFold")
        self.cv_method_combobox.setToolTip("KFold: Basic k-fold splitting. StratifiedKFold: Preserves class distribution. RepeatedKFold: Repeats k-fold multiple times. GroupKFold: Groups-aware splitting.")
        self.cv_method_combobox.currentTextChanged.connect(self._update_cv_options_visibility)
        layout.addWidget(self.cv_method_combobox, 8, 3)

        # Row 9: CV options
        cv_options_label = QtWidgets.QLabel("CV Options:")
        cv_options_label.setToolTip("Configure cross-validation method parameters.")
        layout.addWidget(cv_options_label, 9, 0)

        cv_options_row = QtWidgets.QHBoxLayout()

        self.cv_n_splits_label = QtWidgets.QLabel("n_splits:")
        self.cv_n_splits_spinbox = QtWidgets.QSpinBox()
        self.cv_n_splits_spinbox.setRange(2, 20)
        self.cv_n_splits_spinbox.setValue(5)
        self.cv_n_splits_spinbox.setToolTip("Number of folds for k-fold splitting.")
        cv_options_row.addWidget(self.cv_n_splits_label)
        cv_options_row.addWidget(self.cv_n_splits_spinbox)

        self.cv_n_repeats_label = QtWidgets.QLabel("n_repeats:")
        self.cv_n_repeats_spinbox = QtWidgets.QSpinBox()
        self.cv_n_repeats_spinbox.setRange(1, 50)
        self.cv_n_repeats_spinbox.setValue(10)
        self.cv_n_repeats_spinbox.setToolTip("Number of times to repeat k-fold (for Repeated methods).")
        cv_options_row.addWidget(self.cv_n_repeats_label)
        cv_options_row.addWidget(self.cv_n_repeats_spinbox)

        cv_options_row.addStretch(1)
        layout.addLayout(cv_options_row, 9, 1, 1, 5)

        # Row 10: Scoring metric
        scoring_label = QtWidgets.QLabel("Scoring Metric/Criteria:")
        scoring_label.setToolTip("Select the metric used to evaluate model performance during hyperparameter search.")
        layout.addWidget(scoring_label, 10, 0)

        self.scoring_combobox = QtWidgets.QComboBox()
        self.scoring_options = [
            "accuracy",
            "balanced_accuracy",
            "average_precision",
            "neg_brier_score",
            "f1_micro",
            "f1_macro",
            "f1_weighted",
            "f1_samples",
            "neg_log_loss",
            "precision",
            "recall",
            "jaccard",
            "matthews_corrcoef",
            "roc_auc",
            "roc_auc_ovo",
            "roc_auc_ovr_weighted",
            "roc_auc_ovo_weighted",
            "d2_brier_score",
        ]
        self.scoring_combobox.addItems(self.scoring_options)
        self.scoring_combobox.setCurrentText("accuracy")
        self.scoring_combobox.setToolTip("Choose the scoring metric for hyperparameter search cross-validation.")
        layout.addWidget(self.scoring_combobox, 10, 1)

        # Row 11: Random Forest Parameters label
        param_label = QtWidgets.QLabel("Random Forest Parameters:")
        param_label.setToolTip("Configure the hyperparameter ranges for random forest training.")
        layout.addWidget(param_label, 11, 0)

        # Each numeric hyperparameter has two interchangeable inputs occupying the
        # same grid cell: a min/max range (used by RandomizedSearchCV) and an
        # explicit comma-separated value list (used by GridSearchCV). Only one is
        # visible at a time, toggled by _update_search_method_visibility().

        # Row 11: n_estimators (number of trees)
        n_estimators_label = QtWidgets.QLabel("n_estimators:")
        n_estimators_label.setToolTip("Number of trees in the forest. More trees improve accuracy but increase training time and memory usage.")
        layout.addWidget(n_estimators_label, 12, 0)

        self.n_estimators_range_widget = QtWidgets.QWidget()
        n_estimators_layout = QtWidgets.QHBoxLayout(self.n_estimators_range_widget)
        n_estimators_layout.setContentsMargins(0, 0, 0, 0)
        self.n_estimators_min_spinbox = QtWidgets.QSpinBox()
        self.n_estimators_min_spinbox.setRange(1, 2000)
        self.n_estimators_min_spinbox.setValue(100)
        n_estimators_layout.addWidget(self.n_estimators_min_spinbox)

        n_est_max_label = QtWidgets.QLabel("to")
        n_est_max_label.setToolTip("Maximum number of trees to search.")
        self.n_estimators_max_spinbox = QtWidgets.QSpinBox()
        self.n_estimators_max_spinbox.setRange(1, 2000)
        self.n_estimators_max_spinbox.setValue(500)
        n_estimators_layout.addWidget(n_est_max_label)
        n_estimators_layout.addWidget(self.n_estimators_max_spinbox)
        n_estimators_layout.addStretch(1)
        layout.addWidget(self.n_estimators_range_widget, 12, 1, 1, 5)

        self.n_estimators_list_input = QtWidgets.QLineEdit()
        self.n_estimators_list_input.setText("50, 100, 150, 200, 300, 400, 500, 1000")
        self.n_estimators_list_input.setPlaceholderText("Comma-separated values, e.g. 50, 100, 200")
        self.n_estimators_list_input.setToolTip("Specific n_estimators values for GridSearchCV to search over. Separate values with commas.")
        layout.addWidget(self.n_estimators_list_input, 12, 1, 1, 5)

        # Row 12: max_depth (tree depth)
        max_depth_label = QtWidgets.QLabel("max_depth:")
        max_depth_label.setToolTip("Maximum depth of each tree. Deeper trees capture complex patterns but are prone to overfitting.")
        layout.addWidget(max_depth_label, 13, 0)

        self.max_depth_range_widget = QtWidgets.QWidget()
        max_depth_layout = QtWidgets.QHBoxLayout(self.max_depth_range_widget)
        max_depth_layout.setContentsMargins(0, 0, 0, 0)
        self.max_depth_min_spinbox = QtWidgets.QSpinBox()
        self.max_depth_min_spinbox.setRange(1, 100)
        self.max_depth_min_spinbox.setValue(3)
        max_depth_layout.addWidget(self.max_depth_min_spinbox)

        max_d_max_label = QtWidgets.QLabel("to")
        max_d_max_label.setToolTip("Maximum maximum depth to search.")
        self.max_depth_max_spinbox = QtWidgets.QSpinBox()
        self.max_depth_max_spinbox.setRange(1, 100)
        self.max_depth_max_spinbox.setValue(15)
        max_depth_layout.addWidget(max_d_max_label)
        max_depth_layout.addWidget(self.max_depth_max_spinbox)
        max_depth_layout.addStretch(1)
        layout.addWidget(self.max_depth_range_widget, 13, 1, 1, 5)

        self.max_depth_list_input = QtWidgets.QLineEdit()
        self.max_depth_list_input.setText("None, 10, 20, 50")
        self.max_depth_list_input.setPlaceholderText("Comma-separated values, e.g. None, 10, 20")
        self.max_depth_list_input.setToolTip("Specific max_depth values for GridSearchCV to search over. Use 'None' for unlimited depth. Separate values with commas.")
        layout.addWidget(self.max_depth_list_input, 13, 1, 1, 5)

        # Row 13: min_samples_split
        min_split_label = QtWidgets.QLabel("min_samples_split:")
        min_split_label.setToolTip("Minimum number of samples required to split a node. Higher values prevent learning noise and reduce overfitting.")
        layout.addWidget(min_split_label, 14, 0)

        self.min_samples_split_range_widget = QtWidgets.QWidget()
        min_split_layout = QtWidgets.QHBoxLayout(self.min_samples_split_range_widget)
        min_split_layout.setContentsMargins(0, 0, 0, 0)
        self.min_samples_split_min_spinbox = QtWidgets.QSpinBox()
        self.min_samples_split_min_spinbox.setRange(1, 50)
        self.min_samples_split_min_spinbox.setValue(2)
        min_split_layout.addWidget(self.min_samples_split_min_spinbox)

        min_s_max_label = QtWidgets.QLabel("to")
        min_s_max_label.setToolTip("Maximum min_samples_split value to search.")
        self.min_samples_split_max_spinbox = QtWidgets.QSpinBox()
        self.min_samples_split_max_spinbox.setRange(1, 50)
        self.min_samples_split_max_spinbox.setValue(10)
        min_split_layout.addWidget(min_s_max_label)
        min_split_layout.addWidget(self.min_samples_split_max_spinbox)
        min_split_layout.addStretch(1)
        layout.addWidget(self.min_samples_split_range_widget, 14, 1, 1, 5)

        self.min_samples_split_list_input = QtWidgets.QLineEdit()
        self.min_samples_split_list_input.setText("2, 5, 10")
        self.min_samples_split_list_input.setPlaceholderText("Comma-separated values, e.g. 2, 5, 10")
        self.min_samples_split_list_input.setToolTip("Specific min_samples_split values for GridSearchCV to search over. Separate values with commas.")
        layout.addWidget(self.min_samples_split_list_input, 14, 1, 1, 5)

        # Row 14: min_samples_leaf
        min_leaf_label = QtWidgets.QLabel("min_samples_leaf:")
        min_leaf_label.setToolTip("Minimum number of samples required at leaf nodes. Higher values increase bias but reduce variance and prevent overfitting.")
        layout.addWidget(min_leaf_label, 15, 0)

        self.min_samples_leaf_range_widget = QtWidgets.QWidget()
        min_leaf_layout = QtWidgets.QHBoxLayout(self.min_samples_leaf_range_widget)
        min_leaf_layout.setContentsMargins(0, 0, 0, 0)
        self.min_samples_leaf_min_spinbox = QtWidgets.QSpinBox()
        self.min_samples_leaf_min_spinbox.setRange(1, 50)
        self.min_samples_leaf_min_spinbox.setValue(1)
        min_leaf_layout.addWidget(self.min_samples_leaf_min_spinbox)

        min_l_max_label = QtWidgets.QLabel("to")
        min_l_max_label.setToolTip("Maximum min_samples_leaf value to search.")
        self.min_samples_leaf_max_spinbox = QtWidgets.QSpinBox()
        self.min_samples_leaf_max_spinbox.setRange(1, 50)
        self.min_samples_leaf_max_spinbox.setValue(5)
        min_leaf_layout.addWidget(min_l_max_label)
        min_leaf_layout.addWidget(self.min_samples_leaf_max_spinbox)
        min_leaf_layout.addStretch(1)
        layout.addWidget(self.min_samples_leaf_range_widget, 15, 1, 1, 5)

        self.min_samples_leaf_list_input = QtWidgets.QLineEdit()
        self.min_samples_leaf_list_input.setText("1, 2, 5, 10")
        self.min_samples_leaf_list_input.setPlaceholderText("Comma-separated values, e.g. 1, 2, 5, 10")
        self.min_samples_leaf_list_input.setToolTip("Specific min_samples_leaf values for GridSearchCV to search over. Separate values with commas.")
        layout.addWidget(self.min_samples_leaf_list_input, 15, 1, 1, 5)

        # Row 16: criterion
        self.criterion_combobox = QtWidgets.QComboBox()
        self.criterion_options = ["gini", "entropy"]
        self.criterion_combobox.addItems(self.criterion_options)
        self.criterion_combobox.setCurrentText("gini")
        self.criterion_combobox.setToolTip("Choose the criterion for split quality measurement.")
        layout.addWidget(self.criterion_combobox, 10, 2)

        # Row 16b: max_features
        max_features_label = QtWidgets.QLabel("max_features:")
        max_features_label.setToolTip("Number of features to consider for each split. 'sqrt': sqrt(n_features), 'log2': log2(n_features), None: all features.")
        layout.addWidget(max_features_label, 16, 0)

        self.max_features_combobox = QtWidgets.QComboBox()
        self.max_features_options = ["sqrt", "log2", None]
        self.max_features_combobox.addItems(["sqrt", "log2", "None"])
        self.max_features_combobox.setCurrentText("sqrt")
        self.max_features_combobox.setToolTip("Choose how many features to consider at each split.")
        layout.addWidget(self.max_features_combobox, 16, 1)

        # Row 16c: class_weight (only for standard RandomForest)
        class_weight_label = QtWidgets.QLabel("class_weight:")
        class_weight_label.setToolTip("Adjust weights inversely proportional to class frequency. Useful for handling imbalanced classes (standard RF only).")
        layout.addWidget(class_weight_label, 16, 2)

        self.class_weight_combobox = QtWidgets.QComboBox()
        self.class_weight_options = ["None", "balanced", "balanced_subsample", "Custom"]
        self.class_weight_combobox.addItems(self.class_weight_options)
        self.class_weight_combobox.setCurrentText("None")
        self.class_weight_combobox.setToolTip("None: no weighting. balanced: weight inversely to class frequency. balanced_subsample: same but per subsample. Custom: specify per-class weights.")
        self.class_weight_combobox.currentTextChanged.connect(self._update_class_weight_visibility)
        layout.addWidget(self.class_weight_combobox, 16, 3)

        # Row 17: bootstrap and max_samples
        bootstrap_label = QtWidgets.QLabel("bootstrap:")
        bootstrap_label.setToolTip("Whether to use bootstrap samples when building trees. If False, the whole dataset is used to build each tree.")
        layout.addWidget(bootstrap_label, 17, 0)

        self.bootstrap_combobox = QtWidgets.QComboBox()
        self.bootstrap_combobox.addItems(["True", "False"])
        self.bootstrap_combobox.setCurrentText("True")
        self.bootstrap_combobox.setToolTip("Enable or disable bootstrapping for tree building.")
        layout.addWidget(self.bootstrap_combobox, 17, 1)

        max_samples_label = QtWidgets.QLabel("max_samples:")
        max_samples_label.setToolTip("Proportion of samples to draw for each tree (only used when bootstrap=True). 0: use all samples, 0.5: use 50% of samples.")
        layout.addWidget(max_samples_label, 17, 2)

        self.max_samples_spinbox = QtWidgets.QDoubleSpinBox()
        self.max_samples_spinbox.setRange(0.0, 1.0)
        self.max_samples_spinbox.setSingleStep(0.1)
        self.max_samples_spinbox.setDecimals(2)
        self.max_samples_spinbox.setValue(1.0)
        self.max_samples_spinbox.setToolTip("Proportion of samples per tree (0.0-1.0). 0.0 or None uses all samples, 0.5 uses 50%, 1.0 uses 100%.")
        layout.addWidget(self.max_samples_spinbox, 17, 3)

        # Row 18: Action buttons
        self.train_btn = QtWidgets.QPushButton("Train model")
        self.train_btn.clicked.connect(self.train_requested.emit)
        layout.addWidget(self.train_btn, 18, 0, 1, 2)

        self.load_settings_btn = QtWidgets.QPushButton("Load settings from JSON")
        self.load_settings_btn.setToolTip("Load training settings from a previously saved JSON settings file")
        layout.addWidget(self.load_settings_btn, 18, 2, 1, 2)

        self.status_label = QtWidgets.QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #00FF00;")
        layout.addWidget(self.status_label, 19, 0, 1, 4)

        # Row 19: Results
        results_label = QtWidgets.QLabel("Results:")
        layout.addWidget(results_label, 20, 0)

        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(50)
        layout.addWidget(self.results_text, 21, 0, 1, 6)
        
        # Initialize CSV paths list
        self.csv_file_paths = []
        
        # Initialize CV options visibility
        self._update_cv_options_visibility()

        # Initialize class weight visibility
        self._update_class_weight_visibility()

        # Initialize search-method-dependent hyperparameter inputs
        self._update_search_method_visibility()

    def _remove_selected_csv_files(self):
        """Remove selected CSV files from the list."""
        for item in self.csv_files_list.selectedItems():
            row = self.csv_files_list.row(item)
            if 0 <= row < len(self.csv_file_paths):
                self.csv_file_paths.pop(row)
            self.csv_files_list.takeItem(row)

    def _clear_all_csv_files(self):
        """Clear all CSV files from the list."""
        self.csv_file_paths = []
        self.csv_files_list.clear()

    def get_csv_path(self):
        """Get the first CSV path (for backward compatibility)."""
        return self.csv_file_paths[0] if self.csv_file_paths else ""

    def get_csv_paths(self):
        """Get all CSV file paths as a list."""
        return list(self.csv_file_paths)

    def add_csv_path(self, path):
        """Add a single CSV file path to the list."""
        if path and path not in self.csv_file_paths:
            self.csv_file_paths.append(path)
            self.csv_files_list.addItem(os.path.basename(path))
            # Display full path as tooltip
            item = self.csv_files_list.item(self.csv_files_list.count() - 1)
            if item:
                item.setToolTip(path)

    def bind_manager(self, manager):
        """Connect this panel's signals to the training manager."""
        self.csv_browse_requested.connect(manager.on_csv_browse)
        self.output_location_browse_requested.connect(manager.on_output_browse)
        self.train_requested.connect(manager.train_requested)
        self.load_settings_btn.clicked.connect(manager.on_load_settings)

    def add_csv_paths(self, paths):
        """Add multiple CSV file paths to the list."""
        for path in paths:
            self.add_csv_path(path)

    def set_csv_path(self, path, set_output_default=True):
        """Set a single CSV file path (clears existing paths)."""
        self.csv_file_paths = []
        self.csv_files_list.clear()
        if path:
            self.add_csv_path(path)
            
            if set_output_default:
                self.output_location_input.setText(os.path.dirname(path))

    def set_csv_paths(self, paths, set_output_default=True):
        """Set multiple CSV file paths."""
        self.csv_file_paths = []
        self.csv_files_list.clear()
        for path in paths:
            self.add_csv_path(path)
            
        if set_output_default and paths:
            self.output_location_input.setText(os.path.dirname(paths[0]))

    def get_output_location(self):
        return self.output_location_input.text().strip()

    def set_output_location(self, path):
        self.output_location_input.setText(path)

    def get_output_filename(self):
        return self.output_filename_input.text().strip()

    def set_output_filename(self, name):
        self.output_filename_input.setText(name)

    def get_add_date_to_filename(self):
        """Get whether to add date to filename."""
        return self.add_date_checkbox.isChecked()

    def get_add_settings_to_filename(self):
        """Get whether to add settings abbreviation to filename."""
        return self.add_settings_checkbox.isChecked()

    def get_test_size(self):
        return float(self.test_size_spinbox.value())

    def get_random_state(self):
        return int(self.random_state_spinbox.value())

    def get_n_iter(self):
        return int(self.n_iter_spinbox.value())

    def get_model_type(self):
        """Get the selected model type."""
        return self.model_type_combobox.currentText()

    def get_search_method(self):
        """Get the selected search method."""
        return self.search_method_combobox.currentText()

    def get_cv_method(self):
        """Get the selected cross-validation method."""
        return self.cv_method_combobox.currentText()

    def get_cv_n_splits(self):
        """Get the n_splits parameter for cross-validation."""
        return int(self.cv_n_splits_spinbox.value())

    def get_cv_n_repeats(self):
        """Get the n_repeats parameter for repeated cross-validation methods."""
        return int(self.cv_n_repeats_spinbox.value())

    def get_criterion(self):
        """Get the selected criterion for split quality measurement."""
        return self.criterion_combobox.currentText()

    def get_max_features(self):
        """Get the selected max_features option."""
        text = self.max_features_combobox.currentText()
        return None if text == "None" else text

    def get_class_weight(self):
        """Get the selected class_weight option or custom weights dict."""
        text = self.class_weight_combobox.currentText()
        if text == "None":
            return None
        elif text == "Custom":
            # Collect weights from individual input fields
            weights_dict = {}
            for class_name, weight_input in self.class_weight_inputs.items():
                weights_dict[class_name] = weight_input.value()
            if not weights_dict:
                raise ValueError("No classes available for custom weights")
            return weights_dict
        else:
            return text

    def _update_class_weight_visibility(self):
        """Show/hide weight input fields based on class_weight selection."""
        show_weights = self.class_weight_combobox.currentText() == "Custom"
        for weight_input in self.class_weight_inputs.values():
            weight_input.setVisible(show_weights)

    def _update_search_method_visibility(self):
        """Swap range inputs for explicit value-list inputs based on the search method.

        RandomizedSearchCV samples from a min/max range, so the range spinboxes are
        shown. GridSearchCV searches over the specific values you list, so the
        comma-separated value-list inputs are shown instead.
        """
        is_grid = self.search_method_combobox.currentText() == "GridSearchCV"
        range_widgets = (
            self.n_estimators_range_widget,
            self.max_depth_range_widget,
            self.min_samples_split_range_widget,
            self.min_samples_leaf_range_widget,
        )
        list_inputs = (
            self.n_estimators_list_input,
            self.max_depth_list_input,
            self.min_samples_split_list_input,
            self.min_samples_leaf_list_input,
        )
        for range_widget in range_widgets:
            range_widget.setVisible(not is_grid)
        for list_input in list_inputs:
            list_input.setVisible(is_grid)
        # n_iter only applies to RandomizedSearchCV (GridSearchCV is exhaustive)
        self.n_iter_spinbox.setEnabled(not is_grid)

    def _update_cv_options_visibility(self):
        """Show/hide CV options based on the selected method."""
        cv_method = self.cv_method_combobox.currentText()
        # All methods use n_splits
        self.cv_n_splits_label.setVisible(True)
        self.cv_n_splits_spinbox.setVisible(True)
        # Only Repeated methods use n_repeats
        show_n_repeats = "Repeated" in cv_method
        self.cv_n_repeats_label.setVisible(show_n_repeats)
        self.cv_n_repeats_spinbox.setVisible(show_n_repeats)

    def get_scoring(self):
        """Get the selected scoring metric."""
        return self.scoring_combobox.currentText()

    def get_bootstrap(self):
        """Get the bootstrap setting."""
        return self.bootstrap_combobox.currentText() == "True"

    def get_max_samples(self):
        """Get the max_samples setting. Returns None if spinbox is 0.0."""
        value = self.max_samples_spinbox.value()
        return None if value == 0.0 else value

    @staticmethod
    def _parse_value_list(text, allow_none=False):
        """Parse a comma-separated list of integers (and optionally 'None')."""
        values = []
        for token in text.split(","):
            token = token.strip()
            if not token:
                continue
            if allow_none and token.lower() in ("none", "null"):
                values.append(None)
            else:
                values.append(int(token))
        return values

    def get_param_distributions_text(self):
        """Generate JSON parameter distributions from the active hyperparameter inputs.

        For GridSearchCV the explicit value lists are used; for RandomizedSearchCV
        the min/max ranges are used.
        """
        if self.get_search_method() == "GridSearchCV":
            param_dist = {
                "n_estimators": {
                    "type": "values",
                    "values": self._parse_value_list(self.n_estimators_list_input.text()),
                },
                "max_depth": {
                    "type": "values",
                    "values": self._parse_value_list(self.max_depth_list_input.text(), allow_none=True),
                },
                "min_samples_split": {
                    "type": "values",
                    "values": self._parse_value_list(self.min_samples_split_list_input.text()),
                },
                "min_samples_leaf": {
                    "type": "values",
                    "values": self._parse_value_list(self.min_samples_leaf_list_input.text()),
                },
            }
        else:
            param_dist = {
                "n_estimators": {
                    "type": "randint",
                    "low": self.n_estimators_min_spinbox.value(),
                    "high": self.n_estimators_max_spinbox.value(),
                },
                "max_depth": {
                    "type": "randint",
                    "low": self.max_depth_min_spinbox.value(),
                    "high": self.max_depth_max_spinbox.value(),
                },
                "min_samples_split": {
                    "type": "randint",
                    "low": self.min_samples_split_min_spinbox.value(),
                    "high": self.min_samples_split_max_spinbox.value(),
                },
                "min_samples_leaf": {
                    "type": "randint",
                    "low": self.min_samples_leaf_min_spinbox.value(),
                    "high": self.min_samples_leaf_max_spinbox.value(),
                },
            }
        return json.dumps(param_dist)

    def set_status(self, text, style="color: #00FF00;"):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(style)

    def set_results_text(self, text):
        self.results_text.setPlainText(text)

    def append_results_text(self, text):
        self.results_text.append(text)

    def set_class_options(self, class_names, checked=True):
        """Populate class checkboxes from class names."""
        while self.classes_layout.count():
            item = self.classes_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        self.class_checkboxes = {}
        self.class_weight_inputs = {}
        for idx, class_name in enumerate(class_names):
            checkbox = QtWidgets.QCheckBox(class_name)
            checkbox.setChecked(checked)
            row = idx
            col = 0
            self.classes_layout.addWidget(checkbox, row, col)
            self.class_checkboxes[class_name] = checkbox

            # Add weight input field next to checkbox
            weight_input = QtWidgets.QDoubleSpinBox()
            weight_input.setRange(0.0, 100.0)
            weight_input.setValue(1.0)
            weight_input.setSingleStep(0.1)
            weight_input.setDecimals(2)
            weight_input.setMaximumWidth(80)
            weight_input.setVisible(False)  # Hidden by default, shown when Custom is selected
            weight_input.setToolTip(f"Weight for class '{class_name}' when using custom class weights. Higher values give more importance (less likely to misclassify) to this class during training.")
            col = 1
            self.classes_layout.addWidget(weight_input, row, col)
            self.class_weight_inputs[class_name] = weight_input

    def check_all_labels(self, checked=True):
        """Check or uncheck all class toggles."""
        for checkbox in self.class_checkboxes.values():
            checkbox.setChecked(checked)

    def get_selected_labels(self):
        """Get all selected class names."""
        return [
            class_name
            for class_name, checkbox in self.class_checkboxes.items()
            if checkbox.isChecked()
        ]
    
    def save_settings_to_dict(self):
        """Export all current settings as a dictionary."""
        settings = {
            "csv_paths": self.get_csv_paths(),
            "output_location": self.get_output_location(),
            "output_filename": self.get_output_filename(),
            "add_date_to_filename": self.get_add_date_to_filename(),
            "add_settings_to_filename": self.get_add_settings_to_filename(),
            "test_size": self.get_test_size(),
            "random_state": self.get_random_state(),
            "n_iter": self.get_n_iter(),
            "model_type": self.get_model_type(),
            "search_method": self.get_search_method(),
            "cv_method": self.get_cv_method(),
            "cv_n_splits": self.get_cv_n_splits(),
            "cv_n_repeats": self.get_cv_n_repeats(),
            "criterion": self.get_criterion(),
            "max_features": self.get_max_features(),
            "scoring": self.get_scoring(),
            "bootstrap": self.get_bootstrap(),
            "max_samples": self.get_max_samples(),
            "n_estimators_min": self.n_estimators_min_spinbox.value(),
            "n_estimators_max": self.n_estimators_max_spinbox.value(),
            "max_depth_min": self.max_depth_min_spinbox.value(),
            "max_depth_max": self.max_depth_max_spinbox.value(),
            "min_samples_split_min": self.min_samples_split_min_spinbox.value(),
            "min_samples_split_max": self.min_samples_split_max_spinbox.value(),
            "min_samples_leaf_min": self.min_samples_leaf_min_spinbox.value(),
            "min_samples_leaf_max": self.min_samples_leaf_max_spinbox.value(),
            "n_estimators_list": self.n_estimators_list_input.text(),
            "max_depth_list": self.max_depth_list_input.text(),
            "min_samples_split_list": self.min_samples_split_list_input.text(),
            "min_samples_leaf_list": self.min_samples_leaf_list_input.text(),
            "selected_classes": self.get_selected_labels(),
            "class_weight": str(self.get_class_weight()),  # Convert to string for JSON serialization
        }
        return settings
    
    def load_settings_from_dict(self, settings):
        """Load settings from a dictionary."""
        try:
            if "csv_paths" in settings:
                self.set_csv_paths(settings["csv_paths"], set_output_default=False)
            
            if "output_location" in settings:
                self.set_output_location(settings["output_location"])
            
            if "output_filename" in settings:
                self.set_output_filename(settings["output_filename"])
            
            if "add_date_to_filename" in settings:
                self.add_date_checkbox.setChecked(settings["add_date_to_filename"])
            
            if "add_settings_to_filename" in settings:
                self.add_settings_checkbox.setChecked(settings["add_settings_to_filename"])
            
            if "test_size" in settings:
                self.test_size_spinbox.setValue(settings["test_size"])
            
            if "random_state" in settings:
                self.random_state_spinbox.setValue(settings["random_state"])
            
            if "n_iter" in settings:
                self.n_iter_spinbox.setValue(settings["n_iter"])
            
            if "model_type" in settings:
                self.model_type_combobox.setCurrentText(settings["model_type"])
            
            if "search_method" in settings:
                self.search_method_combobox.setCurrentText(settings["search_method"])
            
            if "cv_method" in settings:
                self.cv_method_combobox.setCurrentText(settings["cv_method"])
            
            if "cv_n_splits" in settings:
                self.cv_n_splits_spinbox.setValue(settings["cv_n_splits"])
            
            if "cv_n_repeats" in settings:
                self.cv_n_repeats_spinbox.setValue(settings["cv_n_repeats"])
            
            if "criterion" in settings:
                self.criterion_combobox.setCurrentText(settings["criterion"])
            
            if "max_features" in settings:
                max_feat = settings["max_features"]
                self.max_features_combobox.setCurrentText("None" if max_feat is None else max_feat)
            
            if "scoring" in settings:
                self.scoring_combobox.setCurrentText(settings["scoring"])
            
            if "bootstrap" in settings:
                self.bootstrap_combobox.setCurrentText("True" if settings["bootstrap"] else "False")
            
            if "max_samples" in settings:
                self.max_samples_spinbox.setValue(settings["max_samples"] if settings["max_samples"] else 0.0)
            
            if "n_estimators_min" in settings:
                self.n_estimators_min_spinbox.setValue(settings["n_estimators_min"])
            
            if "n_estimators_max" in settings:
                self.n_estimators_max_spinbox.setValue(settings["n_estimators_max"])
            
            if "max_depth_min" in settings:
                self.max_depth_min_spinbox.setValue(settings["max_depth_min"])
            
            if "max_depth_max" in settings:
                self.max_depth_max_spinbox.setValue(settings["max_depth_max"])
            
            if "min_samples_split_min" in settings:
                self.min_samples_split_min_spinbox.setValue(settings["min_samples_split_min"])
            
            if "min_samples_split_max" in settings:
                self.min_samples_split_max_spinbox.setValue(settings["min_samples_split_max"])
            
            if "min_samples_leaf_min" in settings:
                self.min_samples_leaf_min_spinbox.setValue(settings["min_samples_leaf_min"])
            
            if "min_samples_leaf_max" in settings:
                self.min_samples_leaf_max_spinbox.setValue(settings["min_samples_leaf_max"])

            if "n_estimators_list" in settings:
                self.n_estimators_list_input.setText(settings["n_estimators_list"])

            if "max_depth_list" in settings:
                self.max_depth_list_input.setText(settings["max_depth_list"])

            if "min_samples_split_list" in settings:
                self.min_samples_split_list_input.setText(settings["min_samples_split_list"])

            if "min_samples_leaf_list" in settings:
                self.min_samples_leaf_list_input.setText(settings["min_samples_leaf_list"])

            # Re-apply visibility in case the loaded search method changed it
            self._update_search_method_visibility()

            if "selected_classes" in settings:
                # First check all, then uncheck those not in selected_classes
                self.check_all_labels(False)
                selected = set(settings["selected_classes"])
                for class_name, checkbox in self.class_checkboxes.items():
                    if class_name in selected:
                        checkbox.setChecked(True)
            
            self.set_status("Settings loaded successfully", "color: #00FF00;")
        except Exception as exc:
            self.set_status(f"Error loading settings: {str(exc)}", "color: #FF6666;")
