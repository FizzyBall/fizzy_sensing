"""
Control panels for the Fizzy IMU 3D Gizmo application.

This module provides separate control panel classes that can be independently
instantiated and integrated into the main window.
"""

from .analyse_manager import AnalyseManager
from .featurization_manager import FeaturizationManager
from .graph_manager import GraphManager
from .live_classification_manager import LiveClassificationManager
from .recording_manager import RecordingManager
from .training_manager import TrainingManager
from .audio_manager import AudioManager
from .csv_manager import CSVManager
from .classification_analysis_manager import ClassificationAnalysisManager
from .actions_manager import ActionsManager
__all__ = ['AnalyseManager', 'FeaturizationManager', 'GraphManager', 'LiveClassificationManager', 'RecordingManager', 'TrainingManager', 'AudioManager', 'CSVManager', 'ClassificationAnalysisManager', 'ActionsManager']
