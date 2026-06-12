"""
Control panels for the Fizzy IMU 3D Gizmo application.

This module provides separate control panel classes that can be independently
instantiated and integrated into the main window.
"""

from .control_panel import ControlPanel
from .recording_panel import RecordingPanel
from .analyse_panel import AnalysePanel
from .featurization_panel import FeaturizationPanel
from .training_panel import TrainingPanel
from .live_classification_panel import LiveClassificationPanel
from .classification_analysis_panel import ClassificationAnalysisPanel
from .actions_panel import ActionsPanel
__all__ = ['ControlPanel', 'RecordingPanel', 'AnalysePanel', 'FeaturizationPanel', 'TrainingPanel', 'LiveClassificationPanel', 'ClassificationAnalysisPanel', 'ActionsPanel']
