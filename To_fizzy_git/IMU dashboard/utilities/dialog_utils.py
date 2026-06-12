"""Shared file/folder dialog helpers that remember the most recently used location.

Every "browse" button in the UI routes through these wrappers so the next dialog
opens at the folder you last navigated to. The remembered folder is stored with
QSettings, so it also persists between runs of the application.
"""

import os

from PyQt6 import QtCore, QtWidgets

_SETTINGS = QtCore.QSettings("Fizzy", "FizzyIMUDashboard")
_LAST_DIR_KEY = "dialogs/last_dir"


def _resolve_start_dir(default_dir=""):
    """Return the folder dialogs should open in.

    Prefers the most recently used folder; otherwise falls back to the caller's
    default, then to the current working directory.
    """
    last = _SETTINGS.value(_LAST_DIR_KEY, "", type=str)
    if last and os.path.isdir(last):
        return last
    if default_dir and os.path.isdir(default_dir):
        return default_dir
    return ""


def _remember(path):
    """Store the folder containing (or equal to) ``path`` as the most recent dir."""
    if not path:
        return
    directory = path if os.path.isdir(path) else os.path.dirname(path)
    if directory and os.path.isdir(directory):
        _SETTINGS.setValue(_LAST_DIR_KEY, directory)


def get_open_file_name(parent, caption, default_dir="", file_filter="All Files (*)"):
    """``QFileDialog.getOpenFileName`` that opens at and records the last-used folder."""
    filepath, selected = QtWidgets.QFileDialog.getOpenFileName(
        parent, caption, _resolve_start_dir(default_dir), file_filter
    )
    if filepath:
        _remember(filepath)
    return filepath, selected


def get_open_file_names(parent, caption, default_dir="", file_filter="All Files (*)"):
    """``QFileDialog.getOpenFileNames`` that opens at and records the last-used folder."""
    filepaths, selected = QtWidgets.QFileDialog.getOpenFileNames(
        parent, caption, _resolve_start_dir(default_dir), file_filter
    )
    if filepaths:
        _remember(filepaths[0])
    return filepaths, selected


def get_save_file_name(parent, caption, default_path="", file_filter="All Files (*)"):
    """``QFileDialog.getSaveFileName`` that opens at the last-used folder while
    keeping the suggested filename from ``default_path``."""
    suggested = default_path
    last = _SETTINGS.value(_LAST_DIR_KEY, "", type=str)
    if last and os.path.isdir(last) and default_path:
        suggested = os.path.join(last, os.path.basename(default_path))
    filepath, selected = QtWidgets.QFileDialog.getSaveFileName(
        parent, caption, suggested, file_filter
    )
    if filepath:
        _remember(filepath)
    return filepath, selected


def get_existing_directory(parent, caption, default_dir=""):
    """``QFileDialog.getExistingDirectory`` that opens at and records the last-used folder."""
    directory = QtWidgets.QFileDialog.getExistingDirectory(
        parent, caption, _resolve_start_dir(default_dir)
    )
    if directory:
        _remember(directory)
    return directory
