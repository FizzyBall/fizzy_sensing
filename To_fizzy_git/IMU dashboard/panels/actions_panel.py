"""
Actions Panel - Toggle buttons for high-level robot actions (Random, Wiggle, Forward).
"""

from PyQt6 import QtWidgets, QtCore


class ActionsPanel(QtWidgets.QGroupBox):
    """
    Panel exposing toggle buttons for high-level robot actions
    (Random, Wiggle, Forward). Buttons are mutually exclusive: enabling one
    disables the others. Clicking the active button again disables it.

    The currently active action name is exposed as `self.active_action`
    (one of "random", "wiggle", "forward", or None). It is also emitted via
    the `action_changed` signal.
    """

    action_changed = QtCore.pyqtSignal(object)  # emits str or None

    def __init__(self, parent=None):
        super().__init__("Actions", parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self._set_contents_visible)
        self._set_contents_visible(self.isChecked())
        self.active_action = None  # None | "random" | "wiggle" | "forward"

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.btn_random = QtWidgets.QPushButton("Random")
        self.btn_wiggle = QtWidgets.QPushButton("Wiggle")
        self.btn_forward = QtWidgets.QPushButton("Forward")
        self.btn_stop = QtWidgets.QPushButton("Stop")

        for b in (self.btn_random, self.btn_wiggle, self.btn_forward):
            b.setCheckable(True)
            layout.addWidget(b)
        layout.addWidget(self.btn_stop)

        self.btn_random.clicked.connect(lambda: self._on_action_clicked("random"))
        self.btn_wiggle.clicked.connect(lambda: self._on_action_clicked("wiggle"))
        self.btn_forward.clicked.connect(lambda: self._on_action_clicked("forward"))
        self.btn_stop.clicked.connect(self._stop)

        # Visual style — make active button stand out
        self._base_style = ""
        self._active_style = (
            "QPushButton:checked { background-color: #4CAF50; color: white; "
            "font-weight: bold; }"
        )
        for b in (self.btn_random, self.btn_wiggle, self.btn_forward):
            b.setStyleSheet(self._active_style)
            
    def _set_contents_visible(self, visible: bool):
        """Show or hide the panel contents while keeping the title checkbox visible."""
        for child in self.findChildren(
            QtWidgets.QWidget,
            options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly,
        ):
            child.setVisible(visible)

    def bind_manager(self, manager):
        """
        Bind the actions manager to this panel.
        
        Args:
            manager: The ActionsManager instance
        """
        self.manager = manager
        self.manager.bind_panel(self)

    def _on_action_clicked(self, name):
        # Map button name → button widget
        buttons = {
            "random": self.btn_random,
            "wiggle": self.btn_wiggle,
            "forward": self.btn_forward,
        }
        clicked_btn = buttons[name]

        if self.active_action == name:
            # Toggle off
            clicked_btn.setChecked(False)
            self.active_action = None
        else:
            # Activate this one, deactivate others
            for other_name, other_btn in buttons.items():
                other_btn.setChecked(other_name == name)
            self.active_action = name

        self.action_changed.emit(self.active_action)

    def _stop(self):
        self.btn_random.setChecked(False)
        self.btn_wiggle.setChecked(False)
        self.btn_forward.setChecked(False)
        self.active_action = None
        self.action_changed.emit(None)
