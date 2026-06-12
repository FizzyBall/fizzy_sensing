"""
Actions Manager Module

Manages the Actions Panel state and interactions.
"""


class ActionsManager:
    """Manages the actions panel state and interactions."""
    
    def __init__(self, dashboard):
        """
        Initialize the actions manager.
        
        Args:
            dashboard: Reference to the main FizzyIMUDashboard instance
        """
        self.dashboard = dashboard
        self.current_action = None
    
    def bind_panel(self, panel):
        """
        Bind the actions panel to this manager.
        
        Args:
            panel: The ActionsPanel instance
        """
        self.panel = panel
        self.panel.action_changed.connect(self._on_action_changed)
    
    def _on_action_changed(self, action):
        """Handle action changes from the panel."""
        self.current_action = action
        # Future: Add any additional logic needed when actions change
        # (e.g., sending commands to the robot, logging, etc.)
    
    def get_current_action(self):
        """Get the currently active action."""
        return self.current_action
    
    def set_action(self, action):
        """
        Programmatically set the current action.
        
        Args:
            action: One of "random", "wiggle", "forward", or None
        """
        if hasattr(self, 'panel'):
            if action is None:
                self.panel._stop()
            elif action in ("random", "wiggle", "forward"):
                self.panel._on_action_clicked(action)
