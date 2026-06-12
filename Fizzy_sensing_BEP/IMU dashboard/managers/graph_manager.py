"""
Graph Manager Module
Handles all graph initialization, data management, and visualization for the IMU 3D Gizmo application.
"""

from collections import deque
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from utilities.fizzy_config import CLASSIFICATION_LABELS, LABEL_COLORS


class GraphManager:
    """
    Manages all graph-related functionality including plot creation, data storage, and updates.
    """
    
    # Display mode constants
    MODE_ALL_DATA = "All Data"
    MODE_ANGLES = "Angles"
    MODE_ACCELERATIONS = "Accelerations"
    MODE_ANGULAR_VELOCITY = "Angular Velocity"
    MODES = [MODE_ALL_DATA, MODE_ANGLES, MODE_ACCELERATIONS, MODE_ANGULAR_VELOCITY]
    
    def __init__(self, plot_widget, data_plotting=200):
        """
        Initialize the GraphManager with a plot widget.
        
        Args:
            plot_widget: PyQtGraph GraphicsLayoutWidget to add plots to
            data_plotting: Number of data points to keep in history (default: 200)
        """
        self.plot_widget = plot_widget
        self.data_plotting = data_plotting
        self.current_display_mode = self.MODE_ALL_DATA
        self.graph_visibility_state = {
            "full_data": True,
            "angle_xyz": True,
            "acc_xyz": True,
            "acc_mag": True,
            "gyro_xyz": True,
            "gyro_mag": True,
            "motor_input": True,
            "fft": False,
        }
        
        # Initialize data deques for storing historical data
        self._init_data_deques()
        
        # FFT state
        self.fft_enabled = False
        self.fft_samples = 128
        self.fft_window_size_value = 128
        self.fft_window_unit = "Samples"
        self.sample_rate = 104.0  # 104 Hz from hardware, updated with actual measurement
        
        # Initialize all plot objects
        self._init_plots()
        
        # Initialize scrubber line (vertical line for analyse mode scrubbing)
        self._init_scrubber_lines()
        
        # Data gap indicators
        self.gap_regions = []  # Store all gap region items
        
        # Analysis window indicator
        self.analysis_window_regions = []  # Store all analysis window region items
        self.analysis_window_width = 1.0  # Default: 1 second window
        
        # Frequency tracking
        self.last_timestamp = None
        self.frequency_estimate = 0.0
        
        # Marker lines storage (for analyse-mode marker vertical lines)
        self.marker_lines = []  # list of (plot, line, text_item)
        
        # Labeling support (when in analyse mode with windowed data)
        self.labels = {}  # {window_idx: label_name}
        self.available_labels = CLASSIFICATION_LABELS
        self.label_colors = LABEL_COLORS
        
        # Window tracking
        self.current_window_idx = -1
        self.window_indices = []  # List of (start_idx, end_idx) tuples
        self.window_size = 128  # Default, updated when loading data
        self.stride = 64  # Default, 50% overlap
        
        # Overlay management
        self.label_region_items = []  # Store LinearRegionItem objects for all plots
        self.labels_visible = True  # Whether existing label overlays are shown
        # When False, label overlays draw only the colored region (no text box).
        # Classification analysis sets this off so the prediction/label comparison
        # box is the single text source; the labeling tab keeps it on.
        self.label_overlay_show_text = True

        # Window hover/click interaction (used by the Analyse and Classification
        # Analysis tabs). Hovering highlights the window under the cursor; clicking
        # invokes window_click_callback(window_idx), which the active tab's manager
        # sets to jump its scrubber to that window.
        self.window_click_callback = None
        self._hover_window_idx = None
        self.window_hover_regions = []  # list of (plot, region)
        self._init_window_hover_highlight()
        self._connect_window_interaction()
    
    def _init_data_deques(self):
        """Initialize all data storage deques."""
        # Time data
        self.t_data = deque(maxlen=self.data_plotting)
        
        # Angle data (degrees)
        self.r_data = deque(maxlen=self.data_plotting)
        self.p_data = deque(maxlen=self.data_plotting)
        self.y_data = deque(maxlen=self.data_plotting)
        
        # Acceleration data (g)
        self.ax_data = deque(maxlen=self.data_plotting)
        self.ay_data = deque(maxlen=self.data_plotting)
        self.az_data = deque(maxlen=self.data_plotting)
        self.acc_mag_data = deque(maxlen=self.data_plotting)
        
        # Angular velocity / Gyro data (rad/s)
        self.wx_data = deque(maxlen=self.data_plotting)
        self.wy_data = deque(maxlen=self.data_plotting)
        self.wz_data = deque(maxlen=self.data_plotting)
        self.gyro_mag_data = deque(maxlen=self.data_plotting)

        # Motor input data (dimensionless, -1 to 1)
        self.motor_input_data = deque(maxlen=self.data_plotting)
    
    def _init_plots(self):
        """Initialize all plot objects and curves."""
        # Configure pyqtgraph appearance
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')
        
        # ----- ALL DATA DISPLAY MODE (Default) -----
        self.angle_plot = self.plot_widget.addPlot(title="Angles (deg)")
        self.angle_plot.showGrid(x=True, y=True)
        self.angle_plot.addLegend()
        self.curve_r = self.angle_plot.plot(pen=(255, 100, 100), name="roll")
        self.curve_p = self.angle_plot.plot(pen=(100, 255, 100), name="pitch")
        self.curve_y = self.angle_plot.plot(pen=(100, 150, 255), name="yaw")
        self.angle_plot.setLabel('left', 'Angle', units='deg')
        
        self.plot_widget.nextRow()
        
        # Acceleration Plot
        self.acc_plot = self.plot_widget.addPlot(title="Linear Accelerations (g)")
        self.acc_plot.showGrid(x=True, y=True)
        self.acc_plot.addLegend()
        self.curve_ax = self.acc_plot.plot(pen=(255, 255, 0), name="acc_x")
        self.curve_ay = self.acc_plot.plot(pen=(255, 0, 255), name="acc_y")
        self.curve_az = self.acc_plot.plot(pen=(0, 255, 255), name="acc_z")
        self.acc_plot.setLabel('left', 'Acceleration', units='g')
        self.acc_plot.setLabel('bottom', 'Time', units='s')

        self.plot_widget.nextRow()
        
        # Angular Velocity Plot
        self.angvel_plot = self.plot_widget.addPlot(title="Angular Velocity (rad/s)")
        self.angvel_plot.showGrid(x=True, y=True)
        self.angvel_plot.addLegend()
        self.curve_wx = self.angvel_plot.plot(pen=(200, 100, 255), name="gyro_x")
        self.curve_wy = self.angvel_plot.plot(pen=(100, 200, 255), name="gyro_y")
        self.curve_wz = self.angvel_plot.plot(pen=(150, 255, 200), name="gyro_z")
        self.angvel_plot.setLabel('left', 'Angular Velocity', units='rad/s')
        self.angvel_plot.setLabel('bottom', 'Time', units='s')

        self.plot_widget.nextRow()

        # Motor Input Plot (All Data mode)
        self.motor_input_plot = self.plot_widget.addPlot(title="Motor Input")
        self.motor_input_plot.showGrid(x=True, y=True)
        self.motor_input_plot.addLegend()
        self.curve_motor_input = self.motor_input_plot.plot(pen=(255, 165, 0), name="motor_input")
        self.motor_input_plot.setLabel('left', 'Power', units='')
        self.motor_input_plot.setLabel('bottom', 'Time', units='s')
        self.motor_input_plot.setYRange(-1.0, 1.0)

        # ----- ANGLES SEPARATE DISPLAY MODE -----
        self.angle_roll_plot = self.plot_widget.addPlot(title="Roll (deg)")
        self.angle_roll_plot.showGrid(x=True, y=True)
        self.angle_roll_plot.addLegend()
        self.curve_roll_sep = self.angle_roll_plot.plot(pen=(255, 100, 100), name="roll")
        self.angle_roll_plot.setLabel('left', 'roll', units='deg')
        self.angle_roll_plot.setVisible(False)
        
        self.plot_widget.nextRow()
        
        self.angle_pitch_plot = self.plot_widget.addPlot(title="Pitch (deg)")
        self.angle_pitch_plot.showGrid(x=True, y=True)
        self.angle_pitch_plot.addLegend()
        self.curve_pitch_sep = self.angle_pitch_plot.plot(pen=(100, 255, 100), name="pitch")
        self.angle_pitch_plot.setLabel('left', 'pitch', units='deg')
        self.angle_pitch_plot.setLabel('bottom', 'Time', units='s')
        self.angle_pitch_plot.setVisible(False)
        
        self.plot_widget.nextRow()
        
        self.angle_yaw_plot = self.plot_widget.addPlot(title="Yaw (deg)")
        self.angle_yaw_plot.showGrid(x=True, y=True)
        self.angle_yaw_plot.addLegend()
        self.curve_yaw_sep = self.angle_yaw_plot.plot(pen=(100, 150, 255), name="yaw")
        self.angle_yaw_plot.setLabel('left', 'yaw', units='deg')
        self.angle_yaw_plot.setLabel('bottom', 'Time', units='s')
        self.angle_yaw_plot.setVisible(False)
        
        # ----- ACCELERATIONS SEPARATE DISPLAY MODE -----
        self.plot_widget.nextRow()
        
        self.acc_x_plot = self.plot_widget.addPlot(title="Acc X (g)")
        self.acc_x_plot.showGrid(x=True, y=True)
        self.acc_x_plot.addLegend()
        self.curve_acc_x_sep = self.acc_x_plot.plot(pen=(255, 255, 0), name="acc_x")
        self.acc_x_plot.setLabel('left', 'acc_x', units='g')
        self.acc_x_plot.setVisible(False)
        
        self.plot_widget.nextRow()
        
        self.acc_y_plot = self.plot_widget.addPlot(title="Acc Y (g)")
        self.acc_y_plot.showGrid(x=True, y=True)
        self.acc_y_plot.addLegend()
        self.curve_acc_y_sep = self.acc_y_plot.plot(pen=(255, 0, 255), name="acc_y")
        self.acc_y_plot.setLabel('left', 'acc_y', units='g')
        self.acc_y_plot.setLabel('bottom', 'Time', units='s')
        self.acc_y_plot.setVisible(False)
        
        self.plot_widget.nextRow()
        
        self.acc_z_plot = self.plot_widget.addPlot(title="Acc Z (g)")
        self.acc_z_plot.showGrid(x=True, y=True)
        self.acc_z_plot.addLegend()
        self.curve_acc_z_sep = self.acc_z_plot.plot(pen=(0, 255, 255), name="acc_z")
        self.acc_z_plot.setLabel('left', 'acc_z', units='g')
        self.acc_z_plot.setVisible(False)
        
        self.plot_widget.nextRow()
        
        self.acc_mag_plot = self.plot_widget.addPlot(title="Acc Magnitude (g)")
        self.acc_mag_plot.showGrid(x=True, y=True)
        self.acc_mag_plot.addLegend()
        self.curve_acc_mag_sep = self.acc_mag_plot.plot(pen=(200, 200, 100), name="acc_mag")
        self.acc_mag_plot.setLabel('left', 'acc_mag', units='g')
        self.acc_mag_plot.setLabel('bottom', 'Time', units='s')
        self.acc_mag_plot.setVisible(False)
        
        # ----- ANGULAR VELOCITY SEPARATE DISPLAY MODE -----
        self.plot_widget.nextRow()
        
        self.gyro_x_plot = self.plot_widget.addPlot(title="Gyro X (rad/s)")
        self.gyro_x_plot.showGrid(x=True, y=True)
        self.gyro_x_plot.addLegend()
        self.curve_gyro_x_sep = self.gyro_x_plot.plot(pen=(200, 100, 255), name="gyro_x")
        self.gyro_x_plot.setLabel('left', 'gyro_x', units='rad/s')
        self.gyro_x_plot.setVisible(False)
        
        self.plot_widget.nextRow()
        
        self.gyro_y_plot = self.plot_widget.addPlot(title="Gyro Y (rad/s)")
        self.gyro_y_plot.showGrid(x=True, y=True)
        self.gyro_y_plot.addLegend()
        self.curve_gyro_y_sep = self.gyro_y_plot.plot(pen=(100, 200, 255), name="gyro_y")
        self.gyro_y_plot.setLabel('left', 'gyro_y', units='rad/s')
        self.gyro_y_plot.setLabel('bottom', 'Time', units='s')
        self.gyro_y_plot.setVisible(False)
        
        self.plot_widget.nextRow()
        
        self.gyro_z_plot = self.plot_widget.addPlot(title="Gyro Z (rad/s)")
        self.gyro_z_plot.showGrid(x=True, y=True)
        self.gyro_z_plot.addLegend()
        self.curve_gyro_z_sep = self.gyro_z_plot.plot(pen=(150, 255, 200), name="gyro_z")
        self.gyro_z_plot.setLabel('left', 'gyro_z', units='rad/s')
        self.gyro_z_plot.setVisible(False)
        
        self.plot_widget.nextRow()
        
        self.gyro_mag_plot = self.plot_widget.addPlot(title="Gyro Magnitude (rad/s)")
        self.gyro_mag_plot.showGrid(x=True, y=True)
        self.gyro_mag_plot.addLegend()
        self.curve_gyro_mag_sep = self.gyro_mag_plot.plot(pen=(150, 200, 200), name="gyro_mag")
        self.gyro_mag_plot.setLabel('left', 'gyro_mag', units='rad/s')
        self.gyro_mag_plot.setLabel('bottom', 'Time', units='s')
        self.gyro_mag_plot.setVisible(False)
        
        # ----- FFT PLOTS (HIDDEN BY DEFAULT) -----
        # These will be shown/hidden based on FFT enabled state and display mode
        self._init_fft_plots()
    
    def _init_fft_plots(self):
        """Initialize FFT (frequency domain) plots for each data type."""
        # ----- FFT FOR ALL DATA MODE -----
        self.plot_widget.nextRow()
        
        self.fft_angle_plot = self.plot_widget.addPlot(title="Angles FFT (Hz)")
        self.fft_angle_plot.showGrid(x=True, y=True)
        self.fft_angle_plot.addLegend()
        self.curve_fft_r = self.fft_angle_plot.plot(pen=(255, 100, 100), name="roll")
        self.curve_fft_p = self.fft_angle_plot.plot(pen=(100, 255, 100), name="pitch")
        self.curve_fft_y = self.fft_angle_plot.plot(pen=(100, 150, 255), name="yaw")
        self.fft_angle_plot.setLabel('left', 'Magnitude', units='')
        self.fft_angle_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.fft_angle_plot.setVisible(False)
        
        self.plot_widget.nextRow()
        
        self.fft_acc_plot = self.plot_widget.addPlot(title="Accelerations FFT (Hz)")
        self.fft_acc_plot.showGrid(x=True, y=True)
        self.fft_acc_plot.addLegend()
        self.curve_fft_ax = self.fft_acc_plot.plot(pen=(255, 255, 0), name="acc_x")
        self.curve_fft_ay = self.fft_acc_plot.plot(pen=(255, 0, 255), name="acc_y")
        self.curve_fft_az = self.fft_acc_plot.plot(pen=(0, 255, 255), name="acc_z")
        self.fft_acc_plot.setLabel('left', 'Magnitude', units='')
        self.fft_acc_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.fft_acc_plot.setVisible(False)
        
        self.plot_widget.nextRow()
        
        self.fft_angvel_plot = self.plot_widget.addPlot(title="Angular Velocity FFT (Hz)")
        self.fft_angvel_plot.showGrid(x=True, y=True)
        self.fft_angvel_plot.addLegend()
        self.curve_fft_wx = self.fft_angvel_plot.plot(pen=(200, 100, 255), name="gyro_x")
        self.curve_fft_wy = self.fft_angvel_plot.plot(pen=(100, 200, 255), name="gyro_y")
        self.curve_fft_wz = self.fft_angvel_plot.plot(pen=(150, 255, 200), name="gyro_z")
        self.fft_angvel_plot.setLabel('left', 'Magnitude', units='')
        self.fft_angvel_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.fft_angvel_plot.setVisible(False)
        
        # ----- FFT FOR INDIVIDUAL ANGLE PLOTS -----
        self.plot_widget.nextRow()
        
        self.fft_roll_plot = self.plot_widget.addPlot(title="Roll FFT (Hz)")
        self.fft_roll_plot.showGrid(x=True, y=True)
        self.fft_roll_plot.addLegend()
        self.curve_fft_roll_sep = self.fft_roll_plot.plot(pen=(255, 100, 100), name="roll")
        self.fft_roll_plot.setLabel('left', 'Magnitude', units='')
        self.fft_roll_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.fft_roll_plot.setVisible(False)
        
        self.plot_widget.nextRow()
        
        self.fft_pitch_plot = self.plot_widget.addPlot(title="Pitch FFT (Hz)")
        self.fft_pitch_plot.showGrid(x=True, y=True)
        self.fft_pitch_plot.addLegend()
        self.curve_fft_pitch_sep = self.fft_pitch_plot.plot(pen=(100, 255, 100), name="pitch")
        self.fft_pitch_plot.setLabel('left', 'Magnitude', units='')
        self.fft_pitch_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.fft_pitch_plot.setVisible(False)
        
        self.plot_widget.nextRow()
        
        self.fft_yaw_plot = self.plot_widget.addPlot(title="Yaw FFT (Hz)")
        self.fft_yaw_plot.showGrid(x=True, y=True)
        self.fft_yaw_plot.addLegend()
        self.curve_fft_yaw_sep = self.fft_yaw_plot.plot(pen=(100, 150, 255), name="yaw")
        self.fft_yaw_plot.setLabel('left', 'Magnitude', units='')
        self.fft_yaw_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.fft_yaw_plot.setVisible(False)
        
        self.plot_widget.nextRow()
        
        self.fft_acc_mag_plot = self.plot_widget.addPlot(title="Acc Magnitude FFT (Hz)")
        self.fft_acc_mag_plot.showGrid(x=True, y=True)
        self.fft_acc_mag_plot.addLegend()
        self.curve_fft_acc_mag_sep = self.fft_acc_mag_plot.plot(pen=(200, 200, 100), name="Magnitude")
        self.fft_acc_mag_plot.setLabel('left', 'Magnitude', units='')
        self.fft_acc_mag_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.fft_acc_mag_plot.setVisible(False)
        
        self.plot_widget.nextRow()
        
        self.fft_gyro_mag_plot = self.plot_widget.addPlot(title="Gyro Magnitude FFT (Hz)")
        self.fft_gyro_mag_plot.showGrid(x=True, y=True)
        self.fft_gyro_mag_plot.addLegend()
        self.curve_fft_gyro_mag_sep = self.fft_gyro_mag_plot.plot(pen=(150, 200, 200), name="Magnitude")
        self.fft_gyro_mag_plot.setLabel('left', 'Magnitude', units='')
        self.fft_gyro_mag_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.fft_gyro_mag_plot.setVisible(False)

        # Leave the next row ready for any additional plots added to the shared layout.
        self.plot_widget.nextRow()
    
    def _init_scrubber_lines(self):
        """Initialize vertical scrubber lines for all plots."""
        # Create a dictionary to store vertical lines for each plot
        self.scrubber_lines = {}
        
        # Line style: white, semi-transparent, dashed
        line_pen = pg.mkPen(color=(255, 255, 255), width=2, style=Qt.PenStyle.DashLine)
        
        # Add vertical lines to all time-domain plots
        # All Data Mode plots
        self.scrubber_lines['angle'] = pg.InfiniteLine(angle=90, movable=False, pen=line_pen)
        self.angle_plot.addItem(self.scrubber_lines['angle'])
        
        self.scrubber_lines['acc'] = pg.InfiniteLine(angle=90, movable=False, pen=line_pen)
        self.acc_plot.addItem(self.scrubber_lines['acc'])
        
        self.scrubber_lines['angvel'] = pg.InfiniteLine(angle=90, movable=False, pen=line_pen)
        self.angvel_plot.addItem(self.scrubber_lines['angvel'])

        self.scrubber_lines['motor_input'] = pg.InfiniteLine(angle=90, movable=False, pen=line_pen)
        self.motor_input_plot.addItem(self.scrubber_lines['motor_input'])

        # Angles Separate Mode plots
        self.scrubber_lines['roll'] = pg.InfiniteLine(angle=90, movable=False, pen=line_pen)
        self.angle_roll_plot.addItem(self.scrubber_lines['roll'])
        
        self.scrubber_lines['pitch'] = pg.InfiniteLine(angle=90, movable=False, pen=line_pen)
        self.angle_pitch_plot.addItem(self.scrubber_lines['pitch'])
        
        self.scrubber_lines['yaw'] = pg.InfiniteLine(angle=90, movable=False, pen=line_pen)
        self.angle_yaw_plot.addItem(self.scrubber_lines['yaw'])
        
        # Accelerations Separate Mode plots
        self.scrubber_lines['acc_x'] = pg.InfiniteLine(angle=90, movable=False, pen=line_pen)
        self.acc_x_plot.addItem(self.scrubber_lines['acc_x'])
        
        self.scrubber_lines['acc_y'] = pg.InfiniteLine(angle=90, movable=False, pen=line_pen)
        self.acc_y_plot.addItem(self.scrubber_lines['acc_y'])
        
        self.scrubber_lines['acc_z'] = pg.InfiniteLine(angle=90, movable=False, pen=line_pen)
        self.acc_z_plot.addItem(self.scrubber_lines['acc_z'])
        
        self.scrubber_lines['acc_mag'] = pg.InfiniteLine(angle=90, movable=False, pen=line_pen)
        self.acc_mag_plot.addItem(self.scrubber_lines['acc_mag'])
        
        # Angular Velocity Separate Mode plots
        self.scrubber_lines['gyro_x'] = pg.InfiniteLine(angle=90, movable=False, pen=line_pen)
        self.gyro_x_plot.addItem(self.scrubber_lines['gyro_x'])
        
        self.scrubber_lines['gyro_y'] = pg.InfiniteLine(angle=90, movable=False, pen=line_pen)
        self.gyro_y_plot.addItem(self.scrubber_lines['gyro_y'])
        
        self.scrubber_lines['gyro_z'] = pg.InfiniteLine(angle=90, movable=False, pen=line_pen)
        self.gyro_z_plot.addItem(self.scrubber_lines['gyro_z'])
        
        self.scrubber_lines['gyro_mag'] = pg.InfiniteLine(angle=90, movable=False, pen=line_pen)
        self.gyro_mag_plot.addItem(self.scrubber_lines['gyro_mag'])

        # Scrubber is analyse-only; keep hidden unless analyse mode enables it.
        self.set_scrubber_visible(False)

    def set_scrubber_visible(self, visible):
        """Show/hide scrubber lines across all time-domain plots."""
        for line in self.scrubber_lines.values():
            line.setVisible(visible)

    def clear_marker_lines(self):
        """Remove any existing marker vertical lines and labels from plots."""
        for tup in self.marker_lines:
            try:
                plot, line, text = tup
                if text is not None:
                    try:
                        plot.removeItem(text)
                    except:
                        pass
                try:
                    plot.removeItem(line)
                except:
                    pass
            except:
                pass
        self.marker_lines.clear()

    def add_marker_line(self, timestamp, label, color=None):
        """Add a vertical marker line (and small label) at `timestamp` across time plots.

        Args:
            timestamp: x-position in seconds
            label: marker label text
            color: optional RGB tuple (r,g,b) or (r,g,b,a)
        """
        # Default color mapping for marker_1..marker_4
        mapping = {
            'marker_1': (255, 0, 0),    # red
            'marker_2': (0, 0, 255),    # blue
            'marker_3': (0, 255, 0),    # green
            'marker_4': (255, 255, 0),  # yellow
        }
        key = str(label).lower() if label is not None else ''
        pen_color = color if color is not None else mapping.get(key, (200, 200, 200))
        pen = pg.mkPen(color=pen_color, width=2)

        plots_to_update = [
            self.angle_plot, self.acc_plot, self.angvel_plot, self.motor_input_plot,
            self.angle_roll_plot, self.angle_pitch_plot, self.angle_yaw_plot,
            self.acc_x_plot, self.acc_y_plot, self.acc_z_plot, self.acc_mag_plot,
            self.gyro_x_plot, self.gyro_y_plot, self.gyro_z_plot, self.gyro_mag_plot
        ]

        for plot in plots_to_update:
            try:
                line = pg.InfiniteLine(angle=90, movable=False, pen=pen)
                plot.addItem(line)
                line.setValue(timestamp)

                # Add a small text label above the plot near the top
                try:
                    y_range = plot.getViewBox().viewRange()[1]
                    y_pos = y_range[1] - 0.05 * (y_range[1] - y_range[0])
                except Exception:
                    y_pos = 0
                txt = pg.TextItem(html=f'<div style="background:black;color:white;padding:2px;border-radius:3px;">{label}</div>', anchor=(0.5, 1))
                txt.setPos(timestamp, y_pos)
                try:
                    txt.setZValue(1000)
                except Exception:
                    pass
                plot.addItem(txt)

                self.marker_lines.append((plot, line, txt))
            except Exception:
                # Ignore individual plot failures
                pass

    def set_graph_visibility_state(self, visibility_state):
        """Apply graph group visibility and refresh the visible plots."""
        if "fft" in visibility_state:
            self.fft_enabled = bool(visibility_state["fft"])
        self.graph_visibility_state.update({key: bool(value) for key, value in visibility_state.items() if key in self.graph_visibility_state})
        self._update_plot_visibility()
        if self.labels:
            self.update_label_overlays()

    def set_graph_visibility(self, graph_key, visible):
        """Toggle a single graph group on or off."""
        if graph_key not in self.graph_visibility_state:
            return
        self.graph_visibility_state[graph_key] = bool(visible)
        self._update_plot_visibility()
        if self.labels:
            self.update_label_overlays()
    
    def _update_plot_visibility(self):
        """Update plot visibility based on the current toggle state."""
        full_data_enabled = self.graph_visibility_state.get("full_data", True)
        all_time_plots = [
            self.angle_plot, self.acc_plot, self.angvel_plot, self.motor_input_plot,
            self.angle_roll_plot, self.angle_pitch_plot, self.angle_yaw_plot,
            self.acc_x_plot, self.acc_y_plot, self.acc_z_plot, self.acc_mag_plot,
            self.gyro_x_plot, self.gyro_y_plot, self.gyro_z_plot, self.gyro_mag_plot,
        ]
        all_fft_plots = [
            self.fft_angle_plot, self.fft_acc_plot, self.fft_angvel_plot,
            self.fft_roll_plot, self.fft_pitch_plot, self.fft_yaw_plot, self.fft_acc_mag_plot,self.fft_gyro_mag_plot,
        ]

        for plot in all_time_plots:
            plot.setVisible(False)
        for plot in all_fft_plots:
            plot.setVisible(False)

        if full_data_enabled and self.graph_visibility_state.get("angle_xyz", False):
            self.angle_plot.setVisible(True)
            if self.fft_enabled:
                self.fft_angle_plot.setVisible(True)

        if full_data_enabled and self.graph_visibility_state.get("acc_xyz", False):
            self.acc_plot.setVisible(True)
            if self.fft_enabled:
                self.fft_acc_plot.setVisible(True)

        if full_data_enabled and self.graph_visibility_state.get("acc_mag", False):
            self.acc_mag_plot.setVisible(True)
            if self.fft_enabled:
                self.fft_acc_mag_plot.setVisible(True)

        if full_data_enabled and self.graph_visibility_state.get("gyro_xyz", False):
            self.angvel_plot.setVisible(True)
            if self.fft_enabled:
                self.fft_angvel_plot.setVisible(True)

        if full_data_enabled and self.graph_visibility_state.get("gyro_mag", False):
            self.gyro_mag_plot.setVisible(True)
            if self.fft_enabled:
                self.fft_gyro_mag_plot.setVisible(True)

        if full_data_enabled and self.graph_visibility_state.get("motor_input", False):
            self.motor_input_plot.setVisible(True)

        # Keep the detailed per-axis plots available for label and marker overlays,
        # but do not show them in the new toggle-driven main layout.
    
    @staticmethod
    def _finite_xy(x, y):
        """Return x, y as float arrays with non-finite (NaN/Inf) pairs removed.

        Auto-ranging a plot over all-NaN data makes pyqtgraph raise
        "Not setting range to [nan, nan]" from inside Qt's repaint, which then
        re-fires every frame and freezes the UI. Stripping non-finite points
        avoids it; an empty result is safe (auto-range becomes a no-op).
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.size == 0 or y_arr.size == 0:
            return x_arr[:0], y_arr[:0]
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        return x_arr[mask], y_arr[mask]

    def _set_fft_curve(self, curve, freq, mag):
        """setData on an FFT curve after stripping non-finite points."""
        fx, fy = self._finite_xy(freq, mag)
        curve.setData(fx, fy)

    def update_scrubber_line_position(self, timestamp):
        """
        Update the position of the scrubber line to the given timestamp.

        Args:
            timestamp: The timestamp (x-value) at which to place the vertical line
        """
        if np.isfinite(timestamp):
            for line in self.scrubber_lines.values():
                line.setValue(timestamp)
        
        # Update analysis window region position
        self.update_analysis_window_position(timestamp)
    
    def init_analysis_window_regions(self):
        """Initialize analysis window region items on all plots."""
        self.clear_analysis_window_regions()
        
        # Window style: semi-transparent light blue background
        window_pen = pg.mkPen(color=(100, 150, 255), width=6)  # Thick light blue border
        window_brush = pg.mkBrush(color=(100, 150, 255, 30))  # Light blue with transparency
        
        plots_to_update = [
            self.angle_plot, self.acc_plot, self.angvel_plot, self.motor_input_plot,
            self.angle_roll_plot, self.angle_pitch_plot, self.angle_yaw_plot,
            self.acc_x_plot, self.acc_y_plot, self.acc_z_plot, self.acc_mag_plot,
            self.gyro_x_plot, self.gyro_y_plot, self.gyro_z_plot, self.gyro_mag_plot
        ]

        for plot in plots_to_update:
            region = pg.LinearRegionItem(values=[0, 1.0],
                                        orientation='vertical',
                                        pen=window_pen,
                                        brush=window_brush,
                                        movable=False)
            self.analysis_window_regions.append((plot, region))
            # ignoreBounds=True: this is just a highlight, it must not drive the
            # plot's auto-range (and must not poison it if the window edges go NaN).
            plot.addItem(region, ignoreBounds=True)
    
    def clear_analysis_window_regions(self):
        """Remove all analysis window regions from plots."""
        for plot, region in self.analysis_window_regions:
            try:
                plot.removeItem(region)
            except:
                pass
        self.analysis_window_regions.clear()
    
    def update_analysis_window_position(self, timestamp):
        """
        Update the analysis window region position behind (to the left of) the scrubber.
        
        Args:
            timestamp: Timestamp for the scrubber (right edge of window)
        """
        window_start = timestamp - self.analysis_window_width
        window_end = timestamp

        if not (np.isfinite(window_start) and np.isfinite(window_end)):
            return
        for plot, region in self.analysis_window_regions:
            region.setRegion([window_start, window_end])

    def set_analysis_window_region(self, start_ts, end_ts):
        """Set the analysis window region explicitly using start/end timestamps.

        This is used when windows are defined by a fixed number of samples
        and the duration must be computed from actual timestamps.
        """
        if not (np.isfinite(start_ts) and np.isfinite(end_ts)):
            return
        for plot, region in self.analysis_window_regions:
            region.setRegion([start_ts, end_ts])
    
    def set_analysis_window_width(self, width):
        """
        Set the analysis window width in seconds.

        Args:
            width: Window width in seconds
        """
        self.analysis_window_width = max(0.01, width)  # Minimum 0.01 seconds

    # --- Window hover / click interaction ---
    def _window_interaction_plots(self):
        """Time-domain plots that participate in window hover/click highlighting."""
        return [
            self.angle_plot, self.acc_plot, self.acc_mag_plot,
            self.angvel_plot, self.gyro_mag_plot, self.motor_input_plot,
        ]

    def _init_window_hover_highlight(self):
        """Create the (hidden) hover-highlight region drawn on the window under the mouse."""
        hover_pen = pg.mkPen(color=(255, 220, 0), width=2)        # yellow border
        hover_brush = pg.mkBrush(color=(255, 220, 0, 45))         # translucent yellow
        self.window_hover_regions = []
        for plot in self._window_interaction_plots():
            region = pg.LinearRegionItem(values=[0, 0], orientation='vertical',
                                         pen=hover_pen, brush=hover_brush, movable=False)
            try:
                region.setZValue(500)  # above data/current-window region, below text boxes
            except Exception:
                pass
            region.setVisible(False)
            # ignoreBounds=True so the highlight never drives the plot's auto-range.
            plot.addItem(region, ignoreBounds=True)
            self.window_hover_regions.append((plot, region))

    def set_window_hover_highlight(self, start_ts, end_ts):
        """Show the hover highlight spanning [start_ts, end_ts] on visible plots."""
        if not (np.isfinite(start_ts) and np.isfinite(end_ts)):
            return
        for plot, region in self.window_hover_regions:
            region.setRegion([start_ts, end_ts])
            region.setVisible(plot.isVisible())

    def clear_window_hover_highlight(self):
        """Hide the hover highlight on all plots."""
        self._hover_window_idx = None
        for _, region in self.window_hover_regions:
            try:
                region.setVisible(False)
            except Exception:
                pass

    def window_index_at_time(self, timestamp):
        """Return the index of the window under `timestamp`, or None.

        Windows can overlap; prefer a window that actually contains the timestamp
        (closest center wins on ties), and fall back to the nearest window center
        when the timestamp lands between windows.
        """
        if not self.window_indices or timestamp is None or not np.isfinite(timestamp):
            return None

        best_idx = None
        best_dist = float("inf")
        for idx, (start, end) in enumerate(self.window_indices):
            if start <= timestamp <= end:
                dist = abs(0.5 * (start + end) - timestamp)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
        if best_idx is not None:
            return best_idx

        for idx, (start, end) in enumerate(self.window_indices):
            dist = abs(0.5 * (start + end) - timestamp)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def _time_at_scene_pos(self, scene_pos):
        """Map a scene position to a time (x) value using whichever visible plot
        contains it, or None when the cursor is not over a participating plot."""
        for plot in self._window_interaction_plots():
            if not plot.isVisible():
                continue
            view_box = plot.getViewBox()
            if view_box is None:
                continue
            try:
                if view_box.sceneBoundingRect().contains(scene_pos):
                    return float(view_box.mapSceneToView(scene_pos).x())
            except Exception:
                continue
        return None

    def _connect_window_interaction(self):
        """Wire up scene-level mouse move/click signals (connected once)."""
        try:
            scene = self.plot_widget.scene()
            scene.sigMouseMoved.connect(self._on_scene_mouse_moved)
            scene.sigMouseClicked.connect(self._on_scene_mouse_clicked)
        except Exception:
            pass

    def _on_scene_mouse_moved(self, scene_pos):
        """Highlight the window under the mouse (no-op when no tab is active)."""
        if self.window_click_callback is None or not self.window_indices:
            return
        timestamp = self._time_at_scene_pos(scene_pos)
        if timestamp is None:
            self.clear_window_hover_highlight()
            return
        idx = self.window_index_at_time(timestamp)
        if idx is None:
            self.clear_window_hover_highlight()
            return
        start, end = self.window_indices[idx]
        self.set_window_hover_highlight(start, end)
        self._hover_window_idx = idx

    def _on_scene_mouse_clicked(self, event):
        """On a plain left-click over a window, ask the active tab to jump to it."""
        if self.window_click_callback is None or not self.window_indices:
            return
        try:
            if event.button() != Qt.MouseButton.LeftButton or event.double():
                return
        except Exception:
            pass
        try:
            scene_pos = event.scenePos()
        except Exception:
            return
        timestamp = self._time_at_scene_pos(scene_pos)
        if timestamp is None:
            return
        idx = self.window_index_at_time(timestamp)
        if idx is None:
            return
        try:
            self.window_click_callback(idx)
        except Exception:
            pass
        
    # --- Labeling Methods ---
    def add_label_region(self, start_time, end_time, label, plot_obj):
        """
        Add a colored LinearRegionItem overlay for a labeled region.
        
        Args:
            start_time: Time for region start
            end_time: Time for region end
            label: Label string (e.g., 'Spin', 'Tap')
            plot_obj: PyQtGraph PlotItem to add overlay to
        """
        if label not in self.label_colors:
            color = (200, 200, 200, 30)  # Default light gray with transparency
        else:
            color = self.label_colors[label]
        region = pg.LinearRegionItem(values=[start_time, end_time], orientation='vertical', movable=False,
                                     brush=pg.mkBrush(*color), pen=pg.mkPen(color, width=1))
        # Create a text label centered above the region instead of InfLineLabel
        plot_obj.addItem(region)

        # Create a TextItem with a black background via HTML/CSS, unless text is
        # suppressed (e.g. classification analysis renders its own comparison box).
        txt = None
        if self.label_overlay_show_text:
            mid = 0.5 * (start_time + end_time)
            # Determine a y position near the top of the plot's current view
            try:
                y_range = plot_obj.getViewBox().viewRange()[1]
                y_pos = y_range[1] - 0.05 * (y_range[1] - y_range[0])
            except Exception:
                y_pos = 0

            html = '<div style="background-color: black; color: white; padding: 2px; border-radius: 3px;">%s</div>' % label
            txt = pg.TextItem(html=html, anchor=(0.5, 0))
            txt.setPos(mid, y_pos)
            # Ensure text (and its background) render above other items
            try:
                txt.setZValue(1000)
            except Exception:
                pass
            # ignoreBounds=True keeps the text out of the view's auto-range calculation.
            # Otherwise each label's text near the top pushes the view top higher, and the
            # next label (positioned relative to the new top) lands higher still.
            plot_obj.addItem(txt, ignoreBounds=True)

        # Store metadata for later cleanup
        region.label = label  # type: ignore
        region.plot_obj = plot_obj  # type: ignore
        region.text_item = txt  # type: ignore
        # Backwards compatibility: ensure attribute exists
        region.text_outline_items = None  # type: ignore
        self.label_region_items.append(region)

        # Respect the current show/hide state for label overlays.
        if not self.labels_visible:
            region.setVisible(False)
            if txt is not None:
                txt.setVisible(False)

        return region
    
    def set_label_overlays_visible(self, visible):
        """Show or hide already-added label overlays without removing them.

        Only affects the colored region overlays for previously labeled windows;
        the current-window indicator (analysis_window_regions) is left untouched.
        """
        self.labels_visible = bool(visible)
        for region in self.label_region_items:
            try:
                region.setVisible(self.labels_visible)
                if hasattr(region, 'text_item') and region.text_item is not None:
                    region.text_item.setVisible(self.labels_visible)
            except Exception:
                pass

    def clear_label_overlays(self):
        """Remove all label overlay regions from plots."""
        for region in self.label_region_items:
            try:
                # Remove associated text label if present
                # Remove associated outline items
                if hasattr(region, 'text_outline_items') and region.text_outline_items:
                    for oi in region.text_outline_items:
                        try:
                            region.plot_obj.removeItem(oi)
                        except:
                            pass

                # Remove associated main text label
                if hasattr(region, 'text_item') and region.text_item is not None:
                    try:
                        region.plot_obj.removeItem(region.text_item)
                    except:
                        pass

                region.plot_obj.removeItem(region)
            except:
                pass
        self.label_region_items.clear()
        
    def _get_visible_plots(self):
        """Get list of visible plots based on current display mode."""
        visible_plots = []
        if self.graph_visibility_state.get("angle_xyz", False):
            visible_plots.append(self.angle_plot)
        if self.graph_visibility_state.get("acc_xyz", False):
            visible_plots.append(self.acc_plot)
        if self.graph_visibility_state.get("acc_mag", False):
            visible_plots.append(self.acc_mag_plot)
        if self.graph_visibility_state.get("gyro_xyz", False):
            visible_plots.append(self.angvel_plot)
        if self.graph_visibility_state.get("gyro_mag", False):
            visible_plots.append(self.gyro_mag_plot)
        return visible_plots
        return []
    
    def update_label_overlays(self):
        """Refresh label overlays on currently visible plots (for display mode changes)."""
        self.clear_label_overlays()
        
        visible_plots = self._get_visible_plots()
            
        # Add region items for each label
        for window_idx, label in self.labels.items():
            if window_idx < len(self.window_indices):
                start_time, end_time = self.window_indices[window_idx]
                for plot in visible_plots:
                    self.add_label_region(start_time, end_time, label, plot)
    
    def add_label_incremental(self, window_idx, label):
        """
        Add a label region for a single window to all visible plots.
        This is O(1) per label instead of O(n), providing massive performance improvement.
        
        Args:
            window_idx: Index of the window to label
            label: Label string for the window
        """
        if window_idx >= len(self.window_indices):
            return
            
        start_time, end_time = self.window_indices[window_idx]
        visible_plots = self._get_visible_plots()
        
        for plot in visible_plots:
            self.add_label_region(start_time, end_time, label, plot)
                    
    def label_current_window(self, label):
        """Label the current window and advance to next."""
        if self.current_window_idx >= 0:
            # Manual labeling always shows the label text box (classification
            # analysis may have turned it off for its own comparison overlay).
            self.label_overlay_show_text = True
            self.labels[self.current_window_idx] = label
            # Use incremental add instead of full redraw for O(1) performance
            self.add_label_incremental(self.current_window_idx, label)
            
    def set_window_parameters(self, window_indices_times, window_size, stride):
        """Set up window tracking for analyse mode."""
        self.window_indices = window_indices_times
        self.window_size = window_size
        self.stride = stride
        self.current_window_idx = 0
    # ------------------------
    
    def clear_gap_regions(self):
        """Remove all gap region indicators from plots."""
        for plot, region, text in self.gap_regions:
            try:
                if text is not None:
                    plot.removeItem(text)
            except:
                pass
            try:
                plot.removeItem(region)
            except:
                pass
        self.gap_regions.clear()
    
    def highlight_data_gaps(self, data_list, gap_threshold=0.1):
        """
        Identify and visualize data gaps greater than the threshold.
        
        Args:
            data_list: List of data dictionaries with 'timestamp' key
            gap_threshold: Minimum gap duration in seconds to highlight (default: 0.1s)
        """
        self.clear_gap_regions()
        
        if not data_list or len(data_list) < 2:
            return
        
        # Find gaps
        gaps = []
        for i in range(len(data_list) - 1):
            current_time = data_list[i]['timestamp']
            next_time = data_list[i + 1]['timestamp']
            dt = next_time - current_time
            
            if dt > gap_threshold:
                # Gap found: from end of current point to start of next point
                gaps.append((current_time, next_time))
        
        if not gaps:
            return
        
        # Create red region items for each gap on each plot
        region_pen = pg.mkPen(color=(255, 0, 0), width=0)  # No border
        region_brush = pg.mkBrush(color=(255, 0, 0, 10))  # Red with transparency
        
        # Add regions to each plot
        plots_to_update = [
            self.angle_plot, self.acc_plot, self.angvel_plot, self.motor_input_plot,
            self.angle_roll_plot, self.angle_pitch_plot, self.angle_yaw_plot,
            self.acc_x_plot, self.acc_y_plot, self.acc_z_plot, self.acc_mag_plot,
            self.gyro_x_plot, self.gyro_y_plot, self.gyro_z_plot, self.gyro_mag_plot
        ]
        
        for gap_start, gap_end in gaps:
            gap_duration = gap_end - gap_start
            label_text = f"Gap ({gap_duration:.3f}s)"
            x_mid = 0.5 * (gap_start + gap_end)
            # Create a separate region for each plot (items can only have one parent)
            for plot in plots_to_update:
                region = pg.LinearRegionItem(values=[gap_start, gap_end], 
                                            orientation='vertical',
                                            pen=region_pen, 
                                            brush=region_brush,
                                            movable=False)
                plot.addItem(region)

                # Add a small label near the top of each gap region.
                text_item = None
                try:
                    y_range = plot.getViewBox().viewRange()[1]
                    y_pos = y_range[1] - 0.05 * (y_range[1] - y_range[0])
                    text_item = pg.TextItem(
                        html=f'<div style="background:black;color:#ff9999;padding:2px;border-radius:3px;">{label_text}</div>',
                        anchor=(0.5, 1)
                    )
                    text_item.setPos(x_mid, y_pos)
                    try:
                        text_item.setZValue(1000)
                    except Exception:
                        pass
                    plot.addItem(text_item)
                except Exception:
                    text_item = None

                self.gap_regions.append((plot, region, text_item))
    
    def append_data(self, t_s, r_deg, p_deg, y_deg, acc_x, acc_y, acc_z,
                    acc_magnitude, gyro_x, gyro_y, gyro_z, gyro_magnitude,
                    motor_input=0.0):
        """
        Append new data point to all data deques and update plots.

        Args:
            t_s: Time in seconds
            r_deg: Roll angle in degrees
            p_deg: Pitch angle in degrees
            y_deg: Yaw angle in degrees
            acc_x, acc_y, acc_z: Linear acceleration components (g)
            acc_magnitude: Magnitude of linear acceleration (g)
            gyro_x, gyro_y, gyro_z: Angular velocity components (rad/s)
            gyro_magnitude: Magnitude of angular velocity (rad/s)
            motor_input: Motor power command (-1 to 1)
        """
        # Update frequency estimate
        if self.last_timestamp is not None:
            dt = t_s - self.last_timestamp
            if dt > 0:
                instantaneous_freq = 1.0 / dt
                self.frequency_estimate = 0.8 * self.frequency_estimate + 0.2 * instantaneous_freq
        self.last_timestamp = t_s
        
        # Append all data
        self.t_data.append(t_s)
        self.r_data.append(r_deg)
        self.p_data.append(p_deg)
        self.y_data.append(y_deg)
        
        self.ax_data.append(acc_x)
        self.ay_data.append(acc_y)
        self.az_data.append(acc_z)
        self.acc_mag_data.append(acc_magnitude)
        
        self.wx_data.append(gyro_x)
        self.wy_data.append(gyro_y)
        self.wz_data.append(gyro_z)
        self.gyro_mag_data.append(gyro_magnitude)
        self.motor_input_data.append(motor_input)

        # Update all plots
        self.update_all_curves()
    
    def update_all_curves(self):
        """Update all plot curves with current data."""
        t_list = list(self.t_data)
        
        # Update All Data mode curves
        self.curve_r.setData(t_list, list(self.r_data))
        self.curve_p.setData(t_list, list(self.p_data))
        self.curve_y.setData(t_list, list(self.y_data))
        
        self.curve_ax.setData(t_list, list(self.ax_data))
        self.curve_ay.setData(t_list, list(self.ay_data))
        self.curve_az.setData(t_list, list(self.az_data))
        
        self.curve_wx.setData(t_list, list(self.wx_data))
        self.curve_wy.setData(t_list, list(self.wy_data))
        self.curve_wz.setData(t_list, list(self.wz_data))
        
        # Update separate angle mode curves
        self.curve_roll_sep.setData(t_list, list(self.r_data))
        self.curve_pitch_sep.setData(t_list, list(self.p_data))
        self.curve_yaw_sep.setData(t_list, list(self.y_data))
        
        # Update separate acceleration mode curves
        self.curve_acc_x_sep.setData(t_list, list(self.ax_data))
        self.curve_acc_y_sep.setData(t_list, list(self.ay_data))
        self.curve_acc_z_sep.setData(t_list, list(self.az_data))
        self.curve_acc_mag_sep.setData(t_list, list(self.acc_mag_data))
        
        # Update separate gyro mode curves
        self.curve_gyro_x_sep.setData(t_list, list(self.wx_data))
        self.curve_gyro_y_sep.setData(t_list, list(self.wy_data))
        self.curve_gyro_z_sep.setData(t_list, list(self.wz_data))
        self.curve_gyro_mag_sep.setData(t_list, list(self.gyro_mag_data))

        # Update motor input curve
        self.curve_motor_input.setData(t_list, list(self.motor_input_data))

        # Update FFT plots if enabled
        if self.fft_enabled:
            self.update_fft_plots()
    
    def clear_all_data(self):
        """Clear all data deques and reset plot curves."""
        # Clear data deques
        self.t_data.clear()
        self.r_data.clear()
        self.p_data.clear()
        self.y_data.clear()
        
        self.ax_data.clear()
        self.ay_data.clear()
        self.az_data.clear()
        self.acc_mag_data.clear()
        
        self.wx_data.clear()
        self.wy_data.clear()
        self.wz_data.clear()
        self.gyro_mag_data.clear()
        self.motor_input_data.clear()

        # Reset frequency tracking
        self.last_timestamp = None
        self.frequency_estimate = 0.0
        
        # Clear all plot curves
        self.curve_r.setData([], [])
        self.curve_p.setData([], [])
        self.curve_y.setData([], [])
        
        self.curve_ax.setData([], [])
        self.curve_ay.setData([], [])
        self.curve_az.setData([], [])
        
        self.curve_wx.setData([], [])
        self.curve_wy.setData([], [])
        self.curve_wz.setData([], [])
        
        # Clear separate angle curves
        self.curve_roll_sep.setData([], [])
        self.curve_pitch_sep.setData([], [])
        self.curve_yaw_sep.setData([], [])
        
        # Clear separate acceleration curves
        self.curve_acc_x_sep.setData([], [])
        self.curve_acc_y_sep.setData([], [])
        self.curve_acc_z_sep.setData([], [])
        self.curve_acc_mag_sep.setData([], [])
        
        # Clear separate gyro curves
        self.curve_gyro_x_sep.setData([], [])
        self.curve_gyro_y_sep.setData([], [])
        self.curve_gyro_z_sep.setData([], [])
        self.curve_gyro_mag_sep.setData([], [])

        # Clear motor input curve
        self.curve_motor_input.setData([], [])

        # Clear FFT curves
        self.curve_fft_r.setData([], [])
        self.curve_fft_p.setData([], [])
        self.curve_fft_y.setData([], [])
        
        self.curve_fft_ax.setData([], [])
        self.curve_fft_ay.setData([], [])
        self.curve_fft_az.setData([], [])
        
        self.curve_fft_wx.setData([], [])
        self.curve_fft_wy.setData([], [])
        self.curve_fft_wz.setData([], [])
        
        self.curve_fft_roll_sep.setData([], [])
        self.curve_fft_pitch_sep.setData([], [])
        self.curve_fft_yaw_sep.setData([], [])
    
        self.curve_fft_gyro_mag_sep.setData([], [])
    
    def set_fft_enabled(self, enabled):
        """Enable or disable FFT display."""
        self.fft_enabled = enabled
        self.graph_visibility_state["fft"] = bool(enabled)
        self._update_plot_visibility()
    
    def clear_fft_graphs(self):
        """Clear all FFT plot data."""
        # Clear all FFT curves
        self.curve_fft_r.setData([], [])
        self.curve_fft_p.setData([], [])
        self.curve_fft_y.setData([], [])
        
        self.curve_fft_ax.setData([], [])
        self.curve_fft_ay.setData([], [])
        self.curve_fft_az.setData([], [])
        
        self.curve_fft_wx.setData([], [])
        self.curve_fft_wy.setData([], [])
        self.curve_fft_wz.setData([], [])
        
        self.curve_fft_roll_sep.setData([], [])
        self.curve_fft_pitch_sep.setData([], [])
        self.curve_fft_yaw_sep.setData([], [])
    
        self.curve_fft_gyro_mag_sep.setData([], [])
    
    def set_window_size(self, size, unit="Samples"):
        """Store the FFT window definition for later timestamp-based resolution."""
        self.fft_window_unit = unit
        self.fft_window_size_value = float(size)
        if unit == "Samples":
            self.fft_samples = max(1, int(round(float(size))))

    def _get_timestamp_window_sample_count(self, timestamps, window_seconds):
        """Resolve the sample count for a time-based window from ordered timestamps."""
        if not timestamps:
            return max(1, int(round(self.fft_samples)))

        timestamp_array = np.asarray(list(timestamps), dtype=float)
        if timestamp_array.size <= 1:
            return int(timestamp_array.size)

        target_timestamp = float(timestamp_array[-1]) - float(window_seconds)
        if target_timestamp <= float(timestamp_array[0]):
            return int(timestamp_array.size)

        insert_idx = int(np.searchsorted(timestamp_array, target_timestamp, side="left"))
        left_idx = max(0, insert_idx - 1)
        right_idx = min(insert_idx, timestamp_array.size - 1)

        left_distance = abs(float(timestamp_array[left_idx]) - target_timestamp)
        right_distance = abs(float(timestamp_array[right_idx]) - target_timestamp)
        start_idx = right_idx if right_distance < left_distance else left_idx
        return max(1, int(timestamp_array.size - start_idx))

    def get_fft_window_sample_count(self, timestamps=None):
        """Return the current FFT window length in samples."""
        if self.fft_window_unit == "Seconds":
            return self._get_timestamp_window_sample_count(timestamps, self.fft_window_size_value)

        return max(1, int(round(self.fft_window_size_value)))
    
    def compute_fft_windowed(self, data_chunk, window_samples=None):
        """
        Compute FFT with Hanning window on a data chunk.
        
        Args:
            data_chunk: numpy array or list of samples
            
        Returns:
            frequencies: frequency axis (Hz)
            magnitude: magnitude spectrum (normalized)
        """
        data_array = np.array(data_chunk, dtype=float)
        window_samples = self.get_fft_window_sample_count() if window_samples is None else max(1, int(window_samples))
        
        if len(data_array) < 2:
            return np.array([0]), np.array([0])
        
        # Ensure data length matches FFT sample size
        if len(data_array) < window_samples:
            # Pad with zeros if not enough data
            data_array = np.pad(data_array, (0, window_samples - len(data_array)), 'constant')
        else:
            # Take only the last window_samples
            data_array = data_array[-window_samples:]
        
        # Create Hanning window
        window = np.hanning(len(data_array))
        
        # Apply window to data
        windowed_data = data_array * window
        
        # Compute FFT
        fft_result = np.fft.fft(windowed_data)
        
        # Compute magnitude spectrum (normalized by window energy)
        magnitude = np.abs(fft_result) / np.sum(window)
        
        # Only keep positive frequencies
        magnitude = magnitude[:len(magnitude)//2]
        
        # Compute frequency axis
        frequencies = np.fft.fftfreq(len(data_array), 1.0/self.sample_rate)[:len(magnitude)]
        
        return frequencies, magnitude
    
    def update_fft_plots(self):
        """Update FFT plots with current data."""
        if not self.fft_enabled or len(self.r_data) < 2:
            return
        
        window_samples = self.get_fft_window_sample_count(self.t_data)
        
        # Compute FFT for each data stream
        freq_r, mag_r = self.compute_fft_windowed(list(self.r_data), window_samples)
        freq_p, mag_p = self.compute_fft_windowed(list(self.p_data), window_samples)
        freq_y, mag_y = self.compute_fft_windowed(list(self.y_data), window_samples)
        
        freq_ax, mag_ax = self.compute_fft_windowed(list(self.ax_data), window_samples)
        freq_ay, mag_ay = self.compute_fft_windowed(list(self.ay_data), window_samples)
        freq_az, mag_az = self.compute_fft_windowed(list(self.az_data), window_samples)
        freq_amag, mag_amag = self.compute_fft_windowed(list(self.acc_mag_data), window_samples)
        
        freq_wx, mag_wx = self.compute_fft_windowed(list(self.wx_data), window_samples)
        freq_wy, mag_wy = self.compute_fft_windowed(list(self.wy_data), window_samples)
        freq_wz, mag_wz = self.compute_fft_windowed(list(self.wz_data), window_samples)
        freq_gmag, mag_gmag = self.compute_fft_windowed(list(self.gyro_mag_data), window_samples)
        
        # Update All Data mode FFT curves
        self.curve_fft_r.setData(freq_r, mag_r)
        self.curve_fft_p.setData(freq_p, mag_p)
        self.curve_fft_y.setData(freq_y, mag_y)
        
        self.curve_fft_ax.setData(freq_ax, mag_ax)
        self.curve_fft_ay.setData(freq_ay, mag_ay)
        self.curve_fft_az.setData(freq_az, mag_az)
        
        self.curve_fft_wx.setData(freq_wx, mag_wx)
        self.curve_fft_wy.setData(freq_wy, mag_wy)
        self.curve_fft_wz.setData(freq_wz, mag_wz)
        
        # Update separate angle mode FFT curves
        self.curve_fft_roll_sep.setData(freq_r, mag_r)
        self.curve_fft_pitch_sep.setData(freq_p, mag_p)
        self.curve_fft_yaw_sep.setData(freq_y, mag_y)
        
        # Update separate acceleration mode FFT curves
        self.curve_fft_acc_mag_sep.setData(freq_amag, mag_amag)
        
        # Update separate gyro mode FFT curves
        self.curve_fft_gyro_mag_sep.setData(freq_gmag, mag_gmag)
    
    def update_fft_plots_from_csv(self, data_list):
        """
        Update FFT plots from CSV data (for analysis mode).
        
        Args:
            data_list: List of data dictionaries with timestamp-indexed data
        """
        if not self.fft_enabled or len(data_list) < 2:
            return
        
        timestamps = [d.get('timestamp', 0.0) for d in data_list if 'timestamp' in d]
        window_samples = self.get_fft_window_sample_count(timestamps)

        # Extract the last window worth of data
        start_idx = max(0, len(data_list) - window_samples)
        subset = data_list[start_idx:]
        
        # Extract data arrays
        rolls = [d['roll'] for d in subset]
        pitches = [d['pitch'] for d in subset]
        yaws = [d['yaw'] for d in subset]
        
        acc_xs = [d['acc_x'] for d in subset]
        acc_ys = [d['acc_y'] for d in subset]
        acc_zs = [d['acc_z'] for d in subset]
        acc_mags = [np.sqrt(d['acc_x']**2 + d['acc_y']**2 + d['acc_z']**2) for d in subset]
        
        gyro_xs = [d['gyro_x'] for d in subset]
        gyro_ys = [d['gyro_y'] for d in subset]
        gyro_zs = [d['gyro_z'] for d in subset]
        gyro_mags = [np.sqrt(d['gyro_x']**2 + d['gyro_y']**2 + d['gyro_z']**2) for d in subset]
        
        # Estimate sample rate from data timestamps
        if len(data_list) > 10:
            time_diffs = np.diff([d['timestamp'] for d in data_list[-10:]])
            if len(time_diffs) > 0 and np.mean(time_diffs) > 0:
                self.sample_rate = 1.0 / np.mean(time_diffs)
        
        # Compute FFTs
        freq_r, mag_r = self.compute_fft_windowed(rolls, window_samples)
        freq_p, mag_p = self.compute_fft_windowed(pitches, window_samples)
        freq_y, mag_y = self.compute_fft_windowed(yaws, window_samples)
        
        freq_ax, mag_ax = self.compute_fft_windowed(acc_xs, window_samples)
        freq_ay, mag_ay = self.compute_fft_windowed(acc_ys, window_samples)
        freq_az, mag_az = self.compute_fft_windowed(acc_zs, window_samples)
        freq_amag, mag_amag = self.compute_fft_windowed(acc_mags, window_samples)
        
        freq_wx, mag_wx = self.compute_fft_windowed(gyro_xs, window_samples)
        freq_wy, mag_wy = self.compute_fft_windowed(gyro_ys, window_samples)
        freq_wz, mag_wz = self.compute_fft_windowed(gyro_zs, window_samples)
        freq_gmag, mag_gmag = self.compute_fft_windowed(gyro_mags, window_samples)
        
        # Update all FFT curves
        self._set_fft_curve(self.curve_fft_r, freq_r, mag_r)
        self._set_fft_curve(self.curve_fft_p, freq_p, mag_p)
        self._set_fft_curve(self.curve_fft_y, freq_y, mag_y)

        self._set_fft_curve(self.curve_fft_ax, freq_ax, mag_ax)
        self._set_fft_curve(self.curve_fft_ay, freq_ay, mag_ay)
        self._set_fft_curve(self.curve_fft_az, freq_az, mag_az)

        self._set_fft_curve(self.curve_fft_wx, freq_wx, mag_wx)
        self._set_fft_curve(self.curve_fft_wy, freq_wy, mag_wy)
        self._set_fft_curve(self.curve_fft_wz, freq_wz, mag_wz)

        self._set_fft_curve(self.curve_fft_roll_sep, freq_r, mag_r)
        self._set_fft_curve(self.curve_fft_pitch_sep, freq_p, mag_p)
        self._set_fft_curve(self.curve_fft_yaw_sep, freq_y, mag_y)

        self._set_fft_curve(self.curve_fft_acc_mag_sep, freq_amag, mag_amag)

        self._set_fft_curve(self.curve_fft_gyro_mag_sep, freq_gmag, mag_gmag)
    
    def update_fft_plots_from_results(self, fft_results):
        """
        Update FFT plots from pre-computed FFT results (from high-frequency thread).
        
        Args:
            fft_results: Dictionary with keys 'r', 'p', 'y', 'ax', 'ay', 'az', 'acc_mag', 
                        'wx', 'wy', 'wz', 'gyro_mag' containing tuples of (frequencies, magnitude)
        """
        if not fft_results:
            return
        
        try:
            # Extract results
            freq_r, mag_r = fft_results['r']
            freq_p, mag_p = fft_results['p']
            freq_y, mag_y = fft_results['y']
            
            freq_ax, mag_ax = fft_results['ax']
            freq_ay, mag_ay = fft_results['ay']
            freq_az, mag_az = fft_results['az']
            freq_amag, mag_amag = fft_results['acc_mag']
            
            freq_wx, mag_wx = fft_results['wx']
            freq_wy, mag_wy = fft_results['wy']
            freq_wz, mag_wz = fft_results['wz']
            freq_gmag, mag_gmag = fft_results['gyro_mag']
            
            # Update All Data mode FFT curves
            self.curve_fft_r.setData(freq_r, mag_r)
            self.curve_fft_p.setData(freq_p, mag_p)
            self.curve_fft_y.setData(freq_y, mag_y)
            
            self.curve_fft_ax.setData(freq_ax, mag_ax)
            self.curve_fft_ay.setData(freq_ay, mag_ay)
            self.curve_fft_az.setData(freq_az, mag_az)
            
            self.curve_fft_wx.setData(freq_wx, mag_wx)
            self.curve_fft_wy.setData(freq_wy, mag_wy)
            self.curve_fft_wz.setData(freq_wz, mag_wz)
            
            # Update separate angle mode FFT curves
            self.curve_fft_roll_sep.setData(freq_r, mag_r)
            self.curve_fft_pitch_sep.setData(freq_p, mag_p)
            self.curve_fft_yaw_sep.setData(freq_y, mag_y)
            
            # Update separate acceleration mode FFT curves
            self.curve_fft_acc_mag_sep.setData(freq_amag, mag_amag)
            
            # Update separate gyro mode FFT curves
            self.curve_fft_gyro_mag_sep.setData(freq_gmag, mag_gmag)
        except Exception as e:
            print(f"Error updating FFT plots from results: {e}")
