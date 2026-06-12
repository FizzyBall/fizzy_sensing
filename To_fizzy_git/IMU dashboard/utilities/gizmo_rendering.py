"""
Gizmo Rendering Module

This module handles all 3D visualization and rendering of the IMU data,
including the orientation gizmo, PCB representation, and acceleration vectors.
"""

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl


class GizmoRenderer:
    """Manages 3D gizmo rendering for IMU orientation and acceleration visualization."""
    
    def __init__(self, view_widget):
        """
        Initialize the gizmo renderer.
        
        Args:
            view_widget: A pyqtgraph.opengl.GLViewWidget instance
        """
        self.view = view_widget
        self._setup_camera()
        self._setup_grid()
        self._setup_pcb_mesh()
        self._setup_axes()
        self._setup_acceleration_vectors()
    
    def _setup_camera(self):
        """Configure camera position and viewing parameters."""
        self.view.setCameraPosition(distance=15, elevation=30, azimuth=45)
    
    def _setup_grid(self):
        """Create background grid."""
        grid = gl.GLGridItem()
        grid.setSize(20, 20, 1)
        grid.setSpacing(1, 1, 1)
        self.view.addItem(grid)
    
    def _setup_pcb_mesh(self):
        """Create PCB/microcontroller mesh representation."""
        # PCB dimensions (x:y:z = 10:20:1) - 10x smaller scale
        pcb_h, pcb_w, pcb_d = 1, 2, 0.1
        small_h, small_w, small_d = 0.3, 0.3, 0.1
        
        # Create vertices for a box centered at origin
        pcb_vertices = np.array([
            [-pcb_w/2, -pcb_h/2, -pcb_d/2],
            [pcb_w/2, -pcb_h/2, -pcb_d/2],
            [pcb_w/2, pcb_h/2, -pcb_d/2],
            [-pcb_w/2, pcb_h/2, -pcb_d/2],
            [-pcb_w/2, -pcb_h/2, pcb_d/2],
            [pcb_w/2, -pcb_h/2, pcb_d/2],
            [pcb_w/2, pcb_h/2, pcb_d/2],
            [-pcb_w/2, pcb_h/2, pcb_d/2],
        ], dtype=np.float32)
        
        # Define faces (triangles) for the box
        pcb_faces = np.array([
            # Bottom face
            [0, 1, 2], [0, 2, 3],
            # Top face
            [4, 6, 5], [4, 7, 6],
            # Front face
            [0, 5, 1], [0, 4, 5],
            # Back face
            [2, 7, 3], [2, 6, 7],
            # Left face
            [0, 3, 7], [0, 7, 4],
            # Right face
            [1, 5, 6], [1, 6, 2],
        ], dtype=np.uint32)
        
        # Create semitransparent greenish PCB color for each face
        pcb_face_colors = np.array([
            [0.2, 0.5, 0.2, 0.4] for _ in range(len(pcb_faces))
        ], dtype=np.float32)
        
        # Create and add mesh
        self.pcb_mesh = gl.GLMeshItem(
            vertexes=pcb_vertices,
            faces=pcb_faces,
            faceColors=pcb_face_colors,
            drawEdges=True,
            edgeColor=(0.4, 0.6, 0.4, 0.7)
        )
        self.view.addItem(self.pcb_mesh)

        # Add a small component on top of the PCB, bottom-right corner
        small_center_x = (-pcb_w / 2) + (small_w / 2)
        small_center_y = (-pcb_h / 2) + (small_h / 2)
        small_center_z = (pcb_d / 2) + (small_d / 2)

        small_vertices = np.array([
            [small_center_x - small_w/2, small_center_y - small_h/2, small_center_z - small_d/2],
            [small_center_x + small_w/2, small_center_y - small_h/2, small_center_z - small_d/2],
            [small_center_x + small_w/2, small_center_y + small_h/2, small_center_z - small_d/2],
            [small_center_x - small_w/2, small_center_y + small_h/2, small_center_z - small_d/2],
            [small_center_x - small_w/2, small_center_y - small_h/2, small_center_z + small_d/2],
            [small_center_x + small_w/2, small_center_y - small_h/2, small_center_z + small_d/2],
            [small_center_x + small_w/2, small_center_y + small_h/2, small_center_z + small_d/2],
            [small_center_x - small_w/2, small_center_y + small_h/2, small_center_z + small_d/2],
        ], dtype=np.float32)

        self.small_mesh = gl.GLMeshItem(
            vertexes=small_vertices,
            faces=pcb_faces,
            faceColors=np.array([[0.55, 0.55, 0.55, 0.85] for _ in range(len(pcb_faces))], dtype=np.float32),
            drawEdges=True,
            edgeColor=(0.6, 0.6, 0.6, 0.9)
        )
        self.view.addItem(self.small_mesh)
    
    def _setup_axes(self):
        """Setup coordinate axis gizmo (RGB for XYZ)."""
        # Red = X, Green = Y, Blue = Z
        self.x_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [5, 0, 0]]),
            color=(1.0, 0.4, 0.4, 1.0),
            width=6.0,
            antialias=True
        )
        self.y_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 5, 0]]),
            color=(0.4, 1.0, 0.4, 1.0),
            width=6.0,
            antialias=True
        )
        self.z_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 5]]),
            color=(0.4, 0.7, 1.0, 1.0),
            width=6.0,
            antialias=True
        )
        
        self.view.addItem(self.x_axis)
        self.view.addItem(self.y_axis)
        self.view.addItem(self.z_axis)
    
    def _setup_acceleration_vectors(self):
        """Setup acceleration vector visualization."""
        # Distinct colors: Yellow=Acc X, Magenta=Acc Y, Cyan=Acc Z, White=Total Acc
        self.acc_x_vec = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 0]]),
            color=(1.0, 1.0, 0.0, 1.0),
            width=6.0,
            antialias=True
        )
        self.acc_y_vec = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 0]]),
            color=(1.0, 0.0, 1.0, 1.0),
            width=6.0,
            antialias=True
        )
        self.acc_z_vec = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 0]]),
            color=(0.0, 1.0, 1.0, 1.0),
            width=6.0,
            antialias=True
        )
        self.acc_total_vec = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 0]]),
            color=(1.0, 1.0, 1.0, 1.0),
            width=8.0,
            antialias=True
        )
        
        self.view.addItem(self.acc_x_vec)
        self.view.addItem(self.acc_y_vec)
        self.view.addItem(self.acc_z_vec)
        self.view.addItem(self.acc_total_vec)
    
    def magnitude_to_color(self, magnitude, max_magnitude=15.0):
        """
        Maps magnitude to a color gradient: green (low) -> yellow -> red (high).
        
        Args:
            magnitude: The magnitude value to map
            max_magnitude: The maximum magnitude for scaling (default 15.0)
            
        Returns:
            tuple: RGBA color values (r, g, b, a)
        """
        # Normalize magnitude to 0-1 range
        normalized = min(magnitude / max_magnitude, 1.0)
        
        if normalized < 0.5:
            # Green to Yellow: increase red
            r = normalized * 2.0
            g = 1.0
            b = 0.0
        else:
            # Yellow to Red: decrease green
            r = 1.0
            g = 2.0 * (1.0 - normalized)
            b = 0.0
        
        return (r, g, b, 1.0)
    
    def apply_minimum_vector_size(self, vector, min_size=0.5):
        """
        Ensures a vector has at least min_size length.
        
        Args:
            vector: Nx3 array with start and end points
            min_size: Minimum vector magnitude (default 0.5)
            
        Returns:
            ndarray: Vector with enforced minimum size
        """
        end_point = vector[1] - vector[0]
        current_magnitude = np.linalg.norm(end_point)
        
        if current_magnitude < min_size:
            # Scale up the vector to minimum size
            if current_magnitude > 0:
                scale_factor = min_size / current_magnitude
                end_point = end_point * scale_factor
            else:
                # If zero, create a small default vector
                end_point = np.array([min_size, 0, 0])
            
            return np.array([[0, 0, 0], end_point])
        return vector
    
    def update_acceleration_vectors(self, acc_x, acc_y, acc_z):
        """
        Update acceleration vector visualizations.
        
        Args:
            acc_x, acc_y, acc_z: Acceleration components (already scaled)
        """
        # Calculate magnitudes for each component
        mag_x = abs(acc_x)
        mag_y = abs(acc_y)
        mag_z = abs(acc_z)
        mag_total = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        # Create vectors with minimum size enforcement
        vec_x = self.apply_minimum_vector_size(
            np.array([[0, 0, 0], [acc_x, 0, 0]]),
            min_size=0.8
        )
        vec_y = self.apply_minimum_vector_size(
            np.array([[0, 0, 0], [0, acc_y, 0]]),
            min_size=0.8
        )
        vec_z = self.apply_minimum_vector_size(
            np.array([[0, 0, 0], [0, 0, acc_z]]),
            min_size=0.8
        )
        vec_total = self.apply_minimum_vector_size(
            np.array([[0, 0, 0], [acc_x, acc_y, acc_z]]),
            min_size=0.8
        )
        
        # Get colors based on magnitude
        color_x = self.magnitude_to_color(mag_x)
        color_y = self.magnitude_to_color(mag_y)
        color_z = self.magnitude_to_color(mag_z)
        color_total = self.magnitude_to_color(mag_total)
        
        # Update acceleration line positions and colors
        self.acc_x_vec.setData(pos=vec_x, color=color_x)
        self.acc_y_vec.setData(pos=vec_y, color=color_y)
        self.acc_z_vec.setData(pos=vec_z, color=color_z)
        self.acc_total_vec.setData(pos=vec_total, color=color_total)
    
    def apply_rotation_transform(self, roll_deg, pitch_deg, yaw_deg):
        """
        Apply rotation transformation to all gizmo elements.
        
        Args:
            roll_deg, pitch_deg, yaw_deg: Euler angles in degrees
        """
        # Create a 3D rotation transform
        transform = pg.Transform3D()
        
        # Apply rotations: Yaw (Z), Pitch (Y), Roll (X)
        transform.rotate(yaw_deg, 0, 0, 1)    # Yaw around Z axis
        transform.rotate(pitch_deg, 0, 1, 0)  # Pitch around Y axis
        transform.rotate(roll_deg, 1, 0, 0)   # Roll around X axis
        
        # Apply transform to all rotating elements
        self.x_axis.setTransform(transform)
        self.y_axis.setTransform(transform)
        self.z_axis.setTransform(transform)
        
        # Rotate the acceleration vectors relative to IMU chassis
        self.acc_x_vec.setTransform(transform)
        self.acc_y_vec.setTransform(transform)
        self.acc_z_vec.setTransform(transform)
        self.acc_total_vec.setTransform(transform)
        
        # Rotate the PCB mesh
        self.pcb_mesh.setTransform(transform)
        self.small_mesh.setTransform(transform)
