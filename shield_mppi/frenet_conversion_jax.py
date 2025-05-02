### Create frenet conversion functions using JAX

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline

class FrenetConverter:
    def __init__(self, waypoints_x: np.array, waypoints_y: np.array, waypoints_psi: np.array):
        self.waypoints_x = jnp.asarray(waypoints_x)
        self.waypoints_y = jnp.asarray(waypoints_y)
        self.waypoints_psi = jnp.asarray(waypoints_psi) # Yaw angle of the waypoints - angle from wpnt[i] to wpnt[i+1]
        self.waypoints_s = None
        self.spline_x = None
        self.spline_y = None
        self.raceline_length = None
        self.waypoints_distance_m = 0.1 # [m]
        self.iter_max = 3
        self.closest_index = None

        self.build_raceline()

    def build_raceline(self):
        self.waypoints_s = [0.0]
        prev_wpnt_x =  self.waypoints_x[0]
        prev_wpnt_y =  self.waypoints_y[0]
        for wpnt_x, wpnt_y in zip(self.waypoints_x[1:], self.waypoints_y[1:]):
            dist = jnp.linalg.norm(jnp.array([wpnt_x - prev_wpnt_x, wpnt_y - prev_wpnt_y]))
            prev_wpnt_x = wpnt_x
            prev_wpnt_y = wpnt_y
            self.waypoints_s.append(self.waypoints_s[-1] + dist)        
        self.waypoints_s = jnp.array(self.waypoints_s)
        self.spline_x = CubicSpline(self.waypoints_s, self.waypoints_x)
        self.spline_y = CubicSpline(self.waypoints_s, self.waypoints_y)
        self.raceline_length = self.waypoints_s[-1]

    def get_frenet(self, x, y, s=None) -> jnp.array:
        # Compute Frenet coordinates for a given (x, y) point
        self.closest_index = self.get_closest_index(x, y)
        if s is None:
            s = self.get_approx_s(x, y)
            s, d = self.get_frenet_coord(x, y, s)
        else:
            s, d = self.get_frenet_coord(x, y, s)
        return jnp.array([s, d])
    
    def get_approx_s(self, x, y) -> float:
        """
        Finds the s-coordinate of the given point by finding the nearest waypoint.
        """
        # Find closest waypoint
        closest_index = self.closest_index
        return self.waypoints_s[closest_index]
    
    def get_frenet_velocities(self, vx, vy, theta) -> jnp.array:
        """
        Returns the Frenet velocities for the given Cartesian velocities.
        
        Args:
            vx (float): x-velocity
            vy (float): y-velocity
            theta (float): orientation of the vehicle
            
        Returns:
            jnp.array: Frenet velocities
        """
        # Compute Frenet velocities for a given (vx, vy) and theta
        s_dot = vx * jnp.cos(theta) + vy * jnp.sin(theta)
        d_dot = -vx * jnp.sin(theta) + vy * jnp.cos(theta)
        return jnp.array([s_dot, d_dot])
    
    def get_closest_index(self, x, y) -> int:
        """
        Finds the index of the closest waypoint to the given (x, y) point.
        
        Args:
            x: (vector of x-coordinates)
            y: (vector of y-coordinates)
            
        Returns:
            int: vector of indices of the closest waypoints
        """
        # closest_indices = jnp.zeros(x.shape, dtype=int)
        # # Compute the distances to all waypoints for all points
        # for i in range(len(x)):
        #     # Compute the distance to all waypoints
        #     distances = jnp.sqrt((self.waypoints_x - x[i])**2 + (self.waypoints_y - y[i])**2)
        #     # Find the index of the closest waypoint
        #     closest_index = jnp.argmin(distances)
        #     # Store the closest index
        #     closest_indices = closest_indices.at[i].set(closest_index)

        x_reshaped = x[:, jnp.newaxis]  # Shape: (60, 1)
        y_reshaped = y[:, jnp.newaxis]  # Shape: (60, 1)

        # Compute distances using broadcasting
        # This creates a matrix of shape (60, 312) where each entry (i, j) is the
        # distance between point i and waypoint j
        distances = jnp.sqrt((x_reshaped - self.waypoints_x)**2 + 
                            (y_reshaped - self.waypoints_y)**2)

        # Find the indices of minimum distances along axis 1 (waypoints)
        closest_indices = jnp.argmin(distances, axis=1)
        

        return closest_indices
    
    def get_frenet_coord(self, x, y, s) -> jnp.array:
        """
        Compute the Frenet coordinates (s, d) for a given (x, y) point and s-coordinate.
        
        Args:
            x (float): x-coordinate
            y (float): y-coordinate
            s (float): s-coordinate
            
        Returns:
            jnp.array: Frenet coordinates [s, d]
        """
        # Compute the closest waypoint index
        closest_index = self.closest_index
        # Compute the distance to the closest waypoint
        dx = x - self.waypoints_x[closest_index]
        dy = y - self.waypoints_y[closest_index]
        # Compute the angle of the closest waypoint
        theta = self.waypoints_psi[closest_index]
        # Compute the Frenet coordinates
        d = dx * jnp.sin(theta) - dy * jnp.cos(theta)
        # Adjust s-coordinate based on the closest waypoint
        s = self.waypoints_s[closest_index] + dx * jnp.cos(theta) + dy * jnp.sin(theta)

        return s, d

    def get_cartesian(self, s, d) -> jnp.array:
        """
        Convert Frenet coordinates (s, d) to Cartesian coordinates (x, y).
        
        Args:
            s (float): s-coordinate
            d (float): d-coordinate
            
        Returns:
            jnp.array: Cartesian coordinates [x, y]
        """
        dx, dy = self.spline_x(s, 1), self.spline_y(s, 1)
        theta = jnp.arctan2(dy, dx)
        normal_theta = theta + jnp.pi / 2
        # Compute the Cartesian coordinates
        x = self.spline_x(s) + d * jnp.cos(normal_theta)
        y = self.spline_y(s) + d * jnp.sin(normal_theta)
        return jnp.array([x, y])