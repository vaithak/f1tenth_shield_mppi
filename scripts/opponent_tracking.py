#!/usr/bin/env python3


"""
Opponent Tracking Node
======================

This ROS 2 node tracks a single opponent vehicle detected by the perception
stack.  A **Frenet** representation (arc-length ``s`` along the reference
centerline and lateral offset ``d``) is used to describe the opponent’s
position on a closed track.  An **Extended Kalman Filter (EKF)** estimates the
state vector

    x = [s, v_s, d, v_d]^T

where
    * ``s``  - longitudinal position along the track (m)
    * ``v_s`` - longitudinal velocity  (m/s)
    * ``d``  - lateral offset from the centerline (m)
    * ``v_d`` - lateral velocity (m/s)

The filter prediction step assumes constant velocity in both directions and
applies a simple proportional damping control in the lateral channel to
prevent divergence when measurements are missing.

Incoming measurements arrive on ``/perception/detection/raw_obstacles``
(``ObstacleArray``).  Only the first obstacle in the array is processed under
the assumption that the race is 1-vs-1.

The posterior state estimate is re-published as
``/perception/obstacles`` (``ObstacleArray``) for downstream planners, and a
visualisation **Marker** is broadcast on
``/perception/static_dynamic_marker_pub``.
"""

# === Standard library ===
from typing import List
import shield_mppi.utils as utils

# === Third-party numerical libs ===
import numpy as np

# === ROS 2 core ===
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    ReliabilityPolicy,
    QoSHistoryPolicy,
)

# === ROS 2 message types ===
from visualization_msgs.msg import Marker, MarkerArray
from f1tenth_icra_race_msgs.msg import ObstacleArray, ObstacleMsg

# === Estimation tools ===
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

# === Local helpers ===
from shield_mppi.frenet_conversion import FrenetConverter

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def normalize_s(x,track_length):
    x = x % (track_length)
    if x > track_length/2:
        x -= track_length
    return x
# ---------------------------------------------------------------------------
# Extended-Kalman-filter-based opponent model
# ---------------------------------------------------------------------------

class OpponentState:
    """Holds EKF instance and helper buffers for a single opponent."""
    track_length: float

    def __init__(
        self,
        rate: int,
        process_var_vs: float,
        process_var_vd: float,
        meas_var_s: float,
        meas_var_d: float,
        meas_var_vs: float,
        meas_var_vd: float,
        P_vs: float,
        P_d: float,
        P_vd: float,
    ) -> None:
        # --- Model constants ------------------------------------------------
        self.rate = rate
        self.dt = 1.0 / rate  # sampling period (s)

        # Simple P-controller gains for lateral damping in *prediction* step
        self.P_vs = P_vs
        self.P_d = P_d
        self.P_vd = P_vd
        self.size = 0.0
        self.is_visible = False
        self.is_visible_ttl = 5

        # --- Kalman filter --------------------------------------------------
        self.kf = EKF(dim_x=4, dim_z=4)

        # State-transition Jacobian (linear because we assume nearly-constant
        # velocity in both axes)
        self.kf.F = np.array(
            [
                [1.0, self.dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, self.dt],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Process-noise covariance – independent planar motions
        q1 = Q_discrete_white_noise(dim=2, dt=self.dt, var=process_var_vs)
        q2 = Q_discrete_white_noise(dim=2, dt=self.dt, var=process_var_vd)
        self.kf.Q = block_diag(q1, q2)

        # Measurement covariance (``s, v_s, d, v_d``)
        self.kf.R = np.diag([meas_var_s, meas_var_vs, meas_var_d, meas_var_vd])

        # Observation matrix (direct observation of every state)
        self.kf.H = np.identity(4)

        # Initial covariance (fairly uninformative)
        self.kf.P *= 5.0

        # --- Runtime helpers -----------------------------------------------
        self.initialized: bool = False  # become *True* after first meas.
        self.vs_filt: List[float] = [0.0] * 5  # small moving window (long.)
        self.vd_filt: List[float] = [0.0] * 5  # small moving window (lat.)

    # ------------------------- Prediction / Update -------------------------

    def residual_h(a, b):
        y = a-b
        y[0] = normalize_s(y[0], OpponentState.track_length)
        return y
    
    def Hjac(x):
        return np.identity(4)

    def hx(x):
        return np.array([normalize_s(x[0],
                         OpponentState.track_length),x[1], x[2], x[3]])

    def predict(self) -> None:
        """EKF prediction step with lateral damping on *d* and *v_d*."""
        # Control input vector ``u`` applies *proportional damping* to the
        # lateral states so the filter does not diverge when no measurements
        # arrive for a few frames.
        u = np.array([0, 0, -self.P_d * self.kf.x[2], -self.P_vd * self.kf.x[3]])
        self.kf.predict(u=u)
        # Keep ``s`` within track length after prediction
        self.kf.x[0] = normalize_s(self.kf.x[0], OpponentState.track_length)

    def update(self, s: float, d: float, vs: float, vd: float) -> None:
        """EKF correction step with the latest noisy measurement."""
        z = np.array([s, vs, d, vd])
        self.kf.update(
            z=z,
            HJacobian=OpponentState.Hjac,
            Hx=OpponentState.hx,
            residual=OpponentState.residual_h,
        )
        self.kf.x[0] = normalize_s(self.kf.x[0], OpponentState.track_length)

        # Update simple rolling averages for smoother velocity outputs
        self.vs_filt.pop(0)
        self.vs_filt.append(self.kf.x[1])
        self.vd_filt.pop(0)
        self.vd_filt.append(self.kf.x[3])

# ---------------------------------------------------------------------------
# ROS 2 Node implementation
# ---------------------------------------------------------------------------

class OpponentTracking(Node):
    """ROS 2 node that wraps :class:`OpponentState` and handles I/O."""

    def __init__(self) -> None:
        super().__init__("opponent_tracking")

        import os
        cwd = os.getcwd()
        self.declare_parameter("waypoint_file", f'{cwd}/src/f1tenth_shield_mppi/waypoints/fitted_2.csv')
        waypoint_file = self.get_parameter("waypoint_file").get_parameter_value().string_value

        waypoints = np.genfromtxt(waypoint_file, delimiter=',')
        waypoint_cols = utils.column_numbers_for_waypoints()

        waypoints_x = waypoints[:, waypoint_cols['x_ref_m']]
        waypoints_y = waypoints[:, waypoint_cols['y_ref_m']]
        waypoints_psi = waypoints[:, waypoint_cols['psi_racetraj_rad']]

        self.track_length = float(waypoints[-1, waypoint_cols['s_racetraj_m']])
        self.converter = FrenetConverter(waypoints_x, waypoints_y, waypoints_psi)



        # ------------------------------------------------------------------
        # QoS setup – keep only the last message, ensure reliability
        # ------------------------------------------------------------------
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        # ----------------------- Parameter interface ----------------------
        # (all have sane defaults; can be overriden via launch-file) --------
        self.declare_parameter("rate", 40)
        self.rate = (
            self.get_parameter("rate").get_parameter_value().integer_value
        )

        # Noise parameters --------------------------------------------------
        self.declare_parameter("process_var_vs", 0.6)
        self.declare_parameter("process_var_vd", 0.6)
        self.declare_parameter("meas_var_s", 0.1)
        self.declare_parameter("meas_var_d", 0.1)
        self.declare_parameter("meas_var_vs", 0.1)
        self.declare_parameter("meas_var_vd", 0.1)
        self.declare_parameter("P_vs", 1.0)
        self.declare_parameter("P_d", 1.0)
        self.declare_parameter("P_vd", 1.0)

    

        # ------------------------------------------------------------------
        # Instantiate the opponent state estimator
        # ------------------------------------------------------------------
        OpponentState.track_length = self.track_length
        self.state = OpponentState(
            rate=self.rate,
            process_var_vs=self.get_parameter("process_var_vs").get_parameter_value().double_value,
            process_var_vd=self.get_parameter("process_var_vd").get_parameter_value().double_value,
            meas_var_s=self.get_parameter("meas_var_s").get_parameter_value().double_value,
            meas_var_d=self.get_parameter("meas_var_d").get_parameter_value().double_value,
            meas_var_vs=self.get_parameter("meas_var_vs").get_parameter_value().double_value,
            meas_var_vd=self.get_parameter("meas_var_vd").get_parameter_value().double_value,
            P_vs=self.get_parameter("P_vs").get_parameter_value().double_value,
            P_d=self.get_parameter("P_d").get_parameter_value().double_value,
            P_vd=self.get_parameter("P_vd").get_parameter_value().double_value,
        )

        # Previous raw ``s`` measurement (for finite-difference velocity)
        self.prev_obs_s: float | None = None
        self.prev_obs_d: float | None = None

        # --------------------------- Subscriptions -------------------------
        self.subscription = self.create_subscription(
            ObstacleArray,
            "/perception/detection/raw_obstacles",
            self.obstacle_callback,
            qos,
        )

        # ---------------------------- Publishers ---------------------------
        self.marker_pub = self.create_publisher(
            MarkerArray, "/perception/static_dynamic_marker_pub", qos
        )
        self.obstacles_pub = self.create_publisher(
            ObstacleArray, "/perception/obstacles", qos
        )

        # ------------------------------ Timer ------------------------------
        self.timer = self.create_timer(1.0 / self.rate, self.loop)

    # ------------------------------------------------------------------
    # Callback for raw perception data
    # ------------------------------------------------------------------
    def obstacle_callback(self, msg: ObstacleArray) -> None:
        """Convert raw obstacle to measurement and feed the EKF."""
        if not msg.obstacles:
            if self.state.is_visible:
                self.state.is_visible_ttl -= 1
                if self.state.is_visible_ttl <= 0:
                    self.state.is_visible = False
                    self.state_is_visible_ttl = 5
            return  # no detections in this frame
        
        self.state.is_visible = True
        self.state.is_visible_ttl = 5

        # For simplicity we only track the obstacle closest to the raceline (using d_center)
        obs = msg.obstacles[0]        
        for curr_obs in msg.obstacles:
            if abs(curr_obs.d_center) < abs(obs.d_center):
                obs = curr_obs
        s = obs.s_center
        d = obs.d_center

        # Approximate longitudinal velocity by finite differencing successive
        # *s* measurements (will be noisy – EKF will smooth it)
        vs = 0.0 if self.prev_obs_s is None else (s - self.prev_obs_s) * self.rate
        self.prev_obs_s = s
        if vs < -1 or vs > 8:
            vs = 0.0

        vd = 0.0 if self.prev_obs_d is None else (d - self.prev_obs_d) * self.rate
        self.prev_obs_d = d
        if vd < -2 or vd > 2:
            vd = 0.0

        # Initialise EKF the first time a measurement arrives
        if not self.state.initialized:
            self.state.kf.x = np.array([s, vs, d, vd])
            self.state.initialized = True
            self.state.size = obs.size
        else:
            self.state.update(s, d, vs, vd)

    # ------------------------------------------------------------------
    # Main loop – prediction + publishing
    # ------------------------------------------------------------------
    def loop(self) -> None:
        if not self.state.initialized:
            return  # Wait until first measurement

        # 1. EKF predict step
        if self.state.is_visible:
            self.state.predict()
        else:
            # Reinitialize the state with zero velocity
            self.state.kf.x = np.zeros(4)
            self.state.initialized = False
            self.state.vs_filt = [0.0] * 5
            self.state.vd_filt = [0.0] * 5            

        # 2. Build *ObstacleArray* message with current estimate
        obstacle_msg = ObstacleArray()
        obstacle_msg.header.stamp = self.get_clock().now().to_msg()
        obstacle_msg.header.frame_id = "map"

        # Compute trace of the covariance matrix
        trace = np.trace(self.state.kf.P)
        if trace < 0.5 and self.state.is_visible:
            obs = ObstacleMsg()
            obs.id = 1  # single-opponent ID

            obs.s_center = self.state.kf.x[0]
            if self.state.kf.x[0] < 0:
                obs.s_center += self.track_length
            obs.d_center = self.state.kf.x[2]
            obs.vs = float(np.mean(self.state.vs_filt))
            obs.vd = float(np.mean(self.state.vd_filt))
            obs.size = self.state.size
            obs.is_static = False
            obs.s_start = (obs.s_center - obs.size / 2.0) % self.track_length
            obs.s_end = (obs.s_center + obs.size / 2.0) % self.track_length
            obs.d_right = obs.d_center - obs.size / 2.0
            obs.d_left = obs.d_center + obs.size / 2.0
            obs.is_visible = self.state.is_visible

            # Covariance diagonal entries for consumers that need uncertainty
            # obs.s_var = float(self.state.kf.P[0, 0])
            # obs.vs_var = float(self.state.kf.P[1, 1])
            # obs.d_var = float(self.state.kf.P[2, 2])
            # obs.vd_var = float(self.state.kf.P[3, 3])

            obstacle_msg.obstacles.append(obs)
            self.obstacles_pub.publish(obstacle_msg)

            # 3. Publish RViz marker for visual debugging
            self.publish_marker(obs)
        else:
            # Clear markers and publish empty obstacle array
            self.marker_pub.publish(self.clearmarkers())
            obstacle_msg.obstacles = []
            self.obstacles_pub.publish(obstacle_msg)


    def clearmarkers(self) -> MarkerArray:
        marker_array = MarkerArray()
        marker = Marker()
        marker.action = 3
        marker_array.markers = [marker]
        return marker_array

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------
    def publish_marker(self, obs: ObstacleMsg) -> None:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = obs.id
        marker.type = Marker.SPHERE

        # Convert Frenet back to xy for RViz display
        if obs.s_center < 0:
            obs.s_center += self.track_length
        x, y = self.converter.get_cartesian(obs.s_center, obs.d_center)
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.orientation.w = 1.0  # no rotation

        marker.scale.x = marker.scale.y = marker.scale.z = obs.size

        # Semi-transparent red
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        self.marker_pub.publish(MarkerArray(markers=[marker]))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = OpponentTracking()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
