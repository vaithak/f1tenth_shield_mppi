#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    ReliabilityPolicy,
    QoSHistoryPolicy,
)
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor, SetParametersResult, ParameterType
from rclpy.parameter import Parameter
from f1tenth_icra_race_msgs.msg import ObstacleArray, ObstacleMsg, OTWpntArray, WpntArray, Wpnt
from geometry_msgs.msg import PointStamped
from tf_transformations import quaternion_from_euler
from tf_transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import time
from typing import List, Any, Tuple
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from shield_mppi.frenet_conversion import FrenetConverter
from shield_mppi import utils


def normalize_s(x, track_length):
    x = x % (track_length)
    return x

class SplineNode(Node):
    def __init__(self):
        super().__init__("spline_node")

        self.declare_parameter("is_sim", True)
        self.is_sim = self.get_parameter("is_sim").get_parameter_value().bool_value

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        # Create parameters for plot and print debugging
        self.declare_parameter('plot_debug', False)
        self.plot_debug = self.get_parameter('plot_debug').value
        self.declare_parameter('print_debug', False)
        self.print_debug = self.get_parameter('print_debug').value

        # Publishers for plotting
        if self.plot_debug:
            self.closest_obs_pub = self.create_publisher(
                Marker, '/planner/avoidance/considered_OBS', qos)
            self.pub_propagated = self.create_publisher(
                Marker, '/planner/avoidance/propagated_OBS', qos)
            self.mrks_pub = self.create_publisher(
                MarkerArray, '/planner/avoidance/markers', qos)

        self.evasion_pub = self.create_publisher(
            OTWpntArray, '/planner/avoidance/otwpnts', qos)

        # Read waypoint file from parameter
        import os
        cwd = os.getcwd()
        self.declare_parameter("waypoint_file", f"{cwd}/src/f1tenth_shield_mppi/waypoints/fitted_2.csv")
        waypoint_file = self.get_parameter("waypoint_file").get_parameter_value().string_value

        self.waypoints = np.genfromtxt(waypoint_file, delimiter=',')
        waypoint_cols = utils.column_numbers_for_waypoints()

        self.waypoints_x = self.waypoints[:, waypoint_cols['x_ref_m']]
        self.waypoints_y = self.waypoints[:, waypoint_cols['y_ref_m']]
        self.waypoints_v = self.waypoints[:, waypoint_cols['vx_racetraj_mps']]
        self.gb_vmax = np.max(self.waypoints_v)
        waypoints_psi = self.waypoints[:, waypoint_cols['psi_racetraj_rad']]
        # waypoints_psi = np.array(utils.convert_psi(waypoints_psi))
        self.wpnts_d_right_array = self.waypoints[:, waypoint_cols['width_right_m']]
        self.wpnts_d_left_array = self.waypoints[:, waypoint_cols['width_left_m']]
        self.wpnts_s_array = self.waypoints[:, waypoint_cols['s_racetraj_m']]
        self.gb_max_idx = len(self.wpnts_s_array)
        if "kappa_racetraj_radpm" in waypoint_cols:
            self.kappa_array = self.waypoints[:, waypoint_cols['kappa_racetraj_radpm']]
        else:
            self.kappa_array = np.zeros(self.gb_max_idx)
        
        self.track_length = float(self.wpnts_s_array[-1])

        # Initialize the converter
        self.converter = FrenetConverter(self.waypoints_x, self.waypoints_y, waypoints_psi)

        self.declare_parameter("publish_rate", 40)
        self.publish_rate = self.get_parameter("publish_rate").get_parameter_value().integer_value
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        self.create_subscription(
            ObstacleArray, '/perception/obstacles', self.obs_cb, QoSProfile(depth=10))
        
        if self.is_sim:
            self.pose_sub = self.create_subscription(
                Odometry,
                '/ego_racecar/odom',
                self.pose_callback,
                qos)
        else:
            self.pose_sub = self.create_subscription(
                Odometry,
                '/pf/pose/odom',
                self.pose_callback,
                qos)
            
        self.car_global_x = None
        self.car_global_y = None
        self.car_global_yaw = None
        self.car_s = None
        self.car_d = None
        self.car_vs = None
        self.car_vd = None
        self.obstacles = None

            
        self.pre_apex_0 = -4.0
        self.pre_apex_1 = -3.0
        self.pre_apex_2 = -1.5
        self.post_apex_0 = 2.0
        self.post_apex_1 = 3.0
        self.post_apex_2 = 4.0
        self.evasion_dist = 0.3
        self.obs_traj_tresh = 0.35
        self.spline_bound_mindist = 0.0
        self.fixed_pred_time = 0.15
        self.kd_obs_pred = 1.0
        self.lookahead = 7.0

        # declare parameters and get parameter values
        self.declare_parameter("pre_apex_0", self.pre_apex_0)
        self.declare_parameter("pre_apex_1", self.pre_apex_1)
        self.declare_parameter("pre_apex_2", self.pre_apex_2)
        self.declare_parameter("post_apex_0", self.post_apex_0)
        self.declare_parameter("post_apex_1", self.post_apex_1)
        self.declare_parameter("post_apex_2", self.post_apex_2)
        self.declare_parameter("evasion_dist", self.evasion_dist)
        self.declare_parameter("obs_traj_tresh", self.obs_traj_tresh)
        self.declare_parameter("spline_bound_mindist", self.spline_bound_mindist)
        self.declare_parameter("fixed_pred_time", self.fixed_pred_time)
        self.declare_parameter("kd_obs_pred", self.kd_obs_pred)
        self.declare_parameter("lookahead", self.lookahead)
        self.pre_apex_0 = self.get_parameter("pre_apex_0").get_parameter_value().double_value
        self.pre_apex_1 = self.get_parameter("pre_apex_1").get_parameter_value().double_value
        self.pre_apex_2 = self.get_parameter("pre_apex_2").get_parameter_value().double_value
        self.post_apex_0 = self.get_parameter("post_apex_0").get_parameter_value().double_value
        self.post_apex_1 = self.get_parameter("post_apex_1").get_parameter_value().double_value
        self.post_apex_2 = self.get_parameter("post_apex_2").get_parameter_value().double_value
        self.evasion_dist = self.get_parameter("evasion_dist").get_parameter_value().double_value
        self.obs_traj_tresh = self.get_parameter("obs_traj_tresh").get_parameter_value().double_value
        self.spline_bound_mindist = self.get_parameter("spline_bound_mindist").get_parameter_value().double_value
        self.fixed_pred_time = self.get_parameter("fixed_pred_time").get_parameter_value().double_value
        self.kd_obs_pred = self.get_parameter("kd_obs_pred").get_parameter_value().double_value
        self.lookahead = self.get_parameter("lookahead").get_parameter_value().double_value

        # Storing previous values for the last switch time and side
        self.last_switch_time = self.get_clock().now().to_msg()
        self.last_ot_side = ""


    def obs_cb(self, msg: ObstacleArray):
        self.obstacles = msg

    def pose_callback(self, pose_msg):
        # Get the current x, y position of the vehicle
        pose = pose_msg.pose.pose
        self.car_global_x = pose.position.x
        self.car_global_y = pose.position.y
        self.car_global_yaw = euler_from_quaternion([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ])[2]
        if self.print_debug:
            self.get_logger().info(f'Pose: {self.car_global_x}, {self.car_global_y}, {self.car_global_yaw}')

        # Convert the global coordinates to Frenet coordinates
        s, d = self.converter.get_frenet(np.array([self.car_global_x]), np.array([self.car_global_y]))
        self.car_s = normalize_s(s[0], self.track_length)
        self.car_d = d[0]
        vs, vd = self.converter.get_frenet_velocities(np.array([pose_msg.twist.twist.linear.x]), np.array([pose_msg.twist.twist.linear.y]), self.car_global_yaw)

        self.car_vs = vs[0]
        self.car_vd = vd[0]

    def timer_callback(self):
        # Check if the car is in a valid state
        if self.car_global_x is None or self.car_global_y is None:
            return
        if self.obstacles is None:
            return

        # Sample data
        wpnts = OTWpntArray()
        mrks = MarkerArray()

        # If obs then do splining around it
        if len(self.obstacles.obstacles) > 0:
            wpnts, mrks = self.do_spline(self.obstacles)

        # Add a delete marker to the marker array
        del_mrk = Marker()
        del_mrk.header.stamp = self.get_clock().now().to_msg()
        del_mrk.action = Marker.DELETEALL
        mrks.markers.insert(0, del_mrk)

        # Publish wpnts and markers
        self.evasion_pub.publish(wpnts)
        self.mrks_pub.publish(mrks)


    #################### UTILS FUNCTIONS####################
    def _predict_obs_movement(self, obs: ObstacleMsg, mode: str = "constant") -> ObstacleMsg:
        """
        Predicts the movement of an obstacle based on the current state and mode.

        TODO: opponent prediction should be completely isolated for added modularity       

        Args:
            obs (Obstacle): The obstacle to predict the movement for.
            mode (str, optional): The mode for predicting the movement. Defaults to "constant".

        Returns:
            Obstacle: The updated obstacle with the predicted movement.
        """
        # propagate opponent by time dependent on distance
        dist_in_front = normalize_s(obs.s_center - self.car_s, self.track_length)
        if dist_in_front < self.lookahead:  # TODO make param
            # distance in s coordinate
            rel_speed = np.clip(self.car_vs - obs.vs, 0.1, 10)
            ot_time_distance = np.clip(dist_in_front / rel_speed, 0, 5) * 0.5

            delta_s = ot_time_distance * obs.vs
            delta_d = ot_time_distance * obs.vd
            # delta_d = -(obs.d_center + delta_d) * np.exp(-np.abs(self.kd_obs_pred * obs.d_center))

            # update
            obs.s_start += delta_s
            obs.s_center += delta_s
            obs.s_end += delta_s
            obs.s_start = normalize_s(obs.s_start, self.track_length)
            obs.s_center = normalize_s(obs.s_center, self.track_length)
            obs.s_end = normalize_s(obs.s_end, self.track_length)

            obs.d_left += delta_d
            obs.d_center += delta_d
            obs.d_right += delta_d

            if self.plot_debug:
                resp = self.converter.get_cartesian([obs.s_center], [obs.d_center])
                marker = self.xy_to_point(resp[0], resp[1], opponent=True)
                self.pub_propagated.publish(marker)

        return obs


    def _more_space(self, obstacle: ObstacleMsg, opp_wpnt_idx: int) -> Tuple[str, float]:
        width_left = self.wpnts_d_left_array[opp_wpnt_idx]
        width_right = self.wpnts_d_right_array[opp_wpnt_idx]
        left_gap = abs(width_left - obstacle.d_left)
        right_gap = abs(width_right + obstacle.d_right)
        min_space = self.evasion_dist + self.spline_bound_mindist

        if right_gap > min_space and left_gap < min_space:
            # Compute apex distance to the right of the opponent
            d_apex_right = obstacle.d_right - self.evasion_dist
            # If we overtake to the right of the opponent BUT the apex is to the left of the raceline, then we set the apex to 0
            if d_apex_right > 0:
                d_apex_right = 0
            return "right", d_apex_right

        elif left_gap > min_space and right_gap < min_space:
            # Compute apex distance to the left of the opponent
            d_apex_left = obstacle.d_left + self.evasion_dist
            # If we overtake to the left of the opponent BUT the apex is to the right of the raceline, then we set the apex to 0
            if d_apex_left < 0:
                d_apex_left = 0
            return "left", d_apex_left
        else:
            candidate_d_apex_left = obstacle.d_left + self.evasion_dist
            candidate_d_apex_right = obstacle.d_right - self.evasion_dist

            if abs(candidate_d_apex_left) < abs(candidate_d_apex_right):
                # If we overtake to the left of the opponent BUT the apex is to the right of the raceline, then we set the apex to 0
                if candidate_d_apex_left < 0:
                    candidate_d_apex_left = 0
                return "left", candidate_d_apex_left
            else:
                # If we overtake to the right of the opponent BUT the apex is to the left of the raceline, then we set the apex to 0
                if candidate_d_apex_right > 0:
                    candidate_d_apex_right = 0
                return "right", candidate_d_apex_right



    def do_spline(self, obstacles: ObstacleArray) -> Tuple[WpntArray, MarkerArray]:
        """
        Creates an evasion trajectory for the closest obstacle by splining between pre- and post-apex points.

        This function takes as input the obstacles to be evaded, and a list of global waypoints that describe a reference raceline.
        It only considers obstacles that are within a threshold of the raceline and generates an evasion trajectory for each of these obstacles.
        The evasion trajectory consists of a spline between pre- and post-apex points that are spaced apart from the obstacle.
        The spatial and velocity components of the spline are calculated using the `Spline` class, and the resulting waypoints and markers are returned.

        Args:
        - obstacles (ObstacleArray): An array of obstacle objects to be evaded.
        - gb_wpnts (WpntArray): A list of global waypoints that describe a reference raceline.
        - state (Odometry): The current state of the car.

        Returns:
        - wpnts (WpntArray): An array of waypoints that describe the evasion trajectory to the closest obstacle.
        - mrks (MarkerArray): An array of markers that represent the waypoints in a visualization format.

        """
        # Return wpnts and markers
        mrks = MarkerArray()
        wpnts = OTWpntArray()

        # Only use obstacles that are within a threshold of the raceline, else we don't care about them
        close_obs = self._obs_filtering(obstacles=obstacles)

        # If there are obstacles within the lookahead distance, then we need to generate an evasion trajectory considering the closest one
        if len(close_obs) > 0:
            # Get the closest obstacle handling wraparound
            closest_obs = min(
                close_obs, key=lambda obs: normalize_s(obs.s_center - self.car_s, self.track_length)
            )

            # Get Apex for evasion that is further away from the trackbounds
            if closest_obs.s_end < closest_obs.s_start:
                s_apex = (closest_obs.s_end + self.track_length +
                          closest_obs.s_start) / 2
                s_apex = normalize_s(s_apex, self.track_length)
            else:
                s_apex = (closest_obs.s_end + closest_obs.s_start) / 2
            # Approximate next 20 indexes of global wpnts with wrapping => 2m and compute which side is the outside of the raceline
            obstacle_idx = utils.find_closest_index(self.wpnts_s_array, s_apex)
            gb_idxs = [(obstacle_idx + i) % self.gb_max_idx for i in range(20)]
            kappas = np.array([self.kappa_array[gb_idx] for gb_idx in gb_idxs])
            outside = "left" if np.sum(kappas) < 0 else "right"
            # Choose the correct side and compute the distance to the apex based on left of right of the obstacle
            more_space, d_apex = self._more_space(closest_obs, obstacle_idx)

            # Publish the point around which we are splining
            if self.plot_debug:
                mrk = self.xy_to_point(
                    x=self.waypoints_x[obstacle_idx], y=self.waypoints_y[obstacle_idx], opponent=False)
                self.closest_obs_pub.publish(mrk)

            # Choose wpnts from global trajectory for splining with velocity
            evasion_points = []
            spline_params = [
                self.pre_apex_0,
                self.pre_apex_1,
                self.pre_apex_2,
                0,
                self.post_apex_0,
                self.post_apex_1,
                self.post_apex_2,
            ]
            for i, dst in enumerate(spline_params):
                # scale dst linearly between 1 and 1.5 depending on the speed normalised to the max speed
                dst = dst * np.clip(1.0 + self.car_vs / self.gb_vmax, 1, 1.5)
                # If we overtake on the outside, we smoothen the spline
                if outside == more_space:
                    si = s_apex + dst * 1.75  # TODO make parameter
                else:
                    si = s_apex + dst
                di = d_apex if dst == 0 else 0
                evasion_points.append([si, di])
            # Convert to nump
            evasion_points = np.array(evasion_points)

            # Spline spatialy for d with s as base
            spline_resolution = 0.25
            spatial_spline = Spline(
                x=evasion_points[:, 0], y=evasion_points[:, 1])
            evasion_s = np.arange(
                evasion_points[0, 0], evasion_points[-1, 0], spline_resolution)
            # Clipe the d to the apex distance
            if d_apex < 0:
                evasion_d = np.clip(spatial_spline(evasion_s), d_apex, 0)
            else:
                evasion_d = np.clip(spatial_spline(evasion_s), 0, d_apex)

            # Handle Wrapping of s
            evasion_s = normalize_s(evasion_s, self.track_length)

            # Do frenet conversion via conversion service for spline and create markers and wpnts
            danger_flag = False
            resp = self.converter.get_cartesian(evasion_s, evasion_d)

            # Check if a side switch is possible
            if not self._check_ot_side_possible(more_space):
                danger_flag = True

            for i in range(evasion_s.shape[0]):
                gb_wpnt_i = utils.find_closest_index(self.wpnts_s_array, evasion_s[i])
                # Check if wpnt is too close to the trackbounds but only if spline is actually off the raceline
                if abs(evasion_d[i]) > spline_resolution:
                    tb_dist = self.wpnts_d_left_array[gb_wpnt_i] if more_space == "left" else self.wpnts_d_right_array[gb_wpnt_i]
                    # Check if the spline is too close to the trackbounds
                    if abs(evasion_d[i]) > abs(tb_dist) - self.spline_bound_mindist:
                        self.get_logger().info(
                            "Evasion trajectory too close to TRACKBOUNDS, aborting evasion"
                        )
                        danger_flag = True
                        break
                # Get V from gb wpnts and go slower if we are going through the inside
                # TODO make speed scaling ros param
                vi = self.waypoints_v[gb_wpnt_i] if outside == more_space else self.waypoints_v[gb_wpnt_i] * 0.8
                wpnts.wpnts.append(
                    self.xyv_to_wpnts(
                        x=resp[0, i], y=resp[1, i], s=evasion_s[i], d=evasion_d[i], v=vi, wpnts=wpnts)
                )
                mrks.markers.append(self.xyv_to_markers(
                    x=resp[0, i], y=resp[1, i], v=vi, mrks=mrks))

            # Fill the rest of OTWpnts
            wpnts.header.stamp = self.get_clock().now().to_msg()
            wpnts.header.frame_id = "map"
            if not danger_flag:
                wpnts.ot_side = more_space
                wpnts.ot_line = outside
                wpnts.side_switch = True if self.last_ot_side != more_space else False
                wpnts.last_switch_time = self.last_switch_time

                # Update the last switch time and the last side
                if self.last_ot_side != more_space:
                    self.last_switch_time = self.get_clock().now().to_msg()
                self.last_ot_side = more_space
            else:
                wpnts.wpnts = []
                mrks.markers = []
                # This fools the statemachine to cool down
                wpnts.side_switch = True
                self.last_switch_time = self.get_clock().now().to_msg()
                self.last_ot_side = more_space
        return wpnts, mrks


    def _obs_filtering(self, obstacles: ObstacleArray) -> List[ObstacleMsg]:
        # Only use obstacles that are within a threshold of the raceline, else we don't care about them
        obs_on_traj = [obs for obs in obstacles.obstacles if abs(
            obs.d_center) < self.obs_traj_tresh]

        # Only use obstacles that within self.lookahead in front of the car
        close_obs = []
        for obs in obs_on_traj:
            obs = self._predict_obs_movement(obs)
            # Handle wraparound
            dist_in_front = normalize_s(obs.s_center - self.car_s, self.track_length)
            # dist_in_back = abs(dist_in_front % (-self.track_length)) # distance from ego to obstacle in the back
            if dist_in_front < self.lookahead:
                close_obs.append(obs)
        return close_obs
    
    def _check_ot_side_possible(self, more_space) -> bool:
        # TODO make rosparam for cur_d threshold
        if abs(self.car_d) > 4.9 and more_space != self.last_ot_side:
            self.get_logger().info("Can't switch sides, because we are not on the raceline")
            return False
        return True

    #################### VISUALIZATION FUNCTIONS####################
    def xyv_to_markers(self, x: float, y: float, v: float, mrks: MarkerArray) -> Marker:
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.type = mrk.CYLINDER
        mrk.scale.x = 0.1
        mrk.scale.y = 0.1
        mrk.scale.z = float(v / self.gb_vmax)
        mrk.color.a = 1.0
        mrk.color.b = 0.75
        mrk.color.r = 0.75

        mrk.id = len(mrks.markers)
        mrk.pose.position.x = float(x)
        mrk.pose.position.y = float(y)
        mrk.pose.position.z = 0.0
        mrk.pose.orientation.w = 1.0

        return mrk

    def xy_to_point(self, x: float, y: float, opponent=True) -> Marker:
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.type = mrk.SPHERE
        mrk.scale.x = 0.5
        mrk.scale.y = 0.5
        mrk.scale.z = 0.5
        mrk.color.a = 0.8
        mrk.color.b = 0.65
        mrk.color.r = 1.0 if opponent else 0.0
        mrk.color.g = 0.65

        mrk.pose.position.x = float(x)
        mrk.pose.position.y = float(y)
        mrk.pose.position.z = 0.0
        mrk.pose.orientation.w = 1.0

        return mrk

    def xyv_to_wpnts(self, s: float, d: float, x: float, y: float, v: float, wpnts: OTWpntArray) -> Wpnt:
        wpnt = Wpnt()
        wpnt.id = len(wpnts.wpnts)
        wpnt.x_m = float(x)
        wpnt.y_m = float(y)
        wpnt.s_m = float(s)
        wpnt.d_m = float(d)
        wpnt.vx_mps = float(v)
        return wpnt
    

def main(args=None):
    rclpy.init(args=args)
    node = SplineNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    





            
        

        



        


