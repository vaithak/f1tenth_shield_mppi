#!/usr/bin/env python3

import rclpy
import numpy as np
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    ReliabilityPolicy,
    QoSHistoryPolicy,
)
from scipy.spatial.transform import Rotation
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from f1tenth_icra_race_msgs.msg import ObstacleArray, OTWpntArray, WpntArray, Wpnt
from visualization_msgs.msg import Marker, MarkerArray
from shield_mppi import utils, frenet_conversion, state_helpers
from tf_transformations import euler_from_quaternion

def normalize_s(s, track_length):
    return s % track_length

def time_to_float(time_instant: Time):
    return time_instant.sec + time_instant.nanosec * 1e-9

def create_wpnts_from_np_array(wpnts_x, wpnts_y, wpnts_v, wpnts_s, wpnts_d, wpnts_psi):
    wpnts = []
    n = len(wpnts_x)
    for i in range(n):
        wpnt = Wpnt()
        wpnt.id = i
        wpnt.x_m = float(wpnts_x[i])
        wpnt.y_m = float(wpnts_y[i])
        wpnt.vx_mps = float(wpnts_v[i])
        wpnt.s_m = float(wpnts_s[i])
        wpnt.d_m = float(wpnts_d[i])
        wpnt.psi_rad = float(wpnts_psi[i])
        wpnts.append(wpnt)
    return wpnts


class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')

        # Subscriptions
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
        # self.kappa_array = self.waypoints[:, waypoint_cols['kappa_racetraj_radpm']]
        self.track_length = self.waypoints[-1, waypoint_cols['s_racetraj_m']]
        self.glb_wpnts = create_wpnts_from_np_array(
            self.waypoints_x, self.waypoints_y, self.waypoints_v,
            self.wpnts_s_array, self.wpnts_d_right_array, waypoints_psi
        )
        self.num_glb_wpnts = len(self.glb_wpnts)

        self.converter = frenet_conversion.FrenetConverter(self.waypoints_x, self.waypoints_y, waypoints_psi)

        # Create parameters for plot and print debugging
        self.declare_parameter('plot_debug', False)
        self.plot_debug = self.get_parameter('plot_debug').value
        self.declare_parameter('print_debug', False)
        self.print_debug = self.get_parameter('print_debug').value
        self.declare_parameter("is_sim", True)
        self.is_sim = self.get_parameter("is_sim").get_parameter_value().bool_value
        self.declare_parameter("rate_hz", 50)
        self.rate_hz = self.get_parameter("rate_hz").get_parameter_value().integer_value
        self.timer = self.create_timer(1.0 / self.rate_hz, self.main_loop_callback)

        # Other parameters
        self.declare_parameter("gb_ego_width_m", 0.15)
        self.gb_ego_width_m = self.get_parameter("gb_ego_width_m").get_parameter_value().double_value
        self.declare_parameter("lateral_width_m", 0.3)
        self.lateral_width_m = self.get_parameter("lateral_width_m").get_parameter_value().double_value
        self.declare_parameter("overtaking_horizon_m", 6.0)
        self.overtaking_horizon_m = self.get_parameter("overtaking_horizon_m").get_parameter_value().double_value
        self.declare_parameter("spline_hyst_timer_sec", 0.3)
        self.spline_hyst_timer_sec = self.get_parameter("spline_hyst_timer_sec").get_parameter_value().double_value
        self.declare_parameter("n_loc_wpnts", 50)
        self.n_loc_wpnts = self.get_parameter("n_loc_wpnts").get_parameter_value().integer_value

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
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
            
        self.create_subscription(
            ObstacleArray, '/perception/obstacles', self.obstacle_callback, qos
        )
        self.create_subscription(OTWpntArray, '/planner/avoidance/otwpnts', self.avoidance_cb, qos)

        # Publishers
        self.state_pub = self.create_publisher(String, '/state_machine/state', qos)
        self.loc_wpnt_pub = self.create_publisher(WpntArray, '/state_machine/local_waypoints', qos)
        if self.plot_debug:
            self.vis_loc_wpnt_pub = self.create_publisher(MarkerArray, '/state_machine/local_waypoints/markers', qos)
            self.state_marker_pub = self.create_publisher(Marker, '/state_machine/state_marker', qos)

        self.car_global_x = None
        self.car_global_y = None
        self.car_global_yaw = None
        self.car_s = None
        self.car_d = None
        self.first_visualization = True

        self.obstacles = []
        self.spline_ttl = 1.0 # Follow the spline for 1 second in worst case
        self.spline_ttl_counter = int(self.spline_ttl * self.rate_hz)
        self.avoidance_wpnts = None
        self.last_valid_avoidance_wpnts = None
        self.local_waypoints = WpntArray()

        # Declare parameters for state machine
        self.state_logic = state_helpers.DefaultStateLogic
        
        self.declare_parameter("mode", "head_to_head")
        self.mode = self.get_parameter("mode").get_parameter_value().string_value
        if self.mode == "head_to_head":
            self.state_transition = state_helpers.head_to_head_transition
        elif self.mode == "timetrials":
            self.state_transition = state_helpers.timetrials_transition
        else:
            self.state_transition = state_helpers.dummy_transition
        self.declare_parameter("force_state", "None")
        self.force_state = self.get_parameter("force_state").get_parameter_value().string_value
        self.force_state = state_helpers.string_to_state_type(self.force_state)
        if self.force_state is None:
            self.state = state_helpers.StateType.GB_TRACK
        else:
            self.state = self.force_state


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


    def obstacle_callback(self, msg):
        if len(msg.obstacles) == 0:
            self.obstacles = []
        else:
            self.obstacles = msg.obstacles

    
    def avoidance_cb(self, msg):
        if len(msg.wpnts) > 0:
            self.avoidance_wpnts = msg
            self.spline_ttl_counter = int(self.spline_ttl * self.rate_hz)

    # TODO: Add sectors
    @property
    def _check_ot_sector(self) -> bool:
        # for sector in self.ot_sectors:
        #     if sector['ot_flag']:
        #         if (sector['start'] <= self.car_s / self.waypoints_dist <= (sector['end']+1)):
        #             return True
        # return False
        return True

    @property
    def _check_close_to_raceline(self) -> bool:
        return np.abs(self.car_d) < self.gb_ego_width_m  # [m]
    
    @property
    def _check_ofree(self) -> bool:
        o_free = True

        if self.last_valid_avoidance_wpnts is not None:
            horizon = self.overtaking_horizon_m  # Horizon in front of car_s [m]

            for obs in self.obstacles:
                obs_s = obs.s_center
                # Wrapping madness to check if infront
                dist_to_obj = (obs_s - self.car_s) % self.track_length
                if dist_to_obj < horizon and len(self.last_valid_avoidance_wpnts):
                    obs_d = obs.d_center
                    # Get d wrt to mincurv from the overtaking line
                    avoid_wpnt_idx = np.argmin(
                        np.array([abs(avoid_s.s_m - obs_s) for avoid_s in self.last_valid_avoidance_wpnts])
                    )
                    ot_d = self.last_valid_avoidance_wpnts[avoid_wpnt_idx].d_m
                    ot_obs_dist = ot_d - obs_d
                    if abs(ot_obs_dist) < self.lateral_width_m:
                        o_free = False
                        break
        else:
            o_free = True
        return o_free    
    
    @property
    def _check_gbfree(self) -> bool:
        gb_free = True
        horizon = self.overtaking_horizon_m  # Horizon in front of car_s [m]

        for obs in self.obstacles:
            obs_s = obs.s_center
            gap = (obs_s - self.car_s) % self.track_length
            if gap < horizon:
                obs_d = obs.d_center
                # Get d wrt to mincurv from the overtaking line
                if abs(obs_d) < self.lateral_width_m:
                    gb_free = False
                    #self.get_logger().info(f"GB_FREE False, obs dist to ot lane: {obs_d} m")
                    break

        return gb_free
    
    @property
    def _check_availability_spline_wpts(self) -> bool:
        if self.avoidance_wpnts is None:
            return False
        elif len(self.avoidance_wpnts.wpnts) == 0:
            return False
        # Say no to the ot line if the last switch was less than 0.75 seconds ago
        elif (
            abs(time_to_float(self.avoidance_wpnts.header.stamp) - time_to_float(self.avoidance_wpnts.last_switch_time))
            < self.spline_hyst_timer_sec
        ):
            return False
        else:
            # If the splines are valid update the last valid ones
            self.last_valid_avoidance_wpnts = self.avoidance_wpnts.wpnts.copy()
            return True
        
    def get_spline_wpts(self) -> WpntArray:
        """
        Obtain the waypoints by fusing those obtained by spliner with the
        global ones.
        """
        spline_glob = self.glb_wpnts.copy()

        # Handle wrapping
        if self.last_valid_avoidance_wpnts is not None and len(self.last_valid_avoidance_wpnts) > 0:
            s_start_idx = utils.find_closest_index(
                self.wpnts_s_array,
                self.last_valid_avoidance_wpnts[0].s_m,
            )
            s_end_idx = utils.find_closest_index(
                self.wpnts_s_array,
                self.last_valid_avoidance_wpnts[-1].s_m,
            )
            if self.last_valid_avoidance_wpnts[-1].s_m > self.last_valid_avoidance_wpnts[0].s_m:
                spline_idxs = [s for s in range(s_start_idx, s_end_idx + 1)]
            else:
                # Wrap around the track
                spline_idxs = [s for s in range(s_start_idx, self.gb_max_idx)] +\
                                [s for s in range(0, s_end_idx + 1)]

            # Get the spline waypoints
            for i, idx in enumerate(spline_idxs):
                spline_glob[idx] = self.last_valid_avoidance_wpnts[min(i, len(self.last_valid_avoidance_wpnts) - 1)]

        # If the last valid points have been reset, then we just pass the global waypoints
        else:
            self.get_logger().warn(f"No valid avoidance waypoints, passing global waypoints")
            pass

        # Compute the yaw of the waypoints
        for i in range(len(spline_glob)):
            if i == len(spline_glob) - 1:
                spline_glob[i].psi_rad = spline_glob[i - 1].psi_rad
            else:
                spline_glob[i].psi_rad = np.arctan2(
                    spline_glob[i + 1].y_m - spline_glob[i - 1].y_m,
                    spline_glob[i + 1].x_m - spline_glob[i - 1].x_m,
                )

        return spline_glob


    def visualize_state(self, state: state_helpers.StateType):
        """
        Function that visualizes the state of the car by displaying a colored cube in RVIZ.

        Parameters
        ----------
        action
            Current state of the car to be displayed
        """
        if self.first_visualization:
            self.first_visualization = False
            x0 = self.glb_wpnts[0].x_m
            y0 = self.glb_wpnts[0].y_m
            x1 = self.glb_wpnts[1].x_m
            y1 = self.glb_wpnts[1].y_m
            # compute normal vector of 125% length of trackboundary but to the left of the trajectory
            xy_norm = (
                -np.array([y1 - y0, x0 - x1]) / np.linalg.norm([y1 - y0, x0 - x1]) * 1.25 * self.glb_wpnts[0].d_left
            )

            self.x_viz = x0 + xy_norm[0]
            self.y_viz = y0 + xy_norm[1]

        mrk = Marker()
        mrk.type = mrk.SPHERE
        mrk.id = int(1)
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.color.a = 1.0
        mrk.color.g = 1.0
        mrk.pose.position.x = float(self.x_viz)
        mrk.pose.position.y = float(self.y_viz)
        mrk.pose.position.z = 0.0
        mrk.pose.orientation.w = 1.0
        mrk.scale.x = 1.0
        mrk.scale.y = 1.0
        mrk.scale.z = 1.0

        # Set color and log info based on the state of the car
        if state == state_helpers.StateType.GB_TRACK:
            mrk.color.g = 1.0
        elif state == state_helpers.StateType.OVERTAKE:
            mrk.color.r = 1.0
            mrk.color.g = 1.0
            mrk.color.b = 1.0
        elif state == state_helpers.StateType.TRAILING:
            mrk.color.r = 0.0
            mrk.color.g = 0.0
            mrk.color.b = 1.0
        self.state_marker_pub.publish(mrk)


    def publish_local_waypoints(self, local_wpnts: WpntArray):
        loc_markers = MarkerArray()
        loc_wpnts = local_wpnts

        # set stamp to now         
        loc_wpnts.header.stamp = self.get_clock().now().to_msg()
        loc_wpnts.header.frame_id = "map"

        # Publish the local waypoints
        if len(loc_wpnts.wpnts) == 0:
            self.get_logger().warn("No local waypoints published...")
        else:
            self.loc_wpnt_pub.publish(loc_wpnts)

        if self.plot_debug:
            for i, wpnt in enumerate(loc_wpnts.wpnts):
                mrk = Marker()
                mrk.header.frame_id = "map"
                mrk.type = mrk.SPHERE
                mrk.scale.x = 0.15
                mrk.scale.y = 0.15
                mrk.scale.z = 0.15
                mrk.color.a = 1.0
                mrk.color.g = 1.0

                mrk.id = i
                mrk.pose.position.x = wpnt.x_m
                mrk.pose.position.y = wpnt.y_m
                mrk.pose.position.z = 0.0
                mrk.pose.orientation.w = 1.0
                loc_markers.markers.append(mrk)

            self.vis_loc_wpnt_pub.publish(loc_markers)


    def main_loop_callback(self):
        if self.car_global_x is None or self.car_global_y is None:
            self.get_logger().warn("No pose received yet")
            return

        # transition logic
        if self.force_state:
            self.state = self.force_state
        else:
            self.state = self.state_transition(self)

        msg = String()
        msg.data = str(self.state)
        self.state_pub.publish(msg)
        if self.plot_debug:
            self.visualize_state(state=self.state)

        self.local_waypoints.wpnts = self.state_logic(self)
        self.publish_local_waypoints(self.local_waypoints)

        if self.mode == "head_to_head":
            self.spline_ttl_counter -= 1
            # Once ttl has reached 0 we overwrite the avoidance waypoints with the empty waypoints
            if self.spline_ttl_counter <= 0:
                self.last_valid_avoidance_wpnts = None
                self.avoidance_wpnts = WpntArray()
                self.spline_ttl_counter = -1


def main(args=None):
    rclpy.init(args=args)
    print("State Machine Initialized")
    node = StateMachine()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()