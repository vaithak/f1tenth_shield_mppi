#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from functools import partial
import copy

import numpy as np

import rclpy
from rclpy.node import Node
import tf_transformations
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray

from shield_mppi.trajectory_sampler import MPPICBFStochasticTrajectoriesSampler
from shield_mppi.dynamics import VehicleDynamics
from shield_mppi.frenet_conversion_jax import FrenetConverter
from shield_mppi.utils import nearest_point
from shield_mppi.cost_evaluator import MPPICBFCostEvaluator
from shield_mppi.collision_checker import CollisionChecker
from f1tenth_icra_race_msgs.msg import ObstacleArray, WpntArray
from std_msgs.msg import String


class oneLineJaxRNG:
    def __init__(self, init_num=0) -> None:
        self.rng = jax.random.PRNGKey(init_num)
    
    def new_key(self):
        self.rng, key = jax.random.split(self.rng)
        return key

class ShieldMPPI(Node):
    def __init__(
        self,
        control_horizon=12,
        num_traj=500,
        control_dim=2,
        inverse_temperature=1,
        initial_control_sequence=np.zeros((2, 1)),
        repair_horizon=4,
        repair_steps=4,
        control_bounds=np.array([[-0.2, -1.0], [0.2, 3.0]]),
        mean=np.array([0.0, 0.0]),
        cov=np.array([[0.5, 0.0], [0.0, 2.5]]),
    ):
        super().__init__('shield_mppi_node')
        print("Shield MPPI Node Initialized")

        import os
        cwd = os.getcwd()

        self.jrng = oneLineJaxRNG(42)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('is_sim', True),
                ('plot_debug', False),
                ('print_debug', False),
                ('dt', 0.1),
                ('max_steering_angle', 0.2),
                ('max_speed', 1.5),
                ('goal_tolerance', 0.1),
                ('waypoint_file', f'{cwd}/src/f1tenth_shield_mppi/waypoints/fitted_2.csv'),
            ]
        )

        # Parameters
        self.is_sim = self.get_parameter('is_sim').get_parameter_value().bool_value
        self.plot_debug = self.get_parameter('plot_debug').get_parameter_value().bool_value
        self.print_debug = self.get_parameter('print_debug').get_parameter_value().bool_value
        self.dt = self.get_parameter('dt').get_parameter_value().double_value
        self.max_steering_angle = self.get_parameter('max_steering_angle').get_parameter_value().double_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.goal_tolerance = self.get_parameter('goal_tolerance').get_parameter_value().double_value
        self.waypoint_file = self.get_parameter('waypoint_file').get_parameter_value().string_value
        self.dlk = 0.15 # dist step [m] kinematic
        self.mean = mean
        self.cov = cov
        self.num_traj = num_traj

        waypoints = np.genfromtxt(self.waypoint_file, delimiter=',')
        self.waypoints_x = waypoints[:, 0]
        self.waypoints_y = waypoints[:, 1]
        self.waypoints_psi = waypoints[:, 3]
        self.waypoints_speed = waypoints[:, 2]
        self.frenet_converter = FrenetConverter(
            waypoints_x=self.waypoints_x,
            waypoints_y=self.waypoints_y,
            waypoints_psi=self.waypoints_psi,
        )

        self.inverse_temperature = inverse_temperature
        self.curr_control_sequence = initial_control_sequence
        self.stochastic_trajectories_sampler = MPPICBFStochasticTrajectoriesSampler(
            number_of_trajectories=num_traj,
        )
        self.control_horizon = control_horizon
        self.control_dim = control_dim
        self.repair_horizon = repair_horizon
        self.repair_steps = repair_steps
        self.min_controls = control_bounds[0]
        self.max_controls = control_bounds[1]
        self.control_bounds = control_bounds

        self.dynamics = VehicleDynamics(state_dim=(7, num_traj), delta_t=self.dt, frenet_converter=self.frenet_converter)
        collision_checker = CollisionChecker(
            track_width=0.75,
        )
        self.cost_evaluator = MPPICBFCostEvaluator(
            cbf_alpha=0.9,
            collision_checker=collision_checker,
            Q = np.diag([5.0, 5.0, 0.0, 5.0, 10.0, 0.0, 10.0]),
            QN = np.diag([100.0, 100.0, 0.0, 50.0, 100.0, 0.0, 100.0]),
            R = np.diag([5.0, 1.0]),
            collision_cost=800.0,
            goal_cost=None
        )

        qos = rclpy.qos.QoSProfile(history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
                                   depth=10,
                                   reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
                                   durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE)

        # Subscribers
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
        self.create_subscription(WpntArray, '/state_machine/local_waypoints', self.wpnts_callback, qos)
        
            
        # Subscribe to obstacles topic /perception/detection/raw_obstacles
        self.obstacles_sub = self.create_subscription(
            ObstacleArray,
            '/perception/detection/raw_obstacles',
            self.obstacles_callback,
            qos)
        self.obstacles = None
        self.obstacles_radius = None
        
            
        # Publishers for visualization
        self.reference_pub = self.create_publisher(MarkerArray, '/reference', qos)
        self.trajectory_pub = self.create_publisher(MarkerArray, '/trajectory', qos)

        self.control = np.array([0.0, 0.0])  # steering angle, speed
        # Publisher
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', qos)

    def wpnts_callback(self, msg: WpntArray):
        local_waypoints = []
        for waypoint in msg.wpnts:
            local_waypoints.append([
                waypoint.x_m,
                waypoint.y_m,
                waypoint.vx_mps,
                waypoint.s_m, waypoint.psi_rad, waypoint.ax_mps2])
        local_waypoints = np.array(local_waypoints)
        self.waypoints_x = local_waypoints[:, 0]
        self.waypoints_y = local_waypoints[:, 1]
        self.waypoints_psi = local_waypoints[:, 4] #+ np.pi / 2.0
        self.waypoints_speed = local_waypoints[:, 2]

    def publish_reference(self, ref_traj):
        marker_array = MarkerArray()
        for i in range(ref_traj.shape[1]):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "reference"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = ref_traj[0, i]
            marker.pose.position.y = ref_traj[1, i]
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = np.sin(ref_traj[4, i] / 2.0)
            marker.pose.orientation.w = np.cos(ref_traj[4, i] / 2.0)
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)
        self.reference_pub.publish(marker_array)

    def publish_sampled_trajectories(self, trajectories):
        marker_array = MarkerArray()
        
        for i in range(trajectories.shape[0]):
            # Create one line strip marker per trajectory
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "sampled_trajectory"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            
            # Set proper scale for visualization (thinner lines)
            marker.scale.x = 0.05  # Line width
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            # Set marker lifetime (optional but recommended)
            # marker.lifetime = Duration(seconds=1).to_msg()
            
            # Add all points for this trajectory
            for j in range(trajectories.shape[2]):
                point = Point()
                point.x = float(trajectories[i, 0, j])
                point.y = float(trajectories[i, 1, j])
                point.z = 0.0
                marker.points.append(point)
            
            marker_array.markers.append(marker)

        self.trajectory_pub.publish(marker_array)

    
    def obstacles_callback(self, msg):
        if msg.obstacles is None or len(msg.obstacles) == 0:
            self.obstacles = None
            self.obstacles_radius = None
            return
        
        car_width_margin = 0.10
        obstacles_s = np.array([obstacle.s_center for obstacle in msg.obstacles])
        obstacles_d = np.array([obstacle.d_center for obstacle in msg.obstacles])
        self.obstacles_radius = np.array([obstacle.size + car_width_margin for obstacle in msg.obstacles])

        # Convert to cartesian coordinates
        obstacles_x, obstacles_y = self.frenet_converter.get_cartesian(obstacles_s, obstacles_d)
        self.obstacles = np.array([obstacles_x, obstacles_y]).T
    

    def pose_callback(self, msg):
        pose = msg.pose.pose
        twist = msg.twist.twist

        # Extract slip angle and yaw
        # slip_angle = np.arctan2(twist.linear.y, twist.linear.x)
        yaw = tf_transformations.euler_from_quaternion(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])[2]
        
        s, d = self.frenet_converter.get_frenet(np.array([pose.position.x]), np.array([pose.position.y]))
        
        # Calculate current state
        curr_state_arr = np.array([
            pose.position.x,
            pose.position.y,
            self.control[0],  # steering angle
            twist.linear.x,
            yaw,
            # twist.angular.z,
            # slip_angle
            s[0],
            d[0]
        ])

        if self.waypoints_x is None or self.waypoints_y is None:
            return

        # Plan control
        new_control = self.plan(curr_state_arr)
        self.control += new_control * self.dt
        self.control[0] = np.clip(self.control[0], -self.max_steering_angle, self.max_steering_angle)
        self.control[1] = np.clip(self.control[1], 1.0, self.max_speed)

        # Publish control
        self.publish_control(self.control, msg.header)


    def publish_control(self, control, header):
        drive_msg = AckermannDriveStamped()
        drive_msg.header = header
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = control[0]
        drive_msg.drive.speed = control[1]
        self.drive_pub.publish(drive_msg)

    
    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        calc referent trajectory ref_traj in T steps: [x, y, 0, v, yaw, 0, 0]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((len(state), self.control_horizon))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        # _, _, _, ind = nearest_point(np.array([state[0], state[1]]), np.array([cx, cy]).T)
        ind = 0   
        # print("cx shape", cx.shape)
        # print("cy shape", cy.shape)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[3, 0] = sp[ind]
        ref_traj[4, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(max(1.0, state[3])) * self.dt
        dind = travel / self.dlk
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.control_horizon-1)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[3, :] = sp[ind_list]
        cyaw[cyaw - state[4] > 3.15] = cyaw[cyaw - state[4] > 3.15] - (2 * np.pi)
        cyaw[cyaw - state[4] < -3.15] = cyaw[cyaw - state[4] < -3.15] + (2 * np.pi)
        ref_traj[4, :] = cyaw[ind_list]
        return ref_traj


    def plan(self, state_cur):
        ref_traj = np.zeros((self.dynamics.state_dim[0], self.control_horizon))
        # Get the reference trajectory
        ref_traj = self.calc_ref_trajectory(
            state_cur,
            self.waypoints_x,
            self.waypoints_y,
            self.waypoints_psi,
            self.waypoints_speed
        )
        # Publish reference trajectory
        self.publish_reference(ref_traj)

        # min_controls = [-1.0, -0.1]
        # max_controls = [1.0, 0.4]
        # control_bounds = [min_controls, max_controls]
        # control_bounds = np.asarray(control_bounds)
        v = copy.deepcopy(self.curr_control_sequence)
        warm_start_itr = 1
        for _ in range(warm_start_itr):
            noises = jax.random.multivariate_normal(
                self.jrng.new_key(),
                self.mean,
                self.cov,
                (self.control_horizon - 1) * self.num_traj
            ).T

            trajectories, us, costs = self.stochastic_trajectories_sampler.sample(
                state_cur,
                v,
                ref_traj,
                self.obstacles,
                self.obstacles_radius,
                self.control_horizon,
                self.control_dim,
                self.dynamics,
                self.cost_evaluator,
                control_bounds=jnp.array(self.control_bounds),
                noises=noises,
            )

            # Publish sampled trajectories
            # self.publish_sampled_trajectories(trajectories)

            beta = np.min(costs)
            eta = np.sum(np.exp(-1 / self.inverse_temperature * (costs - beta)))
            omega = 1 / eta * np.exp(-1 / self.inverse_temperature * (costs - beta))
            v = np.sum(
                omega.reshape((us.shape[0], 1, 1)) * us, axis=0
            )  # us shape = (number_of_trajectories, control_dim, control_horizon)
            self.curr_control_sequence = v
        if self.repair_horizon:  # if we use CBF to carry out local repair
            if self.request_repair(state_cur):
                # Only repair if the vehicle is outside the reference trajectory
                v_safe = self.local_repair(v, state_cur)
            else:
                v_safe = v
        else:  # if original MPPI
            v_safe = v
        u = v_safe[:, 0]
        # Control Bounds
        u = np.clip(u, self.min_controls, self.max_controls)
        v = np.delete(v, 0, 1)
        v = np.hstack((v, v[:, -1].reshape(v.shape[0], 1)))
        self.curr_control_sequence = v
        return u


    def request_repair(self, state_cur):
        # If the vehicle is outside the reference trajectory, we need to repair
        # the control sequence to ensure safety
        if self.obstacles is None or self.obstacles.shape[0] == 0:
            if np.abs(state_cur[-1]) > 0.3:
                # Only repair if the vehicle is outside the reference trajectory
                return True
        else:
            # TODO: check if the vehicle is near any obstacles
            return True
        
        return False


    @partial(jax.jit, static_argnums=(0, 3))
    def repair_cost(self, control, state, control_shape):
        print("re-tracing repair cost")
        cost = 0
        state = state.reshape(-1, 1)
        control = control.reshape(control_shape)

        # Loop over the rollout horizon using LAX scan
        controls = control.T  # transpose so leading axis is timestep
        initial_carry = (0.0, state)  # cost and current state
        def scan_fn(carry, control):
            print(f"scan_fn: {carry}")
            running_cost, state = carry
            state_next = self.dynamics.propagate(state, control.reshape(-1, 1)).reshape(-1, 1)
            running_cost += self.cost_evaluator.evaluate_cbf_cost(
                state, dynamics=self.dynamics, state_next=state_next,
                obstacles_list=self.obstacles, obstacles_radius=self.obstacles_radius
            )
            running_cost = jnp.sum(running_cost) # to reduce to scalar
            return (running_cost, state_next), None

        final_carry, _ = jax.lax.scan(scan_fn, initial_carry, controls)
        cost, _ = final_carry

        print("done tracing repair cost")

        return cost

    @partial(jax.jit, static_argnums=(0,))
    def local_repair(self, v, state_cur):
        print(f"re-tracing repair fn")
        v = jnp.array(v[:, :self.repair_horizon])

        result = minimize(
            self.repair_cost,
            v.reshape(-1),
            args=(state_cur, v.shape),
            method="BFGS",
            options={"maxiter": self.repair_steps}
        )

        print("done tracing repair fn, but JIT compilation may take awhile...")

        return result.x.reshape(v.shape)

    def rollout_out(self, state_cur, v):
        trajectory = np.zeros((self.dynamics.state_dim[0], v.shape[1] + 1))
        trajectory = trajectory.at[:, 0].set(state_cur)
        for i in range(v.shape[1]):
            state_next = self.dynamics.propagate(state_cur.reshape(-1, 1), v[:, i].reshape(-1, 1)).reshape(state_cur.shape)
            trajectory = trajectory.at[:, i + 1].set(state_next)
            state_cur = state_next
        return trajectory


def main(args=None):
    rclpy.init(args=args)
    shield_mppi_node = ShieldMPPI()
    rclpy.spin(shield_mppi_node)
    shield_mppi_node.destroy_node()
    rclpy.shutdown()

    
if __name__ == '__main__':
    main()


"""
Fixes:
- waypoint file columns
"""