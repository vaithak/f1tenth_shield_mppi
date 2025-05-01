#!/usr/bin/env python3

import time
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
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from shield_mppi.trajectory_sampler import MPPICBFStochasticTrajectoriesSampler
from shield_mppi.dynamics import VehicleDynamics
from shield_mppi.frenet_conversion_jax import FrenetConverter
from shield_mppi.utils import nearest_point
from shield_mppi.cost_evaluator import MPPICBFCostEvaluator
from shield_mppi.collision_checker import CollisionChecker

def _numpy_to_multiarray(multiarray_type, np_array):
    multiarray = multiarray_type()
    multiarray.layout.dim = [MultiArrayDimension(label='dim%d' % i,
                                                 size=np_array.shape[i],
                                                 stride=np_array.shape[i] * np_array.dtype.itemsize) for i in range(np_array.ndim)];
    multiarray.data = np_array.reshape([1, -1])[0].tolist()
    return multiarray

def _multiarray_to_numpy(pytype, dtype, multiarray):
    dims = tuple(map(lambda x: x.size, multiarray.layout.dim))
    return np.array(multiarray.data, dtype=pytype).reshape(dims).astype(dtype)

to_multiarray_f32 = partial(_numpy_to_multiarray, Float32MultiArray)
to_numpy_f32 = partial(_multiarray_to_numpy, float, np.float32)


class ShieldMPPI(Node):
    def __init__(
        self,
        control_horizon=12,
        num_traj=60,
        control_dim=2,
        inverse_temperature=1,
        initial_control_sequence=np.zeros((2, 1)),
        repair_horizon=4,
        repair_steps=4,
        control_bounds=np.array([[-0.4, -1.0], [0.4, 1.0]]),
    ):
        super().__init__('shield_mppi_node')
        print("Shield MPPI Node Initialized")

        import os
        cwd = os.getcwd()

        self.declare_parameters(
            namespace='',
            parameters=[
                ('is_sim', True),
                ('plot_debug', False),
                ('print_debug', False),
                ('num_samples', 100),
                ('num_steps', 10),
                ('dt', 0.1),
                ('max_steering_angle', 0.5),
                ('max_speed', 4.0),
                ('goal_tolerance', 0.1),
                ('waypoint_file', f'{cwd}/src/shield_mppi/waypoints/fitted_2.csv'),
            ]
        )

        # Parameters
        self.is_sim = self.get_parameter('is_sim').get_parameter_value().bool_value
        self.plot_debug = self.get_parameter('plot_debug').get_parameter_value().bool_value
        self.print_debug = self.get_parameter('print_debug').get_parameter_value().bool_value
        self.num_samples = self.get_parameter('num_samples').get_parameter_value().integer_value
        self.num_steps = self.get_parameter('num_steps').get_parameter_value().integer_value
        self.dt = self.get_parameter('dt').get_parameter_value().double_value
        self.max_steering_angle = self.get_parameter('max_steering_angle').get_parameter_value().double_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.goal_tolerance = self.get_parameter('goal_tolerance').get_parameter_value().double_value
        self.waypoint_file = self.get_parameter('waypoint_file').get_parameter_value().string_value
        self.dlk = 0.27 # dist step [m] kinematic

        self.waypoints = np.genfromtxt(self.waypoint_file, delimiter=',')
        self.waypoints_x = self.waypoints[:, 0]
        self.waypoints_y = self.waypoints[:, 1]
        self.waypoints_psi = self.waypoints[:, 3]
        self.waypoints_speed = self.waypoints[:, 2]
        self.frenet_converter = FrenetConverter(
            waypoints_x=self.waypoints_x,
            waypoints_y=self.waypoints_y,
            waypoints_psi=self.waypoints_psi,
        )

        self.inverse_temperature = inverse_temperature
        self.curr_control_sequence = initial_control_sequence
        self.stochastic_trajectories_sampler = MPPICBFStochasticTrajectoriesSampler(
            number_of_trajectories=num_traj,
            mean=np.zeros((control_dim,)),
            cov=np.eye(control_dim)
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
            track_width=0.5,
        )
        self.cost_evaluator = MPPICBFCostEvaluator(
            cbf_alpha=0.9,
            collision_checker=collision_checker,
            Q = np.diag([5.0, 5.0, 0.0, 3.0, 1.0, 0.0, 10.0]),
            QN = np.diag([10.0, 10.0, 0.0, 5.0, 1.0, 0.0, 1000.0]),
            R = np.diag([10.0, 5.0]),
            collision_cost=1000.0,
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

        self.control = np.array([0.0, 0.0])  # steering angle, speed
        # Publisher
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', qos)


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

        # Plan control
        new_control = self.plan(curr_state_arr)
        self.control += new_control * self.dt
        self.control[0] = np.clip(self.control[0], -self.max_steering_angle, self.max_steering_angle)
        self.control[1] = np.clip(self.control[1], 0.0, self.max_speed)

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
        _, _, _, ind = nearest_point(np.array([state[0], state[1]]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[3, 0] = sp[ind]
        ref_traj[4, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(state[3]) * self.dt
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
        print("Ref Traj: ", ref_traj)
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

        # min_controls = [-1.0, -0.1]
        # max_controls = [1.0, 0.4]
        # control_bounds = [min_controls, max_controls]
        # control_bounds = np.asarray(control_bounds)
        start_control = time.perf_counter()
        v = copy.deepcopy(self.curr_control_sequence)
        warm_start_itr = 1
        for _ in range(warm_start_itr):

            trajectories, us, costs = self.stochastic_trajectories_sampler.sample(
                state_cur,
                v,
                ref_traj,
                self.control_horizon,
                self.control_dim,
                self.dynamics,
                self.cost_evaluator,
                control_bounds=jnp.array(self.control_bounds)
            )

            beta = np.min(costs)
            eta = np.sum(np.exp(-1 / self.inverse_temperature * (costs - beta)))
            omega = 1 / eta * np.exp(-1 / self.inverse_temperature * (costs - beta))
            v = np.sum(
                omega.reshape((us.shape[0], 1, 1)) * us, axis=0
            )  # us shape = (number_of_trajectories, control_dim, control_horizon)
            self.curr_control_sequence = v
        start = time.perf_counter()
        if self.repair_horizon:  # if we use CBF to carry out local repair
            v_safe = self.local_repair(v, state_cur)
            print("Local repair frequency: ", 1 / (time.perf_counter() - start))
        else:  # if original MPPI
            v_safe = v
            print("No local repair step!")
        u = v_safe[:, 0]
        # Control Bounds
        u = np.clip(u, self.min_controls, self.max_controls)
        print("Control: ", u)

        print("Control update frequency: ", 1 / (time.perf_counter() - start_control))

        # if self.renderer is not None:
            # optimal_trajectory = self.rollout_out(state_cur, v)
            # self.renderer.render_trajectories(trajectories, **{"color": "b"})
            # self.renderer.render_trajectories([optimal_trajectory], **{"color": "r"})
        v = np.delete(v, 0, 1)
        v = np.hstack((v, v[:, -1].reshape(v.shape[0], 1)))
        self.curr_control_sequence = v
        return u

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
                state, dynamics=self.dynamics, state_next=state_next
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