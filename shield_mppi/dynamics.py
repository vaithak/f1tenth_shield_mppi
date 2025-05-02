import jax
import jax.numpy as jnp
import numpy as np

params_f1tenth = {'mu': 1.0, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074,
                               'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2,
                               'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0,
                               'width': 0.31, 'length': 0.58}  # F1/10 car
params_f1tenth['a_max'] = 3.0
params_f1tenth['v_min'] = 0.0
params_f1tenth['v_max'] = 6.0

def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    """
    Acceleration constraints, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            accl (float): unconstraint desired acceleration
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max (float): maximum allowed acceleration
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            accl (float): adjusted acceleration
    """

    # positive accl limit
    # pos_limit = jax.lax.select(vel > v_switch, a_max*v_switch/vel, a_max)

    # # accl limit reached?
    # accl = jax.lax.select((vel <= v_min) & (accl <= 0), 0., accl)
    # accl = jax.lax.select((vel >= v_max) & (accl >= 0), 0., accl)
    
    # accl = jax.lax.select(accl <= -a_max, -a_max, accl)
    # accl = jax.lax.select(accl >= pos_limit, pos_limit, accl)

    # Vectorized version using jnp where
    pos_limit = jnp.where(vel > v_switch, a_max*v_switch/vel, a_max)

    accl = jnp.where((vel <= v_min) & (accl <= 0), 0., accl)
    accl = jnp.where((vel >= v_max) & (accl >= 0), 0., accl)

    accl = jnp.where(accl <= -a_max, -a_max, accl)
    accl = jnp.where(accl >= pos_limit, pos_limit, accl)

    return accl

def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    """
    Steering constraints, adjusts the steering velocity based on constraints

        Args:
            steering_angle (float): current steering_angle of the vehicle
            steering_velocity (float): unconstraint desired steering_velocity
            s_min (float): minimum steering angle
            s_max (float): maximum steering angle
            sv_min (float): minimum steering velocity
            sv_max (float): maximum steering velocity

        Returns:
            steering_velocity (float): adjusted steering velocity
    """

    # constraint steering velocity
    # steering_velocity = jax.lax.select((steering_angle <= s_min) & (steering_velocity <= 0), 0., steering_velocity)
    # steering_velocity = jax.lax.select((steering_angle >= s_max) & (steering_velocity >= 0), 0., steering_velocity)
    # steering_velocity = jax.lax.select(steering_velocity <= sv_min, sv_min, steering_velocity)
    # steering_velocity = jax.lax.select(steering_velocity >= sv_max, sv_max, steering_velocity)

    # Vectorized version using jnp where
    steering_velocity = jnp.where(
        (steering_angle <= s_min) & (steering_velocity <= 0), 0., steering_velocity)
    steering_velocity = jnp.where(
        (steering_angle >= s_max) & (steering_velocity >= 0), 0., steering_velocity)
    steering_velocity = jnp.where(steering_velocity <= sv_min, sv_min, steering_velocity)
    steering_velocity = jnp.where(steering_velocity >= sv_max, sv_max, steering_velocity)
    
    return steering_velocity

# @jax.jit
def vehicle_dynamics_ks(x, u_init, C_Sf=20.898, C_Sr=20.898, 
                        lf=0.88392, lr=1.50876, h=0.59436, m=1225.887, I=1538.853371):
    """
    Single Track Kinematic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
                x6: s coordinate in Frenet coordinates
                x7: d coordinate in Frenet coordinates
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    params = params_f1tenth

    # wheelbase
    lf = params['lf']  # distance from spring mass center of gravity to front axle [m]  LENA
    lr = params['lr']  # distance from spring mass center of gravity to rear axle [m]  LENB
    lwb = lf + lr
    # steering constraints
    s_min = params['s_min']  # minimum steering angle [rad]
    s_max = params['s_max']  # maximum steering angle [rad]
    # longitudinal constraints
    v_min = params['v_min']  # minimum velocity [m/s]
    v_max = params['v_max'] # minimum velocity [m/s]
    sv_min = params['sv_min'] # minimum steering velocity [rad/s]
    sv_max = params['sv_max'] # maximum steering velocity [rad/s]
    v_switch = params['v_switch']  # switching velocity [m/s]
    a_max = params['a_max'] # maximum absolute acceleration [m/s^2]
    
    # constraints
    u = jnp.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # system dynamics
    f = jnp.array([x[3]*jnp.cos(x[4]),
         x[3]*jnp.sin(x[4]), 
         u[0],
         u[1],
         x[3]/lwb*jnp.tan(x[2])])
    return f

def RK4_fn(x0, u, Ddt, vehicle_dynamics_fn, args):
    # return x0 + vehicle_dynamics_fn(x0, u, *args) * Ddt # Euler integration
    # RK4 integration
    k1 = vehicle_dynamics_fn(x0, u, *args)
    k2 = vehicle_dynamics_fn(x0 + k1 * 0.5 * Ddt, u, *args)
    k3 = vehicle_dynamics_fn(x0 + k2 * 0.5 * Ddt, u, *args)
    k4 = vehicle_dynamics_fn(x0 + k3 * Ddt, u, *args)
    return x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * Ddt

# @jax.jit
def update_fn(x, u, mppi_DTK):
    x1 = x.copy()
    Ddt = 0.05
    def step_fn(i, x0):
        args = ()
        return RK4_fn(x0, u, Ddt, vehicle_dynamics_ks, args)
    x1 = jax.lax.fori_loop(0, int(mppi_DTK/Ddt), step_fn, x1)
    # return (x1, 0, x1-x)
    return x1

class VehicleDynamics():
    def __init__(self, state_dim=5, delta_t=0.1, frenet_converter=None):
        self.state_dim = state_dim
        self.delta_t = delta_t
        self.frenet_converter = frenet_converter

    def propagate(self, state_cur, control):
        x1 = update_fn(state_cur[:5], control, self.delta_t)
        if x1.ndim == 1:
            x1 = jnp.expand_dims(x1, axis=0)
        s, d = self.frenet_converter.get_frenet(x1[0], x1[1])
        x1 = jnp.array([x1[0], x1[1], x1[2], x1[3], x1[4], s, d])
        return x1