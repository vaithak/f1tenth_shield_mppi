import time
import numpy as np
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


class MPPICBFStochasticTrajectoriesSampler():
    def __init__(
        self,
        number_of_trajectories=None,
    ):
        self.number_of_trajectories = number_of_trajectories

    @partial(jax.jit, static_argnums=(0, 6, 7, 8, 9))
    def sample(
        self,
        state_cur,
        v,
        ref_traj,
        obstacles,
        obstacles_radius,
        control_horizon,
        control_dim,
        dynamics,
        cost_evaluator,
        control_bounds=None,
        noises=None,
    ):
        #  state_cur is the current state, v is the nominal control sequence
        state_start = state_cur.copy()
        costs = jnp.zeros((self.number_of_trajectories, 1, 1))
        state_cur = jnp.tile(
            state_start.reshape((-1, 1)), (1, self.number_of_trajectories)
        )
        trajectories = jnp.zeros(
            (self.number_of_trajectories, state_cur.shape[0], control_horizon)
        )
        trajectories = trajectories.at[:, :, 0].set(jnp.swapaxes(state_cur, 0, 1))
        noises = noises.reshape(
            (control_dim, (control_horizon - 1), self.number_of_trajectories)
        )
        us = jnp.zeros((v.shape[0], v.shape[1], self.number_of_trajectories))
        us = us.at[:, :, :self.number_of_trajectories].set(jnp.expand_dims(v, axis=2))
        us += noises
        # Control Bounds # TODO make the bounds in the config file
        if control_bounds is not None:
            us = us.at[0, :, :].set(jnp.where(us[0, :, :] < control_bounds[1, 0], us[0, :, :], control_bounds[1, 0]))  # upper bound on steering
            us = us.at[0, :, :].set(jnp.where(us[0, :, :] > control_bounds[0, 0], us[0, :, :], control_bounds[0, 0]))  # lower bound on steering
            us = us.at[1, :, :].set(jnp.where(us[1, :, :] < control_bounds[1, 1], us[1, :, :], control_bounds[1, 1]))  # upper bound on throttle
            us = us.at[1, :, :].set(jnp.where(us[1, :, :] > control_bounds[0, 1], us[1, :, :], control_bounds[0, 1]))  # lower bound on throttle

        propagation_time = 0.0
        for j in range(control_horizon - 1):
            start = time.perf_counter()
            state_next = dynamics.propagate(state_cur, us[:, j, :])
            propagation_time += time.perf_counter() - start
            print(f"state_cur: {state_cur.shape}, us: {us[:, j, :].shape}, state_next: {state_next.shape}")
            costs += cost_evaluator.evaluate(
                state_cur, ref_traj[:, -1], us[:, j, :], noises[:, j, :], dynamics=dynamics, state_next=state_next,
                obstacles_list=obstacles, obstacles_radius=obstacles_radius
            )
            state_cur = state_next
            trajectories = trajectories.at[:, :, j + 1].set(jnp.swapaxes(state_cur, 0, 1))
        print(f"costs before terminal: {costs.shape}")
        costs += cost_evaluator.evaluate_terminal_cost(state_cur, ref_traj[:, -1], dynamics=dynamics, obstacles_list=obstacles, obstacles_radius=obstacles_radius)
        us = jnp.moveaxis(us, 2, 0)
        return trajectories, us, costs