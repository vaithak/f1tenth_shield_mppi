import jax
import jax.numpy as jnp
# import numpy as np
from functools import partial


def boundary_barrier_function(state_cur, track_width):
    return state_cur[-1, :]**2 - track_width**2


def obstacle_barrier_function(state_cur, obstacle_x, obstacle_y, obstacle_radius):
    # Compute the distance from the vehicle to the obstacle
    distance = (state_cur[0, :] - obstacle_x)**2 + (state_cur[1, :] - obstacle_y)**2
    # Compute the negative barrier function value
    return obstacle_radius**2 - distance


def combined_barrier_function(state_cur, track_width, obstacles, obstacle_radius):
    # Compute the boundary barrier function value
    boundary_value = boundary_barrier_function(state_cur, track_width)
    print("boundary_value shape", boundary_value.shape)
    if obstacles is None or obstacles.shape[0] == 0:
        return boundary_value
    # Compute the obstacle barrier function values
    obstacle_values = jnp.zeros((state_cur.shape[1], obstacles.shape[0]))
    for i in range(obstacles.shape[0]):
        obstacle_values = obstacle_values.at[:, i].set(
            obstacle_barrier_function(state_cur, obstacles[i, 0], obstacles[i, 1], obstacle_radius[i])
        )
    obstacle_values_comb = jnp.max(obstacle_values, axis=1)
    print("obstacle_values_comb shape", obstacle_values_comb.shape)

    # Combine the two barrier function values
    combined_value = jnp.maximum(boundary_value, obstacle_values_comb).reshape(-1, 1)
    print("combined_barrier_function shape", combined_value.shape)
    return combined_value


class MPPICBFCostEvaluator():
    def __init__(
        self,
        cbf_alpha=0.9,
        collision_checker=None,
        Q=None,
        QN=None,
        R=None,
        collision_cost=None,
        goal_cost=None,
    ):
        self.collision_checker = collision_checker
        self.Q = Q
        self.QN = QN
        self.R = R
        self.collision_cost = collision_cost
        self.goal_cost = goal_cost
        self.cbf_alpha = cbf_alpha


    @partial(jax.jit, static_argnums=(0, 5))
    def evaluate(
        self,
        state_cur,
        curr_goal_state,
        actions=None,
        noises=None,
        dynamics=None,
        obstacles_list=None,
        obstacles_radius=None,
        state_next=None,
    ):
        print("re-tracing evaluate!")
        # This is all the same as AutorallyMPPICostEvaluator.evaluate
        # except that we don't apply the collision cost
        # (we use the CBF cost instead of a collision cost, but that's applied in a
        # different function)
        error_state_right = jnp.expand_dims(
            (state_cur - curr_goal_state.reshape((-1, 1))).T, axis=2
        )
        error_state_left = jnp.expand_dims(
            (state_cur - curr_goal_state.reshape((-1, 1))).T, axis=1
        )
        # 1/2 xQx
        cost = (
            (1 / 2)
            * error_state_left
            @ jnp.tile(jnp.expand_dims(self.Q, axis=0), (state_cur.shape[1], 1, 1))
            @ error_state_right
        )
        if actions is not None:
            actions_left = jnp.expand_dims(actions.T, axis=1)
            actions_right = jnp.expand_dims(actions.T, axis=2)
            if noises is not None:
                noises_left = jnp.expand_dims(noises.T, axis=1)
                noises_right = jnp.expand_dims(noises.T, axis=2)
                # 1/2 eRe
                cost += (
                    1
                    / 2
                    * noises_left
                    @ jnp.tile(
                        jnp.expand_dims(self.R, axis=0), (state_cur.shape[1], 1, 1)
                    )
                    @ noises_right
                )
                # vRe
                cost += (
                    (actions_left - noises_left)
                    @ jnp.tile(
                        jnp.expand_dims(self.R, axis=0), (state_cur.shape[1], 1, 1)
                    )
                    @ noises_right
                )

            # 1/2 uRu
            cost += (
                (1 / 2)
                * actions_left
                @ jnp.tile(jnp.expand_dims(self.R, axis=0), (state_cur.shape[1], 1, 1))
                @ actions_right
            )

        # Get the barrier function value at this state and the next (if provided)
        h_t = combined_barrier_function(state_cur, self.collision_checker.track_width, obstacles_list, obstacles_radius)
        print("h_t shape", h_t.shape)
        if state_next is not None:
            # h_t_plus_1 = self.barrier_nn(state_next.T)
            h_t_plus_1 = combined_barrier_function(
                state_next, self.collision_checker.track_width, obstacles_list, obstacles_radius
            )
        else:
            h_t_plus_1 = h_t

        # We want this to decrease along trajectories
        discrete_time_cbf_condition = h_t_plus_1 - self.cbf_alpha * h_t
        discrete_time_cbf_violation = jnp.maximum(
            discrete_time_cbf_condition, jnp.zeros_like(discrete_time_cbf_condition) - 0.1
        ).reshape(-1, 1, 1)

        if self.collision_cost is not None:
            cost += discrete_time_cbf_violation * self.collision_cost
        else:
            cost += discrete_time_cbf_violation * 1000  # default collision cost

        # # Also consider collisions in addition to the CBF
        collisions = self.collision_checker.check(
            state_cur
        )
        collisions = collisions.reshape((-1, 1, 1))
        if self.collision_cost is not None:
            cost += collisions * self.collision_cost
        else:
            cost += collisions * 1000  # default collision cost
        return cost


    @partial(jax.jit, static_argnums=(0, 2))
    def evaluate_cbf_cost(
        self,
        state_cur,
        dynamics,
        state_next,
        obstacles_list=None,
        obstacles_radius=None,
    ):
        print("re-tracing cbf cost...", end="")

        # Get the barrier function value at this state and the next (if provided)
        h_t = combined_barrier_function(state_cur, self.collision_checker.track_width, obstacles_list, obstacles_radius)

        h_t_plus_1 = combined_barrier_function(
            state_next, self.collision_checker.track_width, obstacles_list, obstacles_radius
        )

        # We want this to decrease along trajectories
        discrete_time_cbf_condition = h_t_plus_1 - self.cbf_alpha * h_t
        discrete_time_cbf_violation = jnp.maximum(
            discrete_time_cbf_condition, jnp.zeros_like(discrete_time_cbf_condition) - 0.1
        )

        if self.collision_cost is not None:
            cost = discrete_time_cbf_violation * self.collision_cost
        else:
            cost = discrete_time_cbf_violation * 1000  # default collision cost

        print("Done")

        return cost


    def evaluate_terminal_cost(
        self,
        state_cur,
        terminal_goal_state,
        actions=None,
        obstacles_list=None,
        obstacles_radius=None,
        dynamics=None
    ):
        if state_cur.ndim == 1:
            state_cur = state_cur.reshape((-1, 1))
        # evaluate cost of the final step of the horizon
        error_state_right = jnp.expand_dims(
            (state_cur - terminal_goal_state.reshape((-1, 1))).T, axis=2
        )
        error_state_left = jnp.expand_dims(
            (state_cur - terminal_goal_state.reshape((-1, 1))).T, axis=1
        )
        cost = (
            (1 / 2)
            * error_state_left
            @ jnp.tile(jnp.expand_dims(self.QN, axis=0), (state_cur.shape[1], 1, 1))
            @ error_state_right
        )
        if actions is not None:
            cost += (1 / 2) * actions.T @ self.R @ actions
        
        collisions = self.collision_checker.check(
            state_cur,
            obstacles=obstacles_list,
            obstacles_radius=obstacles_radius,
        )
        collisions = collisions.reshape((-1, 1, 1))
        if self.collision_cost is not None:
            cost += collisions * self.collision_cost
        else:
            cost += collisions * 1000
        # if self.goal_checker.check(
        #     state_cur
        # ):  # True for goal reached, False for goal not reached
        #     if self.goal_cost is not None:
        #         cost += self.goal_cost
        #     else:
        #         cost += -5000  # default goal cost
        return cost
