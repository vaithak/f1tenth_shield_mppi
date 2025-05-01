import jax
import jax.numpy as jnp
# import numpy as np
from functools import partial


def boundary_barrier_function(state_cur, track_width):
    return state_cur[-1, :]**2 - track_width**2


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
        self.cbf_alpha = cbf_alpha
        self.collision_checker = collision_checker
        self.Q = Q
        self.QN = QN
        self.R = R
        self.collision_cost = collision_cost
        self.goal_cost = goal_cost
        self.cbf_alpha = cbf_alpha

    @partial(jax.jit, static_argnums=(0, 5, 6))
    def evaluate(
        self,
        state_cur,
        curr_goal_state,
        actions=None,
        noises=None,
        dyna_obstacle_list=None,
        dynamics=None,
        state_next=None,
    ):
        print("re-tracing evaluate!")
        # This is all the same as AutorallyMPPICostEvaluator.evaluate
        # except that we don't apply the collision cost
        # (we use the CBF cost instead of a collision cost, but that's applied in a
        # different function)
        # map_state = self.global_to_local_coordinate_transform(state_cur, dynamics)
        map_state = state_cur.copy()
        error_state_right = jnp.expand_dims(
            (map_state - curr_goal_state.reshape((-1, 1))).T, axis=2
        )
        error_state_left = jnp.expand_dims(
            (map_state - curr_goal_state.reshape((-1, 1))).T, axis=1
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
        # h_t = self.barrier_nn(map_state.T)
        h_t = boundary_barrier_function(map_state, self.collision_checker.track_width)

        if state_next is not None:
            # map_state_next = self.global_to_local_coordinate_transform(
                # state_next, dynamics
            # )
            map_state_next = state_next.copy()
            # h_t_plus_1 = self.barrier_nn(map_state_next.T)
            h_t_plus_1 = boundary_barrier_function(
                map_state_next, self.collision_checker.track_width
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
            map_state
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
    ):
        print("re-tracing cbf cost...", end="")
        # map_state = self.global_to_local_coordinate_transform(state_cur, dynamics)
        map_state = state_cur.copy()

        # Get the barrier function value at this state and the next (if provided)
        # h_t = self.barrier_nn(map_state.T)
        h_t = boundary_barrier_function(map_state, self.collision_checker.track_width)

        # map_state_next = self.global_to_local_coordinate_transform(
        #     state_next, dynamics
        # )
        map_state_next = state_next.copy()
        # h_t_plus_1 = self.barrier_nn(map_state_next.T)
        h_t_plus_1 = boundary_barrier_function(
            map_state_next, self.collision_checker.track_width
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
        self, state_cur, terminal_goal_state, actions=None, dyna_obstacle_list=None, dynamics=None
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
            state_cur
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
