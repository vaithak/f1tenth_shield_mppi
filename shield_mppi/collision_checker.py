import jax.numpy as jnp

class CollisionChecker():
    def __init__(self, track_width=None):
        self.track_width = track_width


    def set_track_width(self, track_width):
        self.track_width = track_width


    def check(self, state_cur, obstacles=None, obstacles_radius=None):
        boundary_collisions = self.check_collision_with_boundaries(state_cur)
        if obstacles is None or obstacles.shape[0] == 0:
            # if there are no obstacles, return only the boundary collisions
            return boundary_collisions

        obstacle_collisions = self.check_collision_with_obstacles(state_cur, obstacles, obstacles_radius)
        # if the state collided with either a boundary or an obstacle, it returns True
        return boundary_collisions | obstacle_collisions


    def check_collision_with_boundaries(self, state_cur):
        if state_cur.ndim == 1:
            if state_cur[-1] < -self.track_width or self.track_width < state_cur[-1]:
                return True
            return False
        else:
            collisions = jnp.where(
                (state_cur[-1, :] < -self.track_width)
                | (self.track_width < state_cur[-1, :]),
                1,
                0,
            )
            return collisions


    def check_collision_with_obstacles(self, state_cur, obstacles=None, obstacles_radius=None):
        if state_cur.ndim == 1:
            if (jnp.linalg.norm(obstacles - state_cur[0:2], axis=1) < obstacles_radius[jnp.newaxis, :]).any():
                # if the vehicle collide with any one of the obstacles
                return True
            return False
        else:
            state_cur = state_cur.T
            obstacles_reshaped = obstacles[jnp.newaxis, :, :]
            state_reshaped = state_cur[:, jnp.newaxis, 0:2]
            distance_to_obstacles = jnp.linalg.norm(state_reshaped - obstacles_reshaped, axis=2) # of shape (state_num, obstacle_num)
            collisions = (distance_to_obstacles < obstacles_radius[jnp.newaxis, :]).any(axis=1) # if a state's distance to any obstacle is less than radius, it collides
            print("collisions: ", collisions)
            return collisions