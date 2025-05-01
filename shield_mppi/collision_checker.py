import jax.numpy as jnp

class CollisionChecker():
    def __init__(self, track_width=None, obstacles=None, obstacles_radius=None):
        self.track_width = track_width
        self.obstacles = obstacles
        self.obstacles_radius = obstacles_radius

    def set_track_width(self, track_width):
        self.track_width = track_width

    def check(self, state_cur, cartesian_state_cur=None):
        # TODO: still working on it
        if cartesian_state_cur is None:
            return self.check_collision_with_boundaries(state_cur)
        else:
            boundary_collisions = self.check_collision_with_boundaries(state_cur)
            # obstacle_collisions = self.check_collision_with_obstacles(cartesian_state_cur)
            # if the state collided with either a boundary or an obstacle, it returns True
            # return boundary_collisions | obstacle_collisions
            return boundary_collisions

    def check_collision_with_boundaries(self, map_state):
        if map_state.ndim == 1:
            if map_state[-1] < -self.track_width or self.track_width < map_state[-1]:
                return True
            return False
        else:
            collisions = jnp.where(
                (map_state[-1, :] < -self.track_width)
                | (self.track_width < map_state[-1, :]),
                1,
                0,
            )
            return collisions

    # def check_collision_with_obstacles(self, cartesian_state_cur):
    #     if cartesian_state_cur.ndim == 1:
    #         if (np.linalg.norm(self.obstacles - cartesian_state_cur[6:8], axis=1) < self.obstacles_radius[np.newaxis, :]).any():
    #             # if the vehicle collide with any one of the obstacles
    #             return True
    #         return False
    #     else:
    #         cartesian_state_cur = cartesian_state_cur.T
    #         obstacles_reshaped = self.obstacles[np.newaxis, :, :]
    #         cartesian_state_reshaped = cartesian_state_cur[:, np.newaxis, 6:8]
    #         distance_to_obstacles = np.linalg.norm(cartesian_state_reshaped - obstacles_reshaped, axis=2) # of shape (state_num, obstacle_num)
    #         collisions = (distance_to_obstacles < self.obstacles_radius[np.newaxis, :]).any(axis=1) # if a state's distance to any obstacle is less than radius, it collides
    #         return collisions