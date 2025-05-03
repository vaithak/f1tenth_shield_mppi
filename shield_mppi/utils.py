import math
import numpy as np
from numba import njit

@njit(cache=True)
def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world
    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    projections = trajectory[:-1,:] + (t*diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


def column_numbers_for_waypoints():
    """
    Returns the column numbers for the waypoints.
    """
    return {
        "x_ref_m": 0,
        "y_ref_m": 1,
        "width_right_m": 6,
        "width_left_m": 5,
        # "x_normvec_m": 4,
        # "y_normvec_m": 5,
        # "alpha_m": 6,
        "s_racetraj_m": 4,
        "psi_racetraj_rad": 3,
        "vx_racetraj_mps": 2,
        # "ax_racetraj_mps2": 11
    }

def find_closest_index(sorted_arr, val):
    """
    Find the index of the closest value in a sorted array.
    Args:
        sorted_arr: Sorted array.
        val: Value to find.
    Returns:
        Index of the closest value.
    """
    idx = np.searchsorted(sorted_arr, val)
    if idx == 0:
        return 0
    elif idx == len(sorted_arr):
        return len(sorted_arr) - 1
    else:
        left = sorted_arr[idx - 1]
        right = sorted_arr[idx]
        return idx - 1 if abs(val - left) < abs(val - right) else idx