import numpy as np
from numba import jit
from numba import prange


from step_snapshot import Step_Snapshot

"""
Module Name: interpolation.py

This file contains the interpolation functions which allow for interpolation between mesh slices
"""


@jit(nopython = True,  cache = True)
def interpolate1D(xval, data, min_x, delta_x):
    """
    Quick interpolator for 1D values
    Parameters:
        xvals: np array of positions to interpolate values at
        data: the value of the function at each slice
        min_x: the smallest slice position
        delta_x: the change in position between slices
    """
    result = np.zeros(xval.shape)
    x_size = data.shape[0]
    xval = (xval - min_x) / delta_x
    v_c = data
    for i in prange(len(xval)):
        x = xval[i]
        x0 = int(x)
        if x0 == x_size - 1:
            x1 = x0
        else:
            x1 = x0 + 1

        xd = x - x0

        if x0 >= 0 and x1 < x_size:
            result[i] = v_c[x0] * (1 - xd) + v_c[x1] * xd
        else:
            result[i] = 0

    return result

#@jit(nopython = True,  cache = True)
def interpolate3D(ret_tvals, svals, xvals, step_snapshots, step_size):
    """
    Parameters:
        ret_tvals, svals, xvals: arrays containing the retarded time, s, and x coordinates at which to interpolate
        step_snapshots: all step_snapshot objects
        step_size: the s spacing between all steps
    """
    # Loop through each point and interpolate!
    for point_index in range(len(svals)):
        t = ret_tvals[point_index]
        s = svals[point_index]
        x = xvals[point_index]

        # Get the 2 snapshots that border the point in time
        unrounded_step_index = t/step_size
        step_size_float = float(step_size)
        
        # Sometimes, a point is exactly on a snapshot, this is not an issue except if the 
        # snapshot that the point lies on is the final snapshot, in this case we just shift the
        # indices back by one.
        snap_size = len(step_snapshots)-1
        if unrounded_step_index == snap_size:
            left_step_index = int(unrounded_step_index)
            snapshot_left = step_snapshots[left_step_index-1]
            snapshot_right = step_snapshots[left_step_index]

        else:
            left_step_index = int(unrounded_step_index)
            snapshot_left = step_snapshots[left_step_index]
            snapshot_right = step_snapshots[left_step_index+1]

        # Get the indices of the snapshot grids that the point falls within
        left_indices = snapshot_left.position2index(np.array([s,x]), snapshot_left.h_matrices)
        right_indices = snapshot_right.position2index(np.array([s,x]), snapshot_right.h_matrices)


        

    return 0


    # First find the 2 step_snapshots which our

