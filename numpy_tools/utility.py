"""
some utility functions
"""

import numpy as np
from numpy_tools.semantics import *


def as_struct_array(*cols):
    """pack a bunch of columns as a struct"""
    cols = [np.asarray(c) for c in cols]
    rows = len(cols[0])
    data = np.empty(rows, [('f'+str(i), c.dtype, c.shape[1:]) for i,c in enumerate(cols)])
    for i,c in enumerate(cols):
        data['f'+str(i)] = c
    return data

def axis_as_object(arr, axis=-1):
    """
    cast the given axis of an array to a void object
    if the axis to be cast is contiguous, a view is returned, otherwise a copy
    this is useful for efficiently sorting by the content of an axis, for instance
    """
    shape = arr.shape
    arr = np.ascontiguousarray(np.swapaxes(arr, axis, -1))
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * shape[axis]))).reshape(np.delete(shape, axis))
def object_as_axis(arr, dtype, axis=-1):
    """cast an array of void objects to a typed axis"""
    return np.swapaxes(arr.view(dtype).reshape(arr.shape+(-1,)), axis, -1)


def array_as_object(arr):
    """view everything but the first axis as a void object"""
    arr = arr.reshape(len(arr),-1)
    return axis_as_object(arr)
def array_as_typed(arr, dtype, shape):
    """unwrap a void object to its original type and shape"""
    return object_as_axis(arr, dtype).reshape(shape)

##def array_as_struct(arr):
##    return np.ascontiguousarray(arr).view([('f0', arr.dtype, arr.shape[1:])])#.flatten()
##def struct_as_array(arr):
##    return arr.view(arr['f0'].dtype)
