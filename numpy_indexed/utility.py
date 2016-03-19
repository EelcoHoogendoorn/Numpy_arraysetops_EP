"""some utility functions; reinterpret-casts on ndarrays"""

import numpy as np


def as_struct_array(*columns):
    """pack a sequence of columns into a recarray

    Parameters
    ----------
    columns : sequence of key objects

    Returns
    -------
    data : recarray
        recarray containing the input columns as struct fields
    """
    columns = [np.asarray(c) for c in columns]
    rows = len(columns[0])
    names = ['f'+str(i) for i in range(len(columns))]
    dtype = [(names[i], c.dtype, c.shape[1:]) for i,c in enumerate(columns)]
    data = np.empty(rows, dtype)
    for i, c in enumerate(columns):
        data[names[i]] = c
    return data


def axis_as_object(arr, axis=-1):
    """
    cast the given axis of an array to a void object
    if the axis to be cast is contiguous, a view is returned, otherwise a copy is made
    this is useful for efficiently sorting by the content of an axis, for instance

    Parameters
    ----------
    arr : ndarray
        array to
    axis : int
        axis to view as a void object

    Returns
    -------
    ndarray
        array with the given axis viewed as a void object
    """
    shape = arr.shape
    # make axis to be viewed as a void object as contiguous items
    arr = np.ascontiguousarray(np.swapaxes(arr, axis, -1))
    # number of bytes in each void object
    nbytes = arr.dtype.itemsize * shape[axis]
    # void type with the correct number of bytes
    voidtype = np.dtype((np.void, nbytes))
    # return the view as such, with the reduced shape
    return arr.view(voidtype).reshape(np.delete(shape, axis))


def object_as_axis(arr, dtype, axis=-1):
    """
    cast an array of void objects to a typed axis

    Parameters
    ----------
    arr : ndarray, [ndim], void
        array of type np.void
    dtype : numpy dtype object
        the output dtype to cast the input array to
    axis : int
        position to insert the newly formed axis into

    Returns
    -------
    ndarray, [ndim+1], dtype
        output array cast to given dtype
    """
    # view the void objects as typed elements
    arr = arr.view(dtype).reshape(arr.shape + (-1,))
    # put the axis in the specified location
    return np.swapaxes(arr, axis, -1)


def array_as_object(arr):
    """
    view everything but the first axis as a void object

    Parameters
    ----------
    arr : ndarray, [keys, ...], any
        array to be cast to a sequence of objects

    Returns
    -------
    ndarray, [keys], void
        1d array of void objects
    """
    arr = arr.reshape(len(arr), -1)
    return axis_as_object(arr)


def array_as_typed(arr, dtype, shape):
    """
    unwrap a void object to its original type and shape

    Parameters
    ----------
    arr : ndarray, [], void
        array of type void to be reinterpreted
    dtype : np.dtype object
        output dtype
    shape : tuple of int
        output shape

    Returns
    -------
    ndarray, [shape], dtype
        input array reinterpreted as the given shape and dtype
    """
    return object_as_axis(arr, dtype).reshape(shape)
