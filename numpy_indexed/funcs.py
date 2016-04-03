"""some useful functions of arraysetops type not currently present in numpy"""
from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import *

import numpy as np

from numpy_indexed.grouping import GroupBy
from numpy_indexed.index import LexIndex, as_index
from numpy_indexed import semantics


def indices(A, B, axis=semantics.axis_default, assume_contained=False):
    """
    vectorized numpy equivalent of list.index
    find indices such that np.all( A[indices] == B)
    as of yet, does not work on lexindex
    could it ever? would need lex-compatible searchsorted
    perhaps we could cast the lexindex to a struct array,
    but this is better left to the user i feel

    if assume_contained==true, it is assumed that the values in B are indeed present in A
    if not, a key error is raised in case a value is missing

    is using searchedsorted(sorter) wise? or is creating a sorted copy and inverting index more cache friendly?
    probably sepends on the size of values. otoh, binary search has poor cache coherence anyway
    """
    A = np.asarray(A)
    B = np.asarray(B)

    Ai = as_index(A, axis)
    if isinstance(Ai, LexIndex):
        raise ValueError('Composite key objects not supported in indices function')
    # use this for getting Ai.keys and Bi.keys organized the same way;
    # sorting is superfluous though. make sorting a cached property?
    # should we be working with cached properties generally?
    # or we should use sorted values, if searchsorted can exploit this knowledge?
    Bi = as_index(B, axis, base=True)

    # use raw private keys here, rather than public unpacked keys
    insertion = np.searchsorted(Ai._keys, Bi._keys, sorter=Ai.sorter, side='left')

    indices = Ai.sorter[insertion]

    if not assume_contained:
        if not np.alltrue(A[indices] == B):
            raise KeyError('Not all keys in B are present in A')

    return indices


def count(keys, axis=semantics.axis_default):
    """count the number of times each key occurs in the input set

    Arguments
    ---------
    keys : indexable object

    Returns
    -------
    unique : ndarray, [groups, ...]
        unique keys
    count : ndarray, [groups], int
        the number of times each key occurs in the input set

    Notes
    -----
    Can be seen as numpy work-alike of collections.Counter
    Alternatively, as sparse equivalent of count_table
    """
    index = as_index(keys, axis, base=True)
    return index.unique, index.count


def count_table(*keys):
    """count the number of times each key occurs in the input set

    Arguments
    ---------
    keys : tuple of indexable objects, each having the same number of items

    Returns
    -------
    unique : tuple of ndarray, [groups, ...]
        unique keys for each input item
        they form the axes labels of the table
    table : ndarray, [keys[0].groups, ... keys[n].groups], int
        the number of times each key-combination occurs in the input set

    Notes
    -----
    Equivalent to R's pivot table or pandas 'crosstab'
    Alternatively, dense equivalent of the count function
    Should we add weights option?
    """
    indices  = [as_index(k, axis=0) for k in keys]
    uniques  = [i.unique  for i in indices]
    inverses = [i.inverse for i in indices]
    shape    = [i.groups  for i in indices]
    table = np.zeros(shape, np.int)
    np.add.at(table, inverses, 1)
    return tuple(uniques), table


def multiplicity(keys, axis=semantics.axis_default):
    """return the multiplicity of each key, or how often it occurs in the set

    Parameters
    ----------
    keys : indexable object

    Returns
    -------
    ndarray, [keys.size], int
        the number of times each input item occurs in the set
    """
    index = as_index(keys, axis)
    return index.count[index.inverse]


def rank(keys, axis=semantics.axis_default):
    """where each item is in the pecking order.

    Parameters
    ----------
    keys : indexable object

    Returns
    -------
    ndarray, [keys.size], int
        unique integers, ranking the sorting order

    Notes
    -----
    we should have that index.sorted[index.rank] == keys
    """
    index = as_index(keys, axis)
    return index.rank


def incidence(boundary):
    """
    given an Nxm matrix containing boundary info between simplices,
    compute indidence info matrix
    not very reusable; should probably not be in this lib
    """
    return GroupBy(boundary).split(np.arange(boundary.size) // boundary.shape[1])


def mode(points, return_indices=False):
    """compute the mode, or most frequent occuring label in a set

    Parameters
    ----------
    points : ndarray, [n_points, ...]
        input point array. elements of 'points' can have arbitrary shape or dtype
    return_indices : bool
        if True, all indices such that points[indices]==mode holds

    Returns
    -------
    mode : ndarray, [...]
        the most frequently occuring item in the point sequence
    indices : ndarray, [mode_multiplicity], int, optional
        if return_indices is True, all indices such that points[indices]==mode holds
    """
    index = as_index(points)
    bin = np.argmax(index.count)
    maxpoint = index.unique[bin]
    if return_indices:
        indices = index.sorter[index.start[bin]: index.stop[bin]]
        return maxpoint, indices
    else:
        return maxpoint


def sorted(keys, axis):
    """to be implemented"""


def searchsorted():
    """to be implemented"""
