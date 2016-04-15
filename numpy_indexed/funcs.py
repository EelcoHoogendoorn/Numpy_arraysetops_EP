"""This module implements useful functionality on top of the Index class,
which is not currently present in numpy"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import *

import numpy as np

from numpy_indexed.grouping import GroupBy, group_by
from numpy_indexed.index import LexIndex, as_index
from numpy_indexed import semantics


__author__ = "Eelco Hoogendoorn"
__license__ = "LGPL"
__email__ = "hoogendoorn.eelco@gmail.com"


def indices(A, B, axis=semantics.axis_default, missing='raise'):
    """vectorized numpy equivalent of list.index
    find indices such that A[indices] == B

    Paramaters
    ----------
    A : indexable object
        items to search in
    B : indexable object
        items to search for
    missing : {'raise', 'ignore', 'mask'}
        if missing is 'raise', a KeyError is raised if not all elements of B are present in A
        if missing is 'ignore', all elements of B are assumed to be present in A, and output is undefined otherwise
        if missing is 'mask', a masked array is returned, containing only the indices found

    Returns
    -------
    indices : ndarray, [B.size], int
        indices such that A[indices] == B

    Notes
    -----
    as of yet, does not work on lexindex
    could it ever? would need lex-compatible searchsorted
    perhaps we could cast the lexindex to a struct array,
    but this is better left to the user i feel

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
    indices = np.take(Ai.sorter, insertion, mode='clip')

    if missing != 'ignore':
        invalid = Ai._keys[indices] != Bi._keys
        if missing == 'raise' and np.any(invalid):
            raise KeyError('Not all keys in B are present in A')
        if missing == 'mask':
            indices = np.ma.masked_array(indices, invalid)

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


def mode(keys, weights=None, return_indices=False):
    """compute the mode, or most frequent occuring key in a set

    Parameters
    ----------
    keys : ndarray, [n_keys, ...]
        input array. elements of 'keys' can have arbitrary shape or dtype
    weights : ndarray, [n_keys], optional
        if given, the contribution of each key to the mode is weighted by the given weights
    return_indices : bool
        if True, return all indices such that keys[indices]==mode holds

    Returns
    -------
    mode : ndarray, [...]
        the most frequently occuring key in the key sequence
    indices : ndarray, [mode_multiplicity], int, optional
        if return_indices is True, all indices such that points[indices]==mode holds
    """
    index = as_index(keys)
    if weights is None:
        unique, weights = count(index)
    else:
        unique, weights = group_by(index).sum(weights)
    bin = np.argmax(weights)
    _mode = unique[bin]
    if return_indices:
        indices = index.sorter[index.start[bin]: index.stop[bin]]
        return _mode, indices
    else:
        return _mode


def sorted(keys, axis):
    """sort an indexable object and return the sorted keys"""
    return as_index(keys, axis).sorted


def argsort(keys, axis):
    """return the indices that will place the keys in sorted order"""
    return as_index(keys, axis).sorter


def searchsorted(keys, axis, side='left', sorter=None):
    """to be implemented"""
    raise NotImplementedError


def incidence(boundary):
    """
    given an Nxm matrix containing boundary info between simplices,
    compute indidence info matrix
    not very reusable; should probably not be in this lib
    """
    return GroupBy(boundary).split(np.arange(boundary.size) // boundary.shape[1])
