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
    Or better yet; what about general reductions over key-grids?
    """
    indices  = [as_index(k, axis=0) for k in keys]
    uniques  = [i.unique  for i in indices]
    inverses = [i.inverse for i in indices]
    shape    = [i.groups  for i in indices]
    table = np.zeros(shape, np.int)
    np.add.at(table, inverses, 1)
    return tuple(uniques), table


class Table(object):
    """group_by type stuff on dense grids; like a generalized bincount"""
    def __init__(self, *keys):
        self.keys = tuple(keys)
        self.indices  = [as_index(k, axis=0) for k in keys]
        self.uniques  = [i.unique  for i in self.indices]
        self.shape    = [i.groups  for i in self.indices]

    def get_inverses(self, keys):
        return tuple([as_index(k, axis=0).inverse for k in keys])

    def allocate(self, dtype, fill=0):
        arr = np.empty(self.shape, dtype=dtype)
        arr.fill(fill)
        return arr

    def count(self):
        table = self.allocate(np.int)
        np.add.at(table, self.get_inverses(self.indices), 1)
        return tuple(self.uniques), table

    def sum(self, values):
        table = self.allocate(values.dtype)
        keys, values = group_by(self.keys).sum(values)
        table[self.get_inverses(keys)] = values
        return tuple(self.uniques), table

    def mean(self, values):
        table = self.allocate(np.float, np.nan)
        keys, values = group_by(self.keys).mean(values)
        table[self.get_inverses(keys)] = values
        return tuple(self.uniques), table

    def first(self, values):
        table = self.allocate(np.float, np.nan)
        keys, values = group_by(self.keys).first(values)
        table[self.get_inverses(keys)] = values
        return tuple(self.uniques), table

    def last(self, values):
        table = self.allocate(np.float, np.nan)
        keys, values = group_by(self.keys).last(values)
        table[self.get_inverses(keys)] = values
        return tuple(self.uniques), table

    def min(self, values, default=None):
        if default is None:
            try:
                info = np.iinfo(values.dtype)
                default = info.max
            except:
                default = +np.inf
        table = self.allocate(values.dtype, default)
        keys, values = group_by(self.keys).min(values)
        table[self.get_inverses(keys)] = values
        return tuple(self.uniques), table

    def max(self, values, default=None):
        if default is None:
            try:
                info = np.iinfo(values.dtype)
                default = info.min
            except:
                default = -np.inf
        table = self.allocate(values.dtype, default)
        keys, values = group_by(self.keys).max(values)
        table[self.get_inverses(keys)] = values
        return tuple(self.uniques), table

    def unique(self, values):
        """Place each entry in a table, while asserting that each entry occurs once"""
        _, count = self.count()
        if not np.array_equiv(count, 1):
            raise ValueError("Not every entry in the table is assigned a unique value")
        return self.sum(values)


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


def mode(keys, axis=semantics.axis_default, weights=None, return_indices=False):
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
    index = as_index(keys, axis)
    if weights is None:
        unique, weights = count(index)
    else:
        unique, weights = group_by(index).sum(weights)
    bin = np.argmax(weights)
    _mode = unique[bin]     # FIXME: replace with index.take for lexindex compatibility?
    if return_indices:
        indices = index.sorter[index.start[bin]: index.stop[bin]]
        return _mode, indices
    else:
        return _mode


def sort(keys, axis=semantics.axis_default):
    """sort an indexable object and return the sorted keys"""
    return as_index(keys, axis).sorted_keys


def argsort(keys, axis=semantics.axis_default):
    """return the indices that will place the keys in sorted order"""
    return as_index(keys, axis).sorter


def searchsorted(keys, axis=semantics.axis_default, side='left', sorter=None):
    """to be implemented"""
    raise NotImplementedError


def incidence(boundary):
    """
    given an Nxm matrix containing boundary info between simplices,
    compute indidence info matrix
    not very reusable; should probably not be in this lib
    """
    return GroupBy(boundary).split(np.arange(boundary.size) // boundary.shape[1])


def all_unique(keys, axis=semantics.axis_default):
    """Returns true if all keys are unique"""
    index = as_index(keys, axis)
    return index.groups == index.size


def any_unique(keys, axis=semantics.axis_default):
    """returns true if any of the keys is unique"""
    index = as_index(keys, axis)
    return np.any(index.count == 1)


def any_equal(keys, axis=semantics.axis_default):
    """return true if any of the keys equals another; or if not all the keys are unique"""
    return not all_unique(keys, axis)


def all_equal(keys, axis=semantics.axis_default):
    """returns true of all keys are equal"""
    index = as_index(keys, axis)
    return index.groups == 1


def is_uniform(keys, axis=semantics.axis_default):
    """returns true if all keys have equal multiplicity"""
    index = as_index(keys, axis)
    return index.uniform
