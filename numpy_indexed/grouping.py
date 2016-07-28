"""grouping module"""
from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import *

import itertools

import numpy as np
from numpy_indexed.index import as_index


__author__ = "Eelco Hoogendoorn"
__license__ = "LGPL"
__email__ = "hoogendoorn.eelco@gmail.com"


class GroupBy(object):
    """
    GroupBy class

    contains an index of keys, and extends the index functionality with grouping-specific functionality
    """

    def __init__(self, keys, axis=0):
        """
        Parameters
        ----------
        keys : indexable object
            sequence of keys to group by
        axis : int, optional
            axis to regard as the key-sequence, in case keys is multi-dimensional

        See Also
        --------
        numpy_indexed.as_index : for information regarding the casting rules to a valid Index object
        """
        self.index = as_index(keys, axis)

    #forward interesting/'public' index properties
    @property
    def unique(self):
        """unique keys"""
        return self.index.unique
    @property
    def count(self):
        """count of each unique key"""
        return self.index.count
    @property
    def inverse(self):
        """mapping such that unique[inverse]==keys"""
        return self.index.inverse
    @property
    def groups(self):
        """int, number of groups formed by the keys"""
        return self.index.groups

    #some different methods of chopping up a set of values by key
    def split_iterable_as_iterable(self, values):
        """Group iterable into iterables, in the order of the keys

        Parameters
        ----------
        values : iterable of length equal to keys
            iterable of values to be grouped

        Yields
        ------
        iterable of items in values

        Notes
        -----
        Memory consumption depends on the amount of sorting required
        Worst case, if index.sorter[-1] = 0, we need to consume the entire value iterable,
        before we can start yielding any output
        But to the extent that the keys are already sorted, the grouping is lazy
        """
        values = iter(enumerate(values))
        cache = dict()
        def get_value(ti):
            try:
                return cache.pop(ti)
            except:
                while True:
                    i, v = next(values)
                    if i==ti:
                        return v
                    cache[i] = v
        s = iter(self.index.sorter)
        for c in self.count:
            yield (get_value(i) for i in itertools.islice(s, int(c)))

    def split_iterable_as_unordered_iterable(self, values):
        """Group iterable into iterables, without regard for the ordering of self.index.unique
        key-group tuples are yielded as soon as they are complete

        Parameters
        ----------
        values : iterable of length equal to keys
            iterable of values to be grouped

        Yields
        ------
        tuple of key, and a list of corresponding items in values

        Notes
        -----
        This approach is lazy, insofar as grouped values are close in their iterable
        """
        from collections import defaultdict
        cache = defaultdict(list)
        count = self.count
        unique = self.unique
        key = (lambda i: unique[i]) if isinstance(unique, np.ndarray) else (lambda i: tuple(c[i] for c in unique))
        for i,v in zip(self.inverse, values):
            cache[i].append(v)
            if len(cache[i]) == count[i]:
                yield key(i), cache.pop(i)

    def split_sequence_as_iterable(self, values):
        """Group sequence into iterables

        Parameters
        ----------
        values : iterable of length equal to keys
            iterable of values to be grouped

        Yields
        ------
        iterable of items in values

        Notes
        -----
        This is the preferred method if values has random access, but we dont want it completely in memory.
        Like a big memory mapped file, for instance
        """
        print(self.count)
        s = iter(self.index.sorter)
        for c in self.count:
            yield (values[i] for i in itertools.islice(s, int(c)))

    def split_array_as_array(self, values):
        """Group ndarray into ndarray by means of reshaping

        Parameters
        ----------
        values : ndarray_like, [index.size, ...]

        Returns
        -------
        ndarray, [groups, group_size, ...]
            values grouped by key

        Raises
        ------
        AssertionError
            This operation is only possible if index.uniform==True
        """
        if not self.index.uniform:
            raise ValueError("Array can only be split as array if all groups have the same size")
        values = np.asarray(values)
        values = values[self.index.sorter]
        return values.reshape(self.groups, -1, *values.shape[1:])

    def split_array_as_list(self, values):
        """Group values as a list of arrays, or a jagged-array

        Parameters
        ----------
        values : ndarray, [keys, ...]

        Returns
        -------
        list of length self.groups of ndarray, [key_count, ...]
        """
        values = np.asarray(values)
        values = values[self.index.sorter]
        return np.split(values, self.index.slices[1:-1], axis=0)

    def split(self, values):
        """some sensible defaults"""
        try:
            return self.split_array_as_array(values)
        except:
            # FIXME: change to iter in python 3?
            return self.split_array_as_list(values)

    def __call__(self, values):
        """not sure how i feel about this. explicit is better than implict?"""
        return self.unique, self.split(values)


    # ufunc based reduction methods. should they return unique keys by default?
    def reduce(self, values, operator=np.add, axis=0, dtype=None):
        """Reduce the values over identical key groups, using the given ufunc
        reduction is over the first axis, which should have elements corresponding to the keys
        all other axes are treated indepenently for the sake of this reduction

        Parameters
        ----------
        values : ndarray, [keys, ...]
            values to perform reduction over
        operator : numpy.ufunc
            a numpy ufunc, such as np.add or np.sum
        axis : int, optional
            the axis to reduce over
        dtype : output dtype

        Returns
        -------
        ndarray, [groups, ...]
        values reduced by operator over the key-groups
        """
        values = np.take(values, self.index.sorter, axis=axis)
        return operator.reduceat(values, self.index.start, axis=axis, dtype=dtype)


    def sum(self, values, axis=0, dtype=None):
        """compute the sum over each group

        Parameters
        ----------
        values : array_like, [keys, ...]
            values to sum per group
        axis : int, optional
            alternative reduction axis for values
        dtype : output dtype

        Returns
        -------
        unique: ndarray, [groups]
            unique keys
        reduced : ndarray, [groups, ...]
            value array, reduced over groups
        """
        values = np.asarray(values)
        return self.unique, self.reduce(values, axis=axis, dtype=dtype)

    def prod(self, values, axis=0, dtype=None):
        """compute the product over each group

        Parameters
        ----------
        values : array_like, [keys, ...]
            values to multiply per group
        axis : int, optional
            alternative reduction axis for values
        dtype : output dtype

        Returns
        -------
        unique: ndarray, [groups]
            unique keys
        reduced : ndarray, [groups, ...]
            value array, reduced over groups
        """
        values = np.asarray(values)
        return self.unique, self.reduce(values, axis=axis, dtype=dtype, operator=np.multiply)

    def mean(self, values, axis=0, weights=None, dtype=None):
        """compute the mean over each group

        Parameters
        ----------
        values : array_like, [keys, ...]
            values to take average of per group
        axis : int, optional
            alternative reduction axis for values
        weights : ndarray, [keys, ...], optional
            weight to use for each value
        dtype : output dtype

        Returns
        -------
        unique: ndarray, [groups]
            unique keys
        reduced : ndarray, [groups, ...]
            value array, reduced over groups
        """
        values = np.asarray(values)
        if weights is None:
            result = self.reduce(values, axis=axis, dtype=dtype)
            shape = [1] * values.ndim
            shape[axis] = self.groups
            weights = self.count.reshape(shape)
        else:
            weights = np.asarray(weights)
            result = self.reduce(values * weights, axis=axis, dtype=dtype)
            weights = self.reduce(weights, axis=axis, dtype=dtype)
        return self.unique, result / weights

    def var(self, values, axis=0, weights=None, dtype=None):
        """compute the variance over each group

        Parameters
        ----------
        values : array_like, [keys, ...]
            values to take variance of per group
        axis : int, optional
            alternative reduction axis for values

        Returns
        -------
        unique: ndarray, [groups]
            unique keys
        reduced : ndarray, [groups, ...]
            value array, reduced over groups
        """
        values = np.asarray(values)
        unique, mean = self.mean(values, axis, weights, dtype)
        err = values - mean.take(self.inverse, axis)

        if weights is None:
            shape = [1] * values.ndim
            shape[axis] = self.groups
            group_weights = self.count.reshape(shape)
            var = self.reduce(err ** 2, axis=axis, dtype=dtype)
        else:
            weights = np.asarray(weights)
            group_weights = self.reduce(weights, axis=axis, dtype=dtype)
            var = self.reduce(weights * err ** 2, axis=axis, dtype=dtype)

        return unique, var / group_weights

    def std(self, values, axis=0, weights=None, dtype=None):
        """standard deviation over each group

        Parameters
        ----------
        values : array_like, [keys, ...]
            values to take standard deviation of per group
        axis : int, optional
            alternative reduction axis for values

        Returns
        -------
        unique: ndarray, [groups]
            unique keys
        reduced : ndarray, [groups, ...]
            value array, reduced over groups
        """
        unique, var = self.var(values, axis, weights, dtype)
        return unique, np.sqrt(var)

    # FIXME: remove rollaxis stuff in the functions below as well
    def median(self, values, axis=0, average=True):
        """compute the median value over each group.

        Parameters
        ----------
        values : array_like, [keys, ...]
            values to take average of per group
        axis : int, optional
            alternative reduction axis for values
        average : bool, optional
            when average is true, the average of the two cental values is taken for groups with an even key-count

        Returns
        -------
        unique: ndarray, [groups]
            unique keys
        reduced : ndarray, [groups, ...]
            value array, reduced over groups
        """
        mid_2 = self.index.start + self.index.stop
        hi = (mid_2    ) // 2
        lo = (mid_2 - 1) // 2

        #need this indirection for lex-index compatibility
        sorted_group_rank_per_key = self.index.sorted_group_rank_per_key

        def median1d(slc):
            #place values at correct keys; preconditions the upcoming lexsort
            slc    = slc[self.index.sorter]
            #refine value sorting within each keygroup
            sorter = np.lexsort((slc, sorted_group_rank_per_key))
            slc    = slc[sorter]
            return (slc[lo]+slc[hi]) / 2 if average else slc[hi]

        values = np.asarray(values)
        if values.ndim>1:   #is trying to skip apply_along_axis somewhat premature optimization?
            values = np.apply_along_axis(median1d, axis, values)
        else:
            values = median1d(values)
        return self.unique, values

    def min(self, values, axis=0):
        """return the minimum within each group

        Parameters
        ----------
        values : array_like, [keys, ...]
            values to take minimum of per group
        axis : int, optional
            alternative reduction axis for values

        Returns
        -------
        unique: ndarray, [groups]
            unique keys
        reduced : ndarray, [groups, ...]
            value array, reduced over groups
        """
        values = np.asarray(values)
        return self.unique, self.reduce(values, np.minimum, axis)

    def max(self, values, axis=0):
        """return the maximum within each group

        Parameters
        ----------
        values : array_like, [keys, ...]
            values to take maximum of per group
        axis : int, optional
            alternative reduction axis for values

        Returns
        -------
        unique: ndarray, [groups]
            unique keys
        reduced : ndarray, [groups, ...]
            value array, reduced over groups
        """
        values = np.asarray(values)
        return self.unique, self.reduce(values, np.maximum, axis)

    def first(self, values, axis=0):
        """return values at first occurance of its associated key

        Parameters
        ----------
        values : array_like, [keys, ...]
            values to pick the first value of per group
        axis : int, optional
            alternative reduction axis for values

        Returns
        -------
        unique: ndarray, [groups]
            unique keys
        reduced : ndarray, [groups, ...]
            value array, reduced over groups
        """
        values = np.asarray(values)
        return self.unique, np.take(values, self.index.sorter[self.index.start], axis)

    def last(self, values, axis=0):
        """return values at last occurance of its associated key

        Parameters
        ----------
        values : array_like, [keys, ...]
            values to pick the last value of per group
        axis : int, optional
            alternative reduction axis for values

        Returns
        -------
        unique: ndarray, [groups]
            unique keys
        reduced : ndarray, [groups, ...]
            value array, reduced over groups
        """
        values = np.asarray(values)
        return self.unique, np.take(values, self.index.sorter[self.index.stop-1], axis)

    def any(self, values, axis=0):
        """compute if any item evaluates to true in each group

        Parameters
        ----------
        values : array_like, [keys, ...]
            values to take boolean predicate over per group
        axis : int, optional
            alternative reduction axis for values

        Returns
        -------
        unique: ndarray, [groups]
            unique keys
        reduced : ndarray, [groups, ...], np.bool
            value array, reduced over groups
        """
        values = np.asarray(values)
        if not values.dtype == np.bool:
            values = values != 0
        return self.unique, self.reduce(values, axis=axis) > 0

    def all(self, values, axis=0):
        """compute if all items evaluates to true in each group

        Parameters
        ----------
        values : array_like, [keys, ...]
            values to take boolean predicate over per group
        axis : int, optional
            alternative reduction axis for values

        Returns
        -------
        unique: ndarray, [groups]
            unique keys
        reduced : ndarray, [groups, ...], np.bool
            value array, reduced over groups
        """
        values = np.asarray(values)
        return self.unique, self.reduce(values, axis=axis, operator=np.multiply) != 0

    def argmin(self, values):
        """return the index into values corresponding to the minimum value of the group

        Parameters
        ----------
        values : array_like, [keys]
            values to pick the argmin of per group

        Returns
        -------
        unique: ndarray, [groups]
            unique keys
        argmin : ndarray, [groups]
            index into value array, representing the argmin per group
        """
        keys, minima = self.min(values)
        minima = minima[self.inverse]
        # select the first occurence of the minimum in each group
        index = as_index((self.inverse, values == minima))
        return keys, index.sorter[index.start[-self.groups:]]

    def argmax(self, values):
        """return the index into values corresponding to the maximum value of the group

        Parameters
        ----------
        values : array_like, [keys]
            values to pick the argmax of per group

        Returns
        -------
        unique: ndarray, [groups]
            unique keys
        argmax : ndarray, [groups]
            index into value array, representing the argmax per group
        """
        keys, maxima = self.max(values)
        maxima = maxima[self.inverse]
        # select the first occurence of the maximum in each group
        index = as_index((self.inverse, values == maxima))
        return keys, index.sorter[index.start[-self.groups:]]

    #implement iter interface? could simply do zip( group_by(keys)(values)), no?


def group_by(keys, values=None, reduction=None, axis=0):
    """construct a grouping object on the given keys, optionally performing the given reduction on the given values

    Parameters
    ----------
    keys : indexable object
        keys to group by
    values : array_like, optional
        sequence of values, of the same length as keys
        if a reduction function is provided, the given values are reduced by key
        if no reduction is provided, the given values are grouped and split by key
    reduction : lambda, optional
        reduction function to apply to the values in each group
    axis : int, optional
        axis to regard as the key-sequence, in case keys is multi-dimensional

    Returns
    -------
    iterable
        if values is None, a GroupBy object of the given keys object
        if reduction is None, an tuple of a sequence of unique keys and a sequence of grouped values
        else, a sequence of tuples of unique keys and reductions of values over that key-group

    See Also
    --------
    numpy_indexed.as_index : for information regarding the casting rules to a valid Index object
    """
    g = GroupBy(keys, axis)
    if values is None:
        return g
    groups = g.split(values)
    if reduction is None:
        return g.unique, groups
    return [(key, reduction(group)) for key, group in zip(g.unique, groups)]


__all__ = ['group_by']
