"""grouping module"""
import numpy as np
from numpy_tools.index import *
import itertools


class GroupBy(object):
    """
    supports ufunc reduction
    should any other form of reduction be supported?
    not sure it should; more cleanly written externally i think, on a grouped iterables
    """
    def __init__(self, keys, axis = 0):
        #we could inherit from Index, but the multiple specializations make
        #holding a reference to an index object more appropriate
        # note that we dont have backwards compatibility issues with groupby,
        #so we are going to use axis = 0 as a default
        #the multi-dimensional structure of a keys object is usualy meaningfull,
        #and value arguments are not automatically flattened either

        self.index = as_index(keys, axis)

    #forward interesting/'public' index properties
    @property
    def unique(self):
        return self.index.unique
    @property
    def count(self):
        return self.index.count
    @property
    def inverse(self):
        return self.index.inverse
    @property
    def rank(self):
        return self.index.rank


    #some different methods of chopping up a set of values by key
    #not sure they are all equally relevant, but i actually have real world use cases for most of them

    def split_iterable_as_iterable(self, values):
        """
        grouping of an iterable. memory consumption depends on the amount of sorting required
        worst case, if index.sorter[-1] = 0, we need to consume the entire value iterable,
        before we can start yielding any output
        but to the extent that the keys come presorted, the grouping is lazy
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
            yield (get_value(i) for i in itertools.islice(s, c))

    def split_iterable_as_unordered_iterable(self, values):
        """
        group values, without regard for the ordering of self.index.unique
        consume values as they come, and yield key-group pairs as soon as they complete
        thi spproach is lazy, insofar as grouped values are close in their iterable
        """
        from collections import defaultdict
        cache = defaultdict(list)
        count = self.count
        unique = self.unique
        key = (lambda i: unique[i]) if isinstance(unique, np.ndarray) else (lambda i: tuple(c[i] for c in unique))
        for i,v in itertools.izip(self.inverse, values):
            cache[i].append(v)
            if len(cache[i]) == count[i]:
                yield key(i), cache.pop(i)

    def split_sequence_as_iterable(self, values):
        """
        this is the preferred method if values has random access,
        but we dont want it completely in memory.
        like a big memory mapped file, for instance
        """
        s = iter(self.index.sorter)
        for c in self.count:
            yield (values[i] for i in itertools.islice(s, c))

    def split_array_as_array(self, values):
        """
        return grouped values as an ndarray
        returns an array of shape [groups, groupsize, ungrouped-axes]
        this is only possible if index.uniform==True
        """
        assert(self.index.uniform)
        values = np.asarray(values)
        values = values[self.index.sorter]
        return values.reshape(self.index.groups, -1, *values.shape[1:])

    def split_array_as_list(self, values):
        """return grouped values as a list of arrays, or a jagged-array"""
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
        """
        not sure how i feel about this. explicit is better than implict?
        also, add py2 py3 split here
        """
        return self.unique, self.split(values)


    # ufunc based reduction methods. should they return unique keys by default?

    def reduce(self, values, operator = np.add):
        """
        reduce the values over identical key groups, using the ufunc operator
        reduction is over the first axis, which should have elements corresponding to the keys
        all other axes are treated indepenently for the sake of this reduction
        """
        values = values[self.index.sorter]
        return operator.reduceat(values, self.index.start)

##        if values.ndim>1:
##            return np.apply_along_axis(
##                lambda slc: operator.reduceat(slc, self.index.start),
##                0, values)
##        else:
##            return operator.reduceat(values, self.index.start)
    def at(self, values, operator = np.add):
        """
        reduction via at
        theoretically, this may be faster than reduceat
        however, tests are not kind to that notion
        infact it appears about an order of magnitude slower than reduceat

        only works for add and multiply so far.
        maximum and minimum have silly identity
        would need to use np.iinfo and np.finfo; kinda messy
        just included here for consideration
        """
        inverse = self.index.inverse
        groups = self.index.groups

        def reduce(slc):
            r = np.empty(groups, values.dtype)
            r.fill(operator.identity)
            operator.at(r, inverse, slc)
            return r

        if values.ndim>1:
            return np.apply_along_axis(reduce, 0, values)
        else:
            return reduce(values)


    def sum(self, values, axis=0):
        """compute the sum over each group"""
        values = np.asarray(values)
        if axis: values = np.rollaxis(values, axis)
        return self.unique, self.reduce(values)

    def mean(self, values, axis=0):
        """compute the mean over each group"""
        values = np.asarray(values)
        if axis: values = np.rollaxis(values, axis)
        count = self.count.reshape(-1,*(1,)*(values.ndim-1))
        return self.unique, self.reduce(values) / count

    def var(self, values, axis=0):
        """compute the variance over each group"""
        values = np.asarray(values)
        if axis: values = np.rollaxis(values, axis)
        count = self.count.reshape(-1,*(1,)*(values.ndim-1))
        mean = self.reduce(values) / count
        err = values - mean[self.inverse]
        return self.unique, self.reduce(err**2) / count

    def std(self, values, axis=0):
        """standard deviation over each group"""
        unique, var = self.var(values, axis)
        return unique, np.sqrt(var)

    def median(self, values, axis=0, average=True):
        """
        compute the median value over each group.
        when average is true, the average is the two cental values is taken
        for groups with an even key-count
        """
        values = np.asarray(values)

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
        if axis: values = np.rollaxis(values, axis)
        if values.ndim>1:   #is trying to skip apply_along_axis somewhat premature optimization?
            values = np.apply_along_axis(median1d, 0, values)
        else:
            values = median1d(values)
        return self.unique, values

    def min(self, values, axis=0):
        """return the minimum within each group"""
        values = np.asarray(values)
        if axis: values = np.rollaxis(values, axis)
        return self.unique, self.reduce(values, np.minimum)

    def max(self, values, axis=0):
        """return the maximum within each group"""
        values = np.asarray(values)
        if axis: values = np.rollaxis(values, axis)
        return self.unique, self.reduce(values, np.maximum)

    def first(self, values, axis=0):
        """return values at first occurance of its associated key"""
        values = np.asarray(values)
        if axis: values = np.rollaxis(values, axis)
        return self.unique, values[self.index.sorter[self.index.start]]

    def last(self, values, axis=0):
        """return values at last occurance of its associated key"""
        values = np.asarray(values)
        if axis: values = np.rollaxis(values, axis)
        return self.unique, values[self.index.sorter[self.index.stop-1]]

    #implement iter interface? could simply do zip( group_by(keys)(values)), no?


def group_by(keys, values=None, reduction=None, axis=0):
    """
    slightly higher level interface to grouping
    """
    g = GroupBy(keys, axis)
    if values is None:
        return g
    groups = g.split(values)
    if reduction is None:
        return g.unique, groups
    return [(key,reduction(group)) for key, group in itertools.izip(g.unique, groups)]


__all__ = ['group_by']