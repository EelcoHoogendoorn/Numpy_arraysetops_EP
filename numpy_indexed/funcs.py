"""
some novel suggested functions of arraysetops type,
or functions which intimiately relate to the indexing mechanism
"""

from numpy_indexed.grouping import GroupBy
from numpy_indexed.index import LexIndex, as_index
from numpy_indexed import semantics
import numpy as np


def indices(A, B, axis=semantics.axis_default, assume_contained=True):
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
    #use this for getting Ai.keys and Bi.keys organized the same way;
    #sorting is superfluous though. make sorting a cached property?
    #should we be working with cached properties generally?
    Bi = as_index(B, axis, base=True)

    # use raw private keys here, rather than public unpacked keys
    I = np.searchsorted(Ai._keys, Bi._keys, sorter=Ai.sorter, side='left')

    indices = Ai.sorter[I]

    if not assume_contained:
        if not np.alltrue(A[indices] == B):
            raise KeyError('Not all keys in B are present in A')

    return indices


def count(keys, axis = semantics.axis_default):
    """
    numpy work-alike of collections.Counter
    sparse equivalent of count_table

    note: do we want utility functions for things like finding the most common key? max_count?
    """
    index = as_index(keys, axis, base = True)
    return index.unique, index.count


def count_table(*keys):
    """
    R's pivot table or pandas 'crosstab'
    dense equivalent of the count function
    """
    indices  = [as_index(k, axis=0) for k in keys]
    uniques  = [i.unique  for i in indices]
    inverses = [i.inverse for i in indices]
    shape    = [i.groups  for i in indices]
    t = np.zeros(shape, np.int)
    np.add.at(t, inverses, 1)
    return tuple(uniques), t


def multiplicity(keys, axis=semantics.axis_default):
    """
    return the multiplicity of each key, or how often it occurs in the set
    given how often i use multiplicity, id like to have it in the numpy namespace
    it is also quite useful for rewriting some common arraysetops
    """
    index = as_index(keys, axis)
    return index.count[index.inverse]


def rank(keys, axis=semantics.axis_default):
    """
    where each item is in the pecking order.
    not sure this should be part of the public api, cant think of any use-case right away
    plus, we have a namespace conflict, though its kindof unpythonic to have both np.ndim and np.rank
    """
    index = as_index(keys, axis)
    return index.rank


def incidence(boundary):
    """
    given an Nxm matrix containing boundary info between simplices,
    compute indidence info matrix
    not to be part of numpy API, to be sure, just something im playing with
    """
    return GroupBy(boundary).group(np.arange(boundary.size) // boundary.shape[1])
