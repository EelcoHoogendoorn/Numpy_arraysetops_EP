"""
this is a rewrite of numpy arraysetops module using the indexing class hierarchy
"""

from numpy_indexed.funcs import *
from numpy_indexed.index import *
from numpy_indexed import semantics


def unique(keys, axis=semantics.axis_default, return_index=False, return_inverse=False, return_count=False):
    """
    compute the set of unique keys

    Parameters
    ----------
    keys : indexable key object
        keys object to find unique keys within
    axis : int
        if keys is a multi-dimensional array, the axis to regard as the sequence of key objects
    return_index : bool
        if True, return indexes such that keys[index] == unique
    return_inverse : bool
        if True, return the indices such that unique[inverse] == keys
    return_count : bool
        if True, return the number of times each unique key occurs in the input

    Notes
    -----
    the kwargs are there to provide a backwards compatible interface to numpy.unique
    but arguably, it is cleaner to call index and its properties directly,
    should more than unique values be desired as output
    """
    stable = return_index or return_inverse
    index = as_index(keys, axis, base = not stable, stable = stable)

    ret = index.unique,
    if return_index:
        ret = ret + (index.index,)
    if return_inverse:
        ret = ret + (index.inverse,)
    if return_count:
        ret = ret + (index.count,)
    return ret[0] if len(ret) == 1 else ret


def count_selected(A, B, axis=semantics.axis_default):
    """
    count how often the elements of B occur in A
    returns an array of unsigned integers, one for each key in B

    Parameters
    ----------
    A : indexable key object
        items to do lookup in
    B : indexable key object
        items to look up in A
    axis : int
        if keys is a multi-dimensional array, the axis to regard as the sequence of key objects

    Notes
    -----
    should this be a private function, or part of the API?

    this is perhaps not the most efficient way of doing it, but it is rather elegant
    alternatively, compute intersection, while computing idx to map back to original space
    dont need to as_index first

    can also use searchsorted based approach; diff between left and right interval
    """

    Ai = as_index(A, axis=axis)
    Bi = as_index(B, axis=axis)

    query_multiplicity = multiplicity(Bi, axis=0)
    joint_multiplicity = multiplicity(_set_concatenate((Bi.keys, Ai.keys)), axis=0)[:Bi.size]
    return joint_multiplicity - query_multiplicity


def contains(A, B, axis=semantics.axis_default):
    """
    test if B is contained in A; 'does A contain B?'

    Parameters
    ----------
    A : indexable key sequence
        items to search in
    B : indexable key sequence
        items to search for

    Returns
    -------
    a bool array with length of B

    Notes
    -----
    like np.in1d

    isnt this better implemented using searchsorted?
    """
    return count_selected(A, B, axis=axis) > 0


def _set_preprocess(sets, **kwargs):
    """
    common code for all set operations that has been factored out
    this should return an index object; simply multi-upcast
    """
    axis            = kwargs.get('axis', semantics.axis_default)
    assume_unique   = kwargs.get('assume_unique', False)

    if assume_unique:
        sets = [as_index(s, axis=axis).unique for s in sets]
    else:
        sets = [as_index(s, axis=axis).unique for s in sets]
    return sets


def _set_concatenate(sets):
    """
    concat set objects. handles both arrays and tuples of arrays
    """
    def con(set):
        # if not all():
        #     raise ValueError('concatenated keys must have the same dtype')
        try:
            return np.concatenate([s for s in sets if len(s)])
        except ValueError:
            return set[0]

    if any(not isinstance(s, tuple) for s in sets):
        #assume all arrays
        return con(sets)
    else:
        #assume all tuples
        return tuple(con(s) for s in zip(*sets))


def _set_count(sets, sc, **kwargs):
    """
    return the elements which occur sc times over the sequence of sets
    used by both exclusive and intersection
    """
    sets = _set_preprocess(sets, **kwargs)
    i = as_index(_set_concatenate(sets), axis=0, base=True)
    # FIXME : this does not work for lex-keys
    return i.unique[i.count == sc]


def union(*sets, **kwargs):
    """all unique items which occur in any one of the sets"""
    sets = _set_preprocess(sets, **kwargs)
    return as_index( _set_concatenate(sets), axis=0, base=True).unique


def intersection(*sets, **kwargs):
    """perform intersection on an sequence of sets; items which are in all sets"""
    return _set_count(sets, len(sets), **kwargs)


def exclusive(*sets, **kwargs):
    """
    return items which are exclusive to one of the sets;
    generalization of xor
    what to do with repeated items in original sets?
    assume_unique kwarg allows toggling
    """
    return _set_count(sets, 1, **kwargs)


def difference(*sets, **kwargs):
    """
    substracts all tail sets from the head set

    alt implementation: compute union of tail, then union with head, then use set_count(1)
    """
    head, tail = sets[0], sets[1:]
    idx = as_index(head, **kwargs)
    lhs = idx.unique
    rhs = [intersection(idx, s, **kwargs) for s in tail]
    return exclusive(lhs, *rhs, axis=0, assume_unique = True)


__all__ = ['unique', 'count_selected', 'contains', 'union', 'intersection', 'exclusive', 'difference']