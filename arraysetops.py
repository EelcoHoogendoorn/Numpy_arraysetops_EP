

"""
rewriet of classic arraysetops using the indexing class hierarchy
these need some rewriting still
we need to upcast to index object at the entry point,
and downcast to key objects when returning
internally, we should be dealing only with index objects
most of these operations require concatenation; this needs to be
dealt with from with the index object
"""


from index import *
from funcs import *

def unique(keys, return_index = False, return_inverse = False, return_count = False, axis = axis_default):
    """
    backwards compatible interface with numpy.unique
    in the long term i think the kwargs should be deprecated though
    cleaner to call index and its properties directly,
    should you want something beyond simply unique values
    """
    index = as_index(keys, axis, base = not (return_index or return_inverse))

    ret = index.unique,
    if return_index:
        ret = ret + (index.index,)
    if return_inverse:
        ret = ret + (index.inverse,)
    if return_count:
        ret = ret + (index.count,)
    return ret[0] if len(ret) == 1 else ret


def count_selected(A, B, axis=axis_default):
    """
    count how often the elements of B occur in A
    returns an array of unsigned integers, one for each key in B

    this isnt the most efficient way of doing it, but it is rather elegant

    alternatively, compute intersection, while computing idx to map back to original space
    dont need to as_index first
    """

    Ai = as_index(A, axis=axis)
    Bi = as_index(B, axis=axis)

    query_multiplicity = multiplicity(Bi, axis=0)
    joint_multiplicity = multiplicity(np.concatenate((Bi.keys, Ai.keys)), axis=0)[:Bi.size]
    return joint_multiplicity - query_multiplicity

def contains(A, B, axis=axis_default):
    """
    test if B is contained in A; 'does A contain B?'
    returns a bool array with length of B
    like np.in1d
    """
    return count_selected(A, B, axis=axis) > 0

def _set_preprocess(sets, **kwargs):
    """
    common code for all set operations that has been factored out
    this should return an index object; simply multi-upcast
    """
    axis            = kwargs.get('axis', axis_default)
    assume_unique   = kwargs.get('assume_unique', False)

    if assume_unique:
        sets = [as_index(s, axis=axis).keys for s in sets]
    else:
        sets = [as_index(s, axis=axis).unique for s in sets]
    return sets

def _set_concatenate(sets):
    """
    concat indices?
    concat set objects. handles both arrays and tuples of arrays
    integrate this into index-object instead?
    nope, cant be part of class, since we need multiple operands
    how to preserve nested index structure?
    in any case, this function should taken and return indices
    """
    if any(not isinstance(s, tuple) for s in sets):
        #assume all arrays
        return np.concatenate(sets)
    else:
        #assume all tuples
        return tuple(np.concatenate(s) for s in zip(*sets))


def _set_count(sets, sc, **kwargs):
    """
    return the elements which occur sc times
    """
    sets = _set_preprocess(sets, **kwargs)
    i = as_index(_set_concatenate(sets), axis=0, base=True)
    return i.unique[i.count==sc]
##    u, c = count(_set_concatenate(sets), axis=0)
##    return u[c == sc]

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
    """
    head, tail = sets[0], sets[1:]
    idx = as_index(head, **kwargs)
    lhs = idx.unique
    rhs = [intersection(idx, s, **kwargs) for s in tail]
    return exclusive(lhs, *rhs, axis=0, assume_unique = True)
