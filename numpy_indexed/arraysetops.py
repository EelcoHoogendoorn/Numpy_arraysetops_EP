"""this is a rewrite and extension of numpy arraysetops module using the indexing class hierarchy.

the main purpose is to expand functionality to multidimensional arrays,
but it also is much more readable and DRY than numpy.arraysetops
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import *

from numpy_indexed.funcs import *
from numpy_indexed.index import *
from numpy_indexed import semantics


__author__ = "Eelco Hoogendoorn"
__license__ = "LGPL"
__email__ = "hoogendoorn.eelco@gmail.com"


def unique(keys, axis=semantics.axis_default, return_index=False, return_inverse=False, return_count=False):
    """compute the set of unique keys

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
    The kwargs are there to provide a backwards compatible interface to numpy.unique, but arguably,
    it is cleaner to call index and its properties directly, should more than unique values be desired as output
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


def contains(this, that, axis=semantics.axis_default):
    """Returns bool for each element of `that`, indicating if it is contained in `this`

    Parameters
    ----------
    this : indexable key sequence
        sequence of items to test against
    that : indexable key sequence
        sequence of items to test for

    Returns
    -------
    ndarray, [that.size], bool
        returns a bool for each element in `that`, indicating if it is contained in `this`

    Notes
    -----
    Reads as 'this contains that'
    Similar to 'that in this', but with different performance characteristics
    """
    this = as_index(this, axis=axis, lex_as_struct=True, base=True)
    that = as_index(that, axis=axis, lex_as_struct=True)

    left = np.searchsorted(that._keys, this._keys, sorter=that.sorter, side='left')
    right = np.searchsorted(that._keys, this._keys, sorter=that.sorter, side='right')

    flags = np.zeros(that.size + 1, dtype=np.int)
    np.add.at(flags, left, 1)
    np.add.at(flags, right, -1)

    return np.cumsum(flags)[:-1].astype(np.bool)[that.rank]


def in_(this, that, axis=semantics.axis_default):
    """Returns bool for each element of `this`, indicating if it is present in `that`

    Parameters
    ----------
    this : indexable key sequence
        sequence of items to test for
    that : indexable key sequence
        sequence of items to test against

    Returns
    -------
    ndarray, [that.size], bool
        returns a bool for each element in `this`, indicating if it is present in `that`

    Notes
    -----
    Reads as 'this in that'
    Similar to 'that contains this', but with different performance characteristics
    """
    this = as_index(this, axis=axis, lex_as_struct=True, base=True)
    that = as_index(that, axis=axis, lex_as_struct=True)

    left = np.searchsorted(that._keys, this._keys, sorter=that.sorter, side='left')
    right = np.searchsorted(that._keys, this._keys, sorter=that.sorter, side='right')

    return left != right


def indices(this, that, axis=semantics.axis_default, missing='raise'):
    """Find indices such that this[indices] == that
    If multiple indices satisfy this condition, the first index found is returned

    Parameters
    ----------
    this : indexable object
        items to search in
    that : indexable object
        items to search for
    missing : {'raise', 'ignore', 'mask'}
        if `missing` is 'raise', a KeyError is raised if not all elements of `that` are present in `this`
        if `missing` is 'mask', a masked array is returned,
        where items of `that` not present in `this` are masked out
        if `missing` is 'ignore', all elements of `that` are assumed to be present in `this`,
        and output is undefined otherwise

    Returns
    -------
    indices : ndarray, [that.size], int
        indices such that this[indices] == that

    Notes
    -----
    May be regarded as a vectorized numpy equivalent of list.index
    """
    this = as_index(this, axis=axis, lex_as_struct=True)
    # use this for getting this.keys and that.keys organized the same way;
    # sorting is superfluous though. make sorting a cached property?
    # should we be working with cached properties generally?
    # or we should use sorted values, if searchsorted can exploit this knowledge?
    that = as_index(that, axis=axis, base=True, lex_as_struct=True)

    # use raw private keys here, rather than public unpacked keys
    insertion = np.searchsorted(this._keys, that._keys, sorter=this.sorter, side='left')
    indices = np.take(this.sorter, insertion, mode='clip')

    if missing != 'ignore':
        invalid = this._keys[indices] != that._keys
        if missing == 'raise':
            if np.any(invalid):
                raise KeyError('Not all keys in `that` are present in `this`')
        elif missing == 'mask':
            indices = np.ma.masked_array(indices, invalid)
        else:
            raise ValueError("Invalid value for `missing` argument; must be 'raise', 'mask' or 'ignore'.")
    return indices


def _set_preprocess(sets, **kwargs):
    """upcasts a sequence of indexable objects to Index objets according to the given kwargs

    Parameters
    ----------
    sets : iterable of indexable objects
    axis : int, optional
        axis to view as item sequence
    assume_unique : bool, optional
        if we should assume the items sequence does not contain duplicates

    Returns
    -------
    list of Index objects

    Notes
    -----
    common preprocessing for all set operations
    """
    axis            = kwargs.get('axis', semantics.axis_default)
    assume_unique   = kwargs.get('assume_unique', False)

    if assume_unique:
        sets = [as_index(s, axis=axis).unique for s in sets]
    else:
        sets = [as_index(s, axis=axis).unique for s in sets]
    return sets


def _set_concatenate(sets):
    """concatenate indexable objects.

    Parameters
    ----------
    sets : iterable of indexable objects

    Returns
    -------
    indexable object

    handles both arrays and tuples of arrays
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


def _set_count(sets, n, **kwargs):
    """return the elements which occur n times over the sequence of sets

    Parameters
    ----------
    sets : iterable of indexable objects
    n : int
        number of sets the element should occur in

    Returns
    -------
    indexable
        indexable with all elements that occured in n of the sets

    Notes
    -----
    used by both exclusive and intersection
    """
    sets = _set_preprocess(sets, **kwargs)
    i = as_index(_set_concatenate(sets), axis=0, base=True)
    # FIXME : this does not work for lex-keys
    return i.unique[i.count == n]


def union(*sets, **kwargs):
    """all unique items which occur in any one of the sets

    Parameters
    ----------
    sets : tuple of indexable objects

    Returns
    -------
    union of all items in all sets
    """
    sets = _set_preprocess(sets, **kwargs)
    return as_index( _set_concatenate(sets), axis=0, base=True).unique


def intersection(*sets, **kwargs):
    """perform intersection on an sequence of sets; items which are in all sets

    Parameters
    ----------
    sets : tuple of indexable objects

    Returns
    -------
    intersection of all items in all sets
    """
    return _set_count(sets, len(sets), **kwargs)


def exclusive(*sets, **kwargs):
    """return items which are exclusive to one of the sets;

    Parameters
    ----------
    sets : tuple of indexable objects

    Returns
    -------
    items which are exclusive to any one of the sets

    Notes
    -----
    this is a generalization of xor
    what to do with repeated items in original sets?
    assume_unique kwarg allows toggling
    """
    return _set_count(sets, 1, **kwargs)


def difference(*sets, **kwargs):
    """subtracts all tail sets from the head set

    Parameters
    ----------
    sets : tuple of indexable objects
        first set is the head, from which we subtract
        other items form the tail, which are subtracted from head

    Returns
    -------
    items which are in the head but not in any of the tail sets

    Notes
    -----
    alt implementation: compute union of tail, then union with head, then use set_count(1)
    """
    head, tail = sets[0], sets[1:]
    idx = as_index(head, **kwargs)
    lhs = idx.unique
    rhs = [intersection(idx, s, **kwargs) for s in tail]
    return exclusive(lhs, *rhs, axis=0, assume_unique=True)


__all__ = ['unique', 'contains', 'in_', 'indices', 'union', 'intersection', 'exclusive', 'difference']
