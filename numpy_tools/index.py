"""
class hierarchy for indexing a set of keys
the class hierarchy allows for code reuse, while providing specializations for different types of key objects
"""

from builtins import *
from numpy_tools.utility import *
from numpy_tools import semantics
from functools import reduce


"""
A note on naming: 'Index' here refers to the fact that the goal of these classes is to
perform and store precomputations on a set of keys,
such as to accelerate subsequent operations involving these keys.
They are not 'logical' indexes as in pandas;
they are not permanently associated with any other data objects

Note that these classes are not primarily intended to be used directly from the numpy namespace,
but rather are intended for code reuse within a family of higher level operations,
only the latter need to be part of the numpy API.

That said, these classes can also be very useful
in those places where the standard operations do not quite cover your needs,
saving your from completely reinventing the wheel.

notes:
    do we need to work more with cached properties here?
    generally, what is necessary and significant to precompute is pretty obvious
    but sometimes, even the sorting might not be required

    do we need to give index a stable flag?
    for grouping, stable sort is generally desirable,
    wehreas for set operations, we are better off using the fastest sort
"""

class BaseIndex(object):
    """
    minimal indexing functionality
    only provides unique and counts, but with optimal performance
    no grouping, or lex-keys are supported,
    or anything that would require an indirect sort
    """

    def __init__(self, keys):
        """
        keys is a flat array of possibly composite type
        """
        self._keys   = np.asarray(keys).flatten()
        self.sorted = np.sort(self._keys)
        #the slicing points of the bins to reduce over
        self.flag   = self.sorted[:-1] != self.sorted[1:]
        self.slices = np.concatenate((
            [0],
            np.flatnonzero(self.flag)+1,
            [self.size]))

    @property
    def keys(self):
        return self._keys

    @property
    def size(self):
        """number of keys"""
        return self._keys.size

    @property
    def start(self):
        """start index of all bins"""
        return self.slices[:-1]

    @property
    def stop(self):
        """stop index of all bins"""
        return self.slices[1:]

    @property
    def unique(self):
        """all unique keys"""
        return self.sorted[self.start]

    @property
    def groups(self):
        """number of unique keys"""
        return len(self.start)

    @property
    def count(self):
        """number of times each key occurs"""
        return np.diff(self.slices)

    @property
    def uniform(self):
        """returns true if each key occurs an equal number of times"""
        return not np.any(np.diff(self.count))


class Index(BaseIndex):
    """
    index object over a set of keys
    adds support for more extensive functionality, notably grouping
    relies on indirect sorting
    maybe it should be called argindex?
    """

    def __init__(self, keys, stable):
        """
        keys is a flat array of possibly composite type

        if stable is true, stable sorting of the keys is used. stable sorting is required
        uf first and last properties are required
        """
        self.stable  = stable
        self._keys   = np.asarray(keys)
        #find indices which sort the keys; use mergesort for stability, so first and last give correct results
        self.sorter = np.argsort(self._keys, kind='mergesort' if self.stable else 'quicksort')
        #computed sorted keys
        self.sorted = self._keys[self.sorter]
        #the slicing points of the bins to reduce over
        self.flag   = self.sorted[:-1] != self.sorted[1:]
        self.slices = np.concatenate((
            [0],
            np.flatnonzero(self.flag)+1,
            [self.size]))

    @property
    def index(self):
        """ive never run into any use cases for this;
        perhaps it was intended to be used to do group_by(keys).first(values)?
        in any case, included for backwards compatibility with np.unique"""
        return self.sorter[self.start]

    @property
    def rank(self):
        """how high in sorted list each key is"""
        r = np.empty(self.size, np.int)
        r[self.sorter] = np.arange(self.size)
        return r

    @property
    def sorted_group_rank_per_key(self):
        """find a better name for this? enumeration of sorted keys. also used in median implementation"""
        return np.cumsum(np.concatenate(([False], self.flag)))

    @property
    def inverse(self):
        """return index array that maps unique values back to original space"""
        inv = np.empty(self.size, np.int)
        inv[self.sorter] = self.sorted_group_rank_per_key
        return inv


class ObjectIndex(Index):
    """
    given axis enumerates the keys
    all other axes form the keys
    groups will be formed on the basis of bitwise equality between void objects

    should we retire objectindex?
    this can be integrated with regular index ala lexsort, no?
    not sure what is more readable though
    """

    def __init__(self, keys, axis, stable):
        self.axis = axis
        self.dtype = keys.dtype

        keys = np.swapaxes(keys, axis, 0)
        self.shape = keys.shape
        keys = array_as_object(keys)

        super(ObjectIndex, self).__init__(keys, stable)

    @property
    def keys(self):
        keys = array_as_typed(self._keys, self.dtype, self.shape)
        return np.swapaxes(keys, self.axis, 0)

    @property
    def unique(self):
        """the first entry of each bin is a unique key"""
        sorted = array_as_typed(self.sorted, self.dtype, self.shape)
        return np.swapaxes(sorted[self.start], self.axis, 0)


class LexIndex(Index):
    """
    index object based on lexographic ordering of a tuple of key-arrays
    key arrays can be any type, including multi-dimensional, structed or voidobjects
    however, passing such fancy keys to lexindex may not be ideal from a performance perspective,
    as lexsort does not accept them as arguments directly, so we have to index and invert them first

    should you find yourself with such complex keys, it may be more efficient
    to place them into a structured array first

    note that multidimensional columns are indexed by their first column,
    and no per-column axis keyword is supplied,
    customization of column layout will have to be done at the call site
    """

    def __init__(self, keys, stable):
        self._keys   = tuple(np.asarray(key) for key in keys)

        keyviews    = tuple(array_as_object(key) if key.ndim>1 else key for key in self._keys)
        #find indices which sort the keys; complex keys which lexsort does not accept are bootstrapped from Index
        self.sorter = np.lexsort(tuple(Index(key, stable).inverse if key.dtype.kind is 'V' else key for key in keyviews))
        #computed sorted keys
        self.sorted = tuple(key[self.sorter] for key in keyviews)
        #the slicing points of the bins to reduce over
        self.flag   = reduce(
            np.logical_or,
            (s[:-1] != s[1:] for s in self.sorted))
        self.slices = np.concatenate((
            [0],
            np.flatnonzero(self.flag)+1,
            [self.size]))

    @property
    def unique(self):
        """returns a tuple of unique key columns"""
        return tuple(
            (array_as_typed(s, k.dtype, k.shape) if k.ndim>1 else s)[self.start]
                for s, k in zip(self.sorted, self._keys))

    @property
    def size(self):
        return self.sorter.size


class LexIndexSimple(Index):
    """
    simplified LexIndex, which only accepts 1-d arrays of simple dtypes
    the more expressive LexIndex only has some logic overhead,
    in case all columns are indeed simple. not sure this is worth the extra code
    """
    def __init__(self, keys):
        self._keys   = tuple(np.asarray(key) for key in keys)
        self.sorter = np.lexsort(self._keys)
        #computed sorted keys
        self.sorted = tuple(key[self.sorter] for key in self._keys)
        #the slicing points of the bins to reduce over
        self.flag   = reduce(
            np.logical_or,
            (s[:-1] != s[1:] for s in self.sorted))
        self.slices = np.concatenate((
            [0],
            np.flatnonzero(self.flag)+1,
            [self.size]))

    @property
    def unique(self):
        """the first entry of each bin is a unique key"""
        return tuple(s[self.start] for s in self.sorted)

    @property
    def size(self):
        return self.sorter.size


def as_index(keys, axis = semantics.axis_default, base=False, stable=True):
    """
    casting rules for a keys object to an index object

    the preferred semantics is that keys is a sequence of key objects,
    except when keys is an instance of tuple,
    in which case the zipped elements of the tuple are the key objects

    the axis keyword specifies the axis which enumerates the keys
    if axis is None, the keys array is flattened
    if axis is 0, the first axis enumerates the keys
    which of these two is the default depends on whether backwards_compatible == True

    if base==True, the most basic index possible is constructed.
    this avoids an indirect sort; if it isnt required, this has better performance
    """
    if isinstance(keys, Index):
        if type(keys) is BaseIndex and base==False:
            keys = keys.keys    #need to upcast to an indirectly sorted index type
        else:
            return keys         #already done here
    if isinstance(keys, tuple):
        return LexIndex(keys, stable)
    try:
        keys = np.asarray(keys)
    except:
        raise TypeError('Given object does not form a valid set of keys')
    if axis is None:
        keys = keys.flatten()
    if keys.ndim==1:
        if base:
            return BaseIndex(keys)
        else:
            return Index(keys, stable=stable)
    else:
        return ObjectIndex(keys, axis, stable=stable)


__all__ = ['as_index']
