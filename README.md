<img src="https://travis-ci.org/EelcoHoogendoorn/Numpy_arraysetops_EP.svg?branch=master">
Numpy indexed operations
====================

This package contains functionality for indexed operations on numpy ndarrays, providing efficient vectorized functionality such as grouping and set operations.

* Rich and efficient grouping functionality:
  * splitting of values by key-group
  * reductions of values by key-group
* Generalization of existing array set operation to nd-arrays, such as:
  * unique
  * union
  * difference
  * exclusive (xor)
  * contains (in1d)
* Some new functions:
  * indices: numpy equivalent of list.index
  * count: numpy equivalent of collections.Counter
  * multiplicity: number of occurrences of each key in a sequence
  * count_table: like R's table or pandas crosstab, or an ndim version of np.bincount

The generalization of the existing array set operations pertains primarily to the extension of this functionality to different types of key objects, such as keys formed by slices of nd-arrays. For instance, we may wish to find the intersection of several sets of graph edges.

Some brief examples to give an impression hereof:
```python
# three sets of graph edges (doublet of ints)
edges = np.random.randint(0, 9, (3, 100, 2))
# find graph edges exclusive to one of three sets
ex = exclusive(*edges)
print(ex)
# which edges are exclusive to the first set?
print(contains(edges[0], ex))
# where are the exclusive edges relative to the totality of them?
print(indices(union(*edges), ex))
# group and reduce values by identical keys
values = np.random.rand(100, 20)
# and so on...
print(group_by(edges[0]).median(values))
```

## Installation
conda install numpy-indexed -c eelcohoogendoorn

## Design decisions:
This package builds upon a generalization of the design pattern as can be found in numpy.unique. That is, by argsorting an ndarray, subsequent operations can be implemented efficiently.

The sorting and related low level operations are encapsulated into a hierarchy of Index classes, which allows for efficient lookup of many properties for a variety of different key-types. The public API of this package is a quite thin wrapper around these Index objects.

The principal information exposed by an Index object is the required permutations to map between the original and sorted order of the keys. This information can subsequently be used for many purposes, such as efficiently finding the set of unique keys, or efficiently performing group_by logic on an array of corresponding values.

The two complex key types currently supported, beyond standard sequences of sortable primitive types, are array keys and composite keys. For the exact casting rules describing valid sequences of key objects to index objects, see as_index().

## Todo and open questions:
* What about nesting of key objects? This should be possible too, but not fully supported yet
*	What about floating point nd keys? Currently, they are treated as object indices. However, bitwise and floating point equality are not the same thing
*	Add special index classes for things like object arrays of variable length strings?
*	While this package is aimed more at expanding functionality than optimizing performance, the most common code paths might benefit from some specialization, such as the concatenation of sorted sets
*	There may be further generalizations that could be made. merge/join functionality perhaps?
