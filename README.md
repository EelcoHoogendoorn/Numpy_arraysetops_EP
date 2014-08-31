Numpy array set operations Enhancement Proposal draft
====================

This is a draft for a numpy EP, aiming for a comprehensive overhaul of the arraysetops module. Once the design has crystallized, the intent is to bring the tests and documentation up to numpy standards, and merge it into the next numpy release. The planned functionality includes:

* Rich and efficient grouping functionality:
  * splitting of values by key-group
  * reductions of values by key-group
* Some new functions:
  * indices: numpy equivalent of list.index
  * count: numpy equivalent of collections.Counter
  * multiplicity: number of occurances of each key in a sequence
  * count_table: like R's table or pandas crosstab
* Some generalization of existing array set operation, such as:
  * unique 
  * union
  * difference
  * exclusive (xor)
  * contains (in1d)

The generalization of the existing array set operations pertains primarily to the extension of this functionality to different types of key objects. For instance, we may wish to find the intersection of several sets of graph edges. All the functions described above build upon this generalized notion of a key object.

Some brief examples to give an impression hereof:
```python
#three sets of graph edges (doublet of ints)
edges = np.random.randint(0,9,(3,100,2))
#find graph edges exclusive to one of three sets
ex = exclusive(*edges)
print ex
#which edges are exclusive to the first set?
print contains(edges[0], ex)
#where are the exclusive edges relative to the totality of them?
print indices( union(*edges), ex)
#group and reduce values by identical keys
values = np.random.rand(100,20)
print group_by(edges[0]).median(values)
```

## Design decisions:
The functionality proposed here builds heavily on the Index class. An index object encapsulates a set of precomputations on a set of key data, and provides a uniform interface for building set operations on top of this information, abstracting away the details of obtaining this information for a given type of key.
The principal information exposed by an index object is the required permutations to map between the original and sorted order of the keys. This information can subsequently be used for many purposes, such as efficiently finding the set of unique keys, or efficiently performing group_by logic on an array of corresponding values.
The two complex key types currently supported, beyond standard sequences of sortable primitive types, are array keys and composite keys. For the exact casting rules describing valid sequences of key objects to index objects, see as_index().

## Todo and open questions:
* What about nesting of key objects? should be possible too, but not fully supported yet
*	What about floating point nd keys? currently, they will be treated as object indices. However, bitwise and floating point equality are not the same thing 
*	Add special index classes for things like object arrays of variable length strings?
*	While this redesign is aimed more at expanding functionality than optimizing performance, the most common code paths might benefit from some specialization, such as the concatenation of sorted sets into sorted sets
*	In general: are there further generalizations we are still missing? merge/join functionality?

