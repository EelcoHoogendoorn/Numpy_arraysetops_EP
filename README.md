Numpy array set operations Enhancement Proposal draft
====================

This is a draft for a numpy EP, aiming for a comprehensive overhaul of the arraysetops module. The planned functionality includes:

* Rich and efficient grouping functionality:
  * splitting of values by key-group
  * reductions of values by key-group
* Some new functions:
  * indices: numpy equivalent of list.index
  * count: numpy equivalent of collections.Counter
  * multiplicity: number of occurances of each key in a sequence
  * count_table: like R's table
* Some generalization of existing array set operation, such as:
  * unique 
  * union
  * difference
  * exclusive (xor)
  * contains (in1d)
The generalization of the existing array set operations pertains primarily to the extension of this functionality to 'complex key objects'. For instance, we may wish to find the intersection of several sets of graph edges. All the functions described above build upon this generalized notion of a key object.

Some brief examples to give an impression hereof:
```python
#find graph edges (doublet of ints) exclusive to one of three sets
edges = np.random.randint(0,9,(3,100,2)
ex = exclusive(*edges)
print ex
#which edges are exclusive to the first set
print contains(edges[0], ex)
#where are the exclusive edges relative to the totality of them
print indices( union(*edges), ex)
#group and reduce values by identical keys
values = np.random.rand(100,20)
print group_by(edges[0]).median(values):
```

## Design decisions:
The Index class is an abstraction introduced, in order to enable the use of complex key objects in all array set operations.
An index encapsulates a set of precomputations on a set of key data, and provides a uniform interface for building set operations on top of this information, abstracting away the details of obtaining this information for a given type of key.
The principal information exposed by an index object is the required permutations to map between the original and sorted order of the keys. This information can subsequently be used for many purposes, such as efficiently finding the set of unique keys, or efficiently performing group_by logic on an array of corresponding values.

## Todo and open questions:
* what about nesting of key objects? should be possible too, but not fully supported yet
* need to wrap access to key objects in an accessor, since this is nontrivial for lexindices
*	what about floating point nd keys? currently, they will be treated as object indices
*	however, bitwise and floating point equality are not the same thing exactly
*	also, lex indices are not fully supported in set operations yet. perhaps need to encapsulate concatenation behavior inside the index object; think of it as concatting the index objects instead of the key-data
*	add special index classes for things like object arrays of variable length strings?
*	while this redesign is aimed more at expanding functionality than optimizing performance, the most common code paths might benefit from some specialization, such as the concatenation of sorted sets into sorted sets
*	in general: are there further generalizations we are still missing? merge/join functionality?


## A note on pandas and complex set operations:

This module has substantial overlap with pandas' grouping functionality. So whats the reason for implementing it in numpy?
Primarily; the concept of grouping is far more general than pandas' dataframe. There is no reason why numpy ndarrays should not have a solid core of grouping functionality. The recently added ufunc support make that we can now express reducing grouping operations in pure numpy; that is, without any slow python loops or cumbersome C-extensions.
It does raise the question as to where the proper line between pandas and numpy lies. I would argue that evidently, most of pandas functionality has no place in numpy. Then how is grouping different? I feel what lies at the heart of pandas is a permanent conceptual association between various pieces of data, assembled in a dataframe and all its metadata. I think numpy ought to stay well clear of that.
On the other hand, you dont want want to go around creating a pandas dataframe just to plot a radial reduction; These kind of transient single-statement associations between keys and values are very useful, entirely independently of a more heavyweight framework.
Further questions raised from pandas: should we have some form of merge/join functionality too? or is this getting too panda-ey? All the use cases I can think of fail to be pressed in some kind of standard mould, but i might be missing something here

