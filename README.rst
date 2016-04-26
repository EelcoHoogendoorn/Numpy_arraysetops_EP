|Build Status| |Build status|

Numpy indexed operations
========================

This package contains functionality for indexed operations on numpy
ndarrays, providing efficient vectorized functionality such as grouping
and set operations.

-  Rich and efficient grouping functionality:
   -  splitting of values by key-group
   -  reductions of values by key-group

-  Generalization of existing array set operation to nd-arrays, such as:
   -  unique
   -  union
   -  difference
   -  exclusive (xor)
   -  contains / in_ (in1d)

-  Some new functions:
   -  indices: numpy equivalent of list.index
   -  count: numpy equivalent of collections.Counter
   -  mode: find the most frequently occuring items in a set
   -  multiplicity: number of occurrences of each key in a sequence
   -  count\_table: like R's table or pandas crosstab, or an ndim version of np.bincount

Some brief examples to give an impression hereof:

.. code:: python

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

Installation
------------

.. code:: python

    > conda install numpy-indexed -c eelcohoogendoorn

or

.. code:: python

    > pip install numpy-indexed

Design decisions:
-----------------

This package builds upon a generalization of the design pattern as can
be found in numpy.unique. That is, by argsorting an ndarray, many
subsequent operations can be implemented efficiently and in a vectorized
manner.

The sorting and related low level operations are encapsulated into a
hierarchy of Index classes, which allows for efficient lookup of many
properties for a variety of different key-types. The public API of this
package is a quite thin wrapper around these Index objects.

The two complex key types currently supported, beyond standard sequences
of sortable primitive types, are ndarray keys (i.e, finding unique
rows/columns of an array) and composite keys (zipped sequences). For the
exact casting rules describing valid sequences of key objects to index
objects, see as\_index().

Todo and open questions:
------------------------

-  There may be further generalizations that could be built on top of
   these abstractions. merge/join functionality perhaps?

.. |Build Status| image:: https://travis-ci.org/EelcoHoogendoorn/Numpy_arraysetops_EP.svg?branch=master
   :target: https://travis-ci.org/EelcoHoogendoorn/Numpy_arraysetops_EP
.. |Build status| image:: https://ci.appveyor.com/api/projects/status/h7w191ovpa9dcfum?svg=true
   :target: https://ci.appveyor.com/project/clinicalgraphics/numpy-arraysetops-ep
