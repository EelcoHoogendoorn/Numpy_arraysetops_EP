Numpy array set operations_Enhancement Proposal draft
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

Some brief examples:
'''python
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
'''
