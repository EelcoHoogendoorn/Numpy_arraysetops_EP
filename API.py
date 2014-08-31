
"""
public API
"""

from grouping import group_by
from index import as_index
from funcs import *
from arraysetops import *


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

