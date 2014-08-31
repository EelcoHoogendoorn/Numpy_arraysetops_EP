
"""
public API
"""

from . import grouping
from . import index
from .funcs import *
from .arraysetops import *


#just an alias, for those who dont like camelcase
#group_by = GroupBy
#could also turn this into a function with optional values and reduction func.
def group_by(keys, values=None, reduction=None, axis=0):
    g = grouping.GroupBy(keys, axis)
    if values is None:
        return g
    groups = g.split(values)
    if reduction is None:
        return g.unique, groups
    return [(key,reduction(group)) for key, group in itertools.izip(g.unique, groups)]


__all__ = [group_by, index.as_index]