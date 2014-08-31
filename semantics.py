"""
this toggle switches between preferred or backwards compatible semantics
for dealing with key objects. current behavior for the arguments to functions
like np.unique is to flatten any input arrays.
i think a more unified semantics is achieved by interpreting all key arguments as
sequences of key objects, whereby non-flat arrays are simply sequences of keys,
whereby the keys themselves are ndim-1 arrays
for reasons of backwards compatibility, it is probably wise to retain the default within numpy 1.x,
but at least an axis keyword to toggle this behavior would be a welcome addition
"""
backwards_compatible = False
if backwards_compatible:
    axis_default = None
else:
    axis_default = 0
