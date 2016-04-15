"""
This toggle switches between preferred or backwards compatible semantics for dealing with key objects.

The behavior of numpy with respect to arguments to functions like np.unique is to flatten any input arrays.
Arguably, a more unified semantics is achieved by interpreting all key arguments as sequences of key objects,
whereby multi-dimensional arrays are simply interpreted as sequences of (complex) keys.

For reasons of backwards compatibility, one may prefer the same semantics as numpy 1.x though
"""


__author__ = "Eelco Hoogendoorn"
__license__ = "LGPL"
__email__ = "hoogendoorn.eelco@gmail.com"


backwards_compatible = False
if backwards_compatible:
    axis_default = None
else:
    axis_default = 0
