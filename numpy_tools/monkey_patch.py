"""import this module to monkey patch the functionality from this package into the numpy namespace"""

import numpy as np
import numpy_tools.semantics
import numpy_tools as npt

numpy_tools.semantics.backwards_compatible = True
numpy_tools.semantics.axis_default = None

['unique', 'group_by']

np.unique = npt.unique
np.group_by = npt.group_by