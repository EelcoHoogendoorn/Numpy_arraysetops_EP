"""import this module to monkey patch the functionality from this package into the numpy namespace"""

import numpy
import numpy_indexed
numpy.indexed = numpy_indexed
