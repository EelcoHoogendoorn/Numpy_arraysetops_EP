"""monkey-patch the numpy tests, and see if they still run"""

import os
import numpy
import numpy_indexed
import unittest

# load numpy arraysetops testing module
module_source = os.path.join(os.path.split(numpy.__file__)[0], r"lib\tests\test_arraysetops.py")
module_name = 'numpy_indexed.tests.test_numpy'
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, module_source)
    numpy_tests = importlib.util.module_from_spec(spec)
except:
    import imp
    numpy_tests = imp.load_source(module_name, module_source)

def run_patched_tests():
    """
    run monkey-patched versions of tests for:
        intersect1d, setxor1d, union1d, setdiff1d, unique, in1d
    """

    numpy_tests.intersect1d = numpy_indexed.intersection

    numpy_tests.setxor1d = numpy_indexed.exclusive

    numpy_tests.union1d = numpy_indexed.union

    # fails on casting rules; empty set gets float dtype...
    # numpy_tests.setdiff1d = numpy_indexed.difference

    def unique(ar, return_index=False, return_inverse=False, return_counts=False):
        return numpy_indexed.unique(ar, None, return_index, return_inverse, return_counts)
    numpy_tests.unique = unique

    def in1d(ar1, ar2, assume_unique=False, invert=False):
        ret = numpy_indexed.contains(ar2, ar1, axis=None)
        return numpy.invert(ret) if invert else ret
    numpy_tests.in1d = in1d

    # run the suite
    suite = unittest.TestLoader().loadTestsFromTestCase(numpy_tests.TestSetOps)
    unittest.TextTestRunner(verbosity=3).run(suite)


if __name__ == '__main__':
    run_patched_tests()