"""monkey-patch the numpy tests, and see if they still run"""

# NOTE: numpy.lib.tests is not a module
# from numpy.lib.tests import arraysetops

import numpy
import numpy_indexed
import unittest

filename = r"C:\Users\Eelco\Miniconda2\envs\numpy_tools\Lib\site-packages\numpy\lib\tests\test_arraysetops.py"

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location('numpy_indexed.test_numpy', filename)
    numpy_tests = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(numpy_tests)
except:
    import imp
    numpy_tests = imp.load_source('numpy_indexed.test_numpy', filename)

def run_patched_tests():
    """
    run patched versions of
        intersect1d, setxor1d, union1d, setdiff1d, unique, in1d
    """

    numpy_tests.intersect1d = numpy_indexed.intersection

    numpy_tests.setxor1d = numpy_indexed.exclusive

    numpy_tests.union1d = numpy_indexed.union

    # fails on casting rules; empty set gets float dtype...
    # numpy_tests.setdiff1d = numpy_indexed.difference


    def unique(*args, **kwargs):
        kwargs['axis'] = None
        return numpy_indexed.unique(*args, **kwargs)
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