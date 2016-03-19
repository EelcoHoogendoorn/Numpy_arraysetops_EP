"""monkey-patch the numpy tests, and see if they still run"""

# NOTE: numpy.lib.tests is not a module
# from numpy.lib.tests import arraysetops

import numpy_indexed

filename = r"C:\Users\Eelco\Miniconda2\envs\numpy_tools\Lib\site-packages\numpy\lib\tests\test_arraysetops.py"


# module = execfile(filename)
# print(module)
# import importlib.util
# spec = importlib.util.spec_from_file_location("module.name", filename)
# numpy_tests = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(numpy_tests)

exec(open(filename).read())

# in1d = lambda x, y : numpy_indexed.contains(x, y, axis=None)

tester = TestSetOps()

# tester.test_in1d()

quit()

import imp
# numpy_tests = imp.load_module('module.name', filename)
numpy_tests = imp.load_source('numpy_indexed.test_numpy', filename)

numpy_tests.in1d = lambda x, y : numpy_indexed.contains(x, y, axis=None)
# numpy_tests.in1d = None
# numpy_tests.run_module_suite()

tester = numpy_tests.TestSetOps()

tester.test_in1d()