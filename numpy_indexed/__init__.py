"""high level interface"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os

from numpy_indexed.arraysetops import *
from numpy_indexed.grouping import *
from numpy_indexed.funcs import *

pkg_dir = os.path.abspath(os.path.dirname(__file__))

__version__ = '0.2.dev8'
