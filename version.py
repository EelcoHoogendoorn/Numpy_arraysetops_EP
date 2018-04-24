import re
from os import path

__version__ = False  # Prevent 'undefined name' warning.

# Read file that should contain __version__ var, and find it. Though it should be the first line,
# we scan the whole file and exec the version setter to have it available in our context, instead
# of more fragile string parsing and regexing.
with open(path.join(path.dirname(path.abspath(__file__)), 'numpy_indexed', '__init__.py'), 'r') as f:
    for line in f:
        if re.search('^__version__ =', line):
            exec(line)
            break


print(__version__)
