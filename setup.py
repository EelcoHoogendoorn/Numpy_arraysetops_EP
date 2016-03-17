import os
from setuptools import setup

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "numpy_indexed",
    version = "0.1.0",
    author = "Eelco Hoogendoorn",
    author_email = "hoogendoorn.eelco@gmail.com",
    description = ("groupy_by and nd-set operations for numpy"),
    license = "BSD",
    keywords = "numpy group_by",
    url = "http://packages.python.org/numpy_indexed",
    packages=['numpy_indexed'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: Freely Distributable",
    ],
)