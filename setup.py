from setuptools import find_packages, setup


from version import __version__


setup(
    name="numpy-indexed",
    packages=find_packages(),
    version=__version__,
    install_requires=['numpy', 'future'],
    keywords="numpy group_by set-operations indexing",
    description=open("README.rst").readlines()[5],
    long_description=open("README.rst").read(),
    author="Eelco Hoogendoorn",
    author_email="hoogendoorn.eelco@gmail.com",
    url="https://github.com/EelcoHoogendoorn/Numpy_arraysetops_EP",
    license="Freely Distributable",
    platforms='Any',
    classifiers=[
        "Development Status :: 4 - Beta",
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        "Topic :: Utilities",
        'Topic :: Scientific/Engineering',
        "License :: {}".format("Freely Distributable"),
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
    ],
)
