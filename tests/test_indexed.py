"""unit tests.
many functions could still do with better test coverage!
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy.testing as npt
import pytest

from numpy_indexed import *
from numpy_indexed.utility import *


__author__ = "Eelco Hoogendoorn"
__license__ = "LGPL"
__email__ = "hoogendoorn.eelco@gmail.com"


def test_fancy_keys():
    """test Index subclasses"""
    keys        = np.random.randint(0, 2, (20, 3)).astype(np.int8)
    values      = np.random.randint(-1, 2, (20, 4))

    #all these various datastructures should produce the same behavior
    #multiplicity is a nice unit test, since it draws on most of the low level functionality
    if semantics.backwards_compatible:
        assert(np.all(
            multiplicity(keys, axis=0) ==           #void object indexing
            multiplicity(tuple(keys.T))))           #lexographic indexing
        assert(np.all(
            multiplicity(keys, axis=0) ==           #void object indexing
            multiplicity(as_struct_array(keys))))   #struct array indexing
    else:
        assert(np.all(
            multiplicity(keys) ==                   #void object indexing
            multiplicity(tuple(keys.T))))           #lexographic indexing
        assert(np.all(
            multiplicity(keys) ==                   #void object indexing
            multiplicity(as_struct_array(keys))))   #struct array indexing

    #lets go mixing some dtypes!
    floatkeys   = np.zeros(len(keys))
    floatkeys[0] = 8.8
    print('sum per group of identical rows using struct key')
    g = group_by(as_struct_array(keys, floatkeys))
    for e in zip(g.count, *g.sum(values)):
        print(e)
    print('sum per group of identical rows using lex of nd-key')
    g = group_by(( keys, floatkeys))
    for e in zip(zip(*g.unique), g.sum(values)[1]):
        print(e)
    print('sum per group of identical rows using lex of struct key')
    g = group_by((as_struct_array( keys), floatkeys))
    for e in zip(zip(*g.unique), g.sum(values)[1]):
        print(e)

    #showcase enhanced unique functionality
    images = np.random.rand(4,4,4)
    #shuffle the images; this is a giant mess now; how to find the unique ones?
    shuffled = images[np.random.randint(0,4,200)]
    #there you go
    if semantics.backwards_compatible:
        print(unique(shuffled, axis=0))
    else:
        print(unique(shuffled))


def test_compact():
    """demonstrate the most functionality in the least number of lines"""
    key1 = list('abaabb')
    key2 = np.random.randint(0,2,(6, 2))
    values = np.random.rand(6,3)
    g = group_by((key1, key2))
    (unique1, unique2), median = group_by((key1, key2)).median(values)
    print(unique1)
    print(unique2)
    print(median)


def test_indices():
    """test indices function"""
    values = np.random.rand(20)
    idx = [1, 2, 5, 7]

    assert(np.alltrue(indices(values, values[idx]) == idx))
    with pytest.raises(KeyError):
        indices(values, [-1])


def test_indices_object():
    """test indices function with objectindex"""
    A = np.array(
        [[0, 1],
         [0, 2],
         [1, 1],
         [0, 2]])
    B = np.array([[0, 2]])
    assert indices(A, B) == 1
    B = np.array([[1, 2]])
    with pytest.raises(KeyError):
        indices(A, B)
    B = np.array(
        [[0, 2],
         [1, 2]])
    assert len(indices(A, B, missing='mask').compressed()) == 1


def test_indices_lex():
    k1 = ["e", "b", "b", "c", "d", "e", "c", 'a']
    k2 = ["b", "b", "c", "d", "e", "e", 'e', 'e']
    values = [1.2, 4.5, 4.3, 2.0, 5.6, 8.8, 9.1, 1]

    npt.assert_equal(indices((k1, k2), (k1, k2)), np.arange(len(k1)))
    with pytest.raises(KeyError):
        indices((k1, k2), (['d'],['a']))


def test_setops_edgecase():
    """test some edge cases like zero-length, etc"""
    assert np.array_equal(intersection([1], [1]), [1])
    assert np.array_equal(intersection([], []), [])

    assert np.array_equal(difference([1], []), [1])
    assert difference([1], []).dtype == int

    assert np.array_equal(union([], [], []), [])
    assert np.array_equal(exclusive([], []), [])


def test_setops():
    """test generalized classic set operations"""
    # edges exclusive to one of three sets
    edges = np.random.randint(0, 9, size=(3, 100, 2))
    print(exclusive(*edges))

    # difference on object keys
    edges = np.arange(20).reshape(10, 2)
    assert(np.all(difference(edges[:8], edges[-8:]) == edges[:2]))

    # unique on lex keys
    key1 = list('abc')*10
    key2 = np.random.randint(0,9,30)
    print(unique( (key1, key2)))


def test_count_table():
    k = list('aababaababbbaabba')
    i = np.random.randint(0, 10, len(k))
    l, t = count_table(k, i)
    print(l)
    print(t)

    l, t = count_table(*np.random.randint(0,4,(3,1000)))
    print(l)
    print(t)

    l, t = count_table(np.random.randint(0,4,(1000,3)))
    print(l)
    print(t)


def test_mode():
    m, idx = mode([1, 2, 2, 1, 3, 1], return_indices=True)
    assert m == 1
    npt.assert_equal(idx, [0, 3, 5])


def test_void_casting():
    """ensure that axis_as_object and object_as_axis are indeed inverse operations"""
    dummy = np.random.rand(2, 3, 4)
    for a in range(dummy.ndim):
        void = axis_as_object(dummy, a)
        restored = object_as_axis(void, dummy.dtype, a)
        assert (np.alltrue(dummy == restored))


def test_all_any_unique():
    assert all_unique([1, 2, 2, 1, 3, 1]) == False
    assert all_unique(np.eye(3)) == True
    assert any_unique([1, 2, 2, 1, 3, 1]) == True
    assert any_unique([1, 1, 1]) == False


def test_all_any_equal():
    assert all_equal([1, 2, 2, 1, 3, 1]) == False
    assert all_equal([1, 1, 1]) == True
    assert any_equal([1, 2, 2, 1, 3, 1]) == True
    assert any_equal(np.eye(3)) == False


def test_sorted():
    arr = np.random.rand(20)
    npt.assert_equal(sort(arr), np.sort(arr))
    arr = np.arange(20).reshape(10, 2)
    npt.assert_equal(sort(arr[::-1]), arr)
    a, b = np.random.permutation(arr).T
    npt.assert_equal(sort((a, b)), arr.T)


def test_containment_relations():
    this = [1, 1, 1]
    that = [1, 2, 2, 1, 3, 1]
    npt.assert_equal(in_(this, that), True)
    npt.assert_equal(contains(that, this), True)

    this = np.random.randint(0, 9, 20)
    that = np.random.randint(0, 4, 10)

    npt.assert_equal(in_(this, that), contains(that, this))
    npt.assert_equal(in_(that, this), contains(this, that))

    # test empty behavior
    npt.assert_equal(contains(this, []), [])
    npt.assert_equal(contains([], that), False)

    npt.assert_equal(in_(this, []), False)
    npt.assert_equal(in_([], that), [])


def test_regression_contains():
    a = np.array([[4, 2.2, 5], [2, -6.3, 0], [3, 3.6, 8], [5, -9.8, 50]])
    b = np.array([[2.2, 5], [-6.3, 0], [3.6, 8]])
    npt.assert_equal(contains(a[:, 1:], b), True)
    npt.assert_equal(contains(b, a[:, 1:]), [True]*3+[False])

    this = [0, 1, 2, 2, 3, 4]
    that = [2, 3]
    npt.assert_equal(in_(this, that), contains(that, this))
    npt.assert_equal(in_(that, this), contains(this, that))


def test_table():
    k1 = ["e", "b", "b", "c", "d", "e", "c", 'a']
    k2 = ["b", "b", "c", "d", "e", "e", 'e', 'e']
    values = [1.2, 4.5, 4.3, 2.0, 5.6, 8.8, 9.1, 1]
    u, t = Table(k1, k2).mean(values)
    # u, t = Table(k1, k2).count()
    print(u)
    print(t)


def test_remap():
    keys = [1, 2, 3, 4]
    values = [5, 6, 7, 8]
    input = [1, 1, 2, 4]
    output = remap(input, keys, values)
    npt.assert_equal(output, [5, 5, 6, 8])

    with pytest.raises(KeyError):
        output = remap(input, np.array(keys) + 10, values, missing='raise')

    output = remap(input, np.array(keys)+10, values)
    npt.assert_equal(output, input)

    # test with more input
    keys = np.tile(keys, [2, 1]).T
    values = np.tile(values, [2, 1]).T
    input = np.tile(input, [2, 1]).T
    output = remap(input, keys, values)
    npt.assert_equal(output, np.tile([5, 5, 6, 8], [2, 1]).T)


def test_table_max():
    idx = np.array([
        [[0, 0, 1], [0, 1, 2], [0, 2, 3], [3, 0, 16], [0, 4, 5]],
        [[1, 0, 6], [1, 1, 7], [1, 2, 8], [1, 3, 9], [1, 4, 10]],
        [[2, 0, 11], [2, 1, 12], [0, 0, 13], [2, 3, 14], [2, 4, 15]],
        [[3, 0, 4], [3, 1, 17], [3, 2, 18], [3, 3, 19], [3, 4, 20]],
    ])
    result = \
        [[13, 2,  3,  0,  5],
         [6,  7,  8,  9,  10],
         [11, 12,  0,  14, 15],
         [16, 17,  18, 19, 20],]
    idx = idx.reshape(-1, 3)
    u, r = Table(idx[:,0], idx[:,1]).max(idx[:,2], default=0)
    npt.assert_array_equal(r, result)
