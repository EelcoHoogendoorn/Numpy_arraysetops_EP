"""unit tests"""

# FIXME : add logic to run numpy tests with our newly created functions?

import numpy as np
import pytest
import numpy.testing
from numpy_indexed import *
from numpy_indexed.utility import as_struct_array


def test_group_by():

    keys   = ["e", "b", "b", "c", "d", "e", "e", 'a']
    values = [1.2, 4.5, 4.3, 2.0, 5.67, 8.08, 9.01,1]

    print('two methods of splitting by group')
    print('as iterable')
    g = group_by(keys)
    for k,v in zip(g.unique, g.split_sequence_as_iterable(values)):
        print(k, list(v))
    print('as list')
    for k,v in zip(*group_by(keys)(values)):
        print(k, v)

    print('some reducing group operations')
    g = group_by(keys)
    unique_keys, reduced_values = g.median(values)
    print('per group median')
    print(reduced_values)
    unique_keys, reduced_values = g.mean(values)
    print('per group mean')
    print(reduced_values)
    unique_keys, reduced_values = g.std(values)
    print('per group std')
    print(reduced_values)
    reduced_values = g.reduce(np.array(values), np.minimum) #alternate way of calling
    print('per group min')
    print(reduced_values)
    unique_keys, reduced_values = g.max(values)
    print('per group max')
    print(reduced_values)

    print('per group sum using custom reduction')
    print(group_by(keys, values, lambda x:x.sum()))


def test_lex_median():
    """for making sure i squased all bugs related to fancy-keys and median filter implementation"""
    keys1  = ["e", "b", "b", "c", "d", "e", "e", 'a']
    keys2  = ["b", "b", "b", "d", "e", "e", 'e', 'e']
    values = [1.2, 4.5, 4.3, 2.0, 5.6, 8.8, 9.1, 1]

    unique, median = group_by((keys1, keys2)).median(values)
    for i in zip(zip(*unique), median):
        print(i)


def test_dict():
    input = [
    {'dept': '001', 'sku': 'foo', 'transId': 'uniqueId1', 'qty': 100},
    {'dept': '001', 'sku': 'bar', 'transId': 'uniqueId2', 'qty': 200},
    {'dept': '001', 'sku': 'foo', 'transId': 'uniqueId3', 'qty': 300},
    {'dept': '002', 'sku': 'baz', 'transId': 'uniqueId4', 'qty': 400},
    {'dept': '002', 'sku': 'baz', 'transId': 'uniqueId5', 'qty': 500},
    {'dept': '002', 'sku': 'qux', 'transId': 'uniqueId6', 'qty': 600},
    {'dept': '003', 'sku': 'foo', 'transId': 'uniqueId7', 'qty': 700}
    ]

    inputs = dict((k, [i[k] for i in input ]) for k in input[0].keys())
    print(group_by((inputs['dept'], inputs['sku'])).mean(inputs['qty']))


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
    key2 = np.random.randint(0,2,(6,2))
    values = np.random.rand(6,3)
    (unique1, unique2), median = group_by((key1, key2)).median(values)
    print(unique1)
    print(unique2)
    print(median)


def test_timings():
    """small performance test"""
    idx = np.random.randint(0, 1000, 10000)
    g = group_by(idx)
    values = np.random.rand(10000, 100)
    assert (np.allclose(g.at(values), g.reduce(values)))

    from time import clock
    t = clock()
    for i in range(100):
        g.reduce(values)
    print('reducing with np.add over 1000 groups with 10000x100 values repeated 100 times took %f seconds' % (clock()-t))


def test_indices():
    """test indices function"""
    values = np.random.rand(20)
    idx = [1, 2, 5, 7]

    assert(np.alltrue(indices(values, values[idx]) == idx))
    assert(np.alltrue(contains(values, values[idx])))
    with pytest.raises(KeyError):
        indices(values, [-1], assume_contained=False)


def test_indices_object():
    """test indices function with objectindex"""
    A = np.array(
        [[0, 1],
         [0, 2],
         [1, 1],
         [0, 2]])
    B = np.array([[0, 2]])
    assert np.alltrue(indices(A, B) == 1)


def test_setops_edgecase():
    """test some edge cases like zero-length, etc"""
    assert np.array_equal(intersection([1], [1]), [1])
    assert np.array_equal(intersection([], []), [])

    assert np.array_equal(count_selected([], []), [])
    # test repeating values in both arguments
    assert np.array_equal(count_selected([1, 2, 3, 1], [1, 1, 2]), [2, 2, 1])

    # FIXME: this casts the dtype to float
    print(difference([1], []))


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
