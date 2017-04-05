"""Tests for the group_by object"""

import numpy as np
from numpy import testing as npt

from numpy_indexed import group_by


def test_group_by():

    keys   = ["e", "b", "b", "c", "d", "e", "e", 'a']
    values = [1.2, 4.5, 4.3, 2.0, 5.67, 8.08, 9.01,1]

    print('two methods of splitting by group')
    print('as list')
    for k,v in zip(*group_by(keys)(values)):
        print(k, v)
    print('as iterable')
    g = group_by(keys)
    for k,v in zip(g.unique, g.split_sequence_as_iterable(values)):
        print(k, list(v))
    print('iterable as iterable')
    for k, v in zip(g.unique, g.split_iterable_as_iterable(values)):
        print(k, list(v))

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


def test_argmin():
    a = [4, 5, 6, 8, 0, 9, 8, 5, 4, 9]
    b = [2, 0, 0, 1, 1, 2, 2, 2, 2, 2]
    k, u = group_by(b).argmin(a)
    npt.assert_equal(k, [0, 1, 2])
    npt.assert_equal(u, [1, 4, 0])


def test_argmax():
    a = [4, 5, 6, 8, 0, 9, 8, 5, 4, 9]
    b = [0, 0, 0, 0, 1, 2, 2, 2, 2, 2]
    k, u = group_by(b).argmax(a)
    npt.assert_equal(k, [0, 1, 2])
    npt.assert_equal(u, [3, 4, 5])


def test_mode():
    a = [4, 5, 6, 4, 0, 9, 8, 5, 4, 9]
    b = [0, 0, 0, 0, 1, 2, 2, 2, 2, 2]
    k, u = group_by(b).mode(a)
    npt.assert_equal(u, [4, 0, 9])


def test_weighted_mean():
    keys   = ["e", "b", "b", "c", "d", "e", "e", 'a']
    values = [1.2, 4.5, 4.3, 2.0, 5.67, 8.08, 7.99, 1]
    weights = [1,  1,   1,   1,   1,    1,    1,    1]
    g, mean_none = group_by(keys).mean(values, weights=None)
    g, mean_ones = group_by(keys).mean(values, weights=weights)
    g, mean_threes = group_by(keys).mean(values, weights=np.array(weights)*3)

    npt.assert_equal(mean_none, mean_ones)
    npt.assert_allclose(mean_none, mean_threes)     # we incurr some fp error here

    weights = [0,  0,   1,   1,   1,    1,    8,    1]
    g, mean_w = group_by(keys).mean(values, weights=weights)
    npt.assert_allclose(mean_w, [1, 4.3, 2, 5.67, 8])


def test_weighted_std():
    means = [0, 1, 2]
    stds = [1, 2, 3]
    samples = [100, 1000, 10000]

    normal = np.concatenate([np.random.normal(*p) for p in zip(means, stds, samples)])
    keys = np.concatenate([np.ones(s)*s for s in samples])
    weights = np.random.rand(len(normal))

    g, std_w = group_by(keys).std(normal, weights=weights)
    print(std_w)


def test_prod():
    keys   = ["e", "b", "b", "c", "d", "e", "e", 'a']
    values = [1.2, 4.5, 4.3, 2.0, 5.67, 8.08, 0, 1]
    g, p = group_by(keys).prod(values)
    print(g, p)


def test_all():
    keys   = ["e", "b", "b", "c", "d", "e", "e", 'a']
    values = [1.2, 4.5, 4.3, 0,   5.67, 8.08, 0, 1]
    g, p = group_by(keys).all(values)
    print(g, p)


def test_mean_axis():
    """http://stackoverflow.com/questions/38607586/delete-columns-based-on-repeat-value-in-one-row-in-numpy-array"""
    initial_array = np.array(
    [[1, 1, 1, 1, 1, 1, 1, 1, ],
    [0.5, 1, 2.5, 4, 2.5, 2, 1, 3.5,],
    [1, 1.5, 3, 4.5, 3, 2.5, 1.5, 4,],
    [228, 314, 173, 452, 168, 351, 300, 396]])

    unique, final_array = group_by(initial_array[1, :]).mean(initial_array, axis=1)
    print(final_array)