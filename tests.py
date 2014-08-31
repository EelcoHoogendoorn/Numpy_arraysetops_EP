

from API import *




def test_basic():

    keys   = ["e", "b", "b", "c", "d", "e", "e", 'a']
    values = [1.2, 4.5, 4.3, 2.0, 5.67, 8.08, 9.01,1]

##    keys   = np.array(["a", "b", "b", "c", "d", "e", "e",'d','a','c'])
##    values = np.array([1.2, 4.5, 4.3, 2.0, 5.67, 8.08, 9.01, 1,2,3])

    print 'two methods of splitting by group'
    print 'as iterable'
    g = group_by(keys)
    for k,v in zip(g.unique, g.split_sequence_as_iterable(values)):
        print k, list(v)
    print 'as list'
    for k,v in zip(*group_by(keys)(values)):
        print k, v

    print 'some reducing group operations'
    g = group_by(keys)
    unique_keys, reduced_values = g.median(values)
    print 'per group median'
    print reduced_values
    unique_keys, reduced_values = g.mean(values)
    print 'per group mean'
    print reduced_values
    unique_keys, reduced_values = g.std(values)
    print 'per group std'
    print reduced_values
    reduced_values = g.reduce(np.array(values), np.minimum) #alternate way of calling
    print 'per group min'
    print reduced_values
    unique_keys, reduced_values = g.max(values)
    print 'per group max'
    print reduced_values

    print 'per group sum using custom reduction'
    print group_by(keys, values, lambda x:x.sum())



def test_lex_median():
    """
    for making sure i squased all bugs related to fancy-keys and median filter implementation
    """
    keys1  = ["e", "b", "b", "c", "d", "e", "e", 'a']
    keys2  = ["b", "b", "b", "d", "e", "e", 'e', 'e']
##    keys3 = np.random.randint(0,2,(8,2))
    values = [1.2, 4.5, 4.3, 2.0, 5.6, 8.8, 9.1, 1]

    unique, median = group_by((keys1, keys2)).median(values)
    for i in zip(zip(*unique), median):
        print i



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

    inputs = dict( (k, [i[k] for i in input ]) for k in input[0].keys())
    print group_by((inputs['dept'], inputs['sku'])).mean(inputs['qty'])



def test_fancy_keys():
    """
    test Index subclasses
    """
    keys        = np.random.randint(0, 2, (20,3)).astype(np.int8)
    values      = np.random.randint(-1,2,(20,4))


    #all these various datastructures should produce the same behavior
    #multiplicity is a nice unit test, since it draws on most of the low level functionality
    if backwards_compatible:
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
    print 'sum per group of identical rows using struct key'
    g = group_by(as_struct_array(keys, floatkeys))
    for e in zip(g.count, *g.sum(values)):
        print e
    print 'sum per group of identical rows using lex of nd-key'
    g = group_by(( keys, floatkeys))
    for e in zip(zip(*g.unique), g.sum(values)[1]):
        print e
    print 'sum per group of identical rows using lex of struct key'
    g = group_by((as_struct_array( keys), floatkeys))
    for e in zip(zip(*g.unique), g.sum(values)[1]):
        print e

    #showcase enhanced unique functionality
    images = np.random.rand(4,4,4)
    #shuffle the images; this is a giant mess now; how to find the unique ones?
    shuffled = images[np.random.randint(0,4,200)]
    #there you go
    if backwards_compatible:
        print unique(shuffled, axis=0)
    else:
        print unique(shuffled)


def test_compact():
    """demonstrate the most functionality in the least number of lines"""
    key1 = list('abaabb')
    key2 = np.random.randint(0,2,(6,2))
    values = np.random.rand(6,3)
    (unique1, unique2), median = group_by((key1, key2)).median(values)
    print unique1
    print unique2
    print median

def test_timings():
    """test some performance questions"""
    idx = np.random.randint(0, 1000, 10000)
    g = group_by(idx)
    values = np.random.rand(10000, 100)
    assert(np.allclose( g.at(values), g.reduce(values)))
    from time import clock
    t = clock()

    for i in xrange(100):
        g.reduce(values)
    print clock()-t

def test_indices():
    """
    test indices function
    """
    values = np.random.rand(100)
    idx = np.random.randint(0,100, 10)
    assert(np.all(indices(values, values[idx])==idx))
    assert(np.all(contains(values, values[idx])))

def test_setops():
    """
    test generalized classic setops
    """
    edges = np.random.randint(0,9,(3,100,2))
    print exclusive(*edges)

    edges = np.arange(20).reshape(10,2)
    assert(np.all(difference(edges[:8], edges[-8:])==edges[:2]))

    key1 = list('abc')*10
    key2 = np.random.randint(0,9,30)
    print unique( (key1, key2))

def test_funcs():
    t = count_table(*np.random.randint(0,4,(3,1000)))
    print t


if __name__=='__main__':
    test_setops()
    test_funcs()
    test_basic()
    test_lex_median()
    test_dict()
    test_fancy_keys()
    test_compact()
    test_indices()
    test_timings()
