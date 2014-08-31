"""
radial reduction example
"""
import matplotlib.pyplot as pp


x = np.linspace(-2,2, 64)
y = x[:, None]
x = x[None, :]
R = np.sqrt( x**2+y**2)

def airy(r, sigma):
    from scipy.special import j1
    r = r / sigma * np.sqrt(2)
    a = (2*j1(r)/r)**2
    a[r==0] = 1
    return a
def gauss(r, sigma):
    return np.exp(-(r/sigma)**2)

distribution = np.random.choice([gauss, airy])(R, 0.3)
sample = np.random.poisson(distribution*200+10).astype(np.float)

import matplotlib.pyplot as pp
#is this an airy or gaussian function? hard to tell with all this noise!
pp.imshow(sample, interpolation='nearest', cmap='gray')
pp.show()
#radial reduction to the rescue!
#if we are sampling an airy function, you will see a small but significant rise around x=1
g = group_by(np.round(R, 5).flatten())
pp.errorbar(
    g.unique,
    g.mean(sample.flatten())[1],
    g.std (sample.flatten())[1] / np.sqrt(g.count))
pp.xlim(0,2)
pp.show()
