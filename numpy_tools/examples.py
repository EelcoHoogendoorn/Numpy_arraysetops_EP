
"""
examples:

some more complex examples of actual use cases than found in simple tests
"""
import matplotlib.pyplot as pp
from API import *


def test_count_table():
    k = 'aababaababbbaabba'
    k = [c for c in k]
    i = np.random.randint(0,10, len(k))
    print(count_table(k, i)[1])


def test_radial():
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


def test_meshing():
    """
    meshing example
    demonstrates the use of multiplicity, and group.median
    """
    #set up some random points, and get their delaunay triangulation
    points = np.random.random((20000,2))*2-1
    points = points[np.linalg.norm(points,axis=1) < 1]
    from scipy.spatial.qhull import Delaunay
    d = Delaunay(points)
    tris = d.simplices

    #the operations provided in this module allow us to express potentially complex
    #computational geometry questions elegantly in numpy
    #Delaunay.neighbors could be used as well,
    #but the point is to express this in pure numpy, without additional library functionaoty
    edges = tris[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    sorted_edges = np.where(edges[:,0:1]<edges[:,1:2], edges, edges[:,::-1])
    #we test to see how often each edge occurs, or how many indicent simplices it has
    #this is a very general method of finding the boundary of any topology
    #and we can do so here with only one simple and readable command, multiplicity == 1
    if backwards_compatible:
        boundary_edges = edges[multiplicity(sorted_edges, axis=0)==1]
    else:
        boundary_edges = edges[multiplicity(sorted_edges)==1]
    boundary_points = unique(boundary_edges)

    if False:
        print(boundary_edges)
        print(incidence(boundary_edges))

    #create some random values on faces
    #we want to smooth them over the mesh to create a nice hilly landscape
    face_values   = np.random.normal(size=d.nsimplex)
    #add some salt and pepper noise, to make our problem more interesting
    face_values[np.random.randint(d.nsimplex, size=10)] += 1000

    #start with a median step, to remove salt-and-pepper noise
    #toggle to mean to see the effect of the median filter
    g = group_by(tris.flatten())
    prestep = g.median if True else g.mean
    vertex_values = prestep(np.repeat(face_values, 3))[1]
    vertex_values[boundary_points] = 0

    #actually, we can compute the mean without grouping
    tris_per_vert = g.count
    def scatter(x):
        r = np.zeros(d.npoints, x.dtype)
        for idx in tris.T: np.add.at(r, idx, x)
        return r / tris_per_vert
    def gather(x):
        return x[tris].mean(axis=1)

    #iterate a little
    for i in range(100):
        face_values   = gather(vertex_values)
        vertex_values = scatter(face_values)
        vertex_values[boundary_points] = 0


    #display our nicely rolling hills and their boundary
    x, y = points.T
    pp.tripcolor(x,y, triangles = tris, facecolors = face_values)
    pp.scatter(x[boundary_points], y[boundary_points])
    pp.xlim(-1,1)
    pp.ylim(-1,1)
    pp.axis('equal')
    pp.show()
