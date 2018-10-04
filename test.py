import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit
import simulation


def testInterpolate2d():
    xMarkers = np.arange(-5, 5, 0.25)
    yMarkers = np.arange(-2.5, 2.5, 0.25)
    X, Y = np.meshgrid(xMarkers, yMarkers)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, antialiased=False, alpha=0.5)
    xs = [np.random.uniform(-6, 6) for _ in range(100)]
    ys = [np.random.uniform(-4, 4) for _ in range(100)]
    zs = [simulation.interpolate2d(xMarkers, yMarkers, Z, x, y) for (x, y) in zip(xs, ys)]
    ax.scatter(xs, ys, zs, color='red')
    plt.show()


def testRandomPointOnSphere():
    points = np.array([simulation.randomPointOnSphere(10) for _ in range(1000)])
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    plt.show()


def testEnergyConservation():
    pass
    
    
testInterpolate2d()
testRandomPointOnSphere()
testEnergyConservation()
