"""
Test case: straight 1 T magnetic field and 1 GeV proton
  Wolfram alpha: 1 GeV -> 2.6e8 m/s
  r_g = mv_perp/eB = 1.6e-27 * 2.6e8/(1.6e-19 * 1) = 2.6 m
  omega_g = eB/m = 1.6e-19 * 1 / 1.6e-27 = 10^8 rad/s
  tau_g = 2pi/10^8 = 6.2e-8 s for a full turn
"""

import numpy as np
from numba import jit, njit, prange
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d, griddata
import math
import time


def simpleDipole(x_vec):
    """
    B in z direction, constant inside loop, falls off as r^3 outside.
    Not a function of z.
    """
    x = x_vec[0] or 1e-3
    y = x_vec[1] or 1e-3
    z = x_vec[2] or 1e-3
    loopRadius = .5
    BonAxis = 10
    r = (x**2 + y**2)**.5
    if r < loopRadius:
        return np.array([0., 0., BonAxis])
    return np.array([0., 0., BonAxis * loopRadius**3/r**3])


# #@jit(nopython=True)
# def nearestIndex(array, value):
#     return (np.abs(array - value)).argmin()

    
@jit(nopython=True)
def nearestIndex(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    return idx


@jit(nopython=True)
def Bxyz(x_vec, BR, BZ, R, Z):
    """
    Given BR, BZ on an R, Z grid, return interpolated B vector at arbitrary position.
    """
    x, y, z = x_vec[:3]
    r = (x**2 + y**2)**.5

    nr = nearestIndex(R, r)
    nz = nearestIndex(Z, z)
    BRinterp = BR[nz, nr]
    BZinterp = BZ[nz, nr]
    Bx = BRinterp * x/r
    By = BRinterp * y/r
    return np.array([Bx, By, BZinterp])


# # @jit(nopython=True)
# def Bxyz(x_vec, BR, BZ):
#     """
#     Given BR, BZ on an R, Z grid, return interpolated B vector at arbitrary position.
#     """
#     x, y, z = x_vec[:3]
#     r = (x**2 + y**2)**.5
    
#     BRinterp = BR(r, z)
#     BZinterp = BZ(r, z)
#     Bx = BRinterp * x/r
#     By = BRinterp * y/r
#     return np.array([Bx, By, BZinterp])
    

#@jit(nopython=False)
# def fieldFunction(filename):
#     """
#     Read COMSOL output file and return function B(r, z).
#     """
#     with open(filename) as csvfile:
#         vals = []
#         for line in csvfile:
#             if '%' not in line:
#                 vals.append([float(n) for n in line.split()])
#     vals = np.array(vals)
#     r = np.arange(np.min(vals[:,0]), np.max(vals[:,0]), .01)
#     z = np.arange(np.min(vals[:,1]), np.max(vals[:,1]), .01)
#     grid_r, grid_z = np.meshgrid(r, z)
#     grid = griddata(vals[:, :2], vals[:, 2], (grid_r, grid_z))
#     return interp2d(r, z, grid, kind='cubic', fill_value=0.) 

# @jit(nopython=False)
def fieldFunction(filename):
    """
    Read COMSOL output file and return function B(r, z).
    """
    with open(filename) as csvfile:
        vals = []
        for line in csvfile:
            if '%' not in line:
                vals.append([float(n) for n in line.split()])
    vals = np.array(vals)
    r = np.arange(np.min(vals[:,0]), np.max(vals[:,0]), .1)
    z = np.arange(np.min(vals[:,1]), np.max(vals[:,1]), .1)
    grid_r, grid_z = np.meshgrid(r, z)
    grid = griddata(vals[:, :2], vals[:, 2], (grid_r, grid_z))
    return r, z, grid

     
@jit(nopython=True)
def acceleration(xv, BR, BZ, r, z):
    """"
    Return old v and acceleration due to Lorentz force.
    """
    v = xv[3:]
    B = Bxyz(xv[:3], BR, BZ, r, z)
    qm = 1.60217662e-19/1.6726219e-27
    va = np.zeros(6)
    va[:3] = v
    va[3] = qm*(v[1]*B[2]-v[2]*B[1])
    va[4] = qm*(v[2]*B[0]-v[0]*B[2])
    va[5] = qm*(v[0]*B[1]-v[1]*B[0])
    return va


#@jit(nopython=True)
def RKtrajectory():
    """
    Returns trajectory and velocity of particle in each timestep.
    """
    # Initial conditions 
    h = 1e-11
    times = np.arange(0, .5*6.2e-8, h)
    xv = np.zeros((len(times), 6))
    xv[0] = np.array([-3, 0, 0, 2.6e8, 0, 0])
    for i in xrange(0, len(times)-1):
        xvi = xv[i]
        k1 = h * acceleration(xvi, BR, BZ)
        k2 = h * acceleration(xvi + k1/2., BR, BZ)
        k3 = h * acceleration(xvi + k2/2., BR, BZ)
        k4 = h * acceleration(xvi + k3, BR, BZ)
        xv[i+1] = xvi+1./6.*(k1 + 2*k2 + 2*k3 + k4)
        if outOfDomain(x):
            break
    return xv
    
    
@jit(nopython=True)
def randomPointOnCircle(r):
    """
    Special method to pick uniformly distributed points on a sphere.
    Last on page http://mathworld.wolfram.com/SpherePointPicking.html
    """
    x = np.random.normal()
    y = np.random.normal()
    z = np.random.normal()
    point = np.array([x, y, z])
    point *= r/(x**2 + y**2 + z**2)**.5
    return point

    
def testRandomPointOnCircle():
    points = np.zeros((1000, 3))
    for i in range(1000):
        points[i] = randomPointOnCircle(30)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    plt.show()
    
    
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    
def monteCarlo():
    """
    Launch particles from uniformly random points on a sphere centered at dipole.
    Uniformly random angle as long as it points inside sphere (pick another point on the sphere).
    Random energy from distribution.
    
    Keep moving particles unless they leave spherical boundary. (Investigate particles that don't ever leave.)
    
    At each step add particle to tally of square cell in which it finds itself using modulo and index. (Investigate unique particles vs. total points.)
    
    - Adaptive RK step size
    - Drop x or y position because of axisymmetry
    """
    # Initial conditions 
    boundaryRadius = 20
    v0 = 2.6e8
    maxTime = boundaryRadius*3/v0
    h = 1e-11
    times = np.arange(0, maxTime, h)
    r, z, BR = fieldFunction('Brs.txt')
    r, z, BZ = fieldFunction('Bzs.txt')
    #print(r.shape, np.min(r), np.max(r), BR.shape)
    #(201,) 0.0 20.0 (600, 201)
    #print(z.shape, np.min(z), np.max(z), BZ.shape)
    #(600,) -30.000000000000004 29.900000000000848 (600, 201)
    
    # Run simulations with and without magnetic field
    start = time.time()
    r, z, gridOff = MCRK(boundaryRadius, v0, h, times, r, z, BR, BZ, False)
    print('Time elapsed (s):', time.time()-start)
    start = time.time()
    r, z, gridOn = MCRK(boundaryRadius, v0, h, times, r, z, BR, BZ, True)
    print('Time elapsed (s):', time.time()-start)
    
    # Divide cell counts by volume of cell
    for i in range(len(r)-1):
        for j in range(len(z)-1):
            gridOff[j, i] /= (np.pi*(r[i+1]**2-r[i]**2)*(z[j+1]-z[j]))
            gridOn[j, i] /= (np.pi*(r[i+1]**2-r[i]**2)*(z[j+1]-z[j]))
            
    # Plot results
    themax = np.max([np.max(gridOn), np.max(gridOff)])
    plt.figure(figsize=(9,6))
    plt.subplot(131)
    extent = [0, 20, -30, 30]
    plt.imshow(gridOff, vmin=0, vmax=themax, extent=extent, cmap=plt.cm.jet)
    plt.xlim([0, 5])
    plt.ylim([-15, 15])
    plt.colorbar(label='Particles/m^3')
    
    plt.subplot(132)
    plt.imshow(gridOn, vmin=0, vmax=themax, extent=extent, cmap=plt.cm.jet)
    plt.xlim([0, 5])
    plt.ylim([-15, 15])
    plt.colorbar(label='Particles/m^3')
    plt.subplot(133)
    plt.imshow((BR**2+BZ**2)**.5, extent=extent, cmap=plt.cm.jet)
    plt.xlim([0, 5])
    plt.ylim([-15, 15])
    plt.colorbar(label='|B| (T)')
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.plot(r[:150], gridOff[200:400,:150].sum(axis=0), label='No magnets')
    plt.plot(r[:150], gridOn[200:400,:150].sum(axis=0), label='Magnets')
    plt.ylabel('Particles/m^3')
    plt.xlabel('R (m)')
    plt.legend()
    plt.show()
    
    return r, z, gridOn, gridOff
    
    
@jit(nopython=True)
def MCRK(boundaryRadius, v0, h, times, r, z, BR, BZ, accel):    
    totalGrid = np.zeros(BR.shape)
    for particleNumber in range(10):
        if not particleNumber % 100:
            print(particleNumber)
        particleGrid = np.zeros(BR.shape)
        point1 = randomPointOnCircle(boundaryRadius)
        point2 = randomPointOnCircle(boundaryRadius)
        direction = point2-point1
        direction /= np.linalg.norm(direction)
        
        xv = np.zeros(6)
        xv[:3] = point1
        xv[3:] = direction*v0
        
        for i in range(len(times)):
            nr = nearestIndex(r, (xv[0]**2 + xv[1]**2)**.5)
            nz = nearestIndex(z, xv[2])
            if particleGrid[nz, nr] == 0:
                particleGrid[nz, nr] = 1
            if accel:
                k1 = h * acceleration(xv, BR, BZ, r, z)
                k2 = h * acceleration(xv + k1/2., BR, BZ, r, z)
                k3 = h * acceleration(xv + k2/2., BR, BZ, r, z)
                k4 = h * acceleration(xv + k3, BR, BZ, r, z)
                xv += 1./6.*(k1 + 2*k2 + 2*k3 + k4)
            else:
                xv += np.array([xv[3]*h, xv[4]*h, xv[5]*h, 0, 0, 0])
            distance = np.linalg.norm(xv[:3])
            if distance > boundaryRadius:
                # print('out after', i, 'at distance', distance)
                break
        totalGrid += particleGrid
    return r, z, totalGrid


def plotTrajectory(xv):
    """
    Show particle trajectory in 3D.
    """
    # Plot dipole as ring
    u = np.linspace(0, 2*np.pi, num=100)
    loopRadius = .5
    x = loopRadius*np.sin(u)
    y = loopRadius*np.cos(u)
    z = np.zeros(100)

    # Plot particle trajectories
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection='3d')
    ax.plot(xv[:,0], xv[:,1], xv[:,2])
    ax.plot(x, y, z)

    X = xv[:,0]
    Y = xv[:,1]
    Z = xv[:,2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.axis('equal')
    ax.set_aspect('equal')
    plt.show()


def plotField():
    """
    Show field in 2D.
    """
    field = np.zeros((100,100))
    z = np.linspace(-1, 1, num=100)
    y = np.linspace(-1, 1, num=100)
    for i in range(100):
        for j in range(100):
            field[i,j] = np.linalg.norm(Bxyz([.001, y[i], z[i]]))
    plt.figure()
    plt.imshow(field)
    plt.colorbar()
    plt.show()
    
    
"""Module for 2D interpolation over a rectangular mesh
This module
* provides piecewise constant (nearest neighbour) and bilinear interpolation
* is fast (based on numpy vector operations)
* depends only on numpy
* guarantees that interpolated values never exceed the four nearest neighbours
* handles missing values in domain sensibly using NaN
* is unit tested with a range of common and corner cases
See end of this file for documentation of the mathematical derivation used.
"""

import numpy


@jit(nopython=True)
def interpolate2d(x, y, Z, points, mode='linear', bounds_error=False):
    """Fundamental 2D interpolation routine
    Input
        x: 1D array of x-coordinates of the mesh on which to interpolate
        y: 1D array of y-coordinates of the mesh on which to interpolate
        Z: 2D array of values for each x, y pair
        points: Nx2 array of coordinates where interpolated values are sought
        mode: Determines the interpolation order. Options are
              'constant' - piecewise constant nearest neighbour interpolation
              'linear' - bilinear interpolation using the four
                         nearest neighbours (default)
        bounds_error: Boolean flag. If True (default) an exception will
                      be raised when interpolated values are requested
                      outside the domain of the input data. If False, nan
                      is returned for those values
    Output
        1D array with same length as points with interpolated values
    Notes
        Input coordinates x and y are assumed to be monotonically increasing,
        but need not be equidistantly spaced.
        Z is assumed to have dimension M x N, where M = len(x) and N = len(y).
        In other words it is assumed that the x values follow the first
        (vertical) axis downwards and y values the second (horizontal) axis
        from left to right.
        If this routine is to be used for interpolation of raster grids where
        data is typically organised with longitudes (x) going from left to
        right and latitudes (y) from left to right then user
        interpolate_raster in this module
    """

    # Input checks
    x, y, Z, xi, eta = check_inputs(x, y, Z, points, mode, bounds_error)

    # Identify elements that are outside interpolation domain or NaN
    outside = (xi < x[0]) + (eta < y[0]) + (xi > x[-1]) + (eta > y[-1])
    outside += numpy.isnan(xi) + numpy.isnan(eta)

    inside = np.logical_not(outside)
    xi = xi[inside]
    eta = eta[inside]

    # Find upper neighbours for each interpolation point
    idx = numpy.searchsorted(x, xi, side='left')
    idy = numpy.searchsorted(y, eta, side='left')

    # Internal check (index == 0 is OK)
    msg = ('Interpolation point outside domain. This should never happen. '
           'Please email Ole.Moller.Nielsen@gmail.com')
    if len(idx) > 0:
        if not max(idx) < len(x):
            raise RuntimeError(msg)
    if len(idy) > 0:
        if not max(idy) < len(y):
            raise RuntimeError(msg)

    # Get the four neighbours for each interpolation point
    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = y[idy - 1]
    y1 = y[idy]

    z00 = Z[idx - 1, idy - 1]
    z01 = Z[idx - 1, idy]
    z10 = Z[idx, idy - 1]
    z11 = Z[idx, idy]

    # Coefficients for weighting between lower and upper bounds
    oldset = numpy.seterr(invalid='ignore')  # Suppress warnings
    alpha = (xi - x0) / (x1 - x0)
    beta = (eta - y0) / (y1 - y0)
    numpy.seterr(**oldset)  # Restore

    if mode == 'linear':
        # Bilinear interpolation formula
        dx = z10 - z00
        dy = z01 - z00
        z = z00 + alpha * dx + beta * dy + alpha * beta * (z11 - dx - dy - z00)
    else:
        # Piecewise constant (as verified in input_check)

        # Set up masks for the quadrants
        left = alpha < 0.5
        right = -left
        lower = beta < 0.5
        upper = -lower

        lower_left = lower * left
        lower_right = lower * right
        upper_left = upper * left

        # Initialise result array with all elements set to upper right
        z = z11

        # Then set the other quadrants
        z[lower_left] = z00[lower_left]
        z[lower_right] = z10[lower_right]
        z[upper_left] = z01[upper_left]

    # Self test
    if len(z) > 0:
        mz = numpy.nanmax(z)
        mZ = numpy.nanmax(Z)
        msg = ('Internal check failed. Max interpolated value %.15f '
               'exceeds max grid value %.15f ' % (mz, mZ))
        if not(numpy.isnan(mz) or numpy.isnan(mZ)):
            if not mz <= mZ:
                raise RuntimeError(msg)

    # Populate result with interpolated values for points inside domain
    # and NaN for values outside
    r = numpy.zeros(len(points))
    r[inside] = z
    r[outside] = numpy.nan

    return r


@jit(nopython=True)
def interpolate_raster(x, y, Z, points, mode='linear', bounds_error=False):
    """2D interpolation of raster data
    It is assumed that data is organised in matrix Z as latitudes from
    bottom up along the first dimension and longitudes from west to east
    along the second dimension.
    Further it is assumed that x is the vector of longitudes and y the
    vector of latitudes.
    See interpolate2d for details of the interpolation routine
    """

    # Flip matrix Z up-down to interpret latitudes ordered from south to north
    Z = numpy.flipud(Z)

    # Transpose Z to have y coordinates along the first axis and x coordinates
    # along the second axis
    Z = Z.transpose()

    # Call underlying interpolation routine and return
    res = interpolate2d(x, y, Z, points, mode=mode, bounds_error=bounds_error)
    return res


@jit(nopython=True)
def check_inputs(x, y, Z, points, mode, bounds_error):
    """Check inputs for interpolate2d function
    """

    msg = 'Only mode "linear" and "constant" are implemented. I got %s' % mode
    if mode not in ['linear', 'constant']:
        raise RuntimeError(msg)

    try:
        x = numpy.array(x)
    except Exception as e:
        msg = ('Input vector x could not be converted to numpy array: '
               '%s' % str(e))
        raise Exception(msg)

    try:
        y = numpy.array(y)
    except Exception as e:
        msg = ('Input vector y could not be converted to numpy array: '
               '%s' % str(e))
        raise Exception(msg)

    msg = ('Input vector x must be monotoneously increasing. I got '
           'min(x) == %.15f, but x[0] == %.15f' % (min(x), x[0]))
    if not min(x) == x[0]:
        raise RuntimeError(msg)

    msg = ('Input vector y must be monotoneously increasing. '
           'I got min(y) == %.15f, but y[0] == %.15f' % (min(y), y[0]))
    if not min(y) == y[0]:
        raise RuntimeError(msg)

    msg = ('Input vector x must be monotoneously increasing. I got '
           'max(x) == %.15f, but x[-1] == %.15f' % (max(x), x[-1]))
    if not max(x) == x[-1]:
        raise RuntimeError(msg)

    msg = ('Input vector y must be monotoneously increasing. I got '
           'max(y) == %.15f, but y[-1] == %.15f' % (max(y), y[-1]))
    if not max(y) == y[-1]:
        raise RuntimeError(msg)

    try:
        Z = numpy.array(Z)
        m, n = Z.shape
    except Exception as e:
        msg = 'Z must be a 2D numpy array: %s' % str(e)
        raise Exception(msg)

    Nx = len(x)
    Ny = len(y)
    msg = ('Input array Z must have dimensions %i x %i corresponding to the '
           'lengths of the input coordinates x and y. However, '
           'Z has dimensions %i x %i.' % (Nx, Ny, m, n))
    if not (Nx == m and Ny == n):
        raise RuntimeError(msg)

    # Get interpolation points
    points = numpy.array(points)
    xi = points[:, 0]
    eta = points[:, 1]

    if bounds_error:
        msg = ('Interpolation point %f was less than the smallest value in '
               'domain %f and bounds_error was requested.' % (xi[0], x[0]))
        if xi[0] < x[0]:
            raise Exception(msg)

        msg = ('Interpolation point %f was greater than the largest value in '
               'domain %f and bounds_error was requested.' % (xi[-1], x[-1]))
        if xi[-1] > x[-1]:
            raise Exception(msg)

        msg = ('Interpolation point %f was less than the smallest value in '
               'domain %f and bounds_error was requested.' % (eta[0], y[0]))
        if eta[0] < y[0]:
            raise Exception(msg)

        msg = ('Interpolation point %f was greater than the largest value in '
               'domain %f and bounds_error was requested.' % (eta[-1], y[-1]))
        if eta[-1] > y[-1]:
            raise Exception(msg)

    return x, y, Z, xi, eta


if __name__ == '__main__':
    # xv = RKtrajectory()
    # testRandomPointOnCircle()
    monteCarlo()
