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
from scipy.interpolate import griddata
import math
import time


def dipoleR(x_vec):
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
    
    
def dipoleRZ(x_vec):
    """
    Dipole field vector at given R, Z coordinates.
    Currently not working.
    """
    x = x_vec[0] or 1e-3
    y = x_vec[1] or 1e-3
    z = x_vec[2] or 1e-3
    r = (x**2 + y**2 + z**2)**.5
    loopRadius = .5
    BonAxis = 10
    mu_0 = 1.25663706e-6
    I = 2*loopRadius*BonAxis/mu_0
    mu = I*np.pi*loopRadius**2 # magnetic moment
    theta = np.arccos(z/r)
    phi = np.arctan(y/x)
    Bx = mu_0/(4*np.pi) * mu/r**3 * (2*np.cos(theta)*np.sin(theta)*np.cos(phi)+np.sin(theta)*np.cos(theta)*np.cos(phi))
    By = mu_0/(4*np.pi) * mu/r**3 * (2*np.cos(theta)*np.sin(theta)*np.sin(phi)+np.sin(theta)*np.cos(theta)*np.sin(phi))
    Bz = mu_0/(4*np.pi) * mu/r**3 * (2*np.cos(theta)*np.cos(theta)-np.sin(theta)*np.sin(theta))
    return np.array([Bx, By, Bz])


@jit(nopython=True)
def nearestIndex(array, value):
    """
    Return index of closest matching value in array.
    Data must be sorted.
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    return idx


@jit(nopython=True)   
def boundingIndices(array, value):
    """
    Return index of closest and second closest matching values in array.
    """
    ni = nearestIndex(array, value)
    if ni == 0 and value < array[ni]:
        return 0, 0
    elif ni == len(array)-1 and value > array[ni]:
        return ni, ni
    elif math.fabs(value-array[ni+1]) < math.fabs(value-array[ni-1]):
        return ni, ni+1
    else:
        return ni-1, ni


@jit(nopython=True)
def Bxyz(x_vec, BR, BZ, R, Z):
    """
    Given BR, BZ on an R, Z grid, return interpolated B vector at arbitrary position.
    """
    x, y, z = x_vec[:3]
    r = (x**2 + y**2)**.5

    BRinterp = interpolate2d(R, Z, BR, r, z)
    BZinterp = interpolate2d(R, Z, BZ, r, z)
    
    Bx = BRinterp * x/r
    By = BRinterp * y/r
    return np.array([Bx, By, BZinterp])


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
def acceleration(xv, va, qm, BR, BZ, r, z):
    """"
    Return old v and acceleration due to Lorentz force.
    va should be a vector of zeros.
    """
    B = Bxyz(xv[:3], BR, BZ, r, z)
    va[:3] = xv[3:]
    va[3] = qm*(xv[4]*B[2]-xv[5]*B[1])
    va[4] = qm*(xv[5]*B[0]-xv[3]*B[2])
    va[5] = qm*(xv[3]*B[1]-xv[4]*B[0])
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
def randomPointOnSphere(r):
    """
    Special method to pick uniformly distributed points on a sphere.
    Source: http://mathworld.wolfram.com/SpherePointPicking.html (last method)
    """
    x = np.random.normal()
    y = np.random.normal()
    z = np.random.normal()
    point = np.array([x, y, z])
    point *= r/(x**2 + y**2 + z**2)**.5
    return point
    
    
@jit(nopython=True)
def randomPointOnCylinder(radius, length):
    """
    Special method to pick uniformly distributed points on a cylinder.
    Source: http://mathworld.wolfram.com/DiskPointPicking.html
    """
    cylinderArea = (2*np.pi*radius)*(length*2)
    diskArea = np.pi*radius**2
    picker = np.random.uniform(0, cylinderArea + 2*diskArea)
    angle = np.random.uniform(0, 2*np.pi)
    if picker < cylinderArea:
        x = radius*math.cos(angle)
        y = radius*math.sin(angle)
        z = np.random.uniform(-length, length)
    elif cylinderArea < picker < cylinderArea + 2*diskArea:
        r = np.sqrt(np.random.uniform(0, radius**2))
        x = r*math.cos(angle)
        y = r*math.sin(angle)
        z = length
        if diskArea < picker - cylinderArea < 2*diskArea:
            z = -length
    return np.array([x, y, z])

    
def testRandomPointOnCylinder():
    points = np.zeros((1000, 3))
    for i in range(1000):
        points[i] = randomPointOnCylinder(10, 20)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    plt.show()
    
    
def testRandomLinesOnCylinder():
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection='3d')
    for i in range(1000):
        point1 = randomPointOnCylinder(10, 20)
        point2 = randomPointOnCylinder(10, 20)
        x, y, z = zip(point1, point2)
        ax.plot(x, y, z)
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
    zLim = 20
    rLim = 10
    v0 = 2.6e8
    maxTime = zLim*3/v0
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
    r, z, gridOff = MCRK(zLim, rLim, v0, h, times, r, z, BR, BZ, False)
    print('Time elapsed (s):', time.time()-start)
    start = time.time()
    r, z, gridOn = MCRK(zLim, rLim, v0, h, times, r, z, BR, BZ, True)
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
    # plt.xlim([0, 5])
    # plt.ylim([-15, 15])
    plt.colorbar(label='Particles/m^3')
    
    plt.subplot(132)
    plt.imshow(gridOn, vmin=0, vmax=themax, extent=extent, cmap=plt.cm.jet)
    # plt.xlim([0, 5])
    # plt.ylim([-15, 15])
    plt.colorbar(label='Particles/m^3')
    plt.subplot(133)
    plt.imshow((BR**2+BZ**2)**.5, extent=extent, cmap=plt.cm.jet, vmax=5)
    # plt.xlim([0, 5])
    # plt.ylim([-15, 15])
    plt.colorbar(label='|B| (T)')
    plt.tight_layout()
    plt.show()
    
    # plt.figure()
    # plt.plot(r[:150], gridOff[200:400,:150].sum(axis=0), label='No magnets')
    # plt.plot(r[:150], gridOn[200:400,:150].sum(axis=0), label='Magnets')
    # plt.ylabel('Particles/m^3')
    # plt.xlabel('R (m)')
    # plt.legend()
    # plt.show()
    
    return r, z, gridOn, gridOff
    
    
@jit(nopython=True)
def MCRK(zLim, rLim, v0, h, times, r, z, BR, BZ, accel):    
    totalGrid = np.zeros(BR.shape)
    qm = 1.60217662e-19/1.6726219e-27
    for particleNumber in range(1000):
        if not particleNumber % 100:
            print(particleNumber)
        particleGrid = np.zeros(BR.shape)
        point1 = randomPointOnCylinder(rLim, zLim)
        point2 = randomPointOnCylinder(rLim, zLim)
        direction = point2-point1
        direction /= np.linalg.norm(direction)
        
        xv = np.zeros(6)
        xv[:3] = point1
        xv[3:] = direction*v0
        va = np.zeros(6)
        
        for _ in times:
            nr = nearestIndex(r, (xv[0]**2 + xv[1]**2)**.5)
            nz = nearestIndex(z, xv[2])
            if particleGrid[nz, nr] == 0:
                particleGrid[nz, nr] = 1
            if accel:
                k1 = h * acceleration(xv, va, qm, BR, BZ, r, z)
                k2 = h * acceleration(xv + k1/2., va, qm, BR, BZ, r, z)
                k3 = h * acceleration(xv + k2/2., va, qm, BR, BZ, r, z)
                k4 = h * acceleration(xv + k3, va, qm, BR, BZ, r, z)
                xv += 1./6.*(k1 + 2*k2 + 2*k3 + k4)
            else:
                xv += np.array([xv[3]*h, xv[4]*h, xv[5]*h, 0, 0, 0])
            
            # If out of bounds
            if math.fabs(xv[2]) > zLim or (xv[0]**2+xv[1]**2)**.5 > rLim:
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


def testInterpolate2d():
    xMarkers = np.arange(-5, 5, 0.25)
    yMarkers = np.arange(-2.5, 2.5, 0.25)
    X, Y = np.meshgrid(xMarkers, yMarkers)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, antialiased=False, alpha=0.5)
    xs = [np.random.uniform(-6, 6) for i in range(100)]
    ys = [np.random.uniform(-4, 4) for i in range(100)]
    zs = [interpolate2d(xMarkers, yMarkers, Z, x, y) for (x, y) in zip(xs, ys)]
    ax.scatter(xs, ys, zs, color='red')
    plt.show()


@jit(nopython=True)
def interpolate2d(xMarkers, yMarkers, zGrid, x, y):
    """
    Source: http://supercomputingblog.com/graphics/coding-bilinear-interpolation
    """
    xi1, xi2 = boundingIndices(xMarkers, x)
    yi1, yi2 = boundingIndices(yMarkers, y)
        
    # If out of bounds, return closest point on boundary
    if xi1 == xi2 or yi1 == yi2:
        return zGrid[yi1, xi1]
        
    x1, x2 = xMarkers[xi1], xMarkers[xi2]
    y1, y2 = yMarkers[yi1], yMarkers[yi2]
    R1 = ((x2 - x)/(x2 - x1))*zGrid[yi1, xi1] + ((x - x1)/(x2 - x1))*zGrid[yi1, xi2]
    R2 = ((x2 - x)/(x2 - x1))*zGrid[yi2, xi1] + ((x - x1)/(x2 - x1))*zGrid[yi2, xi2]
    return ((y2 - y)/(y2 - y1))*R1 + ((y - y1)/(y2 - y1))*R2
    

if __name__ == '__main__':
    # xv = RKtrajectory()
    monteCarlo()
