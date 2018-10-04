"""
Test case: straight 1 T magnetic field and 1 GeV proton
  Wolfram alpha: 1 GeV -> 2.6e8 m/s
  r_g = mv_perp/eB = 1.6e-27 * 2.6e8/(1.6e-19 * 1) = 2.6 m
  omega_g = eB/m = 1.6e-19 * 1 / 1.6e-27 = 10^8 rad/s
  tau_g = 2pi/10^8 = 6.2e-8 s for a full turn
"""

from __future__ import print_function
import numpy as np
from numba import njit, prange
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


@njit()
def nearestIndex(array, value):
    """
    Return index of closest matching value in array.
    Data must be sorted.
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    return idx


@njit()   
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


@njit()
def Bxyz(x_vec, BR, BZ, R, Z):
    """
    Given BR, BZ on an R, Z grid, return interpolated B vector at arbitrary position.
    """
    x, y, z = x_vec[:3]
    r = (x**2 + y**2)**.5

    BRinterp, BZinterp = interpolate2d2x(R, Z, BR, BZ, r, z)
    
    Bx = BRinterp * x/r
    By = BRinterp * y/r
    return np.array([Bx, By, BZinterp])


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

     
@njit()
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


#@njit()
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
    
    
@njit()
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
    
    
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        
        
def KEtoSpeed(KE, mass):
    """
    Relativistic formula to convert kinetic energy and mass (eV) to speed (m/s).
    """
    return 299792458*(1-(KE/mass+1)**-2)**.5
    
    
def gyroRadius(v, q, B, m):
    return v*m/(q*B)


def gyroPeriod(q, B, m):
    return 2*np.pi*m/(q*B)
    
    
def monteCarlo():
    """
    Launch particles from uniformly random points on a sphere centered at dipole.
    Uniformly random angle as long as it points inside sphere (pick another point on the sphere).
    Random energy from distribution.
    
    Keep moving particles unless they leave spherical boundary. (Investigate particles that don't ever leave.)
    
    At each step add particle to tally of square cell in which it finds itself using modulo and index. (Investigate unique particles vs. total points.)
    
    - Energy conservation, 
    - End position after 2D scattering as a function of step size: make sure it converges
    - Energy, mass spectrum
    - RK step size: smaller than r_L in currently seen field and smaller than inter-grid spacing
    - Cluster computing
    - Plot of deviation with step size
    - Different geometries/simple dipole for surveys
    """
    # B field in R and Z
    r, z, BR = fieldFunction('Brs_LFhabitat.txt')
    r, z, BZ = fieldFunction('Bzs_LFhabitat.txt')
    import pdb; pdb.set_trace()
    Bmagnitude = (BR**2+BZ**2)**.5
    # Coarseness of the output R, Z flux grid
    reductionFactor = 10
    # Radius of spherical simulation boundary used for launching and exiting
    rLim = 30
    # Particle settings
    m = 1 * 1.6726219e-27 # kg *58 for iron
    q = 1 * 1.60217662e-19 # C *26 for iron
    KE0 = 1e14 # eV
    v0 = KEtoSpeed(KE0, m*5.6095887e35) # m/s
    maxTime = rLim*3/v0
    # RK step size
    h = 1e-11
    # Number of particles to launch
    particles = 100000
    maxSteps = int(maxTime/h)
    
    # Print sanity check info
    print('Simulation duration (s):', maxTime)
    print('Timestep duration (s):', h)
    rReduced = np.linspace(np.min(r), np.max(r), len(r)//reductionFactor)
    print('Initial speed (m/s):', v0)
    print('Time between output grid cell centers with speed v0 (s):', (rReduced[1]-rReduced[0])/v0)
    print('Gyro-orbit in max field: radius {} m, period {} s'.format(gyroRadius(v0, q, np.max(Bmagnitude), m), gyroPeriod(q, np.max(Bmagnitude), m)))
    print('Gyro-orbit in min field: radius {} m, period {} s'.format(gyroRadius(v0, q, np.min(Bmagnitude), m), gyroPeriod(q, np.min(Bmagnitude), m)))
    print('Maximum number of timesteps:', maxSteps)
    
    startingPoints = [randomPointOnSphere(rLim) for _ in range(particles)]
    directions = [randomPointOnSphere(1.) for _ in range(particles)]
    
    # Run simulations without magnetic ield 
    start = time.time()
    rReduced, zReduced, gridOff, _ = MCRK(rLim, v0, q, m, h, maxSteps, r, z, BR, BZ, False, particles, reductionFactor, startingPoints, directions)
    print('Time elapsed (s):', time.time()-start)
    np.save('{}particles_no_accel.npy'.format(particles), [rReduced, zReduced, gridOff])
    
    # Run simulation with magnetic field
    start = time.time()
    rReduced, zReduced, gridOn, trappedOn = MCRK(rLim, v0, q, m, h, maxSteps, r, z, BR, BZ, True, particles, reductionFactor, startingPoints, directions)
    print('Time elapsed (s):', time.time()-start)
    np.save('{}particles_accel.npy'.format(particles), [rReduced, zReduced, gridOn])
    
    # Plot results
    themax = np.max([np.max(gridOn), np.max(gridOff)])
    plt.figure(figsize=(18,12))
    plt.subplot(231)
    plt.title('B field off')
    extent = [np.min(r), np.max(r), np.min(z), np.max(z)]
    plt.imshow(gridOff, vmin=0, vmax=themax, extent=extent, cmap=plt.cm.jet)
    plt.colorbar(label='Particles/$\mathregular{m^3}$')
    plt.ylabel('Z (m)')
    
    plt.subplot(232)
    plt.title('B field on, 5 T contour')
    plt.imshow(gridOn, vmin=0, vmax=themax, extent=extent, cmap=plt.cm.jet)
    plt.colorbar(label='Particles/$\mathregular{m^3}$')
    plt.contour(Bmagnitude, levels=[5.], extent=extent, colors='white')
    
    plt.subplot(233)
    plt.title('(B on - B off)/(B off)')
    gridDifference = 100*(gridOn-gridOff)/gridOff
    bwr = plt.cm.bwr
    bwr.set_bad((0, 0, 0, 1))
    plt.imshow(gridDifference, extent=extent, cmap=bwr, vmin=-100, vmax=100)
    plt.colorbar(label='Percent increase')
    plt.contour(Bmagnitude, levels=[5.], extent=extent, colors='#00FFFF') 
    
    plt.subplot(234)
    plt.title('B field')
    plt.imshow(Bmagnitude, extent=extent, cmap=plt.cm.jet, norm=matplotlib.colors.LogNorm())
    plt.colorbar(label='|B| (T)')
    plt.ylabel('Z (m)')
    plt.xlabel('R (m)')
    
    plt.subplot(235)
    plt.title('B field on: trapped particles')
    plt.imshow(trappedOn, extent=extent, cmap=plt.cm.jet)
    plt.colorbar(label='Particles/$\mathregular{m^3}$')
    plt.contour(Bmagnitude, levels=[5.], extent=extent, colors='white')
    plt.xlabel('R (m)')
    
    plt.subplot(236)
    plt.title('B field on: midplane profile')
    plt.plot(rReduced, gridOn[len(zReduced)//2])
    plt.ylabel('Particles/$\mathregular{m^3}$')
    plt.xlabel('R (m)')
    
    plt.tight_layout()
    plt.show()
    
    
@njit(parallel=True)
def MCRK(rLim, v0, q, m, h, maxSteps, r, z, BR, BZ, accel, particles, reductionFactor, startingPoints, directions):
    totalGrid = np.zeros((BR.shape[0]//reductionFactor, BR.shape[1]//reductionFactor))
    trappedGrid = np.zeros((BR.shape[0]//reductionFactor, BR.shape[1]//reductionFactor))
    qm = q/m#1.60217662e-19/(58*1.6726219e-27)
    rReduced = np.linspace(np.min(r), np.max(r), len(r)//reductionFactor)
    rDelta = rReduced[1]-rReduced[0]
    rReduced += rDelta/2. # Use distance to cell centers to count particles
    zReduced = np.linspace(np.min(z), np.max(z), len(z)//reductionFactor)
    zDelta = zReduced[1]-zReduced[0]
    zReduced += zDelta/2. # Use distance to cell centers to count particles
    for particleNumber in prange(particles):
        if not particleNumber % 1000:
            print(particleNumber)
        particleGrid = np.zeros((BR.shape[0]//reductionFactor, BR.shape[1]//reductionFactor))
        
        # Generate random point and direction
        point1 = startingPoints[particleNumber]
        direction = directions[particleNumber]
        trapped = True
        
        xv = np.zeros(6)
        xv[:3] = point1
        xv[3:] = direction*v0
        va = np.zeros(6)
        
        for i in xrange(maxSteps):
            particleR = (xv[0]**2 + xv[1]**2)**.5
            nearestR = nearestIndex(rReduced, particleR)
            nearestZ = nearestIndex(zReduced, xv[2])
            if particleGrid[nearestZ, nearestR] == 0:
                particleGrid[nearestZ, nearestR] = 1
            if accel:
                k1 = h * acceleration(xv, va, qm, BR, BZ, r, z)
                k2 = h * acceleration(xv + k1/2., va, qm, BR, BZ, r, z)
                k3 = h * acceleration(xv + k2/2., va, qm, BR, BZ, r, z)
                k4 = h * acceleration(xv + k3, va, qm, BR, BZ, r, z)
                xv += 1./6.*(k1 + 2.*k2 + 2.*k3 + k4)
            else:
                xv[0] += xv[3]*h
                xv[1] += xv[4]*h
                xv[2] += xv[5]*h
            
            # If out of bounds
            if (particleR**2+xv[2]**2)**.5 > rLim: 
                trapped = False
                break
        totalGrid += particleGrid
        if trapped:
            trappedGrid += particleGrid
        
    # Divide cell counts by volume of cell
    for i in range(len(rReduced)):
        for j in range(len(zReduced)):
            volume = np.pi*((rReduced[i]+rDelta/2.)**2-(rReduced[i]-rDelta/2.)**2)*zDelta
            totalGrid[j, i] /= volume
            trappedGrid[j, i] /= volume
    
    return rReduced, zReduced, totalGrid, trappedGrid
    

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


@njit()
def interpolate2d(xMarkers, yMarkers, zGrid, x, y):
    """
    2D interpolation for a z array defined on an x, y grid.
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
    

@njit()
def interpolate2d2x(xMarkers, yMarkers, zGrid1, zGrid2, x, y):
    """
    2D interpolation for two z arrays defined on the same x, y grid.
    Source: http://supercomputingblog.com/graphics/coding-bilinear-interpolation
    """
    xi1, xi2 = boundingIndices(xMarkers, x)
    yi1, yi2 = boundingIndices(yMarkers, y)
        
    # If out of bounds, return closest point on boundary
    if xi1 == xi2 or yi1 == yi2:
        return zGrid1[yi1, xi1], zGrid2[yi1, xi1]
        
    x1, x2 = xMarkers[xi1], xMarkers[xi2]
    y1, y2 = yMarkers[yi1], yMarkers[yi2]
    c1, c2 = (x2 - x)/(x2 - x1), (x - x1)/(x2 - x1)
    R11 = c1*zGrid1[yi1, xi1] + c2*zGrid1[yi1, xi2]
    R21 = c1*zGrid1[yi2, xi1] + c2*zGrid1[yi2, xi2]
    R12 = c1*zGrid2[yi1, xi1] + c2*zGrid2[yi1, xi2]
    R22 = c1*zGrid2[yi2, xi1] + c2*zGrid2[yi2, xi2]
    return ((y2 - y)/(y2 - y1))*R11 + ((y - y1)/(y2 - y1))*R21, \
           ((y2 - y)/(y2 - y1))*R12 + ((y - y1)/(y2 - y1))*R22
    

if __name__ == '__main__':
    # xv = RKtrajectory()
    monteCarlo()
