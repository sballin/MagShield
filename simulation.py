from __future__ import print_function
import numpy as np
from numba import njit, prange
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate, scipy.integrate
import glob
import math
import time


@njit()
def nearestIndex(array, value):
    """
    Return index of closest matching value in array.
    Data must be sorted but can be irregularly spaced.
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    return idx
    
    
@njit()
def nearestIndexNoSearch(start, stop, step, value):
    """
    TODO: Why is performance here bad?
    """
    if value < start:
        return 0
    elif value > stop:
        return int((stop - start)/step)
    index = int((value - start)/step)
    if value - (start + step*index) > step/2:
        index += 1
    return index


@njit()
def boundingIndices(start, stop, step, value):
    """
    Return indices of bounding values in a regularly spaced array.
    """
    if value < start:
        return 0, 0
    elif value > stop:
        stopIndex = int((stop - start)/step)
        return stopIndex, stopIndex
    lowerIndex = int((value - start)/step)
    return lowerIndex, lowerIndex+1

    
@njit()
def cross(a, b):
    """
    Numba does not support the np.cross function.
    """
    return np.array([a[1]*b[2] - a[2]*b[1],
                     a[2]*b[0] - a[0]*b[2],
                     a[0]*b[1] - a[1]*b[0]])


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
    
    
@njit()
def KEtoSpeed(KE, mass):
    """
    Relativistic formula to convert kinetic energy and mass (eV) to speed (m/s).
    """
    return 299792458*(1-(KE/mass+1)**-2)**.5
        
    
@njit()
def gamma(v):
    c = 299792458
    return 1/(1-(v[0]**2+v[1]**2+v[2]**2)/c**2)**.5
    
    
@njit()    
def gyroRadius(v, q, B, m):
    return v*m/(q*B)


@njit()
def gyroPeriod(q, B, m):
    return 2*np.pi*m/(q*B)


def binnedAverage(x, y, bins=20):
    """
    Define a line from a messy cloud of data using the average of each bin.
    """
    xbins, step = np.linspace(np.min(x), np.max(x), num=bins, retstep=True)
    xbins = (xbins + step/2)[:-1]
    emptyBins = []
    ymeans = []
    for xbi, xb in enumerate(xbins):
        ytotal = 0
        ycount = 0
        for y_i, y_ in enumerate(y):
            if xb - step/2 < x[y_i] < xb + step/2:
                ytotal += y_
                ycount += 1
        if ycount >= 1:
            ymeans.append(ytotal/ycount)
        else:
            emptyBins.append(xbi)
    xbins = np.delete(xbins, emptyBins)
    return xbins, np.array(ymeans)
    
    
@njit()
def interpolate2Dtwice(xMarkers, yMarkers, zGrid1, zGrid2, x, y):
    """
    Linear 2D interpolation for two z arrays defined on the same x, y grid.
    Why twice? This method has is called many, many times.
    Source: http://supercomputingblog.com/graphics/coding-bilinear-interpolation
    """
    xi1, xi2 = boundingIndices(xMarkers[0], xMarkers[-1], xMarkers[1]-xMarkers[0], x)
    yi1, yi2 = boundingIndices(yMarkers[0], yMarkers[-1], yMarkers[1]-yMarkers[0], y)
        
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


def fieldGrid(filename):
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
    grid = scipy.interpolate.griddata(vals[:, :2], vals[:, 2], (grid_r, grid_z))
    return r, z, grid
    
    
def getElementData(element):
    """
    Experimental data obtained from http://lpsc.in2p3.fr/crdb
    """
    energies = []
    fluxes = []
    for filename in glob.glob('GCRdata/{}/*.dat'.format(element)):
        with open(filename, 'r') as f:
            lines = f.read()
        lines = lines.split('\n')
        for line in lines:
            # Ignore comments and newlines
            if not '#' in line and len(line) > 1:
                data = [float(d) for d in line.split()]
                # Ignore 0 energies which cause log problems
                if data[0] > 0:
                    energies.append(data[0])
                    fluxes.append(data[3])
    return np.array(energies), np.array(fluxes)


def getElementFunction(element):
    """
    Return min energy, max energy, and function (all log10).
    """
    energies, fluxes = getElementData(element)
    fluxes = energies*fluxes
    logEnergies = np.log10(energies)
    logFluxes = np.log10(fluxes)
    bins, samples = binnedAverage(logEnergies, logFluxes, bins=10)
    f = scipy.interpolate.interp1d(bins, samples, kind='cubic')
    return min(bins), max(bins), f
    
    
def qmAndVelocitySpectrum(particleCount):
    elements = ['H', 'He', 'C', 'O', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Fe']
    uToKg = 1.660539e-27 # kg
    uToEV = 931.4941e6 # eV/c^2
    mDict = {'H': 1.008, 'He': 4.0026, 'C': 12.011, 'O': 15.999, 'Mg': 24.305,
             'Si': 28.085, 'S': 32.06, 'Ar': 39.948, 'Ca': 40.078, 'Fe': 55.845}
    e = 1.60217662e-19
    qDict = {'H': 1*e, 'He': 2*e, 'C': 6*e, 'O': 8*e, 'Mg': 12*e, 
             'Si': 14*e, 'S': 16*e, 'Ar': 18*e, 'Ca': 20*e, 'Fe': 26*e}
    elemBins = []
    elemPDFs = []
    elemIntegrals = []
    for element in elements:
        fmin, fmax, f = getElementFunction(element)
        fx = np.linspace(fmin, fmax, 100)
        fy = [10**f(fxi) for fxi in fx]
        fy /= np.sum(fy)
        
        elemBins.append(fx)
        elemPDFs.append(fy)
        elemIntegrals.append(scipy.integrate.quad(lambda x: x*10**f(x), fmin, fmax)[0])

    elemIntegrals /= np.sum(elemIntegrals)
    elemChoices = [np.random.choice(elements, p=elemIntegrals) for _ in range(particleCount)]
    qm = [qDict[e]/(mDict[e]*uToKg) for e in elemChoices]
    elemIndices = [elements.index(e) for e in elemChoices]
    v = [KEtoSpeed(1e9*10**np.random.choice(elemBins[e], p=elemPDFs[e]), 
                   mDict[elements[e]]*uToEV) for e in elemIndices]
    return qm, v
    
    
@njit()
def BxyzInterpolated(x_vec, BR, BZ, R, Z):
    """
    Given BR, BZ on an R, Z grid, return interpolated B vector at arbitrary position.
    """
    x, y, z = x_vec[:3]
    r = (x**2 + y**2)**.5

    BRinterp, BZinterp = interpolate2Dtwice(R, Z, BR, BZ, r, z)
    
    Bx = BRinterp * x/r
    By = BRinterp * y/r
    return np.array([Bx, By, BZinterp])

     
@njit()
def acceleration(xv, va, qm, BR, BZ, r, z):
    """"
    Return old v and acceleration due to Lorentz force.
    va should be a vector of zeros.
    """
    B = BxyzInterpolated(xv[:3], BR, BZ, r, z)
    va[:3] = xv[3:]
    va[3] = qm*(xv[4]*B[2]-xv[5]*B[1])
    va[4] = qm*(xv[5]*B[0]-xv[3]*B[2])
    va[5] = qm*(xv[3]*B[1]-xv[4]*B[0])
    return va
    
    
@njit()
def accelerationConstantB(xv, B, qm):
    va = np.zeros(6)
    va[:3] = xv[3:]
    u = qm*cross(xv[3:], B)
    va[3:] = u/gamma(va[:3])
    return va
    
    
@njit()
def BBnext(x, v, B, E, qm, dt):
    """
    Boris Buneman method. vNext is actually v_{n+1/2}, so need x[0] at t = 1/2 delta t. 
    
    Sources
        Boris algorithm: https://www.particleincell.com/2011/vxb-rotation/
        Relativistic correction to Boris algorithm: https://arxiv.org/abs/1710.09164
    """
    vMinus = v + qm*E*0.5*dt
    t = qm*B*0.5*dt/gamma(vMinus)
    tMagnitudeSquared = t[0]*t[0] + t[1]*t[1] + t[2]*t[2]
    s = 2*t/(1+tMagnitudeSquared)
    vPrime = vMinus + cross(vMinus, t)
    vPlus = vMinus + cross(vPrime, s)
    vNext = vPlus + qm*E*0.5*dt
    return x + vNext*dt, vNext
    
        
@njit()    
def RKnext(x, v, B, E, qm, dt):
    """
    Returns trajectory and velocity of particle in next timestep.
    """
    xvi = np.zeros(6)
    xvi[:3] = x
    xvi[3:] = v
    k1 = dt * accelerationConstantB(xvi,         B, qm)
    k2 = dt * accelerationConstantB(xvi + k1/2., B, qm)
    k3 = dt * accelerationConstantB(xvi + k2/2., B, qm)
    k4 = dt * accelerationConstantB(xvi + k3,    B, qm)
    xvNext = xvi+1./6.*(k1 + 2*k2 + 2*k3 + k4) # [x,v] + dt*[v,a]
    return xvNext[:3], xvNext[3:]

    
@njit(parallel=True)
def monteCarloRun(rLim, qms, vs, dt, r, z, BR, BZ, accel, particles, reductionFactor, startingPoints, directions):
    totalGrid = np.zeros((BR.shape[0]//reductionFactor, BR.shape[1]//reductionFactor))
    trappedGrid = np.zeros((BR.shape[0]//reductionFactor, BR.shape[1]//reductionFactor))
    rReduced = np.linspace(np.min(r), np.max(r), len(r)//reductionFactor)
    rDelta = rReduced[1]-rReduced[0]
    rReduced += rDelta/2. # Use distance to cell centers to count particles
    zReduced = np.linspace(np.min(z), np.max(z), len(z)//reductionFactor)
    zDelta = zReduced[1]-zReduced[0]
    zReduced += zDelta/2. # Use distance to cell centers to count particles
    
    habitatCrossings = 0
    GDTcrossings = 0
    
    for particleNumber in prange(particles):
        if not particleNumber % 1000:
            print(particleNumber)
            
        qm = qms[particleNumber]
        v0 = vs[particleNumber]
        timeElapsed = 0
        maxTime = rLim * 3 / v0
        maxSteps = int(maxTime / dt)
        particleGrid = np.zeros((BR.shape[0]//reductionFactor, BR.shape[1]//reductionFactor))
        crossedHabitat = 0
        crossedGDT = 0
        
        # Generate random point and direction
        point1 = startingPoints[particleNumber]
        direction = directions[particleNumber]
        trapped = True
        
        xv = np.zeros(6)
        xv[:3] = point1
        xv[3:] = direction*v0
        va = np.zeros(6)

        for i in range(maxSteps):
            particleR = (xv[0]**2 + xv[1]**2)**.5
            nearestR = nearestIndex(rReduced, particleR)
            nearestZ = nearestIndex(zReduced, xv[2])
            if particleGrid[nearestZ, nearestR] == 0:
                particleGrid[nearestZ, nearestR] = 1
            if accel:
                k1 = dt * acceleration(xv, va, qm, BR, BZ, r, z)
                k2 = dt * acceleration(xv + k1/2., va, qm, BR, BZ, r, z)
                k3 = dt * acceleration(xv + k2/2., va, qm, BR, BZ, r, z)
                k4 = dt * acceleration(xv + k3, va, qm, BR, BZ, r, z)
                xv += 1./6.*(k1 + 2.*k2 + 2.*k3 + k4)
            else:
                xv[0] += xv[3]*dt
                xv[1] += xv[4]*dt
                xv[2] += xv[5]*dt
                
            if 10 < particleR < 14 and -2 < xv[2] < 2:
                crossedHabitat = 1
            if -14 < xv[2] < 14 and particleR < 5:
                crossedGDT = 1
            # If out of bounds
            if (particleR**2+xv[2]**2)**.5 > rLim: 
                trapped = False
                break
        totalGrid += particleGrid
        if trapped:
            trappedGrid += particleGrid
        habitatCrossings += crossedHabitat
        GDTcrossings += crossedGDT
        
    # Divide cell counts by volume of cell
    totalGridUnscaled = totalGrid.copy()
    trappedGridUnscaled = trappedGrid.copy()
    for i in range(len(rReduced)):
        for j in range(len(zReduced)):
            volume = np.pi*((rReduced[i]+rDelta/2.)**2-(rReduced[i]-rDelta/2.)**2)*zDelta
            totalGrid[j, i] /= volume
            trappedGrid[j, i] /= volume
    
    return rReduced, zReduced, totalGrid, trappedGrid, habitatCrossings, GDTcrossings, totalGridUnscaled, trappedGridUnscaled
    
    
def monteCarloSurvey():
    """
    Launch particles from uniformly random points on a sphere centered at dipole.
    Uniformly random angle as long as it points inside sphere (pick another point on the sphere).
    Random energy from distribution.
    
    At each step add particle to tally of square cell in which it finds itself using modulo and index. (Investigate unique particles vs. total points.)
    """
    # B field in R and Z
    r, z, BR = fieldGrid('fields/Brs_oct30.txt')
    _, _, BZ = fieldGrid('fields/Bzs_oct30.txt')
    Bmagnitude = (BR**2+BZ**2)**.5

    # Coarseness of the output R, Z flux grid
    reductionFactor = 1
    # Radius of spherical simulation boundary used for launching and exiting
    rLim = 30
    # Particle settings
    dt = 1e-10; print('dt (s):', dt)

    # Number of particles to launch
    particles = 100000

    qms, vs = qmAndVelocitySpectrum(particles)

    startingPoints = [randomPointOnSphere(rLim) for _ in range(particles)]
    directions = [randomPointOnSphere(1.) for _ in range(particles)]

    # Run simulations without magnetic field
    start = time.time()
    rReduced, zReduced, gridOff, _,  habitatCrossingsOff, GDTcrossingsOff, gridOffUnscaled, _ = monteCarloRun(rLim, qms, vs, dt, r, z, BR, BZ, False, particles, reductionFactor, startingPoints, directions)
    print('Time elapsed (s):', time.time()-start)
    np.save('cache/{}particles_no_accel.npy'.format(particles), [rReduced, zReduced, gridOff])
    
    # Run simulation with magnetic field
    start = time.time()
    _, _, gridOn, trappedOn, habitatCrossingsOn, GDTcrossingsOn, gridOnUnscaled, trappedOnUnscaled = monteCarloRun(rLim, qms, vs, dt, r, z, BR, BZ, True, particles, reductionFactor, startingPoints, directions)
    print('Time elapsed (s):', time.time()-start)
    np.save('cache/{}particles_accel.npy'.format(particles), [rReduced, zReduced, gridOn])
    try:
        print('GDT crossing change: {}%'.format(round(100*(GDTcrossingsOn-GDTcrossingsOff)/GDTcrossingsOff, 3)))
        print('Habitat crossing change: {}%'.format(round(100*(habitatCrossingsOn-habitatCrossingsOff)/habitatCrossingsOff, 3)))
    except Exception as e:
        print(e)
    
    # plotDiff(r, z, Bmagnitude, gridOn, gridOff)
    plot6panel(r, z, rReduced, zReduced, Bmagnitude, gridOnUnscaled, gridOffUnscaled, trappedOnUnscaled)


def plotDiff(r, z, Bmagnitude, gridOn, gridOff):
    plt.figure(figsize=(6, 6))
    plt.title('Flux change')
    gridDifference = 100 * (gridOn - gridOff) / gridOff
    bwr = plt.cm.bwr
    bwr.set_bad((0, 0, 0, 1))
    extent = [np.min(r), np.max(r), np.min(z), np.max(z)]
    plt.imshow(gridDifference, extent=extent, cmap=bwr, vmin=-100, vmax=100)
    plt.colorbar(label='Percent increase')
    plt.contour(Bmagnitude, levels=[5.], extent=extent, colors='#00FFFF')
    plt.xlabel('R (m)')
    plt.ylabel('Z (m)')
    plt.show()


def plot6panel(r, z, rReduced, zReduced, Bmagnitude, gridOn, gridOff, trappedOn):
    themax = np.max([np.max(gridOn), np.max(gridOff)])
    plt.figure(figsize=(18, 12))
    plt.subplot(231)
    plt.title('B field off')
    extent = [np.min(r), np.max(r), np.min(z), np.max(z)]
    plt.imshow(gridOff, vmin=0, extent=extent, cmap=plt.cm.jet)
    plt.colorbar(label='Particles/$\mathregular{m^3}$')
    plt.ylabel('Z (m)')

    plt.subplot(232)
    plt.title('B field on, 5 T contour')
    plt.imshow(gridOn, vmin=0, extent=extent, cmap=plt.cm.jet)
    plt.colorbar(label='Particles/$\mathregular{m^3}$')
    plt.contour(Bmagnitude, levels=[5.], extent=extent, colors='white')

    plt.subplot(233)
    plt.title('(B on - B off)/(B off)')
    gridDifference = 100 * (gridOn - gridOff) / gridOff
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
    plt.plot(rReduced, gridOn[len(zReduced) // 2])
    plt.ylabel('Particles/$\mathregular{m^3}$')
    plt.xlabel('R (m)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    monteCarloSurvey()
