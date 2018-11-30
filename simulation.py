import numpy as np
from numba import njit, prange
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate, scipy.integrate
import glob
import math
import time


settings = {
                'fieldFile':          'nov20.txt',
                'rLim':               50,
                'numParticles':       10000,
                'steppingMethod':     2, # 1: Runge-Kutta, 2: Boris-Buneman
                'fluxGridCoarseness': 1,
                'qmPrescribed':       None, #1.6021766208e-19/1.672621898e-27
                'v0Prescribed':       None  #262326089.7
           }


@njit()
def nearestIndex(array, value):
    """
    Return index of closest matching value in array.
    Data must be sorted but can be irregularly spaced.
    """
    idx = np.searchsorted(array, value, side='left')
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    return idx
    
    
@njit()
def nearestIndexNoSearch(start, stop, step, value):
    """
    TODO: Why is performance here bad? Try specifying signature.
    """
    if value < start:
        index =  0
    elif value > stop:
        index = int((stop - start)/step)
    else:
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
    
    
def randomDirectionCos(negativeSurfaceNormal):
    """
    Tutorial: https://www.particleincell.com/2015/cosine-distribution
    Paper: https://molflow.web.cern.ch/sites/molflow.web.cern.ch/files/1-s2.0-S0042207X02001732-main.pdf
    Geant implementation: https://github.com/Geant4/geant4/blob/6aa23be5171b125c3363b5a4cfa00a57e524598b/source/event/src/G4SPSAngDistribution.cc#L375
    """
    # Pick theta with P(theta) = cos(theta) over [0, pi/2)
    theta = np.arcsin(np.random.random()**.5) 
    # Pick a random direction from which to march outward from the surface normal tip
    normalShift = randomPointOnSphere(1)
    D = cross(normalShift, negativeSurfaceNormal)
    D /= np.linalg.norm(D)
    # March outward by an amount d = |negativeSurfaceNormal| tan(theta)
    D *= np.linalg.norm(negativeSurfaceNormal)*np.tan(theta)
    D += negativeSurfaceNormal
    # Normalize the final direction vector
    return D/np.linalg.norm(D)
    
    
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
def RKva(xv, qm, BR, BZ, r, z):
    va = np.zeros(6)
    va[:3] = xv[3:]
    B = BxyzInterpolated(xv[:3], BR, BZ, r, z)
    va[3:] = qm*cross(xv[3:], B)/gamma(va[:3])
    return va
        
        
@njit()
def RKnext(x, v, qm, BR, BZ, r, z, dt):
    """
    Returns trajectory and velocity of particle in next timestep.
    """
    xv = np.concatenate((x, v))
    k1 = dt * RKva(xv,        qm, BR, BZ, r, z)
    k2 = dt * RKva(xv + k1/2, qm, BR, BZ, r, z)
    k3 = dt * RKva(xv + k2/2, qm, BR, BZ, r, z)
    k4 = dt * RKva(xv + k3,   qm, BR, BZ, r, z)
    xv += 1/6*(k1 + 2*k2 + 2*k3 + k4) # [x,v] + dt*[v,a]
    return xv[:3], xv[3:] 
    
    
@njit()
def BBnext(x, v, qm, B, E, dt):
    """
    Boris Buneman method. vNext is actually v_{n+1/2}, so need x[0] at t = 1/2 delta t. 
    
    Sources
        Boris algorithm: https://www.particleincell.com/2011/vxb-rotation/
        Relativistic correction to Boris algorithm: https://arxiv.org/abs/1710.09164
    """
    vMinus = v + qm*E*0.5*dt
    t = qm*B*0.5*dt/gamma(vMinus)
    tMagnitudeSquared = t[0]**2 + t[1]**2 + t[2]**2
    s = 2*t/(1+tMagnitudeSquared)
    vPrime = vMinus + cross(vMinus, t)
    vPlus = vMinus + cross(vPrime, s)
    vNext = vPlus + qm*E*0.5*dt
    return x + vNext*dt, vNext

    
@njit(parallel=True)
def monteCarloRun(startingPoints, qms, vs, directions, BR, BZ, r, z, rLim, fluxGridCoarseness, steppingMethod):    
    """
    Launch particles from uniformly random points on a sphere centered at dipole.
    Uniformly random angle as long as it points inside sphere (pick another point on the sphere).
    Random energy from distribution.
    
    At each step add particle to tally of square cell in which it finds itself using modulo and index. (Investigate unique particles vs. total points.)
    """
    totalGrid = np.zeros((BR.shape[0]//fluxGridCoarseness, BR.shape[1]//fluxGridCoarseness))
    trappedGrid = np.zeros((BR.shape[0]//fluxGridCoarseness, BR.shape[1]//fluxGridCoarseness))
    rReduced = np.linspace(np.min(r), np.max(r), len(r)//fluxGridCoarseness)
    rDelta = rReduced[1]-rReduced[0]
    rReduced += rDelta/2. # Use distance to cell centers to count particles
    zReduced = np.linspace(np.min(z), np.max(z), len(z)//fluxGridCoarseness)
    zDelta = zReduced[1]-zReduced[0]
    zReduced += zDelta/2. # Use distance to cell centers to count particles
    
    habitatCrossings = 0
    GDTcrossings = 0
    detectorCounts = np.zeros(14)
    
    gridStep = r[1]-r[0]
    
    numParticles = len(qms)
    for particleNumber in prange(numParticles):
        if particleNumber % (numParticles/10) == 0:
            print(particleNumber)
            
        qm = qms[particleNumber]
        v0 = vs[particleNumber]
        dt = (r[1]-r[0])/v0/2
        maxTime = rLim * 3 / v0
        maxSteps = int(maxTime / dt)
        particleGrid = np.zeros((BR.shape[0]//fluxGridCoarseness, BR.shape[1]//fluxGridCoarseness))
        crossedHabitat = 0
        crossedGDT = 0
        particleDetectorCounts = np.zeros(14)
        
        # Generate random point and direction
        point1 = startingPoints[particleNumber]
        direction = directions[particleNumber]
        noAccelStep = 0.99*gridStep*direction
        trapped = True
        
        x = point1.copy() # copy is important... 
        v = direction*v0
        E = np.zeros(3)
        
        if steppingMethod == 2:
            x, _ = RKnext(x, v, qm, BR, BZ, r, z, dt/2)

        for i in range(maxSteps):
            # Count crossings
            particleR = (x[0]**2 + x[1]**2)**.5
            nearestR = nearestIndex(rReduced, particleR)
            nearestZ = nearestIndex(zReduced, x[2])
            particleGrid[nearestZ, nearestR] = 1
            if 9.7 < particleR < 12.3 and -1.3 < x[2] < 1.3:
                crossedHabitat = 1
            if -14 < x[2] < 14 and particleR < 5:
                crossedGDT = 1
            # Will's detectors
            # for det in range(14):
            #     vd = (x[0] - det*1.4, x[1], x[2])
            #     if (vd[0]**2+vd[1]**2+vd[2]**2)**.5 < 0.5:
            #         particleDetectorCounts[det] = 1
                
            # Step
            if steppingMethod == 0:
                x += noAccelStep
            elif steppingMethod == 1:
                x, v = RKnext(x, v, qm, BR, BZ, r, z, dt)
            elif steppingMethod == 2:
                B = BxyzInterpolated(x, BR, BZ, r, z)
                x, v = BBnext(x, v, qm, B, E, dt)
                
            # Stop stepping if out of bounds
            if (particleR**2+x[2]**2)**.5 > rLim + .001: 
                trapped = False
                break
        detectorCounts += particleDetectorCounts
        totalGrid += particleGrid
        if trapped:
            trappedGrid += particleGrid
        habitatCrossings += crossedHabitat
        GDTcrossings += crossedGDT
        
    print("Will's detectors:", detectorCounts)
        
    # Divide cell counts by volume of cell
    totalGridUnscaled = totalGrid.copy()
    trappedGridUnscaled = trappedGrid.copy()
    for i in range(len(rReduced)):
        for j in range(len(zReduced)):
            volume = np.pi*((rReduced[i]+rDelta/2.)**2-(rReduced[i]-rDelta/2.)**2)*zDelta
            totalGrid[j, i] /= volume
            trappedGrid[j, i] /= volume
    
    return rReduced, zReduced, totalGrid, trappedGrid, habitatCrossings, GDTcrossings, totalGridUnscaled, trappedGridUnscaled
    
    
def runSurvey():
    """
    Carry out Monte Carlo simulations with and without fields.
    """
    fieldFile = globals()['settings']['fieldFile']
    # Number of particles to launch
    numParticles = globals()['settings']['numParticles']
    # Radius of spherical simulation boundary used for launching and exiting
    rLim = globals()['settings']['rLim']
    # Particle stepping method
    steppingMethod = globals()['settings']['steppingMethod']
    # Coarseness of output grid that counts particle fluxes in simulation volume
    fluxGridCoarseness = globals()['settings']['fluxGridCoarseness']
    
    # B field in R and Z
    r, z, BR = fieldGrid('fields/Brs_' + fieldFile)
    _, _, BZ = fieldGrid('fields/Bzs_' + fieldFile)
    _, _, habitatBR = fieldGrid('fields/Brs_habitat_' + fieldFile)
    _, _, habitatBZ = fieldGrid('fields/Bzs_habitat_' + fieldFile)
    r = r[:-1]
    z = z[:-1]
    BR = BR[:-1,:-1] # I MAY CAUSE A BUG IN THE FUTURE
    BZ = BZ[:-1,:-1]
    habitatMax = np.max((habitatBR**2+habitatBZ**2)**.5)
    habitatPrescription = 30
    BR += habitatBR*habitatPrescription/habitatMax
    BZ += habitatBZ*habitatPrescription/habitatMax
    print('Habitat prescription (T):', habitatPrescription)
    Bmagnitude = (BR**2+BZ**2)**.5

    qms, vs = qmAndVelocitySpectrum(numParticles)
    if globals()['settings']['qmPrescribed']:
        qms = np.ones(numParticles)*globals()['settings']['qmPrescribed']
    if globals()['settings']['v0Prescribed']:
        vs = np.ones(numParticles)*globals()['settings']['v0Prescribed']

    startingPoints = [randomPointOnSphere(rLim) for _ in range(numParticles)]
    directions = [randomDirectionCos(-sp) for sp in startingPoints]

    # Simulate without magnetic field
    start = time.time()
    rReduced, zReduced, gridOff, _,  habitatCrossingsOff, GDTcrossingsOff, gridOffUnscaled, _ = monteCarloRun(startingPoints, qms, vs, directions, BR, BZ, r, z, rLim, fluxGridCoarseness, 0)
    print('Time elapsed (s):', int(time.time()-start))
    
    # Simulate with magnetic field
    start = time.time()
    _, _, gridOn, trappedOn, habitatCrossingsOn, GDTcrossingsOn, gridOnUnscaled, trappedOnUnscaled = monteCarloRun(startingPoints, qms, vs, directions, BR, BZ, r, z, rLim, fluxGridCoarseness, steppingMethod)
    print('Time elapsed (s):', int(time.time()-start))
    # np.save('cache/{}particles_accel.npy'.format(numParticles), [rReduced, zReduced, gridOn])
    try:
        print('---\nGDT crossing change: {}%'.format(round(100*(GDTcrossingsOn-GDTcrossingsOff)/GDTcrossingsOff, 3)))
        print('Habitat crossing change: {}%\n---'.format(round(100*(habitatCrossingsOn-habitatCrossingsOff)/habitatCrossingsOff, 3)))
    except Exception as e:
        print(e)
    
    # plotDiff(r, z, Bmagnitude, gridOn, gridOff)
    plot6panel(r, z, rReduced, zReduced, Bmagnitude, gridOn, gridOff, trappedOn)


def plotDiff(r, z, Bmagnitude, gridOn, gridOff):
    plt.figure(figsize=(6, 6))
    plt.title('Flux difference')
    gridDifference = 100 * (gridOn - gridOff) / gridOff
    bwr = plt.cm.bwr
    bwr.set_bad((0, 0, 0, 1))
    extent = [np.min(r), np.max(r), np.min(z), np.max(z)]
    plt.imshow(gridDifference, extent=extent, cmap=bwr, vmin=-100, vmax=100)
    plt.colorbar(label='Percent change in flux')
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
    plt.imshow(gridOff, extent=extent, vmax=themax, cmap=plt.cm.jet, norm=matplotlib.colors.LogNorm())
    plt.colorbar(label='Particles/$\mathregular{m^3}$')
    plt.ylabel('Z (m)')

    plt.subplot(232)
    plt.title('B field on, 5 T contour')
    plt.imshow(gridOn, extent=extent, vmax=themax, cmap=plt.cm.jet, norm=matplotlib.colors.LogNorm())
    plt.colorbar(label='Particles/$\mathregular{m^3}$')
    plt.contour(Bmagnitude, levels=[5.], extent=extent, colors='white')

    plt.subplot(233)
    plt.title('(B on - B off)/(B off)')
    # gridDifference = 100 * (gridOn - gridOff) / gridOff
    gridDifference = np.zeros(gridOn.shape)
    for i in range(gridOn.shape[0]):
        for j in range(gridOn.shape[1]):
            if gridOff[i, j] > 0:
                gridDifference[i, j] = 100*(gridOn[i, j]-gridOff[i, j])/gridOff[i, j]
            else:
                gridDifference[i, j] = np.nan
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
    plt.imshow(trappedOn, extent=extent, cmap=plt.cm.jet, norm=matplotlib.colors.LogNorm())
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
    runSurvey()
