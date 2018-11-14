from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import simulation


def _axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        
        
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


def testInterpolate2D():
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
    zs = [simulation.interpolate2D(xMarkers, yMarkers, Z, x, y) for (x, y) in zip(xs, ys)]
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
    
    
def dipoleRZ(x_vec):
    """
    Dipole field vector at given R, Z coordinates.
    This implementation currently appears not to be correct.
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
    

def dipoleR(x_vec):
    """
    B_Z due to a dipole centered at origin facing Z-ward, for testing.
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
    
    
def randomXYonUnitCircle():
    x = np.random.normal()
    y = np.random.normal()
    point = np.array([x, y])
    point *= 1./(x**2 + y**2)**.5
    return point
    
    
def connectRandomPointsPlot2D():
    plt.figure()
    for _ in range(5000):
        x1, y1 = randomXYonUnitCircle()
        x2, y2 = randomXYonUnitCircle()
        plt.plot([x1, x2], [y1, y2], linewidth=.1, color='blue')
    plt.axis('equal')
    plt.show()


def randomStartRandomVelocityPlot2D():
    import matplotlib.patches as patches
    _, ax = plt.subplots()
    patch = patches.Circle((0, 0), radius=1, transform=ax.transData)
    for _ in range(5000):
        x1, y1 = randomXYonUnitCircle()
        x2, y2 = randomXYonUnitCircle()
        x2 = x1 + 2*x2
        y2 = y1 + 2*y2
        line, = ax.plot([x1, x2], [y1, y2], linewidth=.1, color='blue')
        line.set_clip_path(patch)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.axis('equal')
    plt.show()


def randomRadiusPlot2D():
    """
    For funsies: why does this work? Pick an arc segment along the sphere and show all the endpoints for chords that start in the segment. How are they distributed?
    """
    import matplotlib.patches as patches
    _, ax = plt.subplots(figsize=(5,5))
    patch = patches.Circle((0, 0), radius=1, transform=ax.transData)
    for _ in range(1000):
        a = np.random.rand() * 2 * np.pi
        r = np.random.rand()
        xm = r * np.cos(a)
        ym = r * np.sin(a)
        perpSlope = -xm/ym
        x1 = 10 +xm
        y1 = 10*perpSlope +ym
        x2 = -10 +xm
        y2 = -10*perpSlope +ym
        ax.scatter([xm], [ym], s=1, color='red')
        line, = ax.plot([x1, x2], [y1, y2], linewidth=.1, color='blue')
        line.set_clip_path(patch)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')
    plt.show()


def randomRadiusPlot3D():
    """
    Pick uniformly random r, phi, and theta in sphere as midpoint
    Pick an xy vector with random angle
    Take cross product with midpoint to get chord direction
    Step from midpoint to boundary of circle by sqrt(r^2-m^2) in chord direction
    """
    R = 3
    ms = []
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection='3d')
    for _ in range(1000):
        rm = R*np.random.rand()
        xm = simulation.randomPointOnSphere(rm)
        ms.append(xm)
        xp = simulation.randomPointOnSphere(1)
        propdir = np.cross(xm, xp)
        propdir /= np.linalg.norm(propdir)
        xstart = xm + -1*propdir*(R**2-rm**2)**.5
        xend = xm + propdir*(R**2-rm**2)**.5
        ax.plot(*list(zip(xstart, xend)), color='blue', linewidth=.1)
    
    # ax.scatter(*list(zip(*ms)), color='red', s=1)
    _axisEqual3D(ax)
    plt.axis('off')
    plt.show()
    
    
def randomRadiusPlotRZ():
    R = 3
    ms = []
    rms = []
    zms = []
    plt.figure(figsize=(5, 5))
    for _ in range(1000):
        rm = R*np.random.rand()
        xm = simulation.randomPointOnSphere(rm)
        ms.append(xm)
        rms.append((xm[0]**2+xm[1]**2)**.5)
        zms.append(xm[2])
        xp = simulation.randomPointOnSphere(1)
        propdir = np.cross(xm, xp)
        propdir /= np.linalg.norm(propdir)
        points = [xm + i*propdir*(R**2-rm**2)**.5 for i in np.arange(-1, 1, .05)]
        rs = [(p[0]**2+p[1]**2)**.5 for p in points]
        zs = [p[2] for p in points]
        plt.plot(rs, zs, color='blue', linewidth=.1)
    # plt.scatter(rms, zms, s=.5)
    plt.axis('equal')
    # ax.scatter(*list(zip(*ms)), color='red', s=1)
    plt.show()
    
    
def randomHomogRZ():
    """
    Randomly select midpoint R and Z in order to get as uniform an R, Z grid as possible.
    """
    R = 3.
    zs = []
    ms = []
    rms = []
    zms = []
    plt.figure(figsize=(10, 10))
    for _ in range(3000):
        xm = np.array([0., 0., 4.])
        while np.linalg.norm(xm) > R:
            xm[2] = 2*R*(np.random.rand()-0.5)
            rm = R*np.random.rand()**2
            thetam = 2*np.pi*np.random.rand()
            xm[0] = rm*np.cos(thetam)
            xm[1] = rm*np.sin(thetam)
        ms.append(xm)
        rms.append((xm[0]**2+xm[1]**2)**.5)
        zms.append(xm[2])
        xp = simulation.randomPointOnSphere(1)
        propdir = np.cross(xm, xp)
        propdir /= np.linalg.norm(propdir)
        points = [xm + i*propdir*(R**2-rm**2)**.5 for i in np.arange(-1, 1, .05) if np.linalg.norm(xm + i*propdir*(R**2-rm**2)**.5) < R]
        rs = [(p[0]**2+p[1]**2)**.5 for p in points]
        zs = [p[2] for p in points]
        plt.plot(rs, zs, color='blue', linewidth=.1)
    # plt.scatter(rms, zms, s=.5)
    plt.axis('equal')
    # ax.scatter(*list(zip(*ms)), color='red', s=1)
    plt.show()
    
    
def compareUniformity():
    for _ in range(10000):
        # get line points inside sphere using specified method
        # count unique appearances in x and y bins and divide by z height
        pass
        
        
def testOrbit():
    """
    Test case: straight 1 T magnetic field and 1 GeV proton
      Wolfram alpha: 1 GeV -> 2.6e8 m/s
      r_g = mv_perp/eB = 1.6e-27 * 2.6e8/(1.6e-19 * 1) = 2.6 m
      omega_g = eB/m = 1.6e-19 * 1 / 1.6e-27 = 10^8 rad/s
      tau_g = 2pi/10^8 = 6.2e-8 s for a full turn
    """
    q = 1.6e-19
    m = 1.67e-27
    Bmagnitude = 1.
    qm = q/m
    v0 = 2.6e8
    gyroradius = v0/(qm*Bmagnitude)
    gyrofrequency = qm*Bmagnitude
    gyroperiod = 2*np.pi/gyrofrequency
    h = gyroperiod/50 #1e-8
    steps = 10*int(gyroperiod/h)
    print('Radius (m):', gyroradius)
    print('Period (s):', gyroperiod)
    print('Steps:', steps)
    print(h, h*v0)
    
    x = np.zeros((steps, 3))
    x[0] = np.array([v0*h*.5, gyroradius, 0])
    vNext = np.array([v0, 0, 0])
    B = np.array([0., 0, Bmagnitude])
    E = np.array([0., 0, 0])
    for i in range(steps-1):
        x[i+1], vNext = simulation.BBnext2(x[i], vNext, B, E, qm, h)
    print(np.mean(x[:,0]), np.mean(x[:,1]))
    plt.figure(figsize=(5,5))
    # plt.plot((x[:,0]**2+x[:,1]**2)**.5); plt.xlabel('Steps'), plt.ylabel('Radius (m)') 
    plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.axis('equal'); plt.gcf().gca().add_artist(plt.Circle((0,0), radius=gyroradius, color='red', fill=False)); plt.axis('equal');plt.scatter(x[:,0], x[:,1], c=range(steps), s=1, cmap=plt.cm.winter); plt.colorbar(label='Number of timesteps (%d per period)' % int(gyroperiod/h)); plt.title('BBR, Bz = 1 T, red = theoretical')
    plt.show()
    
    
testOrbit()
