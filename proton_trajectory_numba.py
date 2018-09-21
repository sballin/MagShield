# interpolate if it's faster
# test case: straight 1 T magnetic field and 1 GeV proton
# Wolfram alpha: 1 GeV -> 2.6e8 m/s
# r_g = mv_perp/eB = 1.6e-27 * 2.6e8/(1.6e-19 * 1) = 2.6 m
# omega_g = eB/m = 1.6e-19 * 1 / 1.6e-27 = 10^8 rad/s
# tau_g = 2pi/10^8 = 6.2e-8 s for a full turn
# Lorentz force: F = q v cross B


import numpy as np
from numba import jit
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Body:
    def __init__(self, mass=1.67e-27, timesteps=3e7):
        self.dimensions = 3
        self.mass = mass
        self.x = np.zeros((timesteps+1, self.dimensions))
        self.x[0, 0] = 0
        self.x[0, 1] = 0
        self.v = np.zeros((timesteps+1, self.dimensions))
        self.v[0, 0] = 0
        self.v[0, 1] = 2.6e8
        self.v[0, 2] = 0#1.2e8
        self.a = np.zeros((timesteps, self.dimensions))
        
    def accel_B(self, i):
        return 1.6e-19/self.mass*np.cross(self.v[i], np.array([0,0,1]))

    def get_p(self):
        return self.mass*np.array([np.linalg.norm(v) for v in self.v])

    def get_L(self):
        return np.array([np.linalg.norm(x) for x in self.x])*self.get_p()

    def get_KE(self):
        v2 = np.array([np.linalg.norm(v) for v in self.v])**2
        return .5*self.mass*v2

# Initial conditions of our solar system and simulation setup
dt = 1e-11
times = np.arange(0, 5*6.2e-8, dt)
p = Body(timesteps=len(times))

print("Timesteps: ", len(times))

# Leap-frog integration over times
for i in xrange(len(times)):
    p.a[i] += p.accel_B(i)
    if i == 0:
        p.v[i+1] = p.v[i]+.5*p.a[i]*dt
    else:
        p.v[i+1] = p.v[i] + p.a[i]*dt
    p.x[i+1] = p.x[i] + p.v[i+1]*dt

# Plot particle trajectories
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# plt.plot(p.x[:,0], p.x[:,1], p.x[:,2])
# plt.show()

# Obtain values of total KE, L, and p
L = p.get_L()
KE = p.get_KE()
p = p.get_p()

# Plot KE, L, and p
plt.figure()
plt.semilogy(times, L[:-1], label="L (kg*m^2/s)", linewidth=1)
plt.semilogy(times, KE[:-1], label="KE (kg*m^2/s^2)", linewidth=1)
plt.semilogy(times, p[:-1], label="p (kg*m/s)", linewidth=1)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
plt.xlabel('Time (s)')
plt.show()
