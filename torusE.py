"""
For better simulation of surface charge: https://www.particleincell.com/2011/spacecraft-charging
"""
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import math

R = 12
a = 3
scale = 1/(4*np.pi*8.85e-12)*1
Er = lambda theta, phi, r, z: (r-(R+a*math.cos(theta))*math.cos(phi))/((r-(R+a*math.cos(theta))*math.cos(phi))**2+(z-a*math.sin(theta))**2)**(3/2)
Ez = lambda theta, phi, r, z: (z-a*math.sin(theta))/((r-(R-a*math.cos(theta))*math.cos(phi))**2+(z-a*math.sin(theta))**2)**(3/2)

res = 200
rRange = np.linspace(0, 40, res/2)
zRange = np.linspace(-40, 40, res)
ErGrid = np.zeros((len(zRange),len(rRange)))
EzGrid = np.zeros((len(zRange),len(rRange)))
for i in tqdm.tqdm(range(len(rRange))):
    for j in range(len(zRange)):
        ErGrid[j, i] = scipy.integrate.dblquad(Er, 0, 2*math.pi, lambda x: 0, lambda x: 2*math.pi, args=(rRange[i], zRange[j]), epsrel=1e-1)[0]
        EzGrid[j, i] = scipy.integrate.dblquad(Ez, 0, 2*math.pi, lambda x: 0, lambda x: 2*math.pi, args=(rRange[i], zRange[j]), epsrel=1e-1)[0]
        
np.save('ErGrid.npy', ErGrid)
np.save('EzGrid.npy', EzGrid)

plt.figure()
extent = [np.min(rRange), np.max(rRange), np.min(zRange), np.max(zRange)]
plt.imshow(ErGrid, extent=extent)
plt.xlabel('R (m)')
plt.ylabel('Z (m)')
plt.colorbar(label='$\mathregular{E_R}$ (a.u.)')
plt.show()

plt.figure()
plt.imshow(-EzGrid, extent=extent)
plt.xlabel('R (m)')
plt.ylabel('Z (m)')
plt.colorbar(label='$\mathregular{E_Z}$ (a.u.)')
plt.show()
