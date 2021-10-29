import numpy as np
from matplotlib import pyplot as plt

from ice_flow_model import drive_ice_flow

# DOMAIN
L = 50e3        # Domain length (m)
N = 100         # Number of grid centre points
dx = L/N

xc = np.arange(dx/2, L+dx/2, dx)    # Cell centre coordinates
xe = np.arange(0, L+dx, dx)         # Cell edge coordinates
b0 = 1800
zb = b0 - 0.05*xc
t0 = 0
tend = 1e3*360*86400
dt = 5*86400

# Initial conditions - the model is robust enough we can be completely
# naive about the initial state
h0 = np.zeros(xc.shape)
h0[:95] = 50

tt = np.arange(t0, tend+dt, dt)
H2 = drive_ice_flow(tt, xc, h0, zb, zELA=1400)

fig2, ax2 = plt.subplots()
ax2.plot(xc/1e3, zb, 'k')
ax2.plot(xc/1e3, zb + H2[-1])
ax2.grid()
ax2.set_xlabel('x (km)')
ax2.set_ylabel('Surface elevation (m)')

fig3, ax3 = plt.subplots()
mass = np.sum(H2, axis=1)
ax3.plot(tt/86400/360, mass/mass[-1])
ax3.set_xlabel('Time (years)')
ax3.set_ylabel('Relative mass (-)')
ax3.grid()

plt.show()
