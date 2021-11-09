
import numpy as np
from matplotlib import pyplot as plt

from ice_flow_model import drive_ice_flow


# DOMAIN
L = 50e3                            # Domain length (m)
N = 100                             # Number of grid centre points
dx = L/N

xc = np.arange(dx/2, L+dx/2, dx)    # Cell centre coordinates
xe = np.arange(0, L+dx, dx)         # Cell edge coordinates

Gamma = 0.5/(86400*365)           # Mass-balance gradient (m/a)
b0 = 1200
zELA = 1550

zb = b0 - 0*xc                      # Bed elevation

# h0 = 500*np.ones(xc.shape)             # Ice thickness
h0 = 300*np.sin(np.pi*xc/50e3) + 600

t0 = 0
tend = 500*365*86400
dt = 30*86400
tt = np.arange(t0, tend+dt, dt)
H1 = drive_ice_flow(tt, xc, h0, zb, zELA=zELA, method='CN',
    Gamma=Gamma, bcs=('free-flux', 'free-flux'), b='constant')

fig1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

ax1.plot(xc/1e3, H1[0, :], label='t = 0 a')
ax1.plot(xc/1e3, H1[-1, :], label='t = 500 a')
ax1.legend()
ax1.grid()
ax1.set_xlabel('x (km)')
ax1.set_ylabel('h (m)')
ax1.text(-0.075, 1.05, 'a', transform=ax1.transAxes, fontsize=14)

rho = 910
tot_mass = rho*np.sum(H1*dx, axis=1)
ax2.plot(tt/365/86400, tot_mass/tot_mass[-1])
ax2.grid()
ax2.set_xlabel('Time (years)')
ax2.set_ylabel('Relative mass (-)')
ax1.text(-0.075, 1.05, 'b', transform=ax2.transAxes, fontsize=14)

plt.tight_layout()

fig1.savefig('flat_bed.png', dpi=600)

print(np.max(H1[-1, :]))
