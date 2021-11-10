import numpy as np
from matplotlib import pyplot as plt

from ice_flow_model import drive_ice_flow

# DOMAIN
L = 35e3        # Domain length (m)
N = 100         # Number of grid centre points
dx = L/N

xc = np.arange(dx/2, L+dx/2, dx)    # Cell centre coordinates
xe = np.arange(0, L+dx, dx)         # Cell edge coordinates
b0 = 1800
zb = b0 - 0.05*xc
t0 = 0
tend = 1e3*360*86400
dt = 100*86400

# Initial conditions - the model is robust enough we can be completely
# naive about the initial state
h0 = np.zeros(xc.shape)
h0[:95] = 50

tt = np.arange(t0, tend+dt, dt)
H2 = drive_ice_flow(tt, xc, h0, zb, zELA=1400, method='BE')

fig2, (ax2, ax3) = plt.subplots(ncols=2, figsize=(8, 4))
ax2.plot(xc/1e3, zb, color=0.5*np.ones(3))
ax2.plot(xc/1e3, zb + H2[-1])
ax2.grid()
ax2.set_xlabel('x (km)')
ax2.set_ylabel('Elevation (m)')
ax2.text(-0.075, 1.05, 'a', transform=ax2.transAxes, fontsize=14)

mass = np.sum(H2, axis=1)
ax3.plot(tt/86400/360, mass/mass[-1])
ax3.set_xlabel('Time (years)')
ax3.set_ylabel('Relative mass (-)')
ax3.text(-0.075, 1.05, 'b', transform=ax3.transAxes, fontsize=14)
ax3.grid()

plt.tight_layout()
fig2.savefig('valley_slope.png', dpi=600)

print(np.max(H2[-1]))

A_sens = 2.4e-24*np.array([0.5, 0.75, 1, 1.25, 1.5, 2])
N_sens = len(A_sens)
volumes = np.zeros(N_sens)
for i,A in enumerate(A_sens):
    print('Case A = ', A*1e24)
    Hi = drive_ice_flow(tt, xc, h0, zb, zELA=1400, method='BE', A=A)
    volumes[i] = np.sum(Hi[-1, :])

fig, (ax, ax2) = plt.subplots(figsize=(8, 4), ncols=2, sharey=True)
ax.scatter(A_sens/2.4e-24, volumes/volumes[2])
ax.grid()
ax.set_xlabel('A/A$_0$')
ax.set_ylabel('Relative ice volume (-)')
ax.text(-0.075, 1.05, 'a', transform=ax.transAxes, fontsize=14)
print(volumes/volumes[2])

Gamma_0 = 0.007/(365*86400)
Gamma_sens = Gamma_0*np.array([0.5, 0.75, 1, 1.25, 1.5, 2])
M_sens = len(Gamma_sens)
volumes_gamma = np.zeros(M_sens)
for i,G in enumerate(Gamma_sens):
    print('case Gamma = ', G/Gamma_0)
    Hi = drive_ice_flow(tt, xc, h0, zb, zELA=1400, method='BE', Gamma=G)
    volumes_gamma[i] = np.sum(Hi[-1, :])

ax2.scatter(Gamma_sens/Gamma_0, volumes_gamma/volumes_gamma[2])
ax2.grid()
ax2.set_xlabel('$\\Gamma/\\Gamma_0$')
ax2.text(-0.075, 1.05, 'b', transform=ax2.transAxes, fontsize=14)
print(volumes_gamma/volumes_gamma[2])


fig.savefig('valley_sensitivity.png', dpi=600)

plt.show()
