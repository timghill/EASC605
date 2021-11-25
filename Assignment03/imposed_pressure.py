"""
Simple outburst flood model
"""

import numpy as np
from matplotlib import pyplot as plt

# PARAMETERS
rhoi = 917
rhow = 1000
g = 9.81
Lf = 3.34e5
cw = 4.217e3
ct = 7.5e-8
A = 2.4e-24
n = 3
fR = 0.15

gamma = ct*rhow*cw

# DOMAIN
L = 10e3
N = 100
dx = (L/N)
x = np.arange(0, L, dx)

bed_slope = 0
zb = 100 - bed_slope*x
zs_arg = L - x + 2*dx
zs_arg[zs_arg<0] = 0
zs = 600/np.sqrt(L)*np.sqrt(zs_arg)
zs[zs<=zb] = zb[zs<=zb]

pi = rhow*g*(zs - zb)
# print(pi)

# fig, ax = plt.subplots()
# ax.plot(x, zb, 'k')
# ax.plot(x, zs, 'b')
# plt.show()

# FORCING
Q0 = 5


def ds_upwind(x, ghost_value=0):
    dxds = np.zeros(x.shape)
    dxds[1:] = (x[1:] - x[:-1])/dx
    dxds[0] = (x[0] - ghost_value)/dx
    return dxds

def ds_upwind2(x):
    dxds = np.zeros(x.shape)
    dxds[1:] = (x[1:] - x[:-1])/dx
    dxds[0] = dxds[1]
    return dxds


def ds_downwind(x, ghost_value=0):
    dxds = np.zeros(x.shape)
    dxds[:-1] = (x[1:] - x[:-1])/dx
    dxds[-1] = (ghost_value - x[-1])/dx
    return dxds

# IMPOSED PRESSURE CONDITIONS
pw_max = 0.5*pi[0]
pw_min = 0
pw = pw_max - pw_max*((x)/(L))**(1/2)

# INITIAL CONDITIONS
S = 1 + 0*x/L

# Potential gradient is simply pressure gradient
dphids = ds_downwind(pw)

# Calculate initial discharge
# Q = -S**(5/4)*(1/fR/rhow/(np.pi**0.5))**(1/2) * dphids * np.abs(dphids)**(-1/2)

# Create upwind derivative matrix
# D_upwind = np.zeros((N, N))
D_upwind = np.diag(np.ones(N)) - np.diag(np.ones(N-1), k=-1)
D_upwind = D_upwind/dx

D = D_upwind - 1/Lf/rhow*(gamma - 1)*np.vstack(dphids)
y = np.vstack(np.zeros(N))
y[0] = Q0/dx

Q = np.linalg.solve(D, y).flatten()

fig, ax = plt.subplots()
ax.plot(x, Q, label='Calculated')
ax.set_title('Discharge')

# Calculate area
S = -Q**(4/5)*(fR*rhow)**(2/5)*np.pi**(1/5)/(dphids * np.abs(dphids)**(-3/5))

Q_recons = -S**(5/4)*(1/fR/rhow)**(1/2)*np.pi**(-1/4)*dphids*np.abs(dphids)**(-0.5)

ax.plot(x, Q_recons - Q, label='Reconstructed')
ax.legend()

fig, ax = plt.subplots()
ax.plot(x, S)
ax.set_title('S')

plt.show()
#
#
# fig, ax = plt.subplots()
# ax.plot(x, Q)
# ax.set_title('Discharge')
#
# fig, ax = plt.subplots()
# ax.plot(x, pw)
# ax.set_title('Pressure')
#
# fig, ax = plt.subplots()
# ax.plot(x, dphids)
# ax.set_title('Potential gradient')
#
# fig, ax = plt.subplots()
# ax.plot(x, S)
# ax.set_title('S')
#
# plt.show()
