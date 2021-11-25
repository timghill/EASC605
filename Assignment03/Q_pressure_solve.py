"""
Iteratively calculate a consistent Q-pw solution GIVEN a cross-section S
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize

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

dt = 60

bed_slope = 0
zb = 100 - bed_slope*x
zs_arg = L - x + 2*dx
zs_arg[zs_arg<0] = 0
zs = 600/np.sqrt(L)*np.sqrt(zs_arg)
zs[zs<=zb] = zb[zs<=zb]

# Overburden pressure
p_i = rhow*g*(zs - zb)

# FORCING
Q0 = 10


def ds_upwind(x, ghost_value=0):
    dxds = np.zeros(x.shape)
    dxds[1:] = (x[1:] - x[:-1])/dx
    dxds[0] = (x[0] - ghost_value)/dx
    return dxds

def ds_downwind(x, ghost_value=0):
    dxds = np.zeros(x.shape)
    dxds[:-1] = (x[1:] - x[:-1])/dx
    dxds[-1] = (ghost_value - x[-1])/dx
    return dxds

# Create upwind derivative matrix
D_upwind = np.diag(np.ones(N)) - np.diag(np.ones(N-1), k=-1)
D_upwind = D_upwind/dx

D_downwind = 1/dx*(-np.diag(np.ones(N)) + np.diag(np.ones(N-1), k=1))

# IMPOSED PRESSURE CONDITIONS - try to solve for this
pw_max = p_i[0]
pw_min = 0
pw = pw_max - pw_max*((x)/(L))**1
# pw = 0.75*p_i

# INITIAL CONDITIONS
S = 2 + 2*x/L

# Potential gradient is simply pressure gradient
dpwds = ds_downwind(pw)
dphids = dpwds

# Iterate to find a consistent solution for flux Q and cross sectional
# area, given an imposed pressure gradient
itnum = 0
maxiter = 50
err = 1e3
tol = 1e-12

# while err>tol and itnum<maxiter:
for i in range(5):

    # Solve for discharge Q
    D = D_upwind + (1/rhoi - 1/rhow)/Lf*(gamma - 1)*np.diag(dpwds)
    y = np.vstack(2*S*A*( (p_i - pw)/n)**n)
    y[0] = y[0] + Q0/dx # Boundary condition on Q[0]

    Q = np.linalg.solve(D, y).flatten()

    # Now calculate pressure gradient - note the negative sign!
    dpwds = -Q**2*np.sqrt(np.pi)*fR*rhow/(S**5/2)

    # Integrate to calculate pressure
    yp = np.vstack(dpwds)
    p_new = np.linalg.solve(D_downwind, yp).flatten()
    print(np.max(np.abs(pw - p_new))/np.max(p_i))
    pw = p_new

mi = Q/Lf*(gamma-1)*dpwds
dSdt = mi/rhow - 2*S*A*(p_i - pw)**n/n**n

fig, ax = plt.subplots()
ax.plot(x, p_new)
ax.set_title('Pressure')

fig, ax = plt.subplots()
ax.plot(x, Q)
ax.set_title('Discharge')

fig, ax = plt.subplots()
ax.plot(x, dSdt)
ax.set_title('dS dt')

plt.show()
