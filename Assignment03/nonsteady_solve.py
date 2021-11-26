"""
Solve time-dependent R-channel equations
"""

import numpy as np
from matplotlib import pyplot as plt

# PARAMETERS
rhoi = 917          # Density of ice (kg.m-3)
rhow = 1000         # Density of water (kg.m-3)
g = 9.81            # Gravitational acceleration (m.s-2)
Lf = 3.34e5         # Latent heat of fusion (J.kg-1)
cw = 4.217e3        # Heat capacity of water (J.kg-1.K-1
ct = 7.5e-8         # Clausius-Clapeyron (pressure-melting) constant (K.Pa-1)
A = 2.4e-24         # Flow law coefficient (Pa-3.s-1)
n = 3               # Flow law exponent
fR = 0.15           # Darcy-Weisbach friction coefficient
gamma = ct*rhow*cw  # Derived coefficient in energy-balance equation

# DOMAIN
L = 10e3            # Domain length (m)
N = 100             # Number of grid cells
dx = (L/N)          # Derived grid spacing
x = np.arange(0, L, dx)

bed_slope = 0               # Bed slope (-)
zb = 100 - bed_slope*x      # Bed elevation (m)
# Compute suface elevation zs (m)
zs_arg = L - x + 2*dx
zs_arg[zs_arg<0] = 0
zs = 600/np.sqrt(L)*np.sqrt(zs_arg)
zs[zs<=zb] = zb[zs<=zb]

p_i = rhow*g*(zs - zb)      # Overburden ice pressure (Pa)

# Timestepping
dt = 86400
t = 0

# FORCING
Q0 = 10             # Imposed upstream flux from lake drainage

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

# Finite-difference discretizations

# Upwind derivatives - this enforces a BC on the left boundary
D_upwind = np.diag(np.ones(N)) - np.diag(np.ones(N-1), k=-1)
D_upwind = D_upwind/dx

# Downwind derivatives - this enforces a BC on the right boundary
D_downwind = 1/dx*(-np.diag(np.ones(N)) + np.diag(np.ones(N-1), k=1))

# Initial pressure guess - needed in first iteration to solve for
# first guess of discharge Q. This is not an initial condition but a "first
# guess" in the iteration step
pw_min = 0
pw = pw_max - pw_max*((x)/(L))**1
dpwds = ds_downwind(pw)

# INITIAL CONDITIONS
S = 2 + 2*x/L   # True initial condition on channel cross-section

maxiter = 50
tol = 0.1
S_tol = 5e-4/3600

t = 0
S_err = 1
while S_err > S_tol:
    err = 1e3
    itnum = 0
    # Iterate to find a consistent solution for flux Q and cross sectional
    # area, given an imposed pressure gradient
    while err>tol and itnum<maxiter:
    # for i in range(5):

        # Solve for discharge Q
        D = D_upwind + (1/rhoi - 1/rhow)/Lf*(gamma - 1)*np.diag(dpwds)
        y = np.vstack(2*S*A*( (p_i - pw)/n)**n)
        y[0] = y[0] + Q0/dx     # Boundary condition on Q[0]

        Q = np.linalg.solve(D, y).flatten()

        # Now calculate pressure gradient - note the negative sign!
        dpwds = -Q**2*np.sqrt(np.pi)*fR*rhow/(S**5/2)

        # Integrate to calculate pressure
        yp = np.vstack(dpwds)
        p_new = np.linalg.solve(D_downwind, yp).flatten()

        err = np.max(np.abs(pw - p_new))
        pw = p_new
        itnum += 1


    mi = Q/Lf*(gamma-1)*dpwds
    dSdt = mi/rhow - 2*S*A*(p_i - pw)**n/n**n
    S = S + dt*dSdt
    t = t + dt

    S_err = np.max(np.abs(dSdt))

print('Converged after %f days' % (float(t)/86400))


fig, ax = plt.subplots()
ax.plot(x, p_new)
ax.set_title('Pressure')

fig, ax = plt.subplots()
ax.plot(x, Q)
ax.set_title('Discharge')

fig, ax = plt.subplots()
ax.plot(x, S)
ax.set_title('S')

plt.show()
