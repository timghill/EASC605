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

dt = 60

bed_slope = 0
zb = 100 - bed_slope*x
zs_arg = L - x + 2*dx
zs_arg[zs_arg<0] = 0
zs = 600/np.sqrt(L)*np.sqrt(zs_arg)
zs[zs<=zb] = zb[zs<=zb]

# Overburden pressure
pi = rhow*g*(zs - zb)

# FORCING
Q0 = 10


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

# Create upwind derivative matrix
D_upwind = np.diag(np.ones(N)) - np.diag(np.ones(N-1), k=-1)
D_upwind = D_upwind/dx

# IMPOSED PRESSURE CONDITIONS
pw_max = 0.5*pi[0]
pw_min = 0
pw = pw_max - pw_max*((x)/(L))**(1/2)


# INITIAL CONDITIONS
S = 2 + 2*x/L

# Iterate to find a consistent solution for pressure
for j in range(1):
    # Potential gradient is simply pressure gradient
    dphids = ds_downwind(pw)
    # Iterate to find a consistent solution for flux Q and cross sectional
    # area, given an imposed pressure gradient
    for i in range(5):
        D = D_upwind - (1/rhoi - 1/rhow)/Lf*(gamma - 1)*np.vstack(dphids)
        y = np.vstack(2*S*A*( (pw - pi)/n)**n)
        y[0] = y[0] + Q0/dx

        Q = np.linalg.solve(D, y).flatten()

        # Calculate instantaneous area
        S_new = -Q**(4/5)*(fR*rhow)**(2/5)*np.pi**(1/5)/(dphids * np.abs(dphids)**(-3/5))
        if i==0:
            S0 = S.copy()

        print('Cross-sectional area change:', np.max(S - S_new))
        S = S_new

    mi = Q/Lf*(gamma-1)*dphids
    dQds = ds_upwind(Q, Q0)
    dSdt = mi/rhow - dQds
    dpwds = ds_downwind(pw)
    p_arg = 1/(2*S*A)*(mi*(1/rhoi - 1/rhow) + dQds)
    pw = pi - n*np.abs(p_arg)**(1/n - 1)*p_arg

fig, ax = plt.subplots()
ax.plot(x, S0)
ax.plot(x, S)
plt.show()
