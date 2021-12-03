"""
Solve time-dependent R-channel equations
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate

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
# dt = 3600
dt = 3600/2
t = 0
t_end = 86400
t_span = (t, t_end)
t_eval = np.arange(t, t_end, dt)
#
# print(t_span)
# print(t_eval)

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
pw_max = p_i[0]

# INITIAL CONDITIONS
S = 0.2 + 0*x/L   # True initial condition on channel cross-section

# STATE VECTOR: (S, Q, p)
def wrap(S, Q, p):
    v = np.zeros(3*N)
    v[:N] = S
    v[N:2*N] = Q
    v[2*N:] = q
    return v

def unwrap(v):
    S = v[:N]
    Q = v[N:2*N]
    p = v[2*N:]
    return S, Q, p

def Qp_solve(S):
    err = 1e3
    itnum = 0

    pw = pw_max - pw_max*((x)/(L))**1
    dpwds = ds_downwind(pw)

    # Iterate to find a consistent solution for flux Q and cross sectional
    # area, given an imposed pressure gradient
    while err>tol and itnum<maxiter:
    # for i in range(5):
        # Solve for discharge Q
        D = D_upwind + (1/rhoi - 1/rhow)/Lf*(gamma - 1)*np.diag(dpwds)
        y = np.vstack(2*S*A*( (p_i - pw)/n)**n)
        y[y<0] = 0
        y[0] = y[0] + Q_lake/dx     # Boundary condition on Q[0]

        Q = np.linalg.solve(D, y).flatten()

        # Now calculate pressure gradient - note the negative sign!
        dpwds = -Q**2*np.sqrt(np.pi)*fR*rhow/(S**5/2)

        # Integrate to calculate pressure
        yp = np.vstack(dpwds)
        p_new = np.linalg.solve(D_downwind, yp).flatten()

        err = np.max(np.abs(pw - p_new))
        pw = p_new
        itnum += 1

    return (Q, pw, dpwds)

def rhs_channel(t, S):
    # S, Q, pw = unwrap(v)

    err = 1e3
    itnum = 0
    # print(iter)
    # if iter==0:
    #     Q_lake = 10
    # else:
    p_lake = rhow*g*h_lake
    pgrad_lake = (pw[0] - p_lake)/dx
    Q_lake = -S[0]**(5/4)*(1/fR/rhow)**(1/2)*(1/np.pi)**(1/4)*np.abs(pgrad_lake)**(-1/2)*pgrad_lake
    # if Q_lake<0:
    #     Q_lake = 0

    Q, pw, dpwds = Qp_solve(S)

    mi = Q/Lf*(gamma-1)*dpwds
    creep = 2*S*A*(p_i-pw)**n / n**n
    creep[creep<0] = 0
    dSdt = mi/rhow - creep

    return dSdt


maxiter = 50
tol = 0.1
S_tol = 1e-2/86400

h_lake = 300
A_lake = 250*250

t = 0
S_err = 1
# iter = 0
n_steps = int(3*86400/dt)
# while S_err > S_tol:

bunch = scipy.integrate.solve_ivp(rhs_channel, t_span, S, t_eval=t_eval)

S = bunch.y
print(S.shape)

dSdt = rhs_channel(0, S[:, -1])

fig, ax = plt.subplots()
ax.plot(x, S[:, -1])

# fig, ax = plt.subplots()
# ax.plot(x, dSdt)


plt.show()
#
# print('Converged after %f days' % (float(t)/86400))

#
# fig, ax = plt.subplots()
# ax.plot(x, p_new)
# ax.set_title('Pressure')
#
# fig, ax = plt.subplots()
# ax.plot(x, Q)
# ax.set_title('Discharge')
#
# fig, ax = plt.subplots()
# ax.plot(x, S)
# ax.set_title('S')
#
# plt.show()
