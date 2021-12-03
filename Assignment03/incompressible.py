"""
Solve time-dependent R-channel equations
"""

import numpy as np
from matplotlib import pyplot as plt

# PHYSICAL CONSTANTS (can't change these values)
rhoi = 917          # Density of ice (kg.m-3)
rhow = 1000         # Density of water (kg.m-3)
g = 9.81            # Gravitational acceleration (m.s-2)
Lf = 3.34e5         # Latent heat of fusion (J.kg-1)
cw = 4.217e3        # Heat capacity of water (J.kg-1.K-1
ct = 7.5e-8         # Clausius-Clapeyron (pressure-melting) constant (K.Pa-1)
gamma = ct*rhow*cw  # Derived coefficient in energy-balance equation

# PARAMETERS - can change these vaues
default_params = dict(
A = 2.4e-24,        # Flow law coefficient (Pa-3.s-1)
n = 3,              # Flow law exponent
fR = 0.15,          # Darcy-Weisbach friction coefficient
)

def solve_incompressible(params):
    """
    Solve incompressible conduit equations with NO pressure coupling
    """
    A = params['A']
    n = params['n']
    fR = params['fR']

    N = params['N']
    # Finite-difference discretizations

    # Upwind derivatives - this enforces a BC on the left boundary
    D_upwind = np.diag(np.ones(N)) - np.diag(np.ones(N-1), k=-1)
    D_upwind = D_upwind/dx

    # Downwind derivatives - this enforces a BC on the right boundary
    D_downwind = 1/dx*(-np.diag(np.ones(N)) + np.diag(np.ones(N-1), k=1))

    p_i = params['p_i']

    pw = params['init_pw']
    dpwds = np.matmul(D_downwind, np.vstack(pw)).flatten()

    S = params['init_S']

    maxiter = 50
    tol = 0.1
    S_tol = 1e-2/86400

    h_lake = params['init_h_lake']
    A_lake = params['A_lake']

    t, t_end = params['t_span']
    dt = params['dt']
    m = len(params['t_eval'])

    S_out = np.zeros((N, m))
    Q_out = np.zeros((N, m))
    pw_out = np.zeros((N, m))
    h_lake_out = np.zeros(m)

    t = 0

    Q_forcing_handle = params['Q_lake']
    for tindex, t in enumerate(params['t_eval']):
        err = 1e3
        itnum = 0

        Q_lake = Q_forcing_handle(t)

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

        mi = Q/Lf*(gamma-1)*dpwds
        mi[mi<0] = 0

        creep = 2*S*A*(p_i-pw)**n / n**n
        creep[creep<0] = 0

        dSdt = mi/rhow - creep
        t = t + dt

        h_lake = h_lake - dt*Q_lake/A_lake
        if h_lake<0:
            h_lake = 0

        S_err = np.max(np.abs(dSdt))

        S_out[:, tindex] = S
        Q_out[:, tindex] = Q
        pw_out[:, tindex] = pw
        h_lake_out[tindex] = h_lake

        S = S + dt*dSdt

    return S_out, pw_out, Q_out, h_lake_out


if __name__ == '__main__':
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

    params = default_params.copy()
    params['L'] = L
    params['N'] = N
    params['dx'] = dx
    params['x'] = x

    params['zb'] = zb
    params['p_i'] = p_i

    # Timestepping
    dt = 3600
    t = 0
    t_end = 86400*2
    t_span = (t, t_end)
    t_eval = np.arange(t, t_end, dt)

    params['dt'] = dt
    params['t_span'] = t_span
    params['t_eval'] = t_eval

    h_lake = p_i[0]/rhow/g
    p_lake = rhow*g*h_lake
    A_lake = 500*500

    params['A_lake'] = A_lake
    params['init_h_lake'] = h_lake

    # PRESCRIBED LAKE DRAINAGE
    params['Q_lake'] = lambda t: 10

    # Initial pressure guess - needed in first iteration to solve for
    # first guess of discharge Q. This is not an initial condition but a "first
    # guess" in the iteration step
    pw = p_i
    params['init_pw'] = pw

    # INITIAL CONDITIONS
    S = 1 + 0*x/L   # True initial condition on channel cross-section

    params['init_S'] = S

    S, pw, Q, h = solve_incompressible(params)

    fig, ax = plt.subplots()
    ax.plot(x, S[:, -1])

    fig, ax = plt.subplots()
    ax.plot(t_eval/86400, S[0])
    plt.show()
