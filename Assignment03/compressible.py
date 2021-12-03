"""
Solve time-dependent R-channel equations for compressible fluid, including
pressure-coupled lake draining
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate

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
beta = 1e-4,        # Compressibility (Pa-1)
)

# HELPER FUNCTIONS
def wrap(S, pw, h):
    """
    Concatenate S, pw, and h arrays into the state vector
    """
    v = np.zeros(2*N+1)
    v[:N] = S
    v[N:2*N] = pw
    v[-1] = h
    return v

def unwrap(v):
    """
    Split state vector into arrays S, pw, and h
    """
    S = v[:N]
    pw = v[N:2*N]
    h = v[-1]
    return S, pw, h

def rhs_compressible_channel(t, v, params):
    """
    Calculate d(pw)/dt and d(S)/dt for nonsteady, compressible conduit
    equations.
    """
    S, pw, h = unwrap(v)

    A = params['A']
    n = params['n']
    fR = params['fR']
    beta = params['beta']

    p_lake = rhow*g*h
    # Compute pressure gradient at lake outlet
    pgrad_lake = (pw[0] - p_lake)/params['dx']

    # Compute Q discharge from lake (avoiding divide by zero error)
    if pgrad_lake!=0:
        Q_lake = -S[0]**(5/4)*(1/rhow/fR)**(1/2)*(1/np.pi)**(1/4)*np.abs(pgrad_lake)**(-0.5)*pgrad_lake
    else:
        Q_lake = 0

    # Do not allow negative discharge (e.g. lake filling from conduit)
    if Q_lake<0:
        Q_lake = 0

    # Stop draining when h is <= 0
    if h<=0:
        Q_lake = 0

    N = params['N']
    # DERIVATIVE OPERATORS
    # Upwind derivatives - this enforces a BC on the left boundary
    D_upwind = np.diag(np.ones(N)) - np.diag(np.ones(N-1), k=-1)
    D_upwind = D_upwind/dx

    # Downwind derivatives - this enforces a BC on the right boundary
    D_downwind = 1/dx*(-np.diag(np.ones(N)) + np.diag(np.ones(N-1), k=1))

    dpwds = np.matmul(D_downwind, np.vstack(pw)).flatten()
    Q = -S**(5/4)*(1/rhow/fR)**(1/2)*(1/np.pi)**(1/4)*np.abs(dpwds)**(-0.5)*dpwds
    dQds = np.matmul(D_upwind, np.vstack(Q)).flatten()

    # !! Lake discharge BC
    dQds[0] = (Q[0] - Q_lake)/dx

    # Compute components of time derivatives, disallowing freeze-on and
    # creep opening
    mdot = Q/Lf*(gamma - 1)*dpwds
    mdot[mdot<0] = 0

    creep_closure = 2*S*A*(p_i - pw)**n/n**n
    creep_closure[creep_closure<0] = 0

    # Compute time derivatives (area, pressure, and lake level)
    dSdt = mdot/rhoi - creep_closure
    dpwdt = -1/beta/S*(dSdt + dQds - mdot/rhow)
    dhdt = -Q_lake/A_lake

    # Populate state vector
    vprime = np.zeros(2*N+1)
    vprime[:N] = dSdt
    vprime[N:2*N] = dpwdt
    vprime[-1] = dhdt

    return vprime

def solve_compressible(params):
    """
    Solve compressible conduit equations for experimental design
    specified by `params`
    """
    x = params['x']
    L = params['L']
    N = params['N']
    dx = params['dx']

    zb = params['zb']
    p_i = params['p_i']

    dt = params['dt']
    t_eval = params['t_eval']
    t_span = params['t_span']

    A_lake = params['A_lake']

    S = params['init_S']
    pw = params['init_pw']
    h_lake = params['init_h_lake']

    fR = params['fR']

    # DERIVATIVE OPERATORS
    # Upwind derivatives - this enforces a BC on the left boundary
    D_upwind = np.diag(np.ones(N)) - np.diag(np.ones(N-1), k=-1)
    D_upwind = D_upwind/dx

    # Downwind derivatives - this enforces a BC on the right boundary
    D_downwind = 1/dx*(-np.diag(np.ones(N)) + np.diag(np.ones(N-1), k=1))

    v0 = wrap(S, pw, h_lake)

    fun = lambda t, y: rhs_compressible_channel(t, y, params)
    sol_out = scipy.integrate.solve_ivp(fun, t_span,
        v0, t_eval=t_eval, method='BDF')

    v_sol = sol_out.y
    S = v_sol[:N]
    pw = v_sol[N:2*N]
    h_lake = v_sol[-1]

    # Calculate Q at each timestep
    Q = np.zeros(S.shape)
    for i in range(pw.shape[1]):
        dpwds = np.matmul(D_downwind, np.vstack(pw[:, i])).flatten()
        Q[:, i] = -S[:, i]**(5/4)*(1/rhow/fR)**(1/2)*(1/np.pi)**(1/4)*np.abs(dpwds)**(-0.5)*dpwds

    return (S, pw, Q, h_lake)

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
    # dt = 3600
    dt = 86400
    t = 0
    t_end = 86400*60
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

    # Initial pressure guess - needed in first iteration to solve for
    # first guess of discharge Q. This is not an initial condition but a "first
    # guess" in the iteration step
    pw = p_i
    params['init_pw'] = pw

    # INITIAL CONDITIONS
    S = 0.1 + 0*x/L   # True initial condition on channel cross-section

    params['init_S'] = S

    S, pw, Q, h = solve_compressible(params)

    fig, ax = plt.subplots()
    ax.plot(t_eval/86400, Q[0])
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Discharge (m$^3$/s)')
    ax.grid()

    plt.show()
