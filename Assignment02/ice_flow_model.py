"""
One-dimensional ice flow model

Explore time stepping: Explicit, implicit, adaptive
"""
import time
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize

# CONSTANTS
g = 9.81        # Gravity; m/s2
rho = 910       # Ice density: kg/m3
n = 3           # Ice flow exponent: -

# PARAMETERS
A = 2.4e-24     # Ice flow coefficient (Pa-3.s-1)
Gamma = 0.007/(365*86400)   # Mass balance gradient (s-1)
zELA = 1400


def rhs_1d(t, h, zb, dx, Gamma=Gamma, zELA=zELA,
    bcs=('no-flux', 'no-flux'), b='linear', A=A):
    """
    Calculate right hand side (dh/dt) of one-dimensional ice flow model.
    """
    n_center = len(h)
    n_edge = n_center + 1

    zs = zb + h
    # dzdx defined on edges
    dzdx_center = (zs[1:] - zs[:-1])/dx

    # k defined on centers
    k_center = -(2*A)*((rho*g)**n)*(h**(n+2))/(n+2)
    q_edge = np.zeros(n_edge)
    q_edge[1:-1] = (k_center[1:] + k_center[:-1])/2 * dzdx_center**n

    # Calculate mass balance
    if b=='linear':
        bdot = Gamma*(zs - zELA)
    else:
        bdot = Gamma*np.ones(n_center)

    if bcs[0]=='no-flux':
        # No flux boundary conditions
        q_edge[0] = 0
    elif bcs[0]=='free-flux':
        # q_edge[0] = k_center[0]*(dzdx_center[0])**n
        q_edge[0] = q_edge[1]
        q_edge[0] += bdot[0]*dx*np.sign(q_edge[0])

    if bcs[1]=='no-flux':
        q_edge[-1] = 0
    elif bcs[1]=='free-flux':
        # q_edge[-1] = k_center[-1]*dzdx_center[-1]**n
        q_edge[-1] = q_edge[-2]
        q_edge[-1] += bdot[1]*dx*np.sign(q_edge[-1])

    # Time derivative is flux divergence + mass balance
    hprime = -(q_edge[1:] - q_edge[:-1])/dx + bdot
    return hprime

def rhs_1d_print_diff(t, h, zb, dx, Gamma=Gamma, zELA=zELA,
    bcs=('no-flux', 'no-flux'), b='linear', A=A):
    """
    Calculate right hand side (dh/dt) of one-dimensional ice flow model.
    """
    n_center = len(h)
    n_edge = n_center + 1

    zs = zb + h
    # dzdx defined on edges
    dzdx_center = (zs[1:] - zs[:-1])/dx

    # k defined on centers
    k_center = -(2*A)*((rho*g)**n)*(h**(n+2))/(n+2)
    q_edge = np.zeros(n_edge)
    q_edge[1:-1] = (k_center[1:] + k_center[:-1])/2 * dzdx_center**n

    # Calculate mass balance
    if b=='linear':
        bdot = Gamma*(zs - zELA)
    else:
        bdot = Gamma*np.ones(n_center)

    if bcs[0]=='no-flux':
        # No flux boundary conditions
        q_edge[0] = 0
    elif bcs[0]=='free-flux':
        # q_edge[0] = k_center[0]*(dzdx_center[0])**n
        q_edge[0] = q_edge[1]
        q_edge[0] += bdot[0]*dx*np.sign(q_edge[0])

    if bcs[1]=='no-flux':
        q_edge[-1] = 0
    elif bcs[1]=='free-flux':
        # q_edge[-1] = k_center[-1]*dzdx_center[-1]**n
        q_edge[-1] = q_edge[-2]
        q_edge[-1] += bdot[1]*dx*np.sign(q_edge[-1])

    # Time derivative is flux divergence + mass balance
    hprime = -(q_edge[1:] - q_edge[:-1])/dx + bdot

    qval = q_edge[10]
    dval = q_edge[10]/dzdx_center[10]

    print('flux:', qval)
    print('diffusivity:', dval)
    return hprime


def drive_ice_flow(tt, xc, h, zb, Gamma=Gamma, zELA=zELA, method='odeRK', **kwargs):
    """
    Integrate ice-flow model forward in time for given time steps (tt) and
    initial ice depth h. Any arguments passed as keywords arguments are
    forwarded to rhs_1d function.
    """
    tstart = time.time()
    t = tt[0]
    tend = tt[-1]
    dt = tt[1] - tt[0]

    nsteps = int(tend/dt + 1)
    dx = xc[1] - xc[0]
    H = np.zeros((nsteps, len(xc)))
    H[0, :] = h
    i = 1

    # Time stepping parameters
    c2 = 1/2
    c3 = 1/2
    c4 = 1

    b1 = 1/6
    b2 = 1/3
    b3 = 1/3
    b4 = 1/6

    a21 = 0.5
    a31 = 0
    a32 = 0.5

    rhs_fun = lambda t, y: rhs_1d(t, y, zb, dx, Gamma=Gamma, zELA=zELA,**kwargs)
    while t<tend:
        h_old = h
        if method=='odeRK':
            k1 = rhs_fun(t, h)
            k2 = rhs_fun(t + c2*dt, h + dt*a21*k1)
            k3 = rhs_fun(t + c3*dt, h + dt*a31*k1 + dt*a32*k2)
            k4 = rhs_fun(t + c4*dt, h + dt*k3)

            dhdt = b1*k1 + b2*k2 + b3*k3 + b4*k4
            subset = h + dhdt<0
            dhdt[subset] = -h[subset]/dt
            h_new = h + dt*dhdt

        elif method=='BE':
            # Implicit method
            # Define function g which we will find the root of
            # h_new = scipy.optimize.brentq(g, h, maxiter=100)
            g = lambda z: z - h - dt*rhs_fun(t, z)
            h_new = scipy.optimize.newton(g, h, tol=1e-6)
            # dhdt = rhs_fun(t, h_new)
            h_new[h_new<0] = 0

        elif method=='CN':
            g = lambda z: z - h - 0.5*dt*(rhs_fun(t, z) + rhs_fun(t, h))
            h_new = scipy.optimize.newton(g, h, tol=1e-6)
            h_new[h_new<0] = 0

        elif method=='odeFE':
            dhdt = rhs_fun(t, h)
            subset = h + dhdt<0
            dhdt[subset] = -h[subset]/dt
            h_new = h + dt*dhdt

        h = h_new
        H[i, :] = h
        i+=1
        t+=dt

    rhs_1d_print_diff(t, h, zb, dx, Gamma=Gamma, zELA=zELA,**kwargs)

    dhdt = (h_new - h_old)/dt
    print('Maximum dh/dt (m/year):')
    print(np.max(np.abs(dhdt))*86400*365)
    tend = time.time()
    dtime = tend - tstart
    print('Elapsed time: ', dtime)
    return H
