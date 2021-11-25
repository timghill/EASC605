"""
Simple outburst flood model
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
S = 0.1 + 0.1*x/L

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
for i in range(10):
    D = D_upwind + (1/rhoi - 1/rhow)/Lf*(gamma - 1)*np.diag(dphids)
    y = np.vstack(2*S*A*( (p_i - pw)/n)**n)
    y[0] = y[0] + Q0/dx # Boundary condition on Q[0]

    Q = np.linalg.solve(D, y).flatten()

    # Calculate instantaneous area
    S_new = (Q**(4/5)*(fR*rhow)**(2/5)*np.pi**(1/5))/np.abs(dphids)**(2/5)
    err = np.max(np.abs(S - S_new))
    # print('Cross-sectional area change:', np.max(S - S_new))
    S = S_new
    itnum += 1

S = (Q**(4./5)*(fR*rhow)**(2./5)*np.pi**(1./5))/np.abs(dphids)**(2/5)
Q_recons = S**(5/4)*(1/fR/rhow)**(1/2)*(1/np.pi)**(1/4)*np.abs(dphids)**(0.5)
fig, ax = plt.subplots()
ax.plot(x, Q, label='Q')
ax.plot(x, Q_recons, label='Reconstructed')
ax.legend()
ax.set_title('Q verification')\

fig, ax = plt.subplots()
ax.plot(x, S)
ax.set_title('S')

# IMPOSED CONDITIONS for Q and S. Solve for pressure
dQds = ds_upwind(Q, Q0)

# FIRST: use the analytic expression to calculate pw to make sure
# the expression is correct
mi = Q/Lf*(gamma-1)*dpwds
p_arg = (1/(2*S*A))*(mi*(1/rhoi - 1/rhow) + dQds)
pw_recons = p_i - n*np.abs(p_arg)**(1/n-1)*p_arg

fig, ax = plt.subplots()
ax.plot(x, pw, label='pw')
ax.plot(x, pw_recons, label='recons')
ax.legend()
ax.set_title('Pressure verification')

# # Iterative procedure for p
# for j in range(10):
#     dpwds = ds_downwind(pw)
#
#     mi = Q/Lf*(gamma-1)*dpwds
#     p_arg = (1/(2*S*A))*(mi*(1/rhoi - 1/rhow) + dQds)
#     pw_new = p_i - n*np.abs(p_arg)**(1/n-1)*p_arg
#
#     print(np.max(np.abs(pw_new - pw)))
#     pw = pw_new

def objective(z):
    dpwds = np.matmul(D_downwind, np.vstack(z)).flatten()
    mi = Q/Lf*(gamma-1)*dpwds
    return mi/rhoi - 2*S*A*((p_i - z)/n)**n + dQds - mi/rhow

p_guess = pw_max - pw_max*((x)/(L))
R = scipy.optimize.newton(objective, p_guess, maxiter=100, tol=1e-10)

fig, ax = plt.subplots()
ax.plot(x, R, label='Solution')
ax.plot(x, pw, label='Imposed')
ax.legend()
ax.set_title('Pressure solve')
plt.show()
