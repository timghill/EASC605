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
Q0 = 25


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

# INITIAL CONDITIONS
# pw = 0.5*pi
pw_max = 0.5*pi[0]
pw_min = 0
pw = pw_max - pw_max*((x)/(L))**(1/2)
pw0 = pw.copy()

# print(pw)

# fig, ax = plt.subplots()
# ax.plot(x, pw)
# plt.show()

# Probably need to initialize with steady state pw!!
# pw = np.zeros(N)

# print(gamma)

S = 1 + 0/L*(x + dx)
for i in range(3):

    phi = pw# + rhow*g*zb

    # dphids = ds_downwind(phi)
    # dpwds = ds_upwind2(pw)
    dpwds = ds_downwind(pw)
    dphids = dpwds
    dzbds = np.zeros(N)#(1/rhow/g)*(dphids - dpwds)

    r = np.sqrt(S/np.pi)

    Q = -S*(r/fR/rhow)**(1/2)*dphids*np.abs(dphids)**(-0.5)
    dQds = ds_upwind(Q, Q0)

    mi = (Q/Lf)*(-rhow*g*dzbds - (1 - gamma)*dpwds)
    # print(mi)
    p_arg = 1/(2*S*A) * (mi*(1/rhoi - 1/rhow) + dQds)
    pw_new = pi - n*np.abs(p_arg )**((1/n) - 1) * p_arg
    # print(n)
    print(np.max(np.abs(pw_new - pw)))
    pw = pw_new


dSdt = (mi/rhoi) - 2*S*A*( (pi - pw)/n)**n
# print(np.max(dSdt))
fig, ax = plt.subplots()
ax.plot(x, Q)
# ax.plot(x, pw)
# ax.plot(x, pi - pw)
# print(pi - pw)

plt.show()

#
# Q = Q0 + 0/L*(x + dx)
# m = 0
#
# dQds = ds_upwind(Q, ghost_value=Q0)
#
# pw = pi - (1/(2*S*A)*(dQds))**(1/n)
#
# fig, ax = plt.subplots()
# ax.plot(x, pw, label='Water')
# ax.plot(x, pi, label='Ice')
# ax.legend()
# plt.show()
