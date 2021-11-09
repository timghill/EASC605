"""
One-dimensional blatter-pattyn model using sigma coordinates
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg

import cmocean

# CONSTANTS
g = 9.81
rho = 910
n = 3

# PARAMETERS
A = 2.4e-24
# B = 1/A
# B = A**(-1/n)
B = (1/A)**(1/n)
print('Flow parameter:', B)

# PHYSICAL DOMAIN
L = 50e3
N = 25
nz = 10
dx = L/N
dz = 1/nz
x = np.arange(0, L, dx)
z = np.arange(0, 1, dz)

[xm, zm] = np.meshgrid(x, z)
# zb = np.zeros(N)
zb = 250 - 0.05*x

# SIMPLIFICATION!! NEWTONION FLUID
visco = 1e12*np.diag(np.ones(N*nz))    # Pa.s

# INITIAL CONDITIONS
# zs = 750 - 250*np.cos(2*np.pi*x/L)
# zs = 750 - (250/L)*x
zs = zb + 500
# print(zs)
# dSdx = 250*2*np.pi/L*np.sin(2*np.pi*x/L)
dSdx = -0.05 * np.ones(x.shape)
h = zs - zb

# N = 5
Dx = np.diag(np.ones(N*nz)) - np.diag(np.ones(N*nz - 1), k=-1)
for i in range(nz):
    Dx[i*N] = Dx[i*N+1]
Dx *= 1/dx

Dz_outer = -np.diag(np.ones(N*nz)) + np.diag(np.ones(N*nz - N), k=N)
Dz_inner = (np.diag(np.ones(N*nz)) - np.diag(np.ones(N*nz - N), k=-N))
Dz_outer *= 1/dz/500
Dz_inner *= 1/dz/500

Dz = Dz_outer

dHdx_pad = np.zeros((nz, N))

dSdx_pad = np.tile(dSdx, (nz, 1))
h_pad = np.tile(h, (nz, 1))
h_diag = np.diag(h_pad.flatten())
a_x = 1/h_pad*(dSdx_pad - zm*dHdx_pad)
Ax = np.diag(a_x.flatten())

inv_h_diag = np.diag((1/h_pad).flatten())

def calc_D_operator(visco):
    Dx_zeta = Dx
    Dz_zeta = Dz_outer

    D = (4*np.matmul(Dx_zeta, np.matmul(visco, Dx_zeta)) +
            np.matmul(Dz_zeta, np.matmul(visco, Dz_outer)))
    return D

def calc_visco(u):
    dudx = np.matmul(Dx, u)
    dudz = np.matmul(Dz_outer, u)
    eps_eff = np.sqrt( 0.25*(dudx)**2 + 0.5*(dudz)**2)
    visco = 0.5*B*eps_eff**( (1-n)/n)
    return np.diagflat(visco)


Dx_zeta = Dx + np.matmul(Ax, Dz)
zs_pad = np.tile(zs, (nz, 1))
dSdx = np.matmul(Dx, np.vstack(zs_pad.flatten()))
Y = rho*g*np.vstack(dSdx)

# FULL COMPLEXITY MODEL
# Iterate to find a self-consistent viscosity solution
err = 1e8
tol = 1e0/(365*86400)

# First iteration
visco = 1e12*np.diag(np.ones(nz*N))
D = calc_D_operator(visco)
u = np.linalg.solve(D, Y)
itnum = 0
maxiter = 100
while err>tol and itnum<maxiter:
# while itnum<maxiter:
    # Calculate viscosity with previous iteration velocity
    visco = calc_visco(u)

    # Calculate new velocity with updated viscosity
    D = calc_D_operator(visco)
    u_new = np.linalg.solve(D, Y)
    err = np.linalg.norm(u_new - u)

    # Update iteration counter and velocity vector
    itnum += 1
    u = u_new

    print('Iteration: ', itnum, 'Error:', err)


visco = calc_visco(u)
visco_arr = np.diag(visco).reshape((nz, N))

u = u - np.min(u)
fig, ax = plt.subplots()
pc = ax.pcolor(np.log10(visco_arr), cmap=cmocean.cm.turbid)
ax.set_title('log_10 viscosity')
fig.colorbar(pc)

fig, ax = plt.subplots()
u_mat = u.reshape((nz, N))
pc = ax.pcolor(u_mat*356*86400)
ax.set_title('Velocity (m/a)')
fig.colorbar(pc)

fig, ax = plt.subplots()
ax.plot(u_mat[:, 5]*365*86400, z)


plt.show()
