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
print('Flow parameter: %.3e' % B)

# PHYSICAL DOMAIN
L = 50e3
H = 500

N = 25
nz = 10

dx = L/N
dz = H/nz
dzeta = 1/nz

x = np.arange(0, L, dx)

# IMPOSED GEOMETRY
zb = 250 - 0.05*x
zs = zb + H
dSdx = -0.05*np.ones(x.shape)
h = zs - zb

h_diag = np.diagflat(np.tile(h, (nz, 1)).flatten())
inv_h_diag = np.diagflat(np.tile(1/h, (nz, 1)).flatten())


zeta = np.arange(0, 1, dzeta)

[xm, zetam] = np.meshgrid(x, zeta)

zm = zs - h*zetam

# z = np.arange(0, H, dz)

# [xm, zm] = np.meshgrid(x, z)
# zm

# CHANGE OF COORDINATES
# zeta = (zs - zm)/h
# dzeta = 1/nz

print(zeta)

# FD OPERATORS
Dx = np.diag(np.ones(N*nz)) - np.diag(np.ones(N*nz - 1), k=-1)
for i in range(nz):
    Dx[i*N] = Dx[i*N+1]
Dx *= 1/dx

Dz_outer = -np.diag(np.ones(N*nz)) + np.diag(np.ones(N*nz - N), k=N)
Dz_inner = (np.diag(np.ones(N*nz)) - np.diag(np.ones(N*nz - N), k=-N))
Dz_outer *= 1/dz
Dz_inner *= 1/dz


dHdx_pad = np.zeros((nz, N))

dSdx_pad = np.tile(dSdx, (nz, 1))
h_pad = np.tile(h, (nz, 1))
h_diag = np.diag(h_pad.flatten())
a_x = 1/h_pad*(dSdx_pad - zm*dHdx_pad)
Ax = np.diag(a_x.flatten())


# Dzeta = -np.matmul(inv_h_diag, Dz_inner)
Dzeta = -np.matmul(inv_h_diag, Dz_inner)*dz/dzeta
# Dxp = Dx + np.matmul(Ax, Dz)
Dxp = Dx


def calc_D_operator(visco):
    # Dx_zeta = Dx
    # Dz_zeta = Dz_outer
    D = (4*np.matmul(Dxp, np.matmul(visco, Dxp)) +
            np.matmul(Dzeta, np.matmul(visco, Dzeta)))
    return D

def calc_visco(u):
    dudx = np.matmul(Dx, u)
    dudz = np.matmul(Dzeta, u)
    eps_eff = np.sqrt( 0.25*(dudx)**2 + 0.5*(dudz)**2)
    visco = 0.5*B*eps_eff**( (1-n)/n)
    return np.diagflat(visco)

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
hm = zm - zb - dz
fig, ax = plt.subplots()
pc = ax.pcolor(xm, hm, np.log10(visco_arr), cmap=cmocean.cm.turbid)
ax.set_title('log_10 viscosity')
ax.set_ylabel('z (m)')
ax.set_xlabel('x')
fig.colorbar(pc)

fig, ax = plt.subplots()
u_mat = u.reshape((nz, N))
pc = ax.pcolor(xm, zetam, u_mat*356*86400)
ax.set_title('Velocity (m/a)')
ax.set_ylabel('z (m)')
ax.set_xlabel('x (m)')
fig.colorbar(pc)

fig, ax = plt.subplots()
pc = ax.pcolor(xm, hm, u_mat*365*86400)
ax.set_title('Velocity (m/a)')
ax.set_ylabel('z (m)')
ax.set_ylabel('x (m)')
fig.colorbar(pc)

fig, ax = plt.subplots()
ax.plot(u_mat[:, 5]*365*86400, hm[:, 5])

plt.show()
