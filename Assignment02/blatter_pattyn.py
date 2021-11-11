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

def solve_velocity(xm, zetam, h, zs):
    (nz, N) = xm.shape
    dx = xm[0, 1] - xm[0, 0]
    dzeta = zetam[1, 0] - zetam[0, 0]

    # Calculate h and 1/h matrices
    h_diag = np.diagflat(np.tile(h, (nz, 1)).flatten())
    inv_h_diag = np.diagflat(np.tile(1/h, (nz, 1)).flatten())

    # zm = zs - h*zetam

    # FD OPERATORS
    Dx = np.diag(np.ones(N*nz)) - np.diag(np.ones(N*nz - 1), k=-1)
    for i in range(nz):
        Dx[i*N] = Dx[i*N+1]
    Dx *= 1/dx

    # Dx2 = -np.diag(np.ones(N*nz)) + np.diag(np.ones(N*nz-1), k=1)
    # for i in range(nz):
    #     Dx2[i*N + (N-1)] = Dx2[i*N + (N-2)]

    # fig, ax = plt.subplots()
    # ax.pcolor(Dx2)
    # plt.show()

    Dz_outer = -np.diag(np.ones(N*nz)) + np.diag(np.ones(N*nz - N), k=N)
    Dz_inner = (np.diag(np.ones(N*nz)) - np.diag(np.ones(N*nz - N), k=-N))
    Dz_outer *= 1/dzeta
    Dz_inner *= 1/dzeta

    true_Dz_outer = Dz_outer.copy()

    dSdx = np.zeros(N)
    dSdx[1:] = (zs[1:] - zs[:-1])/dx
    dSdx[0] = dSdx[1]

    dHdx = np.zeros(N)
    dHdx[1:] = (h[1:] - h[:-1])/dx
    dHdx[0] = dHdx[1]

    dSdx_pad = np.tile(dSdx, (nz, 1))
    dHdx_pad = np.tile(dHdx, (nz, 1))

    h_pad = np.tile(h, (nz, 1))
    h_diag = np.diag(h_pad.flatten())
    a_x = 1/h_pad*(dSdx_pad - zetam*dHdx_pad)
    Ax = np.diag(a_x.flatten())

    Dzeta = -np.matmul(inv_h_diag, Dz_inner)
    # Dzeta_outer = Dzeta
    # Dzeta_outer[:N] = Dzeta_outer[N:2*N]
    Dzeta_outer = -np.matmul(inv_h_diag, Dz_outer)
    # Dz_outer = (-np.diag(np.ones(N*nz - N), k=-N) + np.diag(np.ones(N*nz - N), k=N))/2/dzeta
    # Dz_outer = Dz_inner.copy()
    # Dz_outer[:N] = Dz_inner[N:2*N]
    # Dzeta_outer = -np.matmul(inv_h_diag, Dz_outer)

    # Dzeta_outer[:(nz-1)*N] = Dzeta[:(nz-1)*N]

    # Dzeta_outer = Dzeta.copy()
    # Dzeta_outer = -Dz_outer
    # for i in range(N*nz):
    #     # if i<(N*nz)-1:
    #     havg = h_pad.flatten()[i-1]
    #     # if i<(nz-1)*N:
    #     #     hk = h_pad.flatten()[i + N]
    #     #     havg = hk
    #     # else:
    #     #     havg = h_pad.flatten()[i]
    #     Dzeta_outer[i] = Dzeta_outer[i]/havg
    # Dzeta_outer[:, (nz-1)*N:] = Dz_outer[:, (nz-1)*N:]
    # Dzeta_outer[(nz-10)*N:, :] = Dz_outer[(nz-10)*N:, :]
    # Dzeta_outer = -Dz_outer/500

    Dxp = Dx + np.matmul(Ax, Dzeta)
    Dxp2 = -Dx.T +np.matmul(Ax, Dzeta)
    # Dxp = Dx

    def calc_D_operator(visco):
        # Dx_zeta = Dx
        # Dz_zeta = Dz_outer
        D = (4*np.matmul(Dxp2, np.matmul(visco, Dxp)) +
                np.matmul(Dzeta, np.matmul(visco, Dzeta_outer)))

        # D = np.matmul(Dzeta, np.matmul(visco, Dzeta_outer))
        # D = np.matmul(Dxp, np.matmul(visco, Dxp))
        return D

    def calc_visco(u):
        dudx = np.matmul(Dxp, u)
        dudx[:N] = dudx[N:2*N]
        dudz = np.matmul(Dzeta_outer, u)
        # dudz[:N] = dudz[N:2*N]
        eps_eff = np.sqrt( 0.5*(dudx)**2 + 0.25*(dudz)**2)
        visco = 0.5*B*eps_eff**( (1-n)/n)
        return np.diagflat(visco)

        # return np.diagflat(1e12*np.ones((nz*N)))

    zs_pad = np.tile(zs, (nz, 1))
    dSdx = np.matmul(Dx, np.vstack(zs_pad.flatten()))
    Y = rho*g*np.vstack(dSdx)

    # FULL COMPLEXITY MODEL
    # Iterate to find a self-consistent viscosity solution
    err = 1e8
    tol = 1e-2/(365*86400)

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

        print('Iteration: ', itnum, 'Error:', err*365*86400)

    visco = calc_visco(u)
    # u = u - np.min(u)
    return (u, visco)

if __name__ == '__main__':
    slab = True
    valley = True

    if slab:
        """
        CASE: SLAB GLACIER + SENSITIVITY TESTS
        """

        # PHYSICAL DOMAIN
        L = 50e3
        H = 500

        N = 2
        nz = 20

        N = 2
        nz = 50

        dx = L/N
        dz = H/nz
        dzeta = 1/nz

        x = np.arange(0, L, dx)

        # IMPOSED GEOMETRY
        zb = 250 - 0.05*x
        zs = zb + H
        # dSdx = -0.05*np.ones(x.shape)
        h = zs - zb
        zeta = np.arange(dzeta, 1+dzeta, dzeta)

        [xm, zetam] = np.meshgrid(x, zeta)
        zm = zs - h*zetam

        u, visco = solve_velocity(xm, zetam, h, zs)
        visco_arr = np.diag(visco).reshape((nz, N))

        # u = u - np.min(u)
        hm = zm - zb
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

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(8, 6), ncols=2, nrows=2)
        pc = ax1.pcolor(xm, hm, u_mat*365*86400, cmap=cmocean.cm.speed)
        ax1.set_title('Velocity (m/a)')
        ax1.set_ylabel('z (m)')
        ax1.set_xlabel('x (m)')
        fig.colorbar(pc, ax=ax1)
        ax1.text(-0.1, 1.1, 'a', transform=ax1.transAxes, fontsize=14)

        pc = ax2.pcolor(xm, hm, np.log10(visco_arr), cmap=cmocean.cm.turbid)
        ax2.set_title('Viscosity (Pa s)')
        ax2.set_xlabel('x (m)')
        fig.colorbar(pc, ax=ax2)
        ax2.text(-0.1, 1.1, 'b', transform=ax2.transAxes, fontsize=14)

        ax3.plot(u_mat[:, 1]*365*86400, hm[:, 1])
        ax3.set_xlabel('u (m/s)')
        ax3.set_ylabel('z (m)')
        ax3.text(-0.1, 1.1, 'c', transform=ax3.transAxes, fontsize=14)
        ax3.grid()

        ax4.plot(np.log10(visco_arr[:, 1]), hm[:, 1])
        ax4.set_xlabel('log$_{10}\\eta$')
        ax4.set_ylabel('z (m)')
        ax4.text(-0.1, 1.1, 'd', transform=ax4.transAxes, fontsize=14)
        ax4.grid()

        print(visco_arr[:, 1])

        plt.tight_layout()

        fig.savefig('blatter_slab.png', dpi=600)

        plt.show()

        L = 50e3
        H = 500

        N = 2
        nz = 100

        dx = L/N
        dz = H/nz
        dzeta = 1/nz

        x = np.arange(0, L, dx)

        # IMPOSED GEOMETRY
        zb = 250 - 0.05*x
        zs = zb + H
        dSdx = -0.05*np.ones(x.shape)
        h = zs - zb
        zeta = np.arange(0, 1, dzeta)

        [xm, zetam] = np.meshgrid(x, zeta)
        zm = zs - h*zetam
        # hm2 = zm - zb - dz￼

        hm2 = zm - zb

        u_hr, visco_hr = solve_velocity(xm, zetam, h, zs)

        fig, ax = plt.subplots()
        pc = ax.pcolor(xm, zetam, u_hr.reshape((nz, N)))
        fig.colorbar(pc)
        ax.set_title('Raw zeta velocity')

        # print(np.min(u_hr))
        # u_hr = u_hr - np.min(u_hr)
        u_mat2 = u_hr.reshape((nz, N))

        fig, ax = plt.subplots()
        ax.plot(u_mat[:, 1]*365*86400, hm[:, 1], label='$\\Delta z = 10 \\mathrm{m}$')
        ax.plot(u_mat2[:, 1]*365*86400, hm2[:, 1], label='$\\Delta z = 2 \\mathrm{m}$')

        u_theory = 2*A*(rho*g)**n/(n+1) * (H**(n+1) - (H - hm[:, 1])**(n+1)) * (0.05)**n
        ax.plot(u_theory*365*86400, hm[:, 1], label='Exact')

        print(u_mat[0, 1]/u_theory[0])
        print(u_mat2[0, 1]/u_theory[0])

        ax.legend()
        ax.set_xlabel('u (m/a)')
        ax.set_ylabel('z (m)')
        ax.grid()

        fig.savefig('blatter_slab_profile.png', dpi=600)

        plt.show()

    if valley:

        """
        CASE: VALLEY GLACIER
        """

        L = 10e3
        N = 25
        nz = 20

        dx = L/N
        x = np.arange(0, L, dx)

        dzeta = 1/nz
        zeta = np.arange(0, 1, dzeta)

        xm, zetam = np.meshgrid(x, zeta)

        zb = 1500 + 750*(x-7.5e3)*(x-5e3)/(7.5e3*5e3)
        zs = 2300 - 0.05*x

        h = zs - zb
        zm = zs - h*zetam

        u, visco = solve_velocity(xm, zetam, h, zs)

        u_mat = u.reshape((nz, N))
        _year = 365*86400
        fig, ax = plt.subplots()
        pc = ax.pcolor(xm, zetam, u_mat*_year, cmap=cmocean.cm.speed)
        fig.colorbar(pc)

        fig, (ax, ax2) = plt.subplots(figsize=(8, 4), ncols=2)
        pc = ax.pcolor(xm/1e3, zm, u_mat*_year, cmap=cmocean.cm.speed)
        cbar = fig.colorbar(pc, ax=ax)
        cbar.set_label('Velocity (m/a)')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('z (m)')
        ax.text(-0.2, 1.1, 'a', transform=ax.transAxes, fontsize=14)
        z_base = zm[-1, :]
        z_lower = 1400*np.ones(z_base.shape)
        # ax.plot(x/1e3, zm[-1, :], 'k')
        ax.fill_between(x/1e3, z_lower, z_base, facecolor=0.5*np.ones(3))

        ax.set_ylim([1400, 2400])

        kk = 10
        hm = zm - zb
        H = h[kk]
        # hi = hm[:, 20]
        u_theory =  2*A*(rho*g)**n/(n+1) * (H**(n+1) - (H - hm[:, kk])**(n+1)) * (0.05)**n
        ax2.plot(u_theory*_year, hm[:, kk], label='SIA')
        ax2.plot(u_mat[:, kk]*_year, hm[:, kk], label='BP')
        ax2.set_xlabel('u (m/a)')
        ax2.set_ylabel('z - z$_b$(m)')
        ax2.grid()
        ax2.legend()
        ax2.text(-0.2, 1.1, 'b', transform=ax2.transAxes, fontsize=14)
        plt.tight_layout()

        fig.savefig('blatter_valley.png', dpi=600)
        plt.show()
