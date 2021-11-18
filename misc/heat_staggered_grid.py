"""
Solve nonlinear heat equation on a one-dimensional staggered grid

Solves the equation:
    du/dt = k(u^n)(du/dt)

for variable power n. n = 0 corresponds to the linear heat equation,
n > 0 varies the strength of the nonlinearity
"""

import numpy as np
from matplotlib import pyplot as plt

# PARAMETERS
D = 1       # Constant diffusivity
power = 1   # Nonlinear exponent. Try n =1 too!

# DOMAIN
L = 5       # Domain length (m)
N = 100     # Number of cells
dx = L/N    # Grid spacing

# Construct center and edge coordinate arrays
xc = np.arange(dx/2, L + dx/2, dx)
xe = np.arange(0, L+dx, dx)

# TIMESTEPPING
t = 0
tend = 1
# Obey maximum CFL stability condition. Explore what happens for larger
# timestep! For even more nonlinear equations (e.g. ice flow), generally
# set this by trial and error
dt = dx**2/D/2.5
# print('Timestep:', dt)

# INITIAL CONDITIONS
u = np.zeros(N)
u[int(N/4):int(3*N/4)] = 1
u0 = u  # Store initial condition vector

# SOLUTION
while t<tend:
    # Calculate derivatives on all interfaces including boundaries
    q_edge = np.zeros(N+1)

    # Interpolate the conserved quantity u onto cell interfaces by
    # averaging the neighbouring cell center values
    u_edge_interp = 0.5*(u[1:] + u[:-1])

    # Calculate du/dx on interfaces
    dudx = (u[1:] - u[:-1])/dx

    # Calculate flux on interior interfaces
    q_interior = u_edge_interp**power * dudx
    q_edge[1:-1] = q_interior

    # Boundary conditions - be explicit about no flux boundary conditions
    q_edge[0] = 0
    q_edge[-1] = 0

    dqdx = (q_edge[1:] - q_edge[:-1])/dx

    dudt = D*dqdx

    # Timestepping - explicit forward Euler
    u = u + dt*dudt
    t = t+dt

# Check mass conservation
m0 = dx*u0.sum()
m = dx*u.sum()

print('Relative mass conservation:', m/m0)

fig, ax = plt.subplots()
ax.plot(xc, u0, label='t = 0')
ax.plot(xc, u, label = 't = 1')
ax.legend()
ax.grid()
ax.set_ylim([0, 1.2])
plt.show()
