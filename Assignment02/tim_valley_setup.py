"""
Parameters and initial conditions for valley glacier test of
shallow ice approximation solver
"""

# DOMAIN
L = 35e3        # Domain length (m)
N = 100         # Number of grid centre points
dx = L/N        # Grid resolution (350 m)

xc = np.arange(dx/2, L+dx/2, dx)    # Cell centre coordinates (175, 525, ... )
xe = np.arange(0, L+dx, dx)         # Cell edge coordinates (0, 350, ... )
b0 = 1800                           # Highest bed elevation (m)
bed_slope = 0.05                    # Bed slope (m/m)
zb = b0 - bed_slope*xc              # Bed elevation (m)
zELA = 1400                         # ELA elevation (m)
Gamma = 0.007/(365*86400)           # Mass balance gradient (s-1)

# TIME STEPPING
t0 = 0                              # Initial time (0)
tend = 1e3*365*86400                # End time: integrate for 1000 years to reach steady state
dt = 5*86400                        # Timestep = 5 days

# Initial conditions - the model is robust enough we can be naive about
# the initial state and let it grow from almost nothing
h0 = 50*np.ones(xc.shape)           # Uniform 50 m thickness

# Zero thickness near domain edge so that no-flow BCs make sense - this is
# not strictly necessary
h0[(N-5):] = 0

"""
My ice flow model codes also includes the following parameters:

# CONSTANTS
g = 9.81                    # Gravity; m/s2
rho = 910                   # Ice density: kg/m3
n = 3                       # Ice flow exponent: -

# PARAMETERS
A = 2.4e-24                 # Ice flow coefficient (Pa-3.s-1)
Gamma = 0.007/(365*86400)   # Mass balance gradient (s-1)
zELA = 1400
"""
