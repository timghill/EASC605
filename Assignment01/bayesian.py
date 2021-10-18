"""
Bayesian parameter inference for energy-balance model
"""

import datetime
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

import EBM
import hourly_daily_averages
import MCMC

# PRIOR PARAMETERS
prior_mean = 2.4e-3
prior_sd = 2
prior_var = prior_sd**2

measure_sigma = 0.2

N_steps = 5000
discard_frac = 0.1


prior_dist = stats.uniform(loc=1e-6, scale = 1e-1)

time = np.load('data/daily/time.npy', allow_pickle=True)
x = [(t - time[0]).days for t in time]

y = 1.05*np.load('data/daily/melt.npy')

# Define the physical system
def simulator(theta):
    # print(theta)
    dt = 3600
    rhow = 1e3
    L_f = 3.34e5
    temp = np.load('data/hourly/temp.npy')
    LWnet = np.load('data/hourly/LWnet.npy')
    pressure = np.load('data/hourly/press.npy')*1e2
    SWin = np.load('data/hourly/SWin.npy')
    SWout = np.load('data/hourly/SWout.npy')
    wind = np.load('data/hourly/wind.npy')
    rain = np.load('data/hourly/rain.npy')
    RH = np.load('data/hourly/RH.npy')
    tt = np.load('data/hourly/time.npy', allow_pickle=True)
    hourly_energy_balance = EBM.energy_balance(temp, RH, wind, pressure, SWin, SWout,
        LWnet, rain, z_0=theta)

    melt = np.zeros(tt.shape)
    # print(melt.shape)
    for i in range(1, len(melt)):
        melt[i] = melt[i-1] + hourly_energy_balance.Qmelt[i-1]*dt/rhow/L_f

    daily_melt = hourly_daily_averages.average_hourly_to_daily(melt)
    # print(daily_melt.shape)

    return daily_melt

z = simulator(2.4e-3)

fig, ax = plt.subplots()
ax.plot(time, z)
ax.plot(time, simulator(1e-12))
ax.plot(time, np.load('data/daily/melt.npy'))

def likelihood(y, eta):
    return np.exp(-0.5*np.linalg.norm(y - eta)**2/(measure_sigma**2))

def posterior(theta):
    eta = simulator(theta)
    return likelihood(y, eta)*prior_dist.pdf(theta)
#
# thetas = np.logspace(-6, -1, 51)
#
# likelis = np.zeros(thetas.shape)
# for i in range(len(thetas)):
#     likelis[i] = np.linalg.norm(y - simulator(thetas[i]))**2
#     likelis[i] =  likelihood(y, simulator(thetas[i]))
#
# fig, ax = plt.subplots()
# ax.plot(thetas, likelis)
# plt.show()

# # print(simulator(0))
#
MCMC_model = MCMC.Model()
MCMC_model.jumping_model = lambda loc: max(1e-6, stats.norm.rvs(loc, scale=5e-4))
MCMC_model.sample_pdf = posterior

steps = int(N_steps)
discard = int(N_steps*discard_frac)
theta0 = 2.4e-3
# theta = MCMC_model.step(theta0)
MCMC_subsample = MCMC_model.chain(theta0, steps=steps, discard=discard)
np.save('data/MCMC.npy', MCMC_subsample)

# MCMC_subsample = np.load('data/MCMC.npy')

#
fig3, ax3 = plt.subplots()
ax3.plot(MCMC_subsample)

KDE = MCMC_model.calculate_pdf(MCMC_subsample)
pos = np.linspace(0, 0.02, 101)
pdist = KDE(pos)

fig2, ax2 = plt.subplots()
ax2.plot(pos, prior_dist.pdf(pos))
ax2.plot(pos, pdist)
ax2.grid()
ax2.set_xlabel('$z_0$')
ax2.set_ylabel('$\\pi(z_0)$')
fig2.savefig('figures/bayesian.png', dpi=600)
plt.show()
