"""
Maximum likelihood estimate
"""

import datetime
import numpy as np
from scipy import stats
from scipy import interpolate
from scipy import integrate
from scipy import optimize
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection
import matplotlib.dates as mdates


import EBM
import hourly_daily_averages
import MCMC
import plots

measure_sigma = 0.1
N = 100
factor = 1.035

time = np.load('data/daily/time.npy', allow_pickle=True)
x = [(t - time[0]).days for t in time]

y = factor*np.load('data/daily/melt.npy')

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

def likelihood(y, eta):
    return np.exp(-0.5*np.linalg.norm(y - eta)**2/(measure_sigma**2))

z0s = np.logspace(-5, -1, N)
# likelis = np.zeros(N)
# for i in range(N):
#     print(i)
#     eta = simulator(z0s[i])
#     likelis[i] = likelihood(y, eta)
# np.save('data/likelihood.npy', likelis)

likelis = np.load('data/likelihood.npy')

F_likeli = interpolate.interp1d(z0s, likelis, kind='cubic', fill_value=np.array([-1]), bounds_error=False)

lambda1 = lambda x: 1e3*F_likeli(x)
res_integrate = integrate.quad(lambda1, z0s[0], z0s[-1])
likeli_area = res_integrate[0]/1e3

F_scaled = lambda x: F_likeli(x)/likeli_area

# Find maximum likelihood estimate
objective = lambda x: -F_scaled(x)
x0_ind = np.argmin(objective(z0s))
x0 = z0s[x0_ind]


print(x0)
res = optimize.minimize(objective, x0, method='SLSQP', tol=1e-14)
z0_mle = res.x
print(res)
print('Maximum likelihood estimate:', z0_mle)

z0_fine = np.logspace(-5, -1, 101)
fig, ax_main = plt.subplots(figsize=(8, 4))
ax = ax_main.inset_axes([0.65, 0.13, 0.3, 0.3], zorder=10)
ax.plot(z0_fine, F_scaled(z0_fine), color='#cb181d')
ax.axvline(z0_mle, color=(0.2, 0.2, 0.2))

percentiles = np.arange(0.05, 1, 0.05)
z0_percentiles = np.zeros(percentiles.shape)

events = []
fun = lambda t, y: F_scaled(t)
t_eval = (z0s[0], z0s[-1])
y0 = np.array([0])
for i,pp in enumerate(percentiles):
    event = lambda t, y: float(y) - pp
    res = integrate.solve_ivp(fun, t_eval, y0, events=event)
    z0_percentiles[i] = float(res.t_events[0])
print(z0_percentiles)
rectangles = [patches.Rectangle((z0_percentiles[0], 0), z0_percentiles[-1]-z0_percentiles[0], 400)]
pc = PatchCollection(rectangles, facecolor='#fb6a4a', alpha=0.5, edgecolor=None)
ax.add_collection(pc)

ax.set_xlabel('$z_0$')
# ax.set_ylabel('$\\pi(z_0)$')
ax.set_xlim([0, 0.02])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.set_yticks([])
# ax.set_ylim([0, 400])
# ax.grid()


# print('5th, 95th percentiles:')
# print(z0_percentiles[0], z0_percentiles[-1])

melt_MLE = simulator(z0_mle)
melt_5th = simulator(z0_percentiles[0])
melt_95th = simulator(z0_percentiles[-1])
melt_ti = factor*np.load('data/daily/ti_melt.npy')

ax_main.plot(time, y, label='Mod. SR50')
ax_main.plot(time, melt_MLE, label='Tuned EBM')
ax_main.plot(time, melt_ti, label='Mod. TI')
ax_main.fill_between(mdates.date2num(time), melt_5th, melt_95th, facecolor='#fec44f', edgecolor=None)
ax_main = plots.set_axes(ax_main)
ax_main = plots.set_xticks(ax_main)
ax_main.set_ylim([0, 2.2])
ax_main.set_xlabel('Date')
ax_main.set_ylabel('Melt (m w.e.)')
ax_main.legend()
fig.savefig('posterior_roughness.png', dpi=600)
plt.show()


def rmse(x):
    return np.sqrt(np.mean( (x-y)**2))

ebm_rmse = rmse(melt_MLE)
ti_rmse = rmse(melt_ti)

print('RMS errors')
print(ebm_rmse, ti_rmse)
