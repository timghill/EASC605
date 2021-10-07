"""
Calculate energy balance and corresponding melt timeseries from
hourly-averaged data.

Uses functions in EBM.py module to calculate hourly energy balance, and
converts this to melt (units of m w.e.). Saves the melt timeseries as
data/hourly/ebm_melt.npy.
"""

import datetime

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import EBM

temp = np.load('data/hourly/temp.npy')
LWnet = np.load('data/hourly/LWnet.npy')
pressure = np.load('data/hourly/press.npy')*1e2
SWin = np.load('data/hourly/SWin.npy')
SWout = np.load('data/hourly/SWout.npy')
wind = np.load('data/hourly/wind.npy')
time = np.load('data/hourly/time.npy', allow_pickle=True)
rain = np.load('data/hourly/rain.npy')
RH = np.load('data/hourly/RH.npy')

# Calculate hourly energy balance
energy_balance = EBM.energy_balance(temp, RH, wind, pressure, SWin, SWout,
    LWnet, rain)

fig, axes = plt.subplots(figsize=(8, 8), nrows=6, sharex=True)

comp_order = ['QSW', 'QLW', 'QH', 'QE', 'Qrain', 'Qmelt']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f']

for i in range(len(comp_order)):
    timeseries = getattr(energy_balance, comp_order[i])
    axes[i].plot(time, getattr(energy_balance, comp_order[i]))
    axes[i].set_ylabel(comp_order[i] + ' (W/m$^2$)')
    axes[i].grid()
    axes[i].text(0.015, 0.85, alphabet[i], transform=axes[i].transAxes)

axes[-1].set_xlabel('Date')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
min_month = 5
max_month = 9
xtick_loc = [datetime.datetime(2008, mm, 1) for mm in range(min_month, max_month+1)]
axes[-1].set_xticks(xtick_loc)
plt.tight_layout()
fig.savefig('figures/energy_balance_partitioning.png', dpi=600)

# Calculate total melt
dt = 3600
rhow = 1e3
L_f = 3.34e5

# By component
tot_QE = np.sum(dt*energy_balance.QE[energy_balance.Qmelt>0])
tot_QH = np.sum(dt*energy_balance.QH[energy_balance.Qmelt>0])
tot_Qrad = np.sum(dt*energy_balance.Qrad[energy_balance.Qmelt>0])
tot_SW = np.sum(dt*energy_balance.QSW[energy_balance.Qmelt>0])
tot_LW = np.sum(dt*energy_balance.QLW[energy_balance.Qmelt>0])
tot_Qrain = np.sum(dt*energy_balance.Qrain[energy_balance.Qmelt>0])
tot_Qmelt = np.sum(dt*energy_balance.Qmelt[energy_balance.Qmelt>0])

frac_QE = tot_QE/tot_Qmelt
frac_QH = tot_QH/tot_Qmelt
frac_LW = tot_LW/tot_Qmelt
frac_SW = tot_SW/tot_Qmelt
frac_Qrain = tot_Qrain/tot_Qmelt

print('Fractional melt components:')
print('QE:\t', frac_QE)
print('QH:\t', frac_QH)
print('QSW:\t', frac_SW)
print('QLW:\t', frac_LW)
print('Qrain:\t', frac_Qrain)

print('\nTotal melt (m w.e.):')
print(tot_Qmelt/rhow/L_f)

melt = np.zeros(time.shape)

for i in range(1, len(melt)):
    melt[i] = melt[i-1] + energy_balance.Qmelt[i-1]*dt/rhow/L_f
np.save('data/hourly/ebm_melt.npy', melt)
