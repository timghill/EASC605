#!/usr/bin/python3
"""
Calculate hourly-averaged meteorological data

Read the clean data (data/clean) and compute hourly averages
"""

import matplotlib
matplotlib.rcParams.update({'font.size': 10})
import numpy as np
import datetime
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

# Time variables
tt = np.load('data/clean/time.npy', allow_pickle=True)
tt_30min = np.load('data/clean/time_30min.npy', allow_pickle=True)
tt_office = np.load('data/clean/time_office.npy', allow_pickle=True)



# 5 min data
temp = np.load('data/clean/temp.npy')
RH = np.load('data/clean/RH.npy')
SWin = np.load('data/clean/SWin.npy')
SWout = np.load('data/clean/SWout.npy')
Rnet = np.load('data/clean/net_rad.npy')
wind = np.load('data/clean/wind_speed.npy')

LWnet = Rnet - (SWin - SWout)

# 30 min data
press = np.load('data/clean/pressure.npy')
rain = np.load('data/clean/rain.npy')


chunk_len = int(60/5)
n_steps = int(len(tt[2:])/chunk_len)
tt_clean = tt[2:]
tt_hourly = np.zeros(n_steps, datetime.datetime)
for i in range(n_steps):
    tt_hourly[i]= tt_clean[i*chunk_len]

tt_office_clean = tt_office[1:]
tt_office_hourly = np.zeros(int(len(tt_office_clean)/2), datetime.datetime)
for i in range(len(tt_office_hourly)):
    tt_office_hourly[i] = tt_office_clean[2*i]
# print(tt_office_hourly.shape)

#
# print(tt_30min[1:10])
# print(tt_30min[-10:])

def average_5min_to_hourly(x):
    # Neglect first observation because it is at 54 min of the hour,
    # next observation is at the hour
    x_clean = x[2:]
    n_obs = len(tt[2:])
    chunk_len = int(60/5)    # Number of observations to average

    n_steps = int(n_obs/chunk_len)
    x_avg = np.zeros(n_steps)
    for i in range(n_steps):
        x_avg[i] = np.nanmean(x_clean[i*chunk_len:(i+1)*chunk_len])

    return x_avg

def average_30min_to_hourly(x):
    chuck_len = 2
    n_obs = len(x)
    n_steps = int(n_obs/chunk_len)
    # print(n_steps)
    x_avg = np.zeros(n_steps)
    for i in range(n_steps):
        x_avg[i] = np.nanmean(x[i*chunk_len:(i+1)*chunk_len])

    return x_avg

def average_30min_off_ice(x):
    x = x[1:]
    n_obs = len(x)
    n_steps = int(n_obs/chunk_len)
    # print(n_steps)
    x_avg = np.zeros(n_steps)
    for i in range(n_steps):
        x_avg[i] = np.nanmean(x[i*chunk_len:(i+1)*chunk_len])

    n_obs_total = int(len(temp[2:])/int(60/5))
    x_avg_total = np.zeros(n_obs_total)
    x_avg_total[2:-2] = x_avg

    # x_avg_total[3:] = x_avg
    return x_avg_total

# Average the fields
hourly_temp = average_5min_to_hourly(temp)
hourly_rh = average_5min_to_hourly(RH)
hourly_SWin = average_5min_to_hourly(SWin)
hourly_SWout = average_5min_to_hourly(SWout)
hourly_LWnet = average_5min_to_hourly(LWnet)
hourly_press = average_30min_to_hourly(press)
hourly_rain = average_30min_off_ice(rain)
hourly_wind = average_5min_to_hourly(wind)

# print(hourly_rain.shape)
# print(hourly_temp.shape)

average_30min_off_ice(rain)

fig, axes = plt.subplots(nrows=7, figsize=(8, 10), sharex=True)
# ax.plot(tt, temp)

ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axes

min_month = 5
max_month = 9
xtick_loc = [datetime.datetime(2008, mm, 1) for mm in range(min_month, max_month+1)]

ax1.plot(tt_hourly, hourly_temp)
ax1.set_ylabel('T ($^\\circ$C)')
ax1.grid()
ax1.text(0.015, 0.8, 'a', transform=ax1.transAxes)

ax2.plot(tt_hourly, hourly_rh)
ax2.set_ylabel('RH (%)')
ax2.grid()
ax2.text(0.015, 0.8, 'b', transform=ax2.transAxes)


ax3.plot(tt_hourly, hourly_SWin)
ax3.set_ylabel('SW$_{in}$')
ax3.grid()
ax3.text(0.015, 0.8, 'c', transform=ax3.transAxes)


ax4.plot(tt_hourly, hourly_SWout)
ax4.set_ylabel('SW$_{out}$')
ax4.grid()
ax4.text(0.015, 0.8, 'd', transform=ax4.transAxes)

ax5.plot(tt_hourly, hourly_LWnet)
ax5.set_ylabel('LW$_{net}$')
ax5.grid()
ax5.text(0.015, 0.8, 'e', transform=ax5.transAxes)

ax6.plot(tt_hourly, hourly_press)
ax6.set_ylabel('P (Pa)')
ax6.grid()
ax6.text(0.015, 0.8, 'f', transform=ax6.transAxes)

ax7.plot(tt_hourly, hourly_rain)
ax7.text(0.015, 0.8, 'g', transform=ax7.transAxes)

ax7.set_ylabel('Rain (mm)')
ax7.grid()

axes[-1].set_xlabel('Date')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
axes[-1].set_xticks(xtick_loc)

plt.tight_layout()

fig.savefig('figures/hourly_average.png', dpi=600)

# Save the arrays
np.save('data/hourly/temp.npy', hourly_temp)
np.save('data/hourly/SWin.npy', hourly_SWin)
np.save('data/hourly/SWout.npy', hourly_SWout)
np.save('data/hourly/LWnet.npy', hourly_LWnet)
np.save('data/hourly/press.npy', hourly_press)
np.save('data/hourly/rain.npy', hourly_rain)
np.save('data/hourly/wind.npy', hourly_wind)
np.save('data/hourly/time.npy', tt_hourly)
np.save('data/hourly/RH.npy', hourly_rh)
