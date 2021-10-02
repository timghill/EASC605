"""
Calculate and plot daily albedo values
"""

import datetime as dt

from matplotlib import pyplot as plt
import numpy as np

SWin = np.load('data/hourly/SWin.npy')
SWout = np.load('data/hourly/SWout.npy')
albedo = SWout/SWin
# albedo[SWin*SWout<100] = np.nan
# albedo[albedo>=1] = np.nan
time = np.load('data/hourly/time.npy', allow_pickle=True)
time_rounded = time.astype('datetime64[D]')

day0 = dt.date(2008, 5, 5)
day1 = dt.date(2008, 9, 13)
ndays = (day1 - day0).days
print(ndays)

time_daily = np.zeros(ndays+1, dt.datetime)
albedo_daily = np.zeros(ndays+1)
for i in range(ndays+1):
    day = day0 + dt.timedelta(days=i)
    tslice = np.logical_and(time_rounded>=day, time_rounded<(day + dt.timedelta(days=1)))
    time_daily[i] = day
    albedo_slice = albedo[tslice]
    albedo_daily[i] = np.nanmean(albedo_slice[albedo_slice<=1])

fig, ax = plt.subplots()
ax.plot(time_daily, albedo_daily)
ax.plot(time, albedo)

fig2, ax2 = plt.subplots()
ax2.plot(time, SWin)
ax2.plot(time, SWout)

fig3, ax3 = plt.subplots()
ax3.plot(time)

plt.show()
