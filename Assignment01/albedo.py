#!/usr/bin/python3
"""
Calculate and plot daily albedo values
"""

import datetime as dt

from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection
import matplotlib.dates as mdates

import numpy as np

_max_albedo = 0.35

height = np.load('data/daily/height.npy')
SWin = np.load('data/daily/SWin.npy')
SWout = np.load('data/daily/SWout.npy')
temp = np.load('data/daily/temp.npy')
albedo = SWout/SWin
# albedo[SWin*SWout<100] = np.nan
albedo[albedo>=1] = np.nan
time = np.load('data/daily/time.npy', allow_pickle=True)

# Find snow-covered days
snow_day_mask = albedo>0.35
snow_day_indices = np.where(albedo>_max_albedo)[0]
snow_days = time[snow_day_mask]

np.save('data/daily/snow_mask.npy', snow_day_mask)

fig, ax = plt.subplots()
ax.plot(time, albedo, color='#756bb1')
ax.axhline(_max_albedo, color=(0.5, 0.5, 0.5), linestyle='--')

n_snow_days = len(snow_days)
rectangles = []
for i in range(n_snow_days-1):
    k = snow_day_indices[i]
    t0 = mdates.date2num(time[k])
    t1 = mdates.date2num(time[k+1])
    R = patches.Rectangle((t0, 0), t1-t0, 1)
    rectangles.append(R)

pc = PatchCollection(rectangles, facecolor='#bcbddc', alpha=0.5, edgecolor=None)
ax.add_collection(pc)
ax.grid()

ax.set_xlabel('Date')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
min_month = 5
max_month = 9
xtick_loc = [dt.datetime(2008, mm, 1) for mm in range(min_month, max_month+1)]
ax.set_xticks(xtick_loc)
ax.set_ylabel('Albedo')
ax.set_ylim([0, 1])

plt.show()
fig.savefig('figures/daily_average_albedo.png', dpi=600, sharex=True)
