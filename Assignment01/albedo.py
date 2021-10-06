"""
Calculate and plot daily albedo values
"""

import datetime as dt

from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection
import matplotlib.dates as mdates

import numpy as np

# PARAMETERS
rho_snow = 500
rho_ice = 910
rho_water = 1000
_day = 86400

height = np.load('data/hourly/height.npy')
ebm_melt = np.load('data/hourly/ebm_melt.npy')
SWin = np.load('data/hourly/SWin.npy')
SWout = np.load('data/hourly/SWout.npy')
temp = np.load('data/hourly/temp.npy')
albedo = SWout/SWin
albedo[SWin*SWout<100] = np.nan
albedo[albedo>=1] = np.nan
time = np.load('data/hourly/time.npy', allow_pickle=True)

day0 = dt.date(2008, 5, 5)
day1 = dt.date(2008, 9, 13)
ndays = (day1 - day0).days

time_daily = np.array([day0 + dt.timedelta(days=i) for i in range(ndays+1)])

chunk_len = 24
chunk_0 = albedo[:5]

albedo_clean = albedo[5:-19]
n_steps = int(albedo_clean.shape[0]/24)

mean_albedo = np.zeros(n_steps+2)
mean_albedo[0] = np.nanmean(chunk_0)

height_clean = height[5:-19]
# print(albedo_clean.shape)
mean_height = np.zeros(n_steps+2)
mean_height[0] = np.nanmean(height[:5])

ebm_clean = ebm_melt[5:-19]
mean_ebm = np.zeros(n_steps+2)
mean_ebm[0] = np.nanmean(ebm_melt[:5])

temp_clean = temp[5:-19]
mean_temp = np.zeros(n_steps+2)
mean_temp[0] = np.nanmean(temp[:5])

PDD = np.zeros(n_steps+2)
PDD[0] = np.sum(temp[:5][temp[:5]>0])

DT = 1/24

for i in range(n_steps+1):
    albedo_vals = albedo_clean[i*chunk_len:(i+1)*chunk_len]
    mean_albedo[i+1] = np.nanmean(albedo_vals)

    height_vals = height_clean[i*chunk_len:(i+1)*chunk_len]
    mean_height[i+1] = np.nanmean(height_vals)

    ebm_vals = ebm_clean[i*chunk_len:(i+1)*chunk_len]
    mean_ebm[i+1] = np.nanmean(ebm_vals)

    temp_vals = temp_clean[i*chunk_len:(i+1)*chunk_len]
    mean_temp[i+1] = np.nanmean(temp_vals)

    PDD[i+1] = PDD[i] + np.sum(temp_vals[temp_vals>0]*DT)

mean_temp[-1] = np.nanmean(temp[-19:])
mean_ebm[-1] = np.nanmean(ebm_melt[-19:])
mean_albedo[-1] = np.nanmean(albedo[-19:])
mean_height[-1] = np.nanmean(height[-19:])
PDD[-1] = PDD[-2]

np.save('data/daily/albedo.npy', mean_albedo)
np.save('data/daily/ebm_melt.npy', mean_ebm)
np.save('data/daily/temp.npy', mean_temp)
np.save('data/daily/PDD.npy', PDD)

# Find snow-covered days
snow_day_mask = mean_albedo>0.35
snow_day_indices = np.where(mean_albedo>0.35)[0]
snow_days = time_daily[snow_day_mask]

np.save('data/daily/snow_mask.npy', snow_day_mask)

fig, ax = plt.subplots()
ax.plot(time_daily, mean_albedo, color='#756bb1')
ax.axhline(0.35, color=(0.5, 0.5, 0.5), linestyle='--')

n_snow_days = len(snow_days)
rectangles = []
for i in range(n_snow_days-1):
    k = snow_day_indices[i]
    t0 = mdates.date2num(time_daily[k])
    t1 = mdates.date2num(time_daily[k+1])
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

fig.savefig('figures/daily_average_albedo.png', dpi=600, sharex=True)

# Calculate melt from SR50 record
inst_melt = np.zeros(mean_height.shape)
tot_melt = np.zeros(mean_height.shape)
for i in range(1, len(mean_height)):
    dh = np.max((0, float(mean_height[i] - mean_height[i-1])))
    if snow_day_mask[i]:
        melt_contrib = dh*rho_snow/rho_water
    else:
        melt_contrib = dh*rho_ice/rho_water
    inst_melt[i] = melt_contrib/_day
    tot_melt[i] = tot_melt[i-1] + melt_contrib

mwe_2_cm_day = 86400*1e2
inst_melt_cm_day = inst_melt*mwe_2_cm_day
# tot_melt_cm_day = tot_melt*mwe_2_cm_day


fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)
ax.plot(time_daily, inst_melt_cm_day)

ax2.plot(time_daily, tot_melt)
ax2.plot(time_daily, mean_ebm)
ax.set_ylabel('$\\dot m$ (cm/day)')
ax.set_ylim([0, 15])
ax.grid()
ax.text(0.02, 0.9, 'a', transform=ax.transAxes)

ax2.set_xlabel('Date')
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
min_month = 5
max_month = 9
xtick_loc = [dt.datetime(2008, mm, 1) for mm in range(min_month, max_month+1)]
ax2.set_xticks(xtick_loc)
ax2.set_ylabel('Melt (m w.e.)')
ax2.set_ylim([0, 2.5])
ax2.grid()
ax2.text(0.02, 0.9, 'b', transform=ax2.transAxes)

np.save('data/daily/time.npy', time_daily)
np.save('data/daily/melt.npy', tot_melt)

fig.savefig('figures/daily_SR50_melt_record.png', dpi=600)
