
import datetime as dt

from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection
import matplotlib.dates as mdates

import numpy as np

rho_snow = 500
rho_ice = 910
rho_water = 1000

sr50_melt = np.load('data/daily/melt.npy')
ebm_melt = np.load('data/daily/ebm_melt.npy')
snow_mask = np.load('data/daily/snow_mask.npy')
temp = np.load('data/daily/temp.npy')
PDD = np.load('data/daily/PDD.npy')
time = np.load('data/daily/time.npy', allow_pickle=True)
print(temp.shape)
print(time.shape)
# TI MODEL
PDD_ice = 0
PDD_snow = 0

melt_ice = 0
melt_snow = 0

ntimes = len(time)
PDD_ice_series = np.zeros(ntimes-1)
PDD_snow_series= np.zeros(ntimes-1)

melt_ice_series = np.zeros(ntimes-1)
melt_snow_series = np.zeros(ntimes-1)

for i in range(len(temp)-1):
    d_melt = sr50_melt[i+1] - sr50_melt[i]
    if d_melt>0:
        if snow_mask[i]:
            PDD_snow += PDD[i+1] - PDD[i]
            melt_snow += d_melt
        else:
            PDD_ice += PDD[i+1] - PDD[i]
            melt_ice += d_melt

    PDD_ice_series[i] = PDD_ice
    PDD_snow_series[i] = PDD_snow
    melt_ice_series[i] = melt_ice
    melt_snow_series[i] = melt_snow


print('PDD factors:')
DDF_snow = melt_snow/PDD_snow
DDF_ice = melt_ice/PDD_ice

print('DDF snow:', DDF_snow)
print('DDF ice:', DDF_ice)

print('LSQ DDF factors')
DDF_lsq_snow = np.linalg.lstsq(np.vstack(PDD_snow_series),
    np.vstack(melt_snow_series))[0][0][0]
DDF_lsq_ice = np.linalg.lstsq(np.vstack(PDD_ice_series),
    np.vstack(melt_ice_series))[0][0][0]

print('DDF snow:', DDF_lsq_snow)
print('DDF ice:', DDF_lsq_ice)
#
# fig, ax = plt.subplots()
# ax.plot(

fig, ax = plt.subplots()
# ax.plot(time[1:], melt_ice_series)
# ax.plot(time[1:], melt_snow_series)

# ax.plot(time[1:], PDD_ice_series)
# ax.plot(time[1:], PDD_snow_series)

ax.plot(PDD_ice_series, melt_ice_series)
ax.plot(PDD_snow_series, melt_snow_series)
ax.set_xlabel('Positive Degree Days (day C)')
ax.set_ylabel('Melt (m w.e.)')

melt_TI = np.zeros(time.shape)
melt_TI[1:] = DDF_lsq_snow*PDD_snow_series + DDF_lsq_ice*PDD_ice_series

np.save('data/daily/ti_melt.npy', melt_TI)

fig, ax = plt.subplots()
ax.plot(time, sr50_melt)
ax.plot(time, melt_TI)

plt.show()
