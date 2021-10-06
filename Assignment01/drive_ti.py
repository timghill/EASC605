#!/usr/bin/python3
"""
Calibrate and run temperature-index melt model
"""

import datetime

from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection
import matplotlib.dates as mdates

import numpy as np

import hourly_daily_averages

# PARAMETERS
rho_snow = 500
rho_ice = 910
rho_water = 1000
_day = 86400

height = np.load('data/daily/height.npy')
snow_mask = np.load('data/daily/snow_mask.npy')
time = np.load('data/daily/time.npy', allow_pickle=True)
PDD = np.load('data/daily/PDD.npy')

# Calculate melt from SR50 record
inst_melt = np.zeros(height.shape)
tot_melt = np.zeros(height.shape)
for i in range(1, len(height)):
    dh = np.max((0, float(height[i] - height[i-1])))
    if snow_mask[i]:
        melt_contrib = dh*rho_snow/rho_water
    else:
        melt_contrib = dh*rho_ice/rho_water
    inst_melt[i] = melt_contrib/_day
    tot_melt[i] = tot_melt[i-1] + melt_contrib

mwe_2_cm_day = 86400*1e2
inst_melt_cm_day = inst_melt*mwe_2_cm_day

# TI MODEL
PDD_ice = 0
PDD_snow = 0

melt_ice = 0
melt_snow = 0

ntimes = len(time)

# Keep track of cumulative PDD as a function of time for snow and ice
PDD_ice_series = np.zeros(ntimes-1)
PDD_snow_series= np.zeros(ntimes-1)

# Measured melt as a function of time for snow and ice
melt_ice_series = np.zeros(ntimes-1)
melt_snow_series = np.zeros(ntimes-1)

for i in range(ntimes-1):
    d_melt = tot_melt[i+1] - tot_melt[i]
    if d_melt>0:
        if snow_mask[i]:
            PDD_snow += PDD[i]
            melt_snow += d_melt
        else:
            PDD_ice += PDD[i]
            melt_ice += d_melt

    PDD_ice_series[i] = PDD_ice
    PDD_snow_series[i] = PDD_snow
    melt_ice_series[i] = melt_ice
    melt_snow_series[i] = melt_snow

# Leastsquares solve for degree-day factors
DDF_lsq_snow = np.linalg.lstsq(np.vstack(PDD_snow_series),
    np.vstack(melt_snow_series), rcond=None)[0][0][0]
DDF_lsq_ice = np.linalg.lstsq(np.vstack(PDD_ice_series),
    np.vstack(melt_ice_series), rcond=None)[0][0][0]

print('DDF snow:', DDF_lsq_snow)
print('DDF ice:', DDF_lsq_ice)

# Now recreate the melt series - very easy!
melt_TI = np.zeros(time.shape)
melt_TI[1:] = DDF_lsq_snow*PDD_snow_series + DDF_lsq_ice*PDD_ice_series

np.save('data/daily/ti_melt.npy', melt_TI)
