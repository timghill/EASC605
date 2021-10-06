import datetime as dt

from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection
import matplotlib.dates as mdates

import numpy as np

SWin = np.load('data/hourly/SWin.npy')
SWout = np.load('data/hourly/SWout.npy')
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
print(n_steps)
mean_albedo = np.zeros(n_steps+2)
mean_albedo[0] = np.nanmean(chunk_0)
mean_albedo[-1] = np.nanmean(albedo[-19:])

for i in range(n_steps+1):
    albedo_vals = albedo_clean[i*chunk_len:(i+1)*chunk_len]
    mean_albedo[i+1] = np.nanmean(albedo_vals)
