import datetime

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np

import hourly_daily_averages
import plots

time = np.load('data/daily/time.npy', allow_pickle=True)
sr50 = np.load('data/daily/melt.npy')
hourly_ebm = np.load('data/hourly/ebm_melt.npy')
ebm = hourly_daily_averages.average_hourly_to_daily(hourly_ebm)
np.save('data/daily/ebm_melt.npy', ebm)
ti = np.load('data/daily/ti_melt.npy')

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(time, sr50, label='SR50')
ax.plot(time, ebm, label='EBM')
ax.plot(time, ti, label='TI')

ax.set_xlabel('Date')
ax.set_ylabel('Melt (m w.e.)')
ax = plots.set_axes(ax)
ax = plots.set_xticks(ax)
ax.legend()

fig.savefig('figures/melt_comparison.png', dpi=600)
