import datetime

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np

import hourly_daily_averages

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

ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
min_month = 5
max_month = 9
xtick_loc = [datetime.datetime(2008, mm, 1) for mm in range(min_month, max_month+1)]
ax.set_xticks(xtick_loc)
ax.legend()
ax.grid()

fig.savefig('figures/melt_comparison.png', dpi=600)
