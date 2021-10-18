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

# Define and calculate error statistics
ebm_tot_err = ebm[-1] - sr50[-1]
ti_tot_err = ti[-1] - sr50[-1]

def rmse(x):
    return np.sqrt(np.mean( (x-sr50)**2))

ebm_rmse = rmse(ebm)
ti_rmse = rmse(ti)
print('REFERENCE MELT')
print(sr50[-1])

print('ERROR STATISTICS')
print('\tEBM\tTI')
print('Total error')
print('\t %.4f\t%.4f' % (ebm_tot_err, ti_tot_err))
print('Relative error')
print('\t %.4f\t%.4f' % (ebm_tot_err/sr50[-1], ti_tot_err/sr50[-1]))
print('RMSE Error')
print('\t %.4f\t%.4f' % (ebm_rmse, ti_rmse))
