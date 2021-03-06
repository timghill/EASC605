#!/usr/bin/python3
"""
Explore weather station and SR50 data before applying energy-balance model

Applies corrections to:

 * SR50 height data: Correctes for instrument reset

 * Radiation: Corrects for negative net SW radiation and strongly negative
net radiation

 * RH: Corrects for one observation of RH = 0

Saves corrected files at original temporal resolution to data/clean
"""

import matplotlib
matplotlib.rcParams.update({'font.size': 10})

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime

import plots

# ------------------------------------------------------------------------
# Read raw AWS data
# ------------------------------------------------------------------------

raw_office_file = 'data/raw/crtd_fix_all_tpmGL1_office_may-sept_2008_5min.txt'
raw_onice_file = 'data/raw/crtd_fix_all_tpmGL1_onice1_may-sept_2008_5min.txt'

AWS_onice = np.loadtxt(raw_onice_file).astype(float)

# Temporal variables
dec_days = AWS_onice[:, 0]
year = AWS_onice[:, 1]
month = AWS_onice[:, 2]
day_of_month = AWS_onice[:, 3]
hour = AWS_onice[:, 4]
minute = AWS_onice[:, 5]
second = AWS_onice[:, 6]

# Met variables at 5 min resolution
wind_speed  =AWS_onice[:, 8]
wind_dir = AWS_onice[:, 9]
wind_dir_sd = AWS_onice[:, 10]
wind_max = AWS_onice[:, 11]
temp = AWS_onice[:, 12]
net_rad = AWS_onice[:, 13]
RH = AWS_onice[:, 14]
SWin = AWS_onice[:, 15]
SWout = AWS_onice[:, 16]

# Met variables at 30 min resolution
rec_num_30min = AWS_onice[:, 17]
SR50_dist = AWS_onice[:, 18]
bar_press = AWS_onice[:, 19]

# Off-ice AWS for rain
AWS_office = np.loadtxt(raw_office_file).astype(float)
off_dec_days = AWS_office[:, 0]
rain = AWS_office[:, -1]

# Make AWS_onice python datetime instance for easier plotting
_date_start = datetime.datetime(2006, 1, 1)
tt = np.zeros(dec_days.shape, datetime.datetime)
tt_office = np.zeros(off_dec_days.shape, datetime.datetime)
for i in range(len(tt)):
    tt[i] = _date_start + datetime.timedelta(days=dec_days[i])

for i in range(len(rain)):
    tt_office[i] = _date_start + datetime.timedelta(days=off_dec_days[i])

# Handle NaN values in 30 min resolution fields
dec_days_30min = dec_days[~np.isnan(rec_num_30min)]
tt_30min = tt[~np.isnan(rec_num_30min)]
SR50_dist = SR50_dist[~np.isnan(rec_num_30min)]
bar_press = bar_press[~np.isnan(rec_num_30min)]

tt_off_30min = tt_office[~np.isnan(rain)]
rain_30min = rain[~np.isnan(rain)]

# ------------------------------------------------------------------------
# Plot raw data
# ------------------------------------------------------------------------

fig, axes = plt.subplots(figsize=(8, 10), nrows=8, sharex=True)
(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = axes

ax1.plot(tt, temp)
ax1.set_ylabel('Temp ($^\\circ$C)')

ax2.plot(tt, wind_speed)
ax2.set_ylabel('U (m/s)')

ax3.plot(tt, RH)
ax3.set_ylabel('RH (%)')

ax4.plot(tt, net_rad)
ax4.set_ylabel('$Q_R$ (W/m$^2$)')

ax5.plot(tt, SWin - SWout)
ax5.set_ylabel('$Q_{SW}$ ((W/m$^2$)')

ax6.plot(tt_30min, SR50_dist)
ax6.set_ylabel('Height (m)')

ax7.plot(tt_30min, bar_press)
ax7.set_ylabel('Pressure (hPa)')

ax8.plot(tt_off_30min, rain_30min)
ax8.set_ylabel('Rain (mm)')

for (i, ax) in enumerate(axes):
    ax = plots.set_axes(ax, panel_label_index=i)

ax8 = plots.set_xticks(ax8)

plt.tight_layout()

fig.savefig('figures/raw_data.png', dpi=600)

# ------------------------------------------------------------------------
# Correct gaps/jumps in raw data
# ------------------------------------------------------------------------
corr_wind_speed = wind_speed.copy()
corr_wind_dir = wind_dir.copy()
corr_wind_dir_sd = wind_dir_sd.copy()
corr_wind_max = wind_max.copy()
corr_temp = temp.copy()
corr_net_rad = net_rad.copy()
corr_RH = RH.copy()
corr_SWin = SWin.copy()
corr_SWout = SWout.copy()
corr_press = bar_press.copy()
corr_height = SR50_dist.copy()
corr_rain = rain_30min.copy()

# Find SR50 jump
delta = corr_height[1:] - corr_height[:-1]
reset_ind = np.argmax(-delta)
print('Found gap in SR50 distance at time %s' % str(tt[reset_ind]))

# Reset so that SR50 record is continuous at jump
SR50_01 = SR50_dist[:reset_ind]
# Readings immediately after jump are also not trustworthy - look ahead
# 5 more readings to find a stable value
offset = SR50_dist[reset_ind] - SR50_dist[reset_ind+5]
SR50_02 = SR50_dist[reset_ind+5:]

corr_height[:reset_ind] = corr_height[:reset_ind]
corr_height[reset_ind:reset_ind+5] = corr_height[reset_ind]
corr_height[reset_ind+5:] = SR50_02 + offset

# Plot SR50 raw and corrected data
fig2, ax2 = plt.subplots()
ax2.plot(tt[reset_ind-100:reset_ind+100],
    SR50_dist[reset_ind-100:reset_ind+100], label='Raw SR50')
ax2.plot(tt[reset_ind-100:reset_ind+100],
    corr_height[reset_ind-100:reset_ind+100], label='Corrected SR50')
ax2.grid()
ax2.set_xlabel('Date')
ax2.set_ylabel('Height (m)')
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

plt.tight_layout()
fig2.savefig('figures/SR50_correction.png', dpi=600)

# Look for strongly negative net radiation, set to zero
rad_corr_ind = corr_net_rad<-100
corr_net_rad[rad_corr_ind] = 0

# Correct for negative net SW radiation
SW_corr_ind = (SWin - SWout)<0
corr_SWout[SW_corr_ind] = corr_SWin[SW_corr_ind]
corr_net_rad[SW_corr_ind] = 0

# Plot the radiation correction
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(tt, net_rad)
ax1 = plots.set_axes(ax1, panel_label_index=0)
ax1.set_ylabel('Raw $Q_R$ (W/m$^2$)')

ax2.plot(tt, corr_net_rad)
ax2.set_ylabel('Corr $Q_R$ (W/m$^2$)')
ax2 = plots.set_axes(ax2, panel_label_index=1)
ax2 = plots.set_xticks(ax2)
ax2.set_xlabel('Date')

plt.tight_layout()
fig.savefig('figures/radiation_correction.png', dpi=600)

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(tt, SWin - SWout)
ax1 = plots.set_axes(ax1, panel_label_index=0)
ax1.set_ylabel('Raw $Q_{SW}$ (W/m$^2$)')

ax2.plot(tt, corr_SWin - corr_SWout)
ax2.grid()
ax2.set_ylabel('Corr $Q_{SW}$ (W/m$^2$)')
ax2 = plots.set_axes(ax2, panel_label_index=1)
ax2 = plots.set_xticks(ax2)
ax2.set_xlabel('Date')

plt.tight_layout()

fig.savefig('figures/SW_radiation_correction.png', dpi=600)

# Fix RH jump
RH_corr_ind = int(np.where(RH<10)[0])
print('Fixing spurious RH at %s' % str(tt[RH_corr_ind]))
corr_RH[RH_corr_ind] = 0.5*(RH[RH_corr_ind-1] + RH[RH_corr_ind+1])

fig3, ax3 = plt.subplots()
plot_delta = 50
ax3.plot(tt[RH_corr_ind-plot_delta:RH_corr_ind+plot_delta],
    RH[RH_corr_ind-plot_delta:RH_corr_ind+plot_delta], label='Raw')
ax3.plot(tt[RH_corr_ind-plot_delta:RH_corr_ind+plot_delta],
    corr_RH[RH_corr_ind-plot_delta:RH_corr_ind+plot_delta], label='Corrected')
ax3.grid()
ax3.set_ylabel('RH (%)')
ax3.set_xlabel('Date')
ax3.legend()
plt.tight_layout()
fig3.savefig('figures/RH_correction.png', dpi=600)

plt.show()

# ------------------------------------------------------------------------
# Save corrected data
# ------------------------------------------------------------------------
np.save('data/clean/temp.npy', temp)
np.save('data/clean/wind_speed.npy', corr_wind_speed)
np.save('data/clean/time.npy', tt)
np.save('data/clean/net_rad.npy', corr_net_rad)
np.save('data/clean/SWin.npy', corr_SWin)
np.save('data/clean/SWout.npy', corr_SWout)
np.save('data/clean/RH.npy', corr_RH)
np.save('data/clean/SR50_height.npy', corr_height)
np.save('data/clean/time_30min.npy', tt_30min)
np.save('data/clean/pressure.npy', corr_press)
np.save('data/clean/rain.npy', rain)
np.save('data/clean/time_office.npy', tt_off_30min)

# Plot all corrected data
fig, axes = plt.subplots(figsize=(8, 10), nrows=8, sharex=True)
(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = axes

ax1.plot(tt, corr_temp)
ax1.set_ylabel('Temp ($^\\circ$C)')

ax2.plot(tt, corr_wind_speed)
ax2.set_ylabel('U (m/s)')

ax3.plot(tt, corr_RH)
ax3.set_ylabel('RH (%)')

# ax4.plot(dec_days, net_rad)
ax4.plot(tt, corr_net_rad)
ax4.set_ylabel('$Q_R$ (W/m$^2$)')

ax5.plot(tt, corr_SWin - corr_SWout)
ax5.set_ylabel('$Q_{SW}$ ((W/m$^2$)')

ax6.plot(tt_30min, corr_height)
ax6.set_ylabel('Height (m)')

ax7.plot(tt_30min, corr_press)
ax7.set_ylabel('Pressure (hPa)')

ax8.plot(tt_off_30min, rain_30min)
ax8.set_ylabel('Rain (mm)')


for (i, ax) in enumerate(axes):
    ax = plots.set_axes(ax, panel_label_index=i)

ax8 = plots.set_xticks(ax8)

ax8.set_xlabel('Date')

plt.tight_layout()

fig.savefig('figures/corrected_data.png', dpi=600)
