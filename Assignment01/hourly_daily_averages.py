#!/usr/bin/python3
"""
Calculate hourly and daily averages of AWS data.

If run as a script, calculates hourly and daily averages of AWS data and
saves them in data/hourly and data/daily directories

Can be imported as a module, in which case user can access the averaging
functions for use in other scripts (e.g. to calculate daily average of
hourly energy-balance model outputs)
"""

import matplotlib
matplotlib.rcParams.update({'font.size': 10})

import numpy as np
import datetime
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

# Averaging functions are specific to the time sampling from the on-ice and
# off-ice weather stations. The average_XX_to_hourly functions calculate
# averages onto a shared time array so that calculating daily averages is
# relatively easier
def average_5min_to_hourly(x):
    """Calculate hourly averages of 5-minute field x

    Assumes x is measured at 5-minute resolution at time samples given
    by data/clean/time.npy
    """
    # Neglect first two observations since they lie just before the
    # nearest hour
    x_clean = x[2:]
    n_obs = len(tt[2:])
    chunk_len = int(60/5)    # Number of observations to average

    n_steps = int(n_obs/chunk_len)
    x_avg = np.zeros(n_steps)
    for i in range(n_steps):
        x_avg[i] = np.nanmean(x_clean[i*chunk_len:(i+1)*chunk_len])

    return x_avg

def average_30min_to_hourly(x):
    """Calculate hourly averages of 30-minute field x

    Assumes x is measured at 30-minute resolution at time samples given
    by data/clean/time_30min.npy
    """
    chunk_len = 2
    n_obs = len(x)
    n_steps = int(n_obs/chunk_len)

    x_avg = np.zeros(n_steps)
    for i in range(n_steps):
        x_avg[i] = np.nanmean(x[i*chunk_len:(i+1)*chunk_len])

    return x_avg


def sum_30min_off_ice(x):
    """Calculate hourly sum of 30-minute field x

    Assumes x is measured at 30-minute resolution at time samples given
    by data/clean/time_office.npy
    """
    # Neglect first observation since it lies just before the
    # nearest hour
    x = x[1:]
    n_obs = len(x)
    n_steps = int(n_obs/chunk_len)
    x_avg = np.zeros(n_steps)
    for i in range(n_steps):
        x_avg[i] = np.nansum(x[i*chunk_len:(i+1)*chunk_len])

    n_obs_total = int(len(temp[2:])/int(60/5))
    x_avg_total = np.zeros(n_obs_total)
    x_avg_total[2:-2] = x_avg

    return x_avg_total

def average_hourly_to_daily(x, metric=np.nanmean):
    """Calculate 24-hour average of hourly field x, assuming x is measured
    with time array data/hourly/time.npy
    """
    chunk_len = 24
    start = 5
    stop = -19

    x_clean = x[start:stop]
    n_steps = int(len(x_clean)/chunk_len)
    x_daily = np.zeros(n_steps+2)
    x_daily[0] = metric(x[:start])
    for i in range(0, n_steps):
        chunk = x_clean[i*chunk_len:(i+1)*chunk_len]
        x_daily[i+1] = metric(chunk)
    x_daily[-1] = metric(x[stop:])
    return x_daily

if __name__ == '__main__':
    # MAIN LOOP
    #
    # If run as a script, calculates hourly and daily averages of AWS
    # Measurements

    # Time variables
    tt = np.load('data/clean/time.npy', allow_pickle=True)
    tt_30min = np.load('data/clean/time_30min.npy', allow_pickle=True)
    tt_office = np.load('data/clean/time_office.npy', allow_pickle=True)

    # 5 min data
    temp = np.load('data/clean/temp.npy')
    RH = np.load('data/clean/RH.npy')
    SWin = np.load('data/clean/SWin.npy')
    SWout = np.load('data/clean/SWout.npy')
    Rnet = np.load('data/clean/net_rad.npy')
    wind = np.load('data/clean/wind_speed.npy')
    height = np.load('data/clean/SR50_height.npy')
    LWnet = Rnet - (SWin - SWout)

    # 30 min data
    press = np.load('data/clean/pressure.npy')

    # 30 min off-ice data
    rain = np.load('data/clean/rain.npy')

    # Calculate hourly and daily time arrays - the averaging functions are
    # not relevant for this

    # Hourly time array
    chunk_len = int(60/5)
    n_steps = int(len(tt[2:])/chunk_len)
    tt_clean = tt[2:]
    tt_hourly = np.zeros(n_steps, datetime.datetime)
    for i in range(n_steps):
        tt_hourly[i]= tt_clean[i*chunk_len]

    # Hourly average of off-ice 30-minute time array
    tt_office_clean = tt_office[1:]
    tt_office_hourly = np.zeros(int(len(tt_office_clean)/2), datetime.datetime)
    for i in range(len(tt_office_hourly)):
        tt_office_hourly[i] = tt_office_clean[2*i]

    # Average the fields
    hourly_temp = average_5min_to_hourly(temp)
    hourly_rh = average_5min_to_hourly(RH)
    hourly_SWin = average_5min_to_hourly(SWin)
    hourly_SWout = average_5min_to_hourly(SWout)
    hourly_LWnet = average_5min_to_hourly(LWnet)
    hourly_press = average_30min_to_hourly(press)
    hourly_rain = sum_30min_off_ice(rain)
    hourly_wind = average_5min_to_hourly(wind)
    hourly_height = average_30min_to_hourly(height)

    fig, axes = plt.subplots(nrows=7, figsize=(8, 8), sharex=True)
    # ax.plot(tt, temp)

    ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axes

    min_month = 5
    max_month = 9
    xtick_loc = [datetime.datetime(2008, mm, 1) for mm in range(min_month, max_month+1)]

    ax1.plot(tt_hourly, hourly_temp)
    ax1.set_ylabel('T ($^\\circ$C)')
    ax1.grid()
    ax1.text(0.015, 0.8, 'a', transform=ax1.transAxes)

    ax2.plot(tt_hourly, hourly_rh)
    ax2.set_ylabel('RH (%)')
    ax2.grid()
    ax2.text(0.015, 0.8, 'b', transform=ax2.transAxes)


    ax3.plot(tt_hourly, hourly_SWin)
    ax3.set_ylabel('SW$_{in}$')
    ax3.grid()
    ax3.text(0.015, 0.8, 'c', transform=ax3.transAxes)


    ax4.plot(tt_hourly, hourly_SWout)
    ax4.set_ylabel('SW$_{out}$')
    ax4.grid()
    ax4.text(0.015, 0.8, 'd', transform=ax4.transAxes)

    ax5.plot(tt_hourly, hourly_LWnet)
    ax5.set_ylabel('LW$_{net}$')
    ax5.grid()
    ax5.text(0.015, 0.8, 'e', transform=ax5.transAxes)

    ax6.plot(tt_hourly, hourly_press)
    ax6.set_ylabel('P (hPa)')
    ax6.grid()
    ax6.text(0.015, 0.8, 'f', transform=ax6.transAxes)

    ax7.plot(tt_hourly, hourly_rain)
    ax7.text(0.015, 0.8, 'g', transform=ax7.transAxes)

    ax7.set_ylabel('Rain (mm)')
    ax7.grid()

    axes[-1].set_xlabel('Date')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[-1].set_xticks(xtick_loc)

    plt.tight_layout()

    fig.savefig('figures/hourly_average.png', dpi=600)

    # Save hourly data
    np.save('data/hourly/temp.npy', hourly_temp)
    np.save('data/hourly/SWin.npy', hourly_SWin)
    np.save('data/hourly/SWout.npy', hourly_SWout)
    np.save('data/hourly/LWnet.npy', hourly_LWnet)
    np.save('data/hourly/press.npy', hourly_press)
    np.save('data/hourly/rain.npy', hourly_rain)
    np.save('data/hourly/wind.npy', hourly_wind)
    np.save('data/hourly/time.npy', tt_hourly)
    np.save('data/hourly/RH.npy', hourly_rh)
    np.save('data/hourly/height.npy', hourly_height)

    # Calculate hourly positive-degree-day
    dt = 1/24
    pdd_hourly = hourly_temp.copy()*dt
    pdd_hourly[pdd_hourly<0] = 0

    pdd_daily = average_hourly_to_daily(pdd_hourly, metric=np.nansum)

    # Calculate daily data
    np.save('data/daily/temp.npy', average_hourly_to_daily(hourly_temp))
    np.save('data/daily/SWin.npy', average_hourly_to_daily(hourly_SWin))
    np.save('data/daily/SWout.npy', average_hourly_to_daily(hourly_SWout))
    np.save('data/daily/height.npy', average_hourly_to_daily(hourly_height))
    np.save('data/daily/PDD.npy', pdd_daily)
    day0 = datetime.date(2008, 5, 5)
    day1 = datetime.date(2008, 9, 13)
    ndays = (day1 - day0).days
    time_daily = np.array([day0 + datetime.timedelta(days=i) for i in range(ndays+1)])
    np.save('data/daily/time.npy', time_daily)
