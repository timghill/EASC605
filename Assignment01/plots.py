import datetime

import matplotlib
matplotlib.rcParams.update({'font.size': 10})

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

def set_axes(ax, panel_label_index=None, text_loc=[0.015, 0.8]):
    ax.grid()
    if panel_label_index is not None:
        letter = chr(ord('a') + panel_label_index)
        ax.text(text_loc[0], text_loc[1], letter, transform=ax.transAxes)
    return ax

def set_xticks(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    min_month = 5
    max_month = 9
    xtick_loc = [datetime.datetime(2008, mm, 1) for mm in range(min_month, max_month+1)]
    ax.set_xticks(xtick_loc)
    return ax
