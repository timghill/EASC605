# Assignment 1 - Glacier melt modelling
Surface energy balance and temperature index modelling for South Glacier, summer 2008

Python scripts are as follows:

 * explore_data.py: Plot raw timeseries and correct for SR50 resetting and other issues
 * hourly_average.py: Calculate hourly average of meteorological fields

To do:

Modularize:

 * Energy-balance model (mostly done)
 * All averaging functions (started)

Clean scripts for:

 * Data inspection/cleaning/averaging (all)
 * Main: Drive EBM, calibrate and run TI model
 * Plotting: Make all the plots
    * Some modularization here, e.g. setting axes xticks, fonts, ...

Clean python files:

 * `clean_input_data.py`: Explore AWS data and correct for data jumps (e.g. in SR50 data) before we use it in melt modelling
 * `hourly_daily_averages.py`
