# Assignment 1 - Glacier melt modelling
Surface energy balance and temperature index modelling for South Glacier, summer 2008

The script `run.sh` runs all the python scripts in the correct order to carry out the complete experiment. For completeness, the individual modules and scripts are:

Modules:

 * `EBM.py`: Core code to solve the energy balance
 * `hourly_daily_averages.py`: Tools for averaging 5- and 30-minute metorological data into hourly and daily bins for energy balance and temperature index models. Can be run as a script to calculate hourly and daily values of the cleaned input data (see `clean_input_data.py`')

Scripts:

 * `clean_input_data.py`: Inspect raw AWS data and carry out required corrections (e.g. correcting for jump in height measurements corresponding to instrument reset) to prep the data for use in melt models
 * `houly_daily_averages.py`: As a script, calculates the hourly and daily values of the cleaned input data
 * `drive_ebm.py`: Calls the functions in `EBM.py` to calculate the energy balance from the hourly-averaged forcing data
 * `albedo.py`: Calculate surface albedo (from ratio of outgoing and incoming SW radiation) and determine when the glacier is snow covered
 * `drive_ti.py`: Calculate melt from height timeseries to calibrate a temperature-index model. Calculate TI-modelled melt.

Plotting:

 * `compare_timeseries.py`: Plot measured melt timeseries and both modelled timeseries
