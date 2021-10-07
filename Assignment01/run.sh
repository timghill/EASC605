#!/bin/bash

python3 clean_input_data.py
python3 hourly_daily_averages.py

python3 drive_ebm.py
python3 albedo.py
python3 drive_ti.py

python3 compare_timeseries.py
