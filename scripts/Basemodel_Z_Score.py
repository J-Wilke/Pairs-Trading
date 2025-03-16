# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Import custom functions
from check_cointegration import check_cointegration
from prep_data import prepare_data

# Configuration: Set tickers and time period
ticker_a = 'MA'  
ticker_b = 'V'   
start_date = '2020-01-01'  
end_date = datetime.today().strftime('%Y-%m-%d')

# Prepare the data using your module function
prepared_data = prepare_data(ticker_a, ticker_b, start_date, end_date)

# Set thresholds for trading signals (in units of standard deviations)
entry_threshold = 2.0   # Enter trade when z_score is above +2 or below -2
exit_threshold = 0.5    # Exit trade when the absolute z_score falls below 0.5

# Function: Generate trading signals based on z_score thresholds.
def generate_signals(prepared_data, entry_thresh, exit_thresh):
    signals = []          # Will store the signal for each date: 1 for long, -1 for short, 0 for no position
    position = 0          # Start with no position
    for z in prepared_data['z_score']:
        # If no current position, check for entry signal
        if position == 0:
            if z < -entry_thresh:
                position = 1   # Enter long position
            elif z > entry_thresh:
                position = -1  # Enter short position
        # If already in a long position, check for exit condition
        elif position == 1:
            if z > -exit_thresh:
                position = 0   # Exit long position
        # If already in a short position, check for exit condition
        elif position == -1:
            if z < exit_thresh:
                position = 0   # Exit short position
        signals.append(position)
    return signals

# Generate signals and add them as a new column in prepared_data
prepared_data['signal'] = generate_signals(prepared_data, entry_threshold, exit_threshold)

# Plot the z_score and generated trading signals for visual inspection
plt.figure(figsize=(12, 6))
plt.plot(prepared_data.index, prepared_data['z_score'], label='Z-Score', color='dodgerblue')
plt.plot(prepared_data.index, prepared_data['signal'], label='Signal', linestyle='--', color='red')
plt.axhline(entry_threshold, color='green', linestyle='--', label='Entry Threshold')
plt.axhline(-entry_threshold, color='green', linestyle='--')
plt.axhline(exit_threshold, color='grey', linestyle=':', label='Exit Threshold')
plt.axhline(-exit_threshold, color='grey', linestyle=':')
plt.legend()
plt.title('Trading Signals Based on Z-Score')
plt.xlabel('Date')
plt.ylabel('Z-Score / Signal')
plt.show()