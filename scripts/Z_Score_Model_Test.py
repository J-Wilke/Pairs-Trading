#!/usr/bin/env python
# This script performs a backtest of a pairs trading strategy.
# It prepares data, generates trading signals based on a z-score model,
# and executes the backtest using the Backtesting library.

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy

# Import modules for data preparation and signal generation
from prep_data import prepare_data
from Basemodel_Z_Score import generate_signals

# Define the PairTrading strategy
class PairTradingZScore(Strategy):
    def init(self):
        # Initialization (optional)
        pass

    def next(self):
        # Retrieve the current trading signal
        signal = self.data.signal[-1]
        
        # For a long signal: if not already in a long position, enter one
        if signal == 1 and not self.position.is_long:
            if self.position:
                self.position.close()
            self.buy()
        # For a short signal: if not already in a short position, enter one
        elif signal == -1 and not self.position.is_short:
            if self.position:
                self.position.close()
            self.sell()
        # For a neutral signal: close any existing position
        elif signal == 0 and self.position:
            self.position.close()

# Convert the prepared dataset into a format compatible with backtesting.py
def prepare_backtest_data(prepared_data, ticker_a):
    # Ensure the index is a DateTimeIndex and sort it
    prepared_data.index = pd.to_datetime(prepared_data.index, errors='coerce')
    prepared_data = prepared_data[prepared_data.index.notna()].sort_index()
    
    # Create the DataFrame for backtesting
    bt_data = prepared_data[['signal']].copy()
    # Convert the log price back to the actual price
    bt_data['Close'] = np.exp(prepared_data[ticker_a])
    bt_data['Open']  = bt_data['Close']
    bt_data['High']  = bt_data['Close']
    bt_data['Low']   = bt_data['Close']
    return bt_data

# Execute the full backtest process
def run_backtest():
    # Parameters
    ticker_a = 'DBC'
    ticker_b = 'GSG'
    start_date = '2020-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Prepare data: check for cointegration, calculate z_score, etc.
    prepared_data = prepare_data(ticker_a, ticker_b, start_date, end_date)
    if prepared_data is None or 'z_score' not in prepared_data.columns:
        print("Data preparation aborted: The pair is not cointegrated or 'z_score' is missing.")
        return

    print("Available columns in prepared_data:", prepared_data.columns.tolist())
    
    # Generate trading signals
    entry_threshold = 1.0
    exit_threshold = 0.25
    prepared_data['signal'] = generate_signals(prepared_data, entry_threshold, exit_threshold)
    
    # Prepare backtesting data
    bt_data = prepare_backtest_data(prepared_data, ticker_a)
    print("Backtesting Data Preview:")
    print(bt_data.head())
    
    # Run the backtest
    bt = Backtest(bt_data, PairTradingZScore, cash=100000, commission=0.001, exclusive_orders=True)
    output = bt.run()
    print(output)

# This block ensures that the backtest is executed only when the script is run directly.
# If the script is imported as a module in another program, the run_backtest() function will not be executed automatically.
if __name__ == '__main__':
    run_backtest()
