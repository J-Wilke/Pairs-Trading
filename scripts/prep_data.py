# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Import custom function
from check_cointegration import check_cointegration

# Configuration: Set tickers and time period
ticker_a = 'DBC'
ticker_b = 'GSG'
start_date = '2020-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')


# Function: Prepares the data by computing additional features if the pair is cointegrated.
def prepare_data(ticker_a, ticker_b, start_date, end_date, significance=0.01, window=60):
    # Run the cointegration test and get the log-transformed price series
    score, pvalue, cointegrated, prices = check_cointegration(ticker_a, ticker_b, start_date, end_date, significance)
    
    if cointegrated:
         # Estimate the hedge ratio (beta) using OLS regression (regress log-price of ticker_a on ticker_b)
        X = sm.add_constant(prices[ticker_b])
        model = sm.OLS(prices[ticker_a], X).fit()
        beta = model.params[ticker_b]
        r_squared = model.rsquared
        print(f"Hedge Ratio (beta) for {ticker_a} and {ticker_b}: {beta:.4f}")
        print(f"R²: {r_squared:.4f}")

        

    
        # Compute the spread: difference between log-price of ticker_a and beta * log-price of ticker_b
        spread = prices[ticker_a] - beta * prices[ticker_b]
        
        
        # Create a new DataFrame with the original log prices
        prepared_data = prices.copy()
        # Add computed spread
        prepared_data['spread'] = spread
        
        # Compute rolling moving averages for each ticker's log price (20-day moving average as example)
        prepared_data[f'{ticker_a}_MA20'] = prepared_data[ticker_a].rolling(20).mean()
        prepared_data[f'{ticker_b}_MA20'] = prepared_data[ticker_b].rolling(20).mean()
        # Compute a moving average for the spread
        prepared_data['spread_MA20'] = spread.rolling(20).mean()
        
        # Compute the rolling z-score of the spread using EWMA
        spread_mean = spread.ewm(span=window, adjust=False).mean()
        spread_std = spread.ewm(span=window, adjust=False).std()
        prepared_data['z_score'] = (spread - spread_mean) / spread_std
        # Optionally, add a moving average of the z-score (20-day example)
        prepared_data['z_score_MA20'] = prepared_data['z_score'].rolling(20).mean()
        
        # Compute daily log returns for each ticker (using log difference)
        for ticker in [ticker_a, ticker_b]:
            prepared_data[f'{ticker}_log_return'] = prepared_data[ticker].diff()
        
        # Compute rolling volatility (standard deviation of daily log returns) over the window
        for ticker in [ticker_a, ticker_b]:
            prepared_data[f'{ticker}_volatility'] = prepared_data[f'{ticker}_log_return'].rolling(window).std()
        
        # Calculate a momentum indicator: Rate of Change (ROC) for the spread over a 5-day period
        prepared_data['spread_ROC_5'] = (spread - spread.shift(5)) / spread.shift(5)
        
        # Add volume (liquidity) measures by re-downloading raw data (you could optimize by returning raw data earlier)
        raw_data = yf.download([ticker_a, ticker_b], start=start_date, end=end_date, auto_adjust=False)
        volume_data = raw_data['Volume']
        # Align the volume data with our prepared_data index
        volume_data = volume_data.loc[prepared_data.index]
        prepared_data[f'{ticker_a}_volume'] = volume_data[ticker_a]
        prepared_data[f'{ticker_b}_volume'] = volume_data[ticker_b]
        
        # Drop any rows with NaN values caused by rolling calculations
        prepared_data.dropna(inplace=True)
        
        # Return the enriched DataFrame
        return prepared_data
    else:
        print("The pair is not cointegrated. Data preparation aborted.")
        return None
    
# Run the data preparation
prepared_data = prepare_data(ticker_a, ticker_b, start_date, end_date)

#Describe Data
print(prepared_data.describe())

#Z-Score Plot
prepared_data['z_score'].plot(title='Spread Z-Score')
plt.show()



