# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import coint

# Function: Checks cointegration of two tickers and returns test results
def check_cointegration(ticker_a, ticker_b, start_date, end_date, significance=0.01):
    # Load data for both tickers simultaneously
    data = yf.download([ticker_a, ticker_b], start=start_date, end=end_date, auto_adjust=False)
    
    # Extract the 'Adj Close' prices and apply log transformation
    prices = np.log(data['Adj Close'])
    
    # Remove rows with missing values, keeping only common dates
    prices.dropna(inplace=True)
    
    # Perform the Engle-Granger cointegration test
    score, pvalue, _ = coint(prices[ticker_a], prices[ticker_b])
    
    # Determine if the p-value is below the significance level
    cointegrated = pvalue < significance
    
    # Print the test results
    print("Engle-Granger Cointegration Test:")
    print("Test Statistic:", round(score, 4))
    print("p-Value:", round(pvalue, 4))
    if cointegrated:
        print(f"The pair {ticker_a}/{ticker_b} is cointegrated (p-value < {significance}).")
    else:
        print(f"The pair {ticker_a}/{ticker_b} is NOT cointegrated (p-value >= {significance}).")
    
    # Return the test results and the price data for further analysis
    return score, pvalue, cointegrated, prices




