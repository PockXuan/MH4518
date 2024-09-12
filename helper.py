import pandas as pd
import numpy as np

'''
This function is used to estimate the parameters of the Geometric Brownian Motion
using the log returns of the stock prices.

Args:
df: pd.DataFrame The dataframe containing the stock prices

Returns:
(mu, sigma_sq, v_daily): The estimated parameters of the Geometric Brownian Motion
    
'''
def parameterEsimation(df: pd.DataFrame) -> tuple[float, float, float]:
    df['log_return'] = np.log(df['Close']).diff()
    v_daily = np.sum(df['log_return']) / (len(df['log_return']) - 1) # v(daily)
    sigma_sq = np.var(df['log_return']) # sigma^2(daily)
    mu = v_daily + sigma_sq / 2
    return mu, sigma_sq, v_daily

def parameterEstimationWithTicker(ticker: str, start: str, end: str) -> tuple[float, float, float]:
    import yfinance as yf
    df = yf.Ticker(ticker).history(start=start, end=end)
    return parameterEsimation(df)
