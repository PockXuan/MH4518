import pandas as pd
import numpy as np
from scipy.stats import norm

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


def calculate_var_cvar(returns, confidence_level=0.95):
    # Calculate the mean and standard deviation of returns
    mu = np.mean(returns)
    sigma = np.std(returns)

    # Calculate the Z-score for the given confidence level
    z_score = norm.ppf(1 - confidence_level)

    # Calculate VaR using the parametric method
    var = mu + z_score * sigma

    # Calculate CVaR (Expected Shortfall)
    cvar = mu - sigma * norm.pdf(z_score) / (1 - confidence_level)

    return var, cvar
