import yfinance as yf
import numpy as np
import pandas as pd

tickers_list = ["UNH", "PFE", "MRK"]
tickers = yf.Tickers(tickers_list)
hist = tickers.download(start="2022-01-01", end="2024-12-31") 

for ticker in tickers_list:
    stock = yf.Ticker(ticker)
    stock_hist = stock.history(start="2022-01-01", end="2024-12-31")
    stock_hist.to_csv(f"datasets/{ticker}.csv")
    # Get all available expiration dates
    expiry_dates = stock.options
    all_options = []

    # Iterate over each expiry date to get the option chain
    for expiry in expiry_dates:
        # Get option chain for the specific expiry date
        options_chain = stock.option_chain(expiry)
        
        # Separate into calls and puts
        calls = options_chain.calls
        calls["expiry"] = expiry
        all_options.append(calls)
        
    # Concatenate all the calls dataframes
    pd.concat(all_options).reset_index(drop=True).to_csv(f"datasets/{ticker}_calls.csv")

# Calculate returns and log-returns
# Load from CSV due to formatting issue when directly loading from yfinance
for ticker in tickers_list:
    stock = pd.read_csv(f"datasets/{ticker}.csv")
    stock["Date"] = pd.to_datetime(stock["Date"], utc=True)
    stock["dt"] = 1 / 252
    # Calculate return as percentage change in 'Close' prices
    stock["return"] = stock["Close"].diff()

    # Calculate log-return
    stock["log_return"] = np.log(stock["Close"] / stock["Close"].shift(1))
    
    stock.to_csv(f"datasets/{ticker}.csv")
