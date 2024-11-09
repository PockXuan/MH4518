import os
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
import scipy as sp
from pingouin import multivariate_normality 
from arch import arch_model

# Set a reference date for processing
REFERENCE_DATE = pd.to_datetime("2023-08-23").date()
dt = 1 / 250

# Function to preprocess data and calculate Garman-Klass variance
def preprocess_stock_data(df):
    """Convert dates, calculate Garman-Klass variance, and take log of stock prices."""
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.date
    return df[['Date', 'Close']]

# Load and preprocess CSV files for stocks and interest rates
def load_data():
    """Load stock data for UNH, PFE, MRK, and interest rates, preprocess, and merge."""
    # Load stock data
    df1 = preprocess_stock_data(pd.read_csv(os.getcwd() + "/MH4518/datasets/UNH.csv"))
    df2 = preprocess_stock_data(pd.read_csv(os.getcwd() + "/MH4518/datasets/PFE.csv"))
    df3 = preprocess_stock_data(pd.read_csv(os.getcwd() + "/MH4518/datasets/MRK.csv"))

    # Rename columns for clarity
    df1.columns = ['Date', 'stock_price_UNH']
    df2.columns = ['Date', 'stock_price_PFE']
    df3.columns = ['Date', 'stock_price_MRK']

    # Merge stock data by Date
    stock_df = df1.merge(df2, on='Date').merge(df3, on='Date')

    # Load and preprocess interest rates
    rates = pd.read_csv(os.getcwd() + "/MH4518/datasets/DGS10.csv").replace('.', np.nan).ffill()
    rates.columns = ['Date', 'true_rate']
    rates['true_rate'] = pd.to_numeric(rates['true_rate']) / 100  # Convert to decimal
    rates['Date'] = pd.to_datetime(rates['Date']).dt.tz_localize(None).dt.date

    # Merge stock and rate data
    stock_df = stock_df.merge(rates, on='Date')

    return stock_df

# Fit GARCH models for each stock and extract residuals and volatilities
def fit_garch_models(data):
    """Fit GARCH models to stock simple returns and extract residuals and conditional volatilities."""
    garch_models = {}
    for stock in ['UNH', 'PFE', 'MRK']:
        garch_models[stock] = arch_model(100 * data[f'simple_return_shock_{stock}'], vol='Garch', p=1, q=1).fit(disp='off')
    return garch_models

# Following the paper at https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/259296/724505_FULLTEXT01.pdf
# Define the DCC-GARCH loss function for parameter optimization
def dcc_garch_log_loss(x, shocks, sigma, q0):
    """Calculate DCC-GARCH model log-likelihood given parameters."""
    a = (np.tanh(x[0]) + 1) / 2
    b = ((np.tanh(x[1]) + 1) / 2) * (1-a)
    nu = np.maximum(x[2], 2)  # Ensure nu > 2 for stability
    iota = x[3:].reshape(-1, 1)
    
    # Initialize correlation matrix
    qt = q0
    
    # Initialize skewed t-distribution parameters
    g1 = sp.special.gamma((nu - 1) / 2)
    g2 = sp.special.gamma(nu / 2)
    iota_norm = np.sum(iota**2)
    denom = np.pi * g2**2 * (nu - (nu - 2) * iota_norm)
    num = 2 * (nu - 2) * (np.pi * g2**2 - (nu - 2) * g1**2) * iota_norm
    k = np.sqrt(1 + (2 * nu * num) / (denom * (nu - (nu - 2) * iota_norm)))
    
    Omega = np.eye(3)
    if not np.isclose(iota_norm, 0):
        Omega = Omega + iota_norm**-1 * (-1 + denom * (k - 1) / num) * iota @ iota.T
    Omega = Omega * (1 - 2 / nu)
    xi = -np.sqrt(nu / np.pi) * (g1 / g2) * (Omega @ iota) / np.sqrt(1 + iota.T @ Omega @ iota)
    loss = (len(shocks) - 1) * (np.log(sp.special.gamma((nu + 3) / 2)) - 0.5 * np.log(np.linalg.det(Omega)) \
            - 1.5 * np.log(np.pi * nu) - np.log(g2))
    
    # Calculate the log-likelihood
    for i in range(1, len(shocks)):
        a_t = shocks[i].reshape(-1, 1)
        qt = (1 - a - b) * q0 + a * (a_t @ a_t.T) + b * qt
        q_t_star_inv = np.diag(1 / np.sqrt(np.diag(qt)))
        R_t = q_t_star_inv @ qt @ q_t_star_inv
        D_t = np.diag(sigma[i])
        H_t = D_t @ R_t @ D_t
        
        # Compute Q_a_t and update loss
        v = np.linalg.solve(np.linalg.cholesky(H_t), a_t) - xi
        Q_a_t = v.T @ np.linalg.solve(Omega, v)
        loss -= 0.5 * (nu + 3) * np.log(1 + Q_a_t / nu)
        loss -= 0.5 * np.log(np.linalg.det(R_t))
    
    return -loss

def multivariate_skew_t(nu, iota, steps, paths):

    # Using Azzalini's skew Student's t-distribution. 
    # Sampling is by a transformation of independent skew-normal and chi-squared variables

    g1 = sp.special.gamma((nu - 1) / 2)
    g2 = sp.special.gamma(nu / 2)
    iota_norm = np.sum(iota**2)
    denom = np.pi * g2**2 * (nu - (nu - 2) * iota_norm)
    num = 2 * (nu - 2) * (np.pi * g2**2 - (nu - 2) * g1**2) * iota_norm
    k = np.sqrt(1 + (2 * nu * num) / (denom * (nu - (nu - 2) * iota_norm)))
    Omega = np.eye(3)
    if not np.isclose(iota_norm, 0):
        Omega = Omega + iota_norm**-1 * (-1 + denom * (k - 1) / num) * iota @ iota.T
    Omega = Omega * (1 - 2 / nu)

    # Skew-normal is done by sampling from one dimension higher then doing a conditional 
    # projection down (Azzalini and Capitanio, 1999)
    # Azzalini, A., & Capitanio, A. (1999). Statistical applications of the multivariate skew normal distribution. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 61(3), 579–602. doi:10.1111/1467-9868.00194 

    d = np.diag(np.sqrt(np.diag(Omega)))
    delta = d @ iota # Skew parameter
    skew_star = (Omega @ delta).flatten()

    def loss(skew_guess, delta):
        skew_guess = skew_guess.reshape(3,1)
        v = np.linalg.solve(Omega, skew_guess)
        v = v / np.sqrt(1 - delta.T @ v)
        return np.sum((v - delta)**2)
    skew_star = sp.optimize.minimize(loss, skew_star, (delta,), method='SLSQP').x

    err_cov = np.block([
        [1, skew_star.reshape(1,-1)],
        [skew_star.reshape(-1,1), Omega]
    ])

    norm = np.random.randn(paths,steps,4,1)
    norm = np.linalg.cholesky(err_cov) @ norm
    norm = (norm[:,:,:3,:] * np.where(norm[:,:,-1,:] < 0, -1, 1).reshape(paths,steps,1,1))
    norm = norm.reshape(paths, steps, 3)

    # Generate chi-squared variables
    chi = stats.chi2.rvs(nu, size=3 * steps * paths).reshape(paths, steps, 3) / nu
    
    # Combine everything
    xi = -np.sqrt(nu / np.pi) * (g1 / g2) * (Omega @ iota) / np.sqrt(1 + iota.T @ Omega @ iota)
    brownian = (xi.reshape(1,1,3) + (norm / np.sqrt(chi))).transpose(0,2,1)

    return brownian

# Forecast future shocks based on the optimized DCC-GARCH parameters
def forecast_dcc_garch(steps, num_paths, r, s0, params, garch_models, shocks, q0):
    """Forecast future shocks using DCC-GARCH model over given steps."""

    a = (np.tanh(params[0]) + 1) / 2
    b = ((np.tanh(params[1]) + 1) / 2) * (1-a)
    nu = params[2]
    iota = params[3:].reshape(-1, 1)
    forecast = np.tile(s0, (num_paths, 1, 1))
        
    univariate_vars = np.vstack([garch_models[stock].forecast(horizon=steps).variance for stock in ['UNH', 'PFE', 'MRK']]).T
    future_shock_process = multivariate_skew_t(nu, iota, steps, num_paths)

    q_t = q0
    shocks = np.array(shocks)
    for i in range(1, len(shocks)):
        a_t = shocks[i,:].reshape(-1, 1)
        q_t = (1 - a - b) * q0 + a * (a_t @ a_t.T) + b * q_t
    
    q0 = q0.reshape(1,3,3)
    q_star_t = np.diag(np.diag(q_t)**-0.5).reshape(1,3,3)
    q_t = q_t.reshape(1,3,3)
    
    for i in range(steps):
        a_t = future_shock_process[:,:,i].reshape(num_paths,3,1)
        
        d_t = np.diag(np.sqrt(univariate_vars[i,:] / 100))
        h_t = d_t @ q_star_t @ q_t @ q_star_t @ d_t
        next_shock = np.linalg.cholesky(h_t) @ a_t
        next_stock_price = np.clip(forecast[:,:,i].reshape(num_paths, 3, 1) * (1 + r * dt + next_shock * np.sqrt(dt) / 100), 0, None)
        forecast = np.concatenate((forecast, next_stock_price.reshape(num_paths,3,1)), axis=2)
        
        q_t =  (1 - a - b) * q0 + a * (a_t @ a_t.transpose(0,2,1)) + b * q_t
        q_star_t = np.apply_along_axis(np.diag, -1, np.diagonal(q_t, axis1=1, axis2=2)**-0.5)
    
    return forecast

class GARCH():

    def __init__(self, starting_date=0):

        self.current_date = 411 + starting_date

        self.data = load_data().iloc[:self.current_date+1]
        for stock in ['UNH', 'PFE', 'MRK']:
            self.data[f'simple_return_shock_{stock}'] = (
                (self.data[f'stock_price_{stock}'] - self.data[f'stock_price_{stock}'].shift(1) * (1 + self.data['true_rate'] * dt)) / (self.data[f'stock_price_{stock}'].shift(1) * np.sqrt(dt))
                )
        self.shocks = self.data[['simple_return_shock_UNH', 'simple_return_shock_PFE', 'simple_return_shock_MRK']].dropna().iloc[1:]
    
    def fit(self, verbose=False):

        self.garch_models = fit_garch_models(self.shocks)

        self.garch_data = {}
        for stock in ['UNH', 'PFE', 'MRK']:
            self.garch_data[stock] = {
                    'std_resid': self.garch_models[stock].std_resid,
                    'conditional_volatility': self.garch_models[stock].conditional_volatility
            }

        # Initial correlation matrix
        self.z = np.column_stack([self.garch_data[stock]['std_resid'] for stock in ['UNH', 'PFE', 'MRK']])
        self.sigma = np.column_stack([self.garch_data[stock]['conditional_volatility'] for stock in ['UNH', 'PFE', 'MRK']])
        self.q0 = np.cov(self.z, rowvar=False)
        self.r = self.data['true_rate'].iloc[self.current_date]
        
        initial_params = [0.25, 0.25, 3, 0, 0, 0]
        self.optimized_params = sp.optimize.minimize(
            dcc_garch_log_loss, initial_params, args=(self.z, self.sigma, self.q0), method='SLSQP',
            bounds=[(-2.3, 2.3), (-2.3, 2.3), (2, None), (None, None), (None, None), (None, None)]
        ).x
        
        if verbose:
            a = (np.tanh(self.optimized_params[0]) + 1) / 2
            b = ((np.tanh(self.optimized_params[1]) + 1) / 2) * (1-a)
            nu = self.optimized_params[2]
            iota = self.optimized_params[3:].reshape(-1, 1)
            print('Optimised parameters:')
            print('a:', a)
            print('b:', b)
            print('nu:', nu)
            print('iota:', iota)
    
    def forecast(self, steps, num_paths):
        s0 = np.array(self.data[['stock_price_UNH', 'stock_price_PFE', 'stock_price_MRK']].iloc[self.current_date]).reshape(1,3,1)
        return forecast_dcc_garch(steps, num_paths, self.r, s0, self.optimized_params, self.garch_models, self.shocks, self.q0)

if __name__=='__main__':
    model = GARCH()
    model.fit(True)
    forecast = model.forecast(250, 1000)

    data = load_data().loc[412:,[f'stock_price_UNH']]
    data = np.array(data)

    # print(forecast.shape)
    # print(simul.shape)

    import matplotlib.pyplot as plt
    plt.plot(np.log(forecast[:,0,:].T), color='blue', alpha=0.3)
    plt.plot(np.log(data[:250,0]), color='red', label="Actual Price")
    plt.show()
