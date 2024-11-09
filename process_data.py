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
    df1 = preprocess_stock_data(pd.read_csv(os.path.join("datasets", "UNH.csv")))
    df2 = preprocess_stock_data(pd.read_csv(os.path.join("datasets", "PFE.csv")))
    df3 = preprocess_stock_data(pd.read_csv(os.path.join("datasets", "MRK.csv")))

    # Rename columns for clarity
    df1.columns = ['Date', 'stock_price_UNH']
    df2.columns = ['Date', 'stock_price_PFE']
    df3.columns = ['Date', 'stock_price_MRK']

    # Merge stock data by Date
    stock_df = df1.merge(df2, on='Date').merge(df3, on='Date')

    # Load and preprocess interest rates
    rates = pd.read_csv(os.path.join("datasets", "DGS10.csv")).replace('.', np.nan).ffill()
    rates.columns = ['Date', 'true_rate']
    rates['true_rate'] = pd.to_numeric(rates['true_rate']) / 100  # Convert to decimal
    rates['Date'] = pd.to_datetime(rates['Date']).dt.tz_localize(None).dt.date

    # Merge stock and rate data
    stock_df = stock_df.merge(rates, on='Date')

    return stock_df

# # Define Mardia's test for multivariate normality
# def mardia_test(data):
#     """ Mardia's test for multivariate normality """
    
#     # Calculate Mardia's skewness and kurtosis
#     data = torch.tensor(data.values)
#     n = data.shape[0]
#     mean = torch.mean(data, dim=0)
#     observations = data.reshape(-1,1,3) - mean
#     sigma = torch.mean(observations.transpose(1,2) @ observations, dim=0)
#     rhs = torch.linalg.solve(sigma, observations.transpose(1,2))
#     prod = observations.reshape(-1,3) @ rhs.reshape(3,-1)
    
#     # Chi-square test for skewness
#     skewness = torch.sum(prod**3) / (6 * n)
#     df_skew = n * (n + 1) * (n + 2) / 6  # degrees of freedom for skewness
#     p_value_skew = 1 - stats.chi2.cdf(skewness, df=df_skew)
    
#     # Normal distribution test for kurtosis
#     kurtosis = np.sqrt(n / (8 * 3 * 5)) * (torch.mean(torch.diag(prod)**2) - 15)
#     p_value_kurt = 2 * (1 - stats.norm.cdf(abs(kurtosis)))
    
#     return p_value_skew, p_value_kurt

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
    loss = (len(z) - 1) * (np.log(sp.special.gamma((nu + 3) / 2)) - 0.5 * np.log(np.linalg.det(Omega)) \
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

    vars = 4 * steps
    # remaining = np.maximum(vars - 21201, 0)
    # sobol_engine = stats.qmc.Sobol(np.minimum(vars, 21201))
    # paths1 = sobol_engine.random(paths)
    # paths2 = np.random.rand(paths, remaining)
    # norm = np.concatenate((paths1, paths2), axis=1).reshape(paths,steps,4,1)
    # norm = (norm - 2 * 1e-6) + 1e-6
    # norm = stats.norm.ppf(norm)
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
        
        d_t = np.diag(np.sqrt(univariate_vars[i,:]))
        next_shock = d_t @ q_star_t @ np.linalg.cholesky(q_t) @ a_t
        next_stock_price = forecast[:,:,i].reshape(num_paths, 3, 1) * (1 + r * dt + next_shock * np.sqrt(dt) / 100)
        forecast = np.concatenate((forecast, next_stock_price.reshape(num_paths,3,1)), axis=2)
        
        q_t =  (1 - a - b) * q0 + a * (a_t @ a_t.transpose(0,2,1)) + b * q_t
        q_star_t = np.apply_along_axis(np.diag, -1, np.diagonal(q_t, axis1=1, axis2=2)**-0.5)
    
    return forecast

# Main execution
merged_df = load_data()

# Find simple returns shock
for stock in ['UNH', 'PFE', 'MRK']:
    merged_df[f'simple_return_shock_{stock}'] = (
        (merged_df[f'stock_price_{stock}'] - merged_df[f'stock_price_{stock}'].shift(1) * (1 + merged_df['true_rate'] * dt)) / (merged_df[f'stock_price_{stock}'].shift(1) * np.sqrt(dt))
        )

# Select and fit GARCH models
shocks = merged_df[['simple_return_shock_UNH', 'simple_return_shock_PFE', 'simple_return_shock_MRK']].dropna().loc[1:]

# print("Goodness of fit")
# print(multivariate_normality(shocks))

# print("Skew")
# mardia_test(shocks)

garch_models = fit_garch_models(shocks)

garch_data = {}
for stock in ['UNH', 'PFE', 'MRK']:
    garch_data[stock] = {
            'std_resid': garch_models[stock].std_resid,
            'conditional_volatility': garch_models[stock].conditional_volatility
    }

# Initial correlation matrix
z = np.column_stack([garch_data[stock]['std_resid'] for stock in ['UNH', 'PFE', 'MRK']])
sigma = np.column_stack([garch_data[stock]['conditional_volatility'] for stock in ['UNH', 'PFE', 'MRK']])
q0 = np.cov(z, rowvar=False)
r = merged_df['true_rate'].iloc[-1]

# Optimize DCC-GARCH parameters
initial_params = [0.25, 0.25, 3, 0, 0, 0]
optimized_params = sp.optimize.minimize(
    dcc_garch_log_loss, initial_params, args=(z, sigma, q0), method='SLSQP',
    bounds=[(-2.3, 2.3), (-2.3, 2.3), (2, None), (None, None), (None, None), (None, None)]
).x

print("Optimized Parameters:")
a = (np.tanh(optimized_params[0]) + 1) / 2
print('a:', a)
print('b:', ((np.tanh(optimized_params[1]) + 1) / 2) * (1-a))
print('nu:', optimized_params[2])
print('iota:', optimized_params[3:].reshape(-1, 1))
print()

# Forecast future shocks
s0 = np.array(merged_df[['stock_price_UNH', 'stock_price_PFE', 'stock_price_MRK']].iloc[-1]).reshape(1,3,1)
forecast = forecast_dcc_garch(250, 1000, r, s0, optimized_params, garch_models, shocks, q0)

import matplotlib.pyplot as plt
plt.plot(np.log(forecast[:,0,:].T))
plt.show()
