import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2, norm, multivariate_normal, qmc
import scipy as sp
from arch import arch_model

# Set a reference date for processing
REFERENCE_DATE = pd.to_datetime("2023-08-23").date()

# Function to preprocess data and calculate Garman-Klass variance
def preprocess_stock_data(df):
    """Convert dates, calculate Garman-Klass variance, and take log of stock prices."""
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.date
    df['garman_klass_var'] = (
        0.5 * (np.log(df['High'] / df['Low']) ** 2) - 
        (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2)
    )
    df['Close'] = np.log(df['Close'])  # Take log of the closing prices
    return df[['Date', 'Close', 'garman_klass_var']]

# Load and preprocess CSV files for stocks and interest rates
def load_data():
    """Load stock data for UNH, PFE, MRK, and interest rates, preprocess, and merge."""
    # Load stock data
    df1 = preprocess_stock_data(pd.read_csv(os.path.join("datasets", "UNH.csv")))
    df2 = preprocess_stock_data(pd.read_csv(os.path.join("datasets", "PFE.csv")))
    df3 = preprocess_stock_data(pd.read_csv(os.path.join("datasets", "MRK.csv")))

    # Rename columns for clarity
    df1.columns = ['Date', 'log_stock_UNH', 'gk_var_UNH']
    df2.columns = ['Date', 'log_stock_PFE', 'gk_var_PFE']
    df3.columns = ['Date', 'log_stock_MRK', 'gk_var_MRK']

    # Merge stock data by Date
    stock_df = df1.merge(df2, on='Date').merge(df3, on='Date')

    # Load and preprocess interest rates
    rates = pd.read_csv(os.path.join("datasets", "DGS10.csv")).replace('.', np.nan).ffill()
    rates.columns = ['Date', 'true_rate']
    rates['true_rate'] = pd.to_numeric(rates['true_rate']) / 100  # Convert to decimal
    rates['Date'] = pd.to_datetime(rates['Date']).dt.tz_localize(None).dt.date

    # Merge stock and rate data
    return stock_df.merge(rates, on='Date')

# Define Mardia's test for multivariate normality
def mardia_test(data):
    """ Mardia's test for multivariate normality """
    
    # Calculate Mardia's skewness and kurtosis
    data = torch.tensor(data.values)
    n = data.shape[0]
    mean = torch.mean(data, dim=0)
    observations = data.reshape(-1,1,3) - mean
    sigma = torch.mean(observations.transpose(1,2) @ observations, dim=0)
    rhs = torch.linalg.solve(sigma, observations.transpose(1,2))
    prod = observations.reshape(-1,3) @ rhs.reshape(3,-1)
    
    # Chi-square test for skewness
    skewness = torch.sum(prod**3) / (6 * n) 
    df_skew = n * (n + 1) * (n + 2) / 6  # degrees of freedom for skewness
    p_value_skew = 1 - chi2.cdf(skewness, df=df_skew)
    
    # Normal distribution test for kurtosis
    kurtosis = np.sqrt(n / (8 * 3 * 5)) * (torch.mean(torch.diag(prod)**2) - 15)
    p_value_kurt = 2 * (1 - norm.cdf(abs(kurtosis)))
    
    return p_value_skew, p_value_kurt

# Fit GARCH models for each stock and extract residuals and volatilities
def fit_garch_models(data):
    """Fit GARCH models to stock log returns and extract residuals and conditional volatilities."""
    garch_models = {}
    for stock in ['UNH', 'PFE', 'MRK']:
        garch_models[stock] = arch_model(100 * data[f'log_return_shock_{stock}'], vol='Garch', p=1, q=1).fit(disp='off')
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

    # Using Azzalini's skew Student's t-distribution. Sampled from a multivariate normal one dimension higher and projected and reflected accordingly
    # Used to model our errors, which inherently have kurtosis and skew
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
    w = (Omega @ iota) / np.sqrt(1 + iota.T @ Omega @ iota)
    err_cov = np.block([
        [1, w.T],
        [w, Omega]
    ])

    # future_shock_process = multivariate_normal.rvs(cov=err_cov, size=steps)
    # future_shock_process = future_shock_process[:,1:] * np.where(future_shock_process[:,0] < 0, -1, 1).reshape(-1,1) # Future errors generated by skewed t

    # return future_shock_process

    vars = 4 * steps
    remaining = np.maximum(vars - 21201, 0)
    sobol_engine = qmc.Sobol(np.minimum(vars, 21201))
    paths1 = sobol_engine.random(paths)
    paths2 = np.random.randn(paths, remaining)
    brownian = np.concatenate((paths1, paths2), axis=1).reshape(paths,steps,4,1)
    brownian = np.linalg.cholesky(err_cov) @ brownian
    brownian = brownian[:,:,:3,:] * np.where(brownian[:,:,-1,:] < 0, -1, 1).reshape(paths,steps,1,1)
    brownian = brownian.reshape(paths, steps, 3).transpose((0,2,1))

    return brownian

# Forecast future shocks based on the optimized DCC-GARCH parameters
def forecast_dcc_garch(steps, num_paths, params, garch_models, shocks, q0):
    """Forecast future shocks using DCC-GARCH model over given steps."""
    a = (np.tanh(params[0]) + 1) / 2
    b = ((np.tanh(params[1]) + 1) / 2) * (1-a)
    nu = params[2]
    iota = params[3:].reshape(-1, 1)
    future_shock = np.empty(shape=(num_paths,3,0))
        
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
        a_t = future_shock_process[:,:,i].reshape(-1,3,1)

        d_t = np.diag(np.sqrt(univariate_vars[i,:]))
        next_shock = d_t @ q_star_t @ np.linalg.cholesky(q_t) @ a_t
        future_shock = np.concatenate((future_shock, next_shock.reshape(num_paths,3,1)), axis=2)
        
        q_t =  (1 - a - b) * q0 + a * (a_t @ a_t.transpose(0,2,1)) + b * q_t
        q_star_t = np.apply_along_axis(np.diag, -1, np.diagonal(q_t, axis1=1, axis2=2)**-0.5)
    
    return future_shock / 100

# Main execution
merged_df = load_data()

# Calculate log-return shocks
for stock in ['UNH', 'PFE', 'MRK']:
    merged_df[f'log_return_shock_{stock}'] = (
        merged_df[f'log_stock_{stock}'] - merged_df[f'log_stock_{stock}'].shift(1)
    )

# Select and fit GARCH models
shocks = merged_df[['log_return_shock_UNH', 'log_return_shock_PFE', 'log_return_shock_MRK']].dropna().loc[123:512]
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

# Optimize DCC-GARCH parameters
initial_params = [0.25, 0.25, 3, 0, 0, 0]
optimized_params = sp.optimize.minimize(
    dcc_garch_log_loss, initial_params, args=(z, sigma, q0), method='SLSQP',
    bounds=[(-2.3, 2.3), (-2.3, 2.3), (2, None), (None, None), (None, None), (None, None)]
).x
# Forecast future shocks
future_shocks = forecast_dcc_garch(10, 3, optimized_params, garch_models, shocks, q0)

print("Optimized Parameters:")
a = (np.tanh(optimized_params[0]) + 1) / 2
print('a:', a)
print('b:', ((np.tanh(optimized_params[1]) + 1) / 2) * (1-a))
print('nu:', optimized_params[2])
print('iota:', optimized_params[3:].reshape(-1, 1))
print()
print("Future Shocks:")
print(future_shocks)
