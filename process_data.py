import pandas as pd
import numpy as np
import torch
import numpy as np
import os
from scipy.stats import chi2, norm, t, multivariate_normal
import scipy as sp
from arch import arch_model

# Define the reference date
reference_date = pd.to_datetime("2023-08-23").date()

# Function to calculate Garman-Klass variance
def preprocess(df):
    # Ensure Date is in datetime format
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.date
    # Calculate Garman-Klass variance for each row
    # df['garman_klass_var'] = 0.5 * (np.log(df['High'] / df['Low']) ** 2) - \
    #                          (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2)
    df['garman_klass_var'] = 0.5 * (np.log(df['High'] / df['Low']) ** 2) - \
                             (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2)
    #Take the log stock price
    df['Close'] = np.log(df['Close'])
    return df[['Date', 'Close', 'garman_klass_var']]

# Load and process each CSV
df1 = pd.read_csv(os.getcwd() + "/MH4518/datasets/UNH.csv")
df2 = pd.read_csv(os.getcwd() + "/MH4518/datasets/PFE.csv")
df3 = pd.read_csv(os.getcwd() + "/MH4518/datasets/MRK.csv")

# Calculate Garman-Klass variance and select relevant columns
df1 = preprocess(df1)
df2 = preprocess(df2)
df3 = preprocess(df3)

# Rename columns to reflect asset names
df1.columns = ['Date', 'log_stock_UNH', 'gk_var_UNH']
df2.columns = ['Date', 'log_stock_PFE', 'gk_var_PFE']
df3.columns = ['Date', 'log_stock_MRK', 'gk_var_MRK']

# Merge DataFrames on Date
merged_df = df1.merge(df2, on='Date').merge(df3, on='Date')

# Select and reorder the columns
stock_price_df = merged_df[['Date', 
                      'log_stock_UNH', 'log_stock_PFE', 'log_stock_MRK', 
                      'gk_var_UNH', 'gk_var_PFE', 'gk_var_MRK']]

rates = pd.read_csv(os.getcwd() + "/MH4518/datasets/DGS10.csv")
rates.replace('.', np.nan, inplace=True)
rates.ffill(inplace=True)
rates.columns = ['Date', 'true_rate']
rates['true_rate'] = pd.to_numeric(rates["true_rate"]) / 100
rates['Date'] = pd.to_datetime(rates['Date']).dt.tz_localize(None).dt.date

merged_df = stock_price_df.merge(rates, on='Date')

# Implement DCC-GARCH model

# Risk-free interest is known on the same day, so no need for estimation

# Deviation of log returns from expected
merged_df['log_return_shock_UNH'] = merged_df['log_stock_UNH'] - merged_df['log_stock_UNH'].shift(1)
merged_df['log_return_shock_PFE'] = merged_df['log_stock_PFE'] - merged_df['log_stock_PFE'].shift(1)
merged_df['log_return_shock_MRK'] = merged_df['log_stock_MRK'] - merged_df['log_stock_MRK'].shift(1)

# Extract shocks
data = merged_df[['log_return_shock_UNH', 'log_return_shock_PFE', 'log_return_shock_MRK']].loc[1:20]

# Step 1: Multivariate Normality Test
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

# Since the kurtosis p-value is 0, we use a skew T-distribution
# print("Mardia's test results:", mardia_test(data))

# Following the paper at https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/259296/724505_FULLTEXT01.pdf
# Step 1: Fit independent GARCH models for each series.
UNH_model = arch_model(100 * data['log_return_shock_UNH'], vol='Garch', p=1, q=1)
UNH_model = UNH_model.fit(disp='off')

PFE_model = arch_model(100 * data['log_return_shock_PFE'], vol='Garch', p=1, q=1)
PFE_model = PFE_model.fit(disp='off')

MRK_model = arch_model(100 * data['log_return_shock_MRK'], vol='Garch', p=1, q=1)
MRK_model = MRK_model.fit(disp='off')

    
z_UNH = UNH_model.std_resid
sigma_UNH = UNH_model.conditional_volatility
z_PFE = PFE_model.std_resid
sigma_PFE = PFE_model.conditional_volatility
z_MRK = MRK_model.std_resid
sigma_MRK = MRK_model.conditional_volatility
z = np.vstack([z_UNH, z_PFE, z_MRK]).T  # Each column represents an asset
sigma = np.vstack([sigma_UNH, sigma_PFE, sigma_MRK]).T
q0 = np.cov(z, rowvar=False)
print(q0)
    
# Step 2: Fit correlation terms
def log_loss(x, t, z, sigma, q0):
    
    a = (np.tanh(x[0]) + 1) / 2
    b = np.minimum((np.tanh(x[1]) + 1) / 2,1-a)
    nu = x[2]
    iota = x[3:].reshape(-1,1)
    
    g1 = sp.special.gamma((nu - 1) / 2)
    g2 = sp.special.gamma(nu / 2)
    iota_norm = np.sum(iota**2)
    denom = np.pi * g2**2 * (nu - (nu - 2) * iota_norm)
    num = 2 * (nu - 2) * (np.pi * g2**2 - (nu - 2) * g1**2) * iota_norm
    k = np.sqrt(1 + (2 * nu * num) / (denom * (nu - (nu - 2) * iota_norm)))
    
    Omega = np.eye(3)
    if not np.isclose(iota_norm, 0):
        Omega = Omega + iota_norm**-1 * (-1 + denom * (k - 1) / num) * iota @ iota.T
        # Omega = Omega + iota_norm**-1 * (-1 + (denom * (nu - (nu - 2) * iota_norm)) * (k - 1) / (4 * nu * num)) * iota @ iota.T
    Omega = Omega * (1 - 2 / nu)
    d = np.diag(np.sqrt(np.diag(Omega)))
    
    xi = -np.sqrt(nu / np.pi) * (g1 / g2) * (Omega @ iota) / np.sqrt(1 + iota.T @ Omega @ iota)
    
    # Initialize Q_t with Q0 and set starting values for a and b
    qt = q0
    loss = (len(z) - 1) * (np.log(sp.special.gamma((nu + 3) / 2)) - 0.5 * np.log(np.linalg.det(Omega)) \
            - 1.5 * np.log(np.pi * nu) - np.log(g2))

    # Loop through timesteps to calculate each correlation matrix
    for i in range(1, len(z)):
        a_t = z[i].reshape(-1, 1)
        qt = (1 - a - b) * q0 + a * (a_t @ a_t.T) + b * qt  # Update Q_t
        q_t_star_inv = np.diag(1 / np.sqrt(np.diag(qt)))  # Diagonal matrix with inverted standard deviations
        R_t = q_t_star_inv @ qt @ q_t_star_inv  # Normalize to get R_t
        
        D_t = np.diag(sigma[i])
        H_t = D_t @ R_t @ D_t
        v = np.linalg.solve(np.linalg.cholesky(H_t), a_t) - xi
        Q_a_t = v.T @ np.linalg.solve(Omega, v)
        
        loss = loss - 0.5 * (nu + 3) * np.log(1 + Q_a_t / nu) - 0.5 * np.log(np.linalg.det(R_t)) \
               + np.log(t.pdf(iota.T @ d.T @ np.linalg.solve(d, v) * np.sqrt((nu + 3) / (Q_a_t + nu)), df=nu+3))
    
    return -loss

params = sp.optimize.minimize(log_loss, np.array([0.25, 0.25, 3, 0, 0, 0]), (t, z, sigma, q0), method='SLSQP', \
                              bounds=[(None, None), (None, None), (2, None), (None, None), (None, None), (None, None)]).x

def forecast(steps, params):
    
    a = (np.tanh(params[0]) + 1) / 2
    b = np.minimum((np.tanh(params[1]) + 1) / 2,1-a)
    nu = params[2]
    iota = params[3:].reshape(-1,1)
    
    # Sample for errors
    g1 = sp.special.gamma((nu - 1) / 2)
    g2 = sp.special.gamma(nu / 2)
    iota_norm = np.sum(iota**2)
    denom = np.pi * g2**2 * (nu - (nu - 2) * iota_norm)
    num = 2 * (nu - 2) * (np.pi * g2**2 - (nu - 2) * g1**2) * iota_norm
    k = np.sqrt(1 + (2 * nu * num) / (denom * (nu - (nu - 2) * iota_norm)))
    Omega = np.eye(3)
    if not np.isclose(iota_norm, 0):
        Omega = Omega + iota_norm**-1 * (-1 + denom * (k - 1) / num) * iota @ iota.T
        # Omega = Omega + iota_norm**-1 * (-1 + (denom * (nu - (nu - 2) * iota_norm)) * (k - 1) / (4 * nu * num)) * iota @ iota.T
    Omega = Omega * (1 - 2 / nu)
    w = (Omega @ iota) / np.sqrt(1 + iota.T @ Omega @ iota)
    err_cov = np.block([
        [1, w.T],
        [w, Omega]
    ])
    future_shock_process = multivariate_normal.rvs(cov=err_cov, size=steps)
    future_shock_process = future_shock_process[:,1:] * np.where(future_shock_process[:,0] < 0, -1, 1).reshape(-1,1) # Future errors generated by skewed t
    future_shock = np.empty(shape=(0,3))
    
    UNH_forecast = UNH_model.forecast(horizon=steps)
    PFE_forecast = PFE_model.forecast(horizon=steps)
    MRK_forecast = MRK_model.forecast(horizon=steps)
    
    UNH_vars = UNH_forecast.variance
    PFE_vars = PFE_forecast.variance
    MRK_vars = MRK_forecast.variance
    univariate_vars = np.vstack([UNH_vars, PFE_vars, MRK_vars]).T
    
    q_t = q0
    for i in range(1, len(z)):
        a_t = z[i].reshape(-1, 1)
        q_t = (1 - a - b) * q0 + a * (a_t @ a_t.T) + b * q_t
    
    for i in range(steps):
        a_t = future_shock_process[i,:]
        
        q_star_t = np.diag(1 / np.sqrt(np.diag(q_t)))
        d_t = np.diag(np.sqrt(univariate_vars[i,:]))
        next_shock = d_t @ q_star_t @ np.linalg.cholesky(q_t) @ a_t
        future_shock = np.concatenate((future_shock, next_shock.reshape(1,3)), axis=0)
        
        q_t =  (1 - a - b) * q0 + a * (a_t @ a_t.T) + b * q_t
    
    return future_shock
        
    
print(params)
print(forecast(5, params))