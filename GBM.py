import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal, qmc, norm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GBM:
    def __init__(self, S, mu, sigma) -> None:
        self.S = S
        self.mu = mu
        self.sigma = sigma
        self.simulated_path = None  # Initialize simulated path to None

    # Exact simulation of the Geometric Brownian Motion
    def simulate(self, T, N, M):  # T: Time horizon, N: Number of time steps, M: Number of paths
        # Parameters
        dt = T / N    # Time increment
        S = np.zeros((N + 1, M))
        S[0] = self.S

        # Simulate the process
        for i in range(1, N + 1):
            epsilon = np.random.normal(0, 1, M)
            S[i] = S[i - 1] * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * epsilon * np.sqrt(dt))
        
        self.simulated_path = S  # Store the simulated path for plotting

    def plot(self):
        if self.simulated_path is None:
            raise ValueError('Simulate the process first')
        # Plot the simulated path
        plt.plot(self.simulated_path)
        plt.title('Geometric Brownian Motion Simulation')
        plt.xlabel('Time')
        plt.ylabel('S(t)')
        plt.show()

class MultiBS:
    def __init__(self, path_1="datasets/UNH.csv", path_2="datasets/PFE.csv", path_3="datasets/MRK.csv", initial_fixing="2023-08-23") -> None:
        asset_1 = pd.read_csv(path_1)
        asset_2 = pd.read_csv(path_2)
        asset_3 = pd.read_csv(path_3)
        self.asset_1 = asset_1
        self.asset_2 = asset_2
        self.asset_3 = asset_3
        self.asset_1["Date"] = pd.to_datetime(self.asset_1["Date"]).dt.strftime("%Y-%m-%d")
        self.asset_2["Date"] = pd.to_datetime(self.asset_2["Date"]).dt.strftime("%Y-%m-%d")
        self.asset_3["Date"] = pd.to_datetime(self.asset_3["Date"]).dt.strftime("%Y-%m-%d")
        log_returns1 = asset_1["log_return"].dropna().values
        log_returns2 = asset_2["log_return"].dropna().values
        log_returns3 = asset_3["log_return"].dropna().values
        start_date = pd.to_datetime(initial_fixing).strftime("%Y-%m-%d")
        barrier_1 = self.asset_1.loc[self.asset_1["Date"] == start_date]["Close"].iloc[0] * 0.59
        barrier_2 = self.asset_2.loc[self.asset_2["Date"] == start_date]["Close"].iloc[0] * 0.59
        barrier_3 = self.asset_3.loc[self.asset_3["Date"] == start_date]["Close"].iloc[0] * 0.59
        self.barrier = np.array([barrier_1, barrier_2, barrier_3])
        
        self.dt = 1 / 250  # daily time increment (assuming 250 trading days per year)
        
        # Stack the asset log return arrays into a single matrix
        self.log_returns_matrix = np.column_stack([log_returns1, log_returns2, log_returns3])

        # Mean and covariance
        self.v = np.mean(self.log_returns_matrix, axis=0) / self.dt
        self.Sigma = torch.tensor(np.cov(self.log_returns_matrix.T) / self.dt)

    # Simulation function for Geometric Brownian Motion (GBM)
    def simulate_multi_GBM_exact(self, m, M=10_000, S0=None):
        if S0 is None:
            S0 = np.array([self.asset_1["Close"].iloc[-m], 
                        self.asset_2["Close"].iloc[-m], 
                        self.asset_3["Close"].iloc[-m]])
            
        # Single asset case to fix broadcasting issue \_(ツ)_/
        if m == 1:
            Z = multivariate_normal.rvs(mean=self.v * self.dt, cov=self.Sigma * self.dt, size=(m,M))
            S = np.zeros((3, m+1, M))
            S[:, 0, :] = S0[:, np.newaxis]
            S[:, 1, :] = S[:, 0, :] * np.exp(Z.T)
            return S
        
        p = len(S0)           # number of assets
        S = np.zeros((p, m+1, M))
        S[:, 0, :] = S0[:, np.newaxis]    
        # Simulate multivariate normal random variables (m x p matrix)
        
        Z = multivariate_normal.rvs(mean=self.v * self.dt, cov=self.Sigma * self.dt, size=(m,M))
        # Iterate and simulate asset prices
        for j in range(1, m + 1):
            S[:, j, :] = S[:, j - 1, :] * np.exp(Z[j - 1, :].T)  # Direct increment of prices from the previous step
        return S
    

    # Simulation function for Geometric Brownian Motion (GBM)
    def simulate_multi_GBM_antithetic(self, m, M=10_000, S0=None):
        if S0 is None:
            S0 = np.array([self.asset_1["Close"].iloc[-m], 
                        self.asset_2["Close"].iloc[-m], 
                        self.asset_3["Close"].iloc[-m]])
        p = len(S0)           # number of assets
        S = np.zeros((p, m+1, M))
        S[:, 0, :] = S0[:, np.newaxis]    
        n = M // 2
        # Simulate multivariate normal random variables (m x p matrix)
        
        Z = multivariate_normal.rvs(mean=self.v * self.dt, cov=self.Sigma * self.dt, size=(m,n))
        antithetic_Z = -Z
        # Iterate and simulate asset prices
        for j in range(1, m+1):
            S[:, j, :n] = S[:, j - 1, :n] * np.exp(Z[j - 1, :].T)  # Direct increment of prices from the previous step
            S[:, j, n:] = S[:, j - 1, n:] * np.exp(antithetic_Z[j - 1, :].T)
        return S


    # Control Variate Simulation using Simplified GBM
    def simulate_multi_GBM_control_variate(self, m, M=10_000, S0=None):
        if S0 is None:
            S0 = np.array([self.asset_1["Close"].iloc[-m], 
                        self.asset_2["Close"].iloc[-m], 
                        self.asset_3["Close"].iloc[-m]])
        p = len(S0)  # Number of assets
        S = np.zeros((p, m + 1, M))
        S[:, 0, :] = S0[:, np.newaxis]

        # Simulate multivariate normal random variables for the complex GBM model
        Z = multivariate_normal.rvs(mean=self.v * self.dt, cov=self.Sigma * self.dt, size=(m, M))

        # Initialize control variate using simplified GBM with fixed drift and volatility
        fixed_drift = np.mean(self.v)  # Use the average drift as the control drift
        fixed_volatility = np.sqrt(np.diag(self.Sigma)).mean()  # Use the average volatility

        control_variate = np.zeros((p, M))
        for j in range(1, m + 1):
            # Update the main GBM simulation
            S[:, j, :] = S[:, j - 1, :] * np.exp(Z[j - 1, :].T)

            # Analytical solution for the simplified GBM (control variate)
            dt = self.dt
            control_variate += S[:, j - 1, :] * (
                np.exp((fixed_drift - 0.5 * fixed_volatility**2) * dt + fixed_volatility * Z[j - 1, :].T) - 1
            )

        # Adjust the simulated paths using the control variate
        control_adjustment = np.mean(control_variate, axis=1)
        S[:, -1, :] -= control_adjustment[:, np.newaxis]

        return S

    # Stratified Sampling Simulation
    def simulate_multi_GBM_stratified(self, m, M=10_000, S0=None):
        if S0 is None:
            S0 = np.array([self.asset_1["Close"].iloc[-m], 
                        self.asset_2["Close"].iloc[-m], 
                        self.asset_3["Close"].iloc[-m]])
        p = len(S0)
        S = np.zeros((p, m + 1, M))
        S[:, 0, :] = S0[:, np.newaxis]

        n_strata = 100
        strata_bounds = norm.ppf(np.linspace(0, 1, n_strata + 1)[1:-1])
        # Cholesky decomposition of the covariance matrix
        L = np.linalg.cholesky(self.Sigma * self.dt)

        # Generate stratified samples for the first timestep (j = 0)
        Z = np.zeros((m, p, M))
        for i in range(M):
            # Select a random stratum
            stratum = np.random.choice(strata_bounds, size=p)
            # Generate uncorrelated stratified samples
            uncorrelated_samples = np.random.normal(loc=self.v * self.dt + stratum, scale=1.0, size=p)
            # Apply Cholesky transform to introduce correlation
            Z[0, :, i] = L @ uncorrelated_samples


        # Standard multivariate normal sampling for subsequent timesteps (j > 1)
        for j in range(1, m):
            Z[j, :, :] = multivariate_normal.rvs(mean=self.v * self.dt, cov=self.Sigma * self.dt, size=M).T

        # Iterate and simulate asset prices
        for j in range(1, m + 1):
            S[:, j, :] = S[:, j - 1, :] * np.exp(Z[j - 1, :, :])

        return S


    # Importance Sampling Simulation
    def simulate_multi_GBM_importance_sampling(self, m, M=10_000, S0=None, shift_factor=1.5):
        if S0 is None:
            S0 = np.array([self.asset_1["Close"].iloc[-m], 
                        self.asset_2["Close"].iloc[-m], 
                        self.asset_3["Close"].iloc[-m]])
        p = len(S0)
        S = np.zeros((p, m + 1, M))
        S[:, 0, :] = S0[:, np.newaxis]

        # Shift the mean for importance sampling
        shifted_mean = self.v * self.dt * shift_factor
        Z = multivariate_normal.rvs(mean=shifted_mean, cov=self.Sigma * self.dt, size=(m, M))

        for j in range(1, m + 1):
            S[:, j, :] = S[:, j - 1, :] * np.exp(Z[j - 1, :].T)

        # Reweight the samples to account for the shifted mean
        weights = np.exp(-shift_factor * self.v * self.dt * m).reshape(-1, 1)
        S[:, -1, :] *= weights
        return S

    
    def simulate_multi_GBM_quasi(self, m, M=10_000, S0=None):
        if S0 is None:
            S0 = np.array([self.asset_1["Close"].iloc[-m], 
                        self.asset_2["Close"].iloc[-m], 
                        self.asset_3["Close"].iloc[-m]])
        p = len(S0)           # number of assets
        S = np.empty((2 * (M // 2), 3, 1 + m))
        S0 = S0.flatten()
        S[:,:,0] = np.tile(S0, (2 * (M // 2), 1, 1)).reshape(-1, p)
        
        n = M // 2
        # Simulate multivariate normal random variables (m x p matrix)
        sobol = qmc.Sobol(p * m)
        Z = norm.ppf(sobol.random(n)).reshape(n, m, p, 1) # Remember to inverse CDF :)
        Z = np.linalg.cholesky(self.Sigma) @ Z
        Z = Z.transpose(1,2)
        Z = np.concatenate((Z,-Z), axis=0).reshape(2 * n, p, m)
        
        # Z = multivariate_normal.rvs(mean=self.v * self.dt, cov=self.Sigma * self.dt, size=(m,n))
        # Iterate and simulate asset prices
        for j in range(m):
            # S[:, j, :n] = S[:, j - 1, :n] * np.exp(Z[j - 1, :].T)  # Direct increment of prices from the previous step

            # S[:, j, n:] = S[:, j - 1, n:] * np.exp(antithetic_Z[j - 1, :].T)
            S[:,:,j+1] = S[:,:,j] * (1 + self.v * self.dt + Z[:,:,j] * np.sqrt(self.dt))
        
        return S.transpose(1,2,0)
    

    # Calibrate using simulated forward data
    def calibrate_model(self, simulated_data):
        # Convert simulated data to log returns
        simulated_log_prices = np.log(simulated_data)
        simulated_log_returns = simulated_log_prices[:, 1:, :] - simulated_log_prices[:, :-1, :] # Shape: (p, m, M)
        
        # Calculate mean and standard deviation of simulated log returns
        mean_simulated_log_returns = np.mean(simulated_log_returns, axis=(1,2))
        std_simulated_log_returns = np.std(simulated_log_returns, axis=(1,2))

        # Calculate forward-looking mu and sigma
        forward_mu = mean_simulated_log_returns / self.dt + (std_simulated_log_returns ** 2) / 2
        forward_sigma = std_simulated_log_returns / np.sqrt(self.dt)

        # Update v
        self.v = forward_mu - 0.5 * (forward_sigma ** 2)
        # Construct Sigma as the full covariance matrix from simulated log returns
        reshaped_returns = simulated_log_returns.reshape(simulated_log_returns.shape[0], -1)  # Shape: (p, m * M)
        self.Sigma = torch.tensor(np.cov(reshaped_returns) / self.dt)

        return self.v, self.Sigma
    

    def plot(self, simulated_data, show_barrier=True):
        if simulated_data is None:
            raise ValueError('No simulated data available. Run the simulation first.')
        
        p, m, M = simulated_data.shape  # number of assets, time steps, number of paths

        # Create subplots for each asset
        fig, axes = plt.subplots(p, 1, figsize=(10, 5 * p), dpi=80, sharex=True)
        
        for i in range(p):
            ax = axes[i] if p > 1 else axes  # Select the appropriate axis if p > 1
            # Plot each path for asset i
            ax.plot(simulated_data[i], alpha=0.7)
            ax.set_title(f'Simulated Paths for Asset {i+1}')
            if show_barrier:
                ax.axhline(y=self.barrier[i], color='red', linestyle='--', label='59% of Initial Value')
                ax.annotate(f'{self.barrier[i]:.2f}', xy=(0, self.barrier[i]),
                        xytext=(-30, 0), textcoords='offset points',
                        color='red', fontsize=10, va='center', ha='right')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Simulated Price')
        
        plt.tight_layout()
        plt.show()


    def backtest(self, M=10_000, start_date="2023-08-23", end_date="2024-08-01"):
        if start_date is None:
            raise ValueError("Please provide a valid start date for backtesting.")

        # Convert the start_date to a pandas datetime format
        start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        # Find the index of the start_date in the asset data
        start_index = self.asset_1.loc[self.asset_1["Date"] == start_date]["Close"].index[0]

        # Convert the end_date to a pandas datetime format
        end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        # Find the index of the end_date in the asset data
        end_index = self.asset_1.loc[self.asset_1["Date"] == end_date]["Close"].index[0]

        # Slice the historical data from start_date onwards
        historical_prices_1 = self.asset_1["Close"].iloc[start_index:end_index+1].values
        historical_prices_2 = self.asset_2["Close"].iloc[start_index:end_index+1].values
        historical_prices_3 = self.asset_3["Close"].iloc[start_index:end_index+1].values
        historical_prices = np.column_stack([historical_prices_1, historical_prices_2, historical_prices_3])

        # Initial prices for backtest (from start_date)
        T = (end_index - start_index)  # Time horizon for backtesting

        # Run forward simulation starting from S0
        simulated_data = self.simulate_multi_GBM_antithetic(T, M, S0=historical_prices[0, :])
        
        # Plot backtesting results
        
        mse = np.mean((simulated_data[:, :, :].mean(axis=2) - historical_prices.T) ** 2, axis=1)
        
        self.plot_backtest_results(simulated_data=simulated_data, historical_prices=historical_prices, mse=mse)

        print("Mean Squared Error (MSE) for each asset:", mse)
        print("Root Mean Squared Error (RMSE) for each asset:", np.sqrt(mse))
        return mse

    
    # Plotting function for backtesting
    def plot_backtest_results(self, simulated_data, historical_prices, mse):
        p, m, M = simulated_data.shape  # number of assets, time steps, number of paths
        
        # Create subplots for each asset
        fig, axes = plt.subplots(p, 1, figsize=(10, 5 * p), dpi=80, sharex=True)
        
        for i in range(p):
            ax = axes[i] if p > 1 else axes  # Select the appropriate axis if p > 1
            # Plot each simulated path for asset i
            for path in range(M):
                ax.plot(simulated_data[i, :, path], color='blue', alpha=0.3)
            # Plot the actual historical price path for asset i
            ax.plot(historical_prices[:, i], color='red', label="Actual Price")
            ax.annotate(f'MSE: {mse[i]:.4f}', xy=(0.95, 0.95), xycoords='axes fraction', horizontalalignment='right', verticalalignment='top', fontsize=12)
            ax.axhline(y=self.barrier[i], color='red', linestyle='--', label='59% of Initial Value')
            ax.annotate(f'{self.barrier[i]:.2f}', xy=(0, self.barrier[i]),
                    xytext=(-30, 0), textcoords='offset points',
                    color='red', fontsize=10, va='center', ha='right')
            ax.set_title(f'Backtest Results for Asset {i+1}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Price')
            ax.legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()


    # Calculate Delta using finite difference method
    def calculate_delta(self, pay_off, m=317, M=10_000, h=0.01, S0=None):
        # Initial asset prices at the start time T
        if S0 is None:
            S0 = np.array([self.asset_1["Close"].iloc[-m],
                        self.asset_2["Close"].iloc[-m],
                        self.asset_3["Close"].iloc[-m]])

        # Initialize delta array
        delta = np.zeros_like(S0)

        # Loop over each asset to compute individual deltas
        for i in range(len(S0)):
            # Create perturbed asset price arrays for plus and minus perturbations
            S0_plus = S0.copy()
            S0_minus = S0.copy()
            
            # Perturb the current asset
            S0_plus[i] *= (1 + h)
            S0_minus[i] *= (1 - h)
            
            # Simulate final values with perturbed prices
            V_S_plus = self.simulate_multi_GBM_antithetic(m, M, S0_plus)
            V_S_minus = self.simulate_multi_GBM_antithetic(m, M, S0_minus)
            
            # Evaluate the payoff for perturbed values
            pay_off_plus = pay_off.evaluate_payoff(np.log(V_S_plus).transpose(2, 0, 1), True)
            pay_off_minus = pay_off.evaluate_payoff(np.log(V_S_minus).transpose(2, 0, 1), True)
            expected_pay_off_plus = pay_off_plus.mean()
            expected_pay_off_minus = pay_off_minus.mean()
            # Calculate delta for the current asset
            delta[i] = (expected_pay_off_plus - expected_pay_off_minus) / (2 * S0[i] * h)

        return delta

    
    def calculate_gamma(self, pay_off, m=317, M=10_000, h=0.01, S0=None):
        if S0 is None:
            S0 = np.array([self.asset_1["Close"].iloc[-m],
                        self.asset_2["Close"].iloc[-m],
                        self.asset_3["Close"].iloc[-m]])
        p = len(S0)
        gamma = np.zeros((p, p))

        # Scale the perturbation size by the underlying prices
        h_scaled = h * S0
        
        # Calculate Gamma for each pair of assets using the new formula
        for i in range(p):
            for j in range(p):                
                # Perturb asset prices for four-point finite difference calculation
                S0_ij_pp = S0.copy()  # S + h*e_i + h*e_j
                S0_ij_pm = S0.copy()  # S + h*e_i - h*e_j
                S0_ij_mp = S0.copy()  # S - h*e_i + h*e_j
                S0_ij_mm = S0.copy()  # S - h*e_i - h*e_j
                
                # Apply perturbations
                S0_ij_pp[i] += h_scaled[i]
                S0_ij_pp[j] += h_scaled[j]

                S0_ij_pm[i] += h_scaled[i]
                S0_ij_pm[j] -= h_scaled[j]

                S0_ij_mp[i] -= h_scaled[i]
                S0_ij_mp[j] += h_scaled[j]

                S0_ij_mm[i] -= h_scaled[i]
                S0_ij_mm[j] -= h_scaled[j]

                # Simulate option values with perturbed prices (using final time step)
                V_S_ij_pp = self.simulate_multi_GBM_antithetic(m, M, S0_ij_pp)
                V_S_ij_pm = self.simulate_multi_GBM_antithetic(m, M, S0_ij_pm)
                V_S_ij_mp = self.simulate_multi_GBM_antithetic(m, M, S0_ij_mp)
                V_S_ij_mm = self.simulate_multi_GBM_antithetic(m, M, S0_ij_mm)

                # Apply the payoff evaluation to each simulated value with log-transformation
                pay_off_ij_pp = pay_off.evaluate_payoff(np.log(V_S_ij_pp).transpose(2, 0, 1), True).mean()
                pay_off_ij_pm = pay_off.evaluate_payoff(np.log(V_S_ij_pm).transpose(2, 0, 1), True).mean()
                pay_off_ij_mp = pay_off.evaluate_payoff(np.log(V_S_ij_mp).transpose(2, 0, 1), True).mean()
                pay_off_ij_mm = pay_off.evaluate_payoff(np.log(V_S_ij_mm).transpose(2, 0, 1), True).mean()

                # Calculate Gamma using the central finite difference formula
                gamma[i, j] = (pay_off_ij_pp - pay_off_ij_pm - pay_off_ij_mp + pay_off_ij_mm) / (4 * h_scaled[i] * h_scaled[j])

        return gamma
