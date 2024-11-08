import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
import torch
from tqdm import tqdm

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
    def __init__(self, path_1="datasets/UNH.csv", path_2="datasets/PFE.csv", path_3="datasets/MRK.csv") -> None:
        asset_1 = pd.read_csv(path_1)
        asset_2 = pd.read_csv(path_2)
        asset_3 = pd.read_csv(path_3)
        self.asset_1 = asset_1
        self.asset_2 = asset_2
        self.asset_3 = asset_3
        log_returns1 = asset_1["log_return"].dropna().values
        log_returns2 = asset_2["log_return"].dropna().values
        log_returns3 = asset_3["log_return"].dropna().values
        
        self.dt = 1 / 250  # daily time increment (assuming 250 trading days per year)
        
        # Stack the asset log return arrays into a single matrix
        self.log_returns_matrix = np.column_stack([log_returns1, log_returns2, log_returns3])

        # Mean and covariance
        self.v = np.mean(self.log_returns_matrix, axis=0) / self.dt
        self.Sigma = torch.tensor(np.cov(self.log_returns_matrix.T) / self.dt)

    # Simulation function for Geometric Brownian Motion (GBM)
    def simulate_multi_GBM_exact(self, T, M=1_000):
        
        m = int(T / self.dt)  # number of periods
        S0 = np.array([self.asset_1["Close"].iloc[-m], 
                       self.asset_2["Close"].iloc[-m], 
                       self.asset_3["Close"].iloc[-m]])
        p = len(S0)           # number of assets
        S = np.zeros((p, m + 1, M))
        S[:, 0, :] = S0[:, np.newaxis]
        
        # Simulate multivariate normal random variables (m x p matrix)
        Z = multivariate_normal.rvs(mean=self.v * self.dt, cov=self.Sigma * self.dt, size=(m,M))
        
        # Iterate and simulate asset prices
        for j in range(1, m + 1):
            S[:, j, :] = S[:, j - 1, :] * np.exp(Z[j - 1, :].T)  # Direct increment of prices from the previous step
        
        return S
    
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
    
    def plot(self, simulated_data):
        if simulated_data is None:
            raise ValueError('No simulated data available. Run the simulation first.')
        
        p, m, M = simulated_data.shape  # number of assets, time steps, number of paths

        # Create subplots for each asset
        fig, axes = plt.subplots(p, 1, figsize=(10, 5 * p), sharex=True)
        
        for i in range(p):
            ax = axes[i] if p > 1 else axes  # Select the appropriate axis if p > 1
            # Plot each path for asset i
            ax.plot(simulated_data[i], alpha=0.7)
            ax.set_title(f'Simulated Paths for Asset {i+1}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Simulated Price')
        
        plt.tight_layout()
        plt.show()

    # Backtesting function
    def backtest(self, T, M):
        # Determine the start date for the backtest
        historical_prices_1 = self.asset_1["Close"].iloc[-int(T / self.dt):].values
        historical_prices_2 = self.asset_2["Close"].iloc[-int(T / self.dt):].values
        historical_prices_3 = self.asset_3["Close"].iloc[-int(T / self.dt):].values
        historical_prices = np.column_stack([historical_prices_1, historical_prices_2, historical_prices_3])
        
        # Initial prices for backtest (T periods ago)
        S0 = historical_prices[0, :]
        
        # Run forward simulation starting from S0
        simulated_data = self.simulate_multi_GBM_exact(T, M)
        
        # Plot backtesting results
        
        # Calculate error metrics (e.g., mean squared error)
        mse = np.mean((simulated_data[:, :len(historical_prices), :].mean(axis=2) - historical_prices.T) ** 2, axis=1)
        
        self.plot_backtest_results(simulated_data=simulated_data, historical_prices=historical_prices, T=T, mse=mse)

        print("Mean Squared Error (MSE) for each asset:", mse)
        return mse
    
    # Plotting function for backtesting
    def plot_backtest_results(self, simulated_data, historical_prices, mse, T=1):
        historical_prices_1 = self.asset_1["Close"].iloc[-int(T / self.dt):].values
        historical_prices_2 = self.asset_2["Close"].iloc[-int(T / self.dt):].values
        historical_prices_3 = self.asset_3["Close"].iloc[-int(T / self.dt):].values
        historical_prices = np.column_stack([historical_prices_1, historical_prices_2, historical_prices_3])
        p, m, M = simulated_data.shape  # number of assets, time steps, number of paths
        
        initial_values = simulated_data[:, 0, 0]  # Taking the first value of each asset (first time step, first path)
        threshold_values = 0.59 * initial_values
        # Create subplots for each asset
        fig, axes = plt.subplots(p, 1, figsize=(10, 5 * p), dpi=80, sharex=True)
        
        for i in range(p):
            ax = axes[i] if p > 1 else axes  # Select the appropriate axis if p > 1
            # Plot each simulated path for asset i
            for path in range(M):
                ax.plot(simulated_data[i, :, path], color='blue', alpha=0.3)
            # Plot the actual historical price path for asset i
            ax.plot(historical_prices[:, i], color='red', label="Actual Price" if i == 0 else None)
            ax.annotate(f'MSE: {mse[i]:.4f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12)
            ax.axhline(y=threshold_values[i], color='red', linestyle='--', label='59% of Initial Value')
            ax.set_title(f'Backtest Results for Asset {i+1}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Price')
            ax.legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()

    def multi_asset_path(self, u, uni, params, r, corr, S0=None, dt=1/250, n=3, verbose=False, *args, **kwargs):
        """
        Multi-Asset Path Simulation using Geometric Brownian Motion (GBM).
        Args:
            u: Array of normally drawn values (shape: num_paths, num_assets, num_timesteps, 2)
            params: Array of Black-Scholes parameters for each asset (shape: n, 1), [sigma]
            r: Risk-free rate
            corr: Correlation matrix between assets
            S0: Initial stock prices (shape: n, 1)
            dt: Time increment (default is 1/250 for daily steps)
            n: Number of assets
            verbose: Display progress bar
        Returns:
            path: Simulated paths for log prices and volatility (shape: num_paths, n, num_timesteps + 1, 2)
        """
        num_paths, num_assets, num_timesteps = u.shape[:3]

        # Simulate paths using the existing simulate_multi_GBM_exact function
        simulated_prices = self.simulate_multi_GBM_exact(T=num_timesteps * dt, M=num_paths)

        # `simulated_prices` shape is (num_assets, num_timesteps + 1, num_paths)
        # Transpose and reshape to (num_paths, num_assets, num_timesteps + 1)
        simulated_prices = torch.tensor(simulated_prices, dtype=torch.float64, device=device).permute(2, 0, 1)

        # Initialize the path array with shape (num_paths, num_assets, num_timesteps + 1, 2)
        path = torch.empty((num_paths, num_assets, num_timesteps + 1, 2), device=device)

        # Fill in the log prices (index 0 in the last dimension)
        path[:, :, :, 0] = torch.log(simulated_prices)

        # Fill in the volatility (index 1 in the last dimension) using the provided sigma parameters
        sigma_arr = params[:, 0].reshape(1, -1).repeat(num_paths, 1)
        volatility = sigma_arr.unsqueeze(-1).repeat(1, 1, num_timesteps + 1)

        path[:, :, :, 1] = volatility

        return path

