import numpy as np
import matplotlib.pyplot as plt

class GBM:
    def __init__(self, S, mu, sigma) -> None:
        self.S = S
        self.mu = mu
        self.sigma = sigma

    # Exact simulation of the Geometric Brownian Motion
    def simulate(self, T, N, M): # T: Time horizon, N: Number of time steps, M: Number of paths
        S = [self.S]
        # Parameters
        dt = T / N    # Time increment

        # Time steps
        t = np.linspace(0, T, N+1)

        # Initialize the process
        S = np.zeros((N+1, M))
        S[0] = self.S

        # Simulate the process
        for i in range(1, N+1):
            # generate M random normal variables
            epsilon = np.random.normal(0, 1, M)
            # Update the process
            S[i] = S[i-1] * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * epsilon * np.sqrt(dt))
        self.simulated_path = S

    def plot(self):
        if self.simulated_path is None:
            raise ValueError('Simulate the process first')
        # Plot the simulated path
        plt.plot(self.simulated_path)
        plt.title('Geometric Brownian Motion Simulation')
        plt.xlabel('Time')
        plt.ylabel('S(t)')
        plt.show()
