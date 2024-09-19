import numpy as np
import numpy.linalg as la
from scipy.stats import norm
from numpy.random import uniform, multivariate_normal
import torch

class Heston():
    
    def __init__(self):
        
        self.params = {'vol_0': 1,
                       'long_vol': 1,
                       'rho': -0.5,
                       'mean_reversion': 1,
                       'vol_vol': 1}
        self.payoff = None
    
    def brownian_motion(self, num_steps, n):
        rho = self.params['rho']
        cov = np.array([[1, rho], [rho, 1]])
        L = la.cholesky(cov)
        
        # Stratified sampling of 100^2 strata for the first step
        bm = uniform(size=(n,n,1,2))/n
        offsets = np.tile(np.arange(n)/n, (n,1))
        offsets = np.expand_dims(np.dstack((offsets, offsets.T)), axis=2)
        bm += offsets
        bm = norm.ppf(bm)
        bm = L @ bm # stratified samples. Each strata has equal weight since they are defined by percentiles
        
        bm = np.concatenate((bm, multivariate_normal([0, 0], cov, (n, n, num_steps-1))), axis=2)
        
        return bm
    
    def sim_path(self, r, S_0, dt, bm, num_steps, n=1000):
        path = np.empty([n,n,num_steps+1,2])
        path[:,:,0,0] = np.log(S_0)
        path[:,:,0,1] = self.params['vol_0']
        
        for i in range(num_steps):
            
            # Naive method to ignore negative volatility
            path[:,:,i+1,1] = path[:,:,i,1] \
                            + self.params['mean_reversion'] * (self.params['long_vol'] - np.clip(path[:,:,i,1], a_min=0)) * dt \
                            + self.params['vol_vol'] * np.sqrt(np.clip(path[:,:,i,1], a_min=0)) * bm[:,:,i,1] * np.sqrt(dt)
            
            # We simulate log price instead because it is cleaner
            path[:,:,i+1,0] = path[:,:,i,0] \
                            + (r[i] - path[:,:,i,1] / 2) * dt \
                            + np.sqrt(path[:,:,i,1]) * bm[:,:,i,0] * np.sqrt(dt)
    
        return path
                
    def control_variate(self, r, S_0, dt, num_steps, n=1000):
        # We use the final positions of the Heston paths as the control variates to the payoffs, i.e. X = payoff(S_T), Y = S_T
        # The paths are simulated twice, first to estimate the control variate coefficient then to do the actual thingy fuck
        assert self.payoff is not None
        
        bm_1 = self.brownian_motion(num_steps, n)
        Y_1 = self.sim_path(r, S_0, dt, bm_1, num_steps)
        Y_mean = np.mean(Y_1[:,:,-1,0])
        X_1 = self.payoff(Y_1)
        X_mean = np.mean(X_1)
        Y_1 = Y_1[:,:,-1,0] # Discard all path data
        c = - np.sum((X_1 - X_mean) * (Y_1 - Y_mean)) / (n**2 - 1) / np.var()
        del X_1, Y_1, bm_1 # Hopefully this makes it faster
        
        bm_2 = self.brownian_motion(num_steps, n)
        Y_2 = self.sim_path(r, S_0, dt, bm_2, num_steps)
        Y_2_anti = self.sim_path(r, S_0, dt, -bm_2, num_steps) # Antithetic variate
        X_2 = self.payoff(Y_2)
        X_2_anti = self.payoff(Y_2_anti)
        del bm_2
        # For now we calculate the antithetic variables as taking the negative of the 2D uncorrelated normal dist
        # Later, we can also try taking each 90 degree rotation for each 2D normal dist variable
        # That will require a bit more change in code since the sign doesn't factor through L in the generation of bm
        
        return X_2 + c * (Y_2 - Y_mean), X_2_anti + c * (Y_2_anti - Y_mean)
        
    def set_payoff(self, payoff_fn):
        # Payoff must take a [n,n,i,2] array and output an [n,n] array corresponding to the payoffs of each path
        self.payoff = payoff_fn    