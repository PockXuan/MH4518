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
    
    def sim_path(self, r, S_0, dt, num_steps, n=1000):
        # r is a list of risk-free interest rates by history
        bm = self.brownian_motion(num_steps, n)
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
                
