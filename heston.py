import torch
import torch.optim as optim

class Heston():
    
    def __init__(self, vol_0=1, long_vol=1, rho=-0.5, mean_reversion=1, vol_vol=1, r=1, S_0=1000):
        
        self.params = {'vol_0': vol_0,
                       'long_vol': long_vol,
                       'rho': rho,
                       'mean_reversion': mean_reversion,
                       'vol_vol': vol_vol,
                       'r': r}
        self.S_0 = S_0
        self.payoff = None
        self.optimizer = optim.Adam(self.params(), lr=1e-3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def brownian_motion(self, m, sobol_engine):
        cov = torch.tensor([[1, self.params['rho']], [self.params['rho'], 1]], device=self.device)
        L = torch.linalg.cholesky(cov)
        
        # Quasi-random sampling of paths
        bm = sobol_engine.draw(2 * m).reshape(m,2,-1)
        
        # Importance sampling by mapping onto a heavier-tailed normal distribution
        # Selection of 2.5 is arbitrary, can experiment with different tail fatnesses
        # This results in a weight of normalpdf(x) / normal(x/2.5)
        bm = torch.distributions.Normal(0, 2.5).icdf(bm)
        
        # Mapping samples to the required correlated distribution
        bm = torch.permute(L @ torch.permute(bm, (0, 2, 1)), (0, 2, 1))
        
        return bm
    
    def sim_path(self, dt, bm, num_steps):
        path = torch.empty((bm.shape[0], 2, bm.shape[2] + 1), device=self.device)
        path[:,0,0] = torch.log(torch.tensor(self.S_0, device=self.device))  
        path[:,1,0] = self.params['vol_0']
        
        for i in range(num_steps):
            
            # Naive method to ignore negative volatility
            path[:,1,i+1] = path[:,1,i] \
                            + self.params['mean_reversion'] * (self.params['long_vol'] - torch.clamp(path[:,1,i], min=0)) * dt \
                            + self.params['vol_vol'] * torch.sqrt(torch.clamp(path[:,1,i], min=0)) * bm[:,1,i] * torch.sqrt(dt)
            
            # We simulate log price instead because it is cleaner
            path[:,0,i+1] = path[:,0,i] \
                            + (self.params['r'] - path[:,1,i] / 2) * dt \
                            + torch.sqrt(torch.clamp(path[:,1,i], min=0)) * bm[:,0,i] * torch.sqrt(dt)
    
        return path
                
    def control_variate(self, dt, num_steps, sobol_engine, m=10000):
        
        # We use the final positions of the Heston paths as the control variates to the payoffs, i.e. X = payoff(S_T), Y = S_T
        # The paths are simulated twice, first to estimate the control variate coefficient then to do the actual thingy
        assert self.payoff is not None
        
        bm_1 = self.brownian_motion(m, sobol_engine)
        Y_1 = self.sim_path(dt, bm_1, num_steps).detach()
        Y_mean = torch.mean(Y_1)
        X_1 = self.payoff(Y_1)
        X_mean = torch.mean(X_1)
        Y_1 = Y_1[:,0,-1] # Discard all path data
        
        # Estimate the correlation coefficient
        X_1 = X_1 - X_mean
        Y_1 = Y_1 - Y_mean
        cov = X_1.T @ Y_1 / (m - 1)
        var = Y_1.T @ Y_1 / (m - 1)
        c = (-cov/var)
        
        # Second actual round now. Yay! :D
        bm_2 = self.brownian_motion(m, sobol_engine)
        Y_2 = self.sim_path(dt, bm_2, num_steps)
        Y_2_anti = self.sim_path(dt, -bm_2, num_steps) # Antithetic variate
        X_2 = self.payoff(Y_2)
        X_2_anti = self.payoff(Y_2_anti)
        
        # For now we calculate the antithetic variables as taking the negative of the 2D uncorrelated normal dist
        # Later, we can also try taking each 90 degree rotation for each 2D normal dist variable
        # That will require a bit more change in code since the sign doesn't factor through L in the generation of bm
        
        return X_2 + c * (Y_2 - Y_mean), X_2_anti + c * (Y_2_anti - Y_mean)
        
    def set_payoff(self, payoff_fn):
        # Payoff must take a (m,2,i) array and output a length (m,) array corresponding to the payoffs of each path
        self.payoff = payoff_fn    
        
    # #TODO#
    # def lesgo(self):
    #     This is just so I know the code for when I do the driver later
    #     sobol_engine = torch.quasirandom.SobolEngine(dimension=num_steps)
