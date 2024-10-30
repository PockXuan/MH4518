import torch
import torch.optim as optim
import normflows as nf
import torch.nn as nn
import numpy as np
from heston_model import HestonModel
from LSL import LSL
from AIS import AIS

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
standard_normal = torch.distributions.Normal(torch.tensor([0]).to(device), torch.tensor([1]).to(device))

class Simulator():
    
    def __init__(self):
        
        self.heston_model = HestonModel()
        self.LSL = LSL()
        self.AIS = None
        
        self.r = None
        
    def calibrate(self, KT_price, S0, rate):
        params, _ = self.heston_model.calibrate(KT_price, S0, rate)
        return params
    
    def single_asset_sim(self, numsteps, numpaths, params, r, S0=1, dt=1/250, psic=1.5, gamma1=0.5, gamma2=0.5, verbose=True):
            
        sobol_engine = torch.quasirandom.SobolEngine(min(2 * numsteps, 21201), True)
        u = sobol_engine.draw(numpaths)
        remaining_steps = max(2 * numsteps - 21201, 0)
        remaining_path = torch.rand((numpaths, remaining_steps))
        u = torch.cat((u, remaining_path), dim=1).reshape(numpaths, numsteps, 2)
        
        uniform = paths
        paths = paths * (1 - 2 * 1e-6) + 1e-6
        paths = standard_normal.icdf(paths)
        
        simulation = self.heston_model.single_asset_path(u, uniform, params, r=r, S0=S0, dt=dt, psic=psic, gamma1=gamma1, gamma2=gamma2, verbose=verbose)
        
        return simulation
    
    def multi_asset_sim(self, numsteps, numpaths, corr, params, r, S0=1, n=3, dt=1/250, psic=1.5, gamma1=0.5, gamma2=0.5, verbose=True):
        
        sobol_engine = torch.quasirandom.SobolEngine(min(2 * n * numsteps, 21201), True)
        u = sobol_engine.draw(numpaths)
        remaining_steps = max(2 * n * numsteps - 21201, 0)
        remaining_path = torch.rand((numpaths, remaining_steps))
        u = torch.cat((u, remaining_path), dim=1).reshape(numpaths, numsteps, n, 2)
        
        uniform = paths
        paths = paths * (1 - 2 * 1e-6) + 1e-6
        paths = standard_normal.icdf(paths)
        
        simulation = self.heston_model.multi_asset_path(u, uniform, params, r=r, corr=corr, S0=S0, dt=dt, psic=psic, gamma1=gamma1, gamma2=gamma2, verbose=verbose)
        
        return simulation
    
    def train_LSL(self, r, corr, params, num_paths=1000, max_epochs=200):
        
        self.LSL.train(r, corr, params, num_paths, max_epochs)
    
    def evaluate_payoff(self, paths, r):
        
        # Remember to train models or load trained weights first
        
        payoff = self.LSL.evaluate_payoff(paths, r)
        
        return payoff

    def train_AIS(self, num_timesteps, S0, r, params, corr, num_paths=254, max_epochs=4000):
        
        # You'll want to have trained the LSL networks by now because AIS calls it
        
        if self.AIS is None:
            self.AIS = AIS(num_timesteps, S0, r, self.LSL)
        
        c, control_mean = self.AIS.train(params, corr, num_paths=num_paths, max_epochs=max_epochs)
        
        return c, control_mean
    
    def AIS_mean(self, c, control_mean, numsteps, numpaths, corr, params, r, S0=1, n=3, dt=1/250, psic=1.5, gamma1=0.5, gamma2=0.5, verbose=True):
    
        UHG_ratio = 2.0432
        Pfizer_ratio = 27.2777
        MC_ratio = 8.9847
        
        # corr and params must be the same as when training
        # c and control_mean should be from the same training instance as train_AIS
        
        sobol_engine = torch.quasirandom.SobolEngine(min(2 * n * numsteps, 21201), True)
        u = sobol_engine.draw(numpaths)
        remaining_steps = max(2 * n * numsteps - 21201, 0)
        remaining_path = torch.rand((numpaths, remaining_steps))
        u = torch.cat((u, remaining_path), dim=1)
        
        uniform = paths
        paths = paths * (1 - 2 * 1e-6) + 1e-6
        paths = standard_normal.icdf(paths)
        
        paths, log_det = self.AIS.AIS.forward_and_log_det(paths)
        
        simulation = self.heston_model.multi_asset_path(u, uniform, params, r=r, corr=corr, dt=dt, psic=psic, gamma1=gamma1, gamma2=gamma2, verbose=verbose)
        payoff = self.LSL.evaluate_payoff(simulation, self.r)
        
        final_worst_stock_idx = torch.min(simulation[:,:,-1,0], dim=1)[1].reshape(-1,1) # This is already in log price
        worst_payout = torch.where(final_worst_stock_idx == 0,
                                    UHG_ratio,
                                    torch.where(final_worst_stock_idx == 1,
                                                Pfizer_ratio,
                                                MC_ratio))**-1
        breached = (simulation[:,:,:,0] < np.log(0.59)).any(2).any(1).reshape(-1,1)
        control = torch.where(breached,
                              worst_payout,
                              1)
        
        cv = payoff + c * (control - control_mean)
        weights = torch.exp(-log_det)
        mean = torch.mean(cv * weights)
        
        return mean
    