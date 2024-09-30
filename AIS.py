import torch
import torch.optim as optim
import normflows as nf
import numpy as np
from tqdm import tqdm
from heston_model import HestonModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
standard_normal = torch.distributions.Normal(torch.tensor([0]).to(device), torch.tensor([1]).to(device))

class AIS():
    
    def __init__(self, num_layers=5):
        
        # The model uses Masked Autoregressive Flow (MAF) layers, which has a fast forward pass and can model complex
        # probability densities. It suffers from a slow inverse pass, but for our purposes that is not necessary. This
        # flow method can be found at https://arxiv.org/abs/1705.07057
        
        self.device = device
        self.to(self.device)
        
        flows = []
        for _ in range(num_layers):
            flows.append(nf.flows.MAF(param_dim=6))
        self.AIS = nf.NormalizingFlow(nf.distributions.DiagGaussian(6), flows) # 6 dimensions for 3 assets
        
        self.market = {
            'K': None,
            'T': None,
            'S0': None,
            'r': None
        }
        
        self.payoff = None
        self.AIS_optimizer = optim.Adam(self.AIS.params(), lr=1e-3, weight_decay=1e-5)
        self.sobol_engine = None
        self.heston_model = HestonModel()
        
    def brownian_motion(self, num_paths, num_steps):
        
        # Here we generate only a 6D Sobol sequence for the initial step, and allow the remaining timesteps to be generated by the AIS
        cov = torch.tensor([[1, self.heston_model.params['rho']], [self.heston_model.params['rho'], 1]], device=self.device)
        L = torch.linalg.cholesky(cov)
        
        # Quasi-random sampling of timesteps, which are mapped to an importance distribution given by the AIS
        sobol_sequences = self.sobol_engine.draw(num_paths).reshape(num_paths,3,1,2) # Indices are (path #, asset #, timestep #, stock or vol)
        base_sequence = standard_normal.icdf(sobol_sequences)
        steps = self.AIS.q0.sample(num_paths * num_steps).reshape(num_paths, 3, -1, 2)
        full_sequence = torch.cat(base_sequence, steps, dim=2)
        
        new_sequence, log_det_jacobian = self.AIS.forward_and_log_det(full_sequence) # Path weight is the magnitude of the determinant of the Jacobian
        weights = torch.exp(log_det_jacobian)
        weights = torch.prod(torch.prod(weights, 3), 2) # This is a 2D tensor with indices (path #, asset #)
        
        # Mapping samples to the required correlated distribution
        new_sequence = torch.permute(new_sequence, (0,2,1,3)).flatten(0,2)
        bm = torch.permute(L @ new_sequence)
        new_sequence = new_sequence.reshape(num_paths, 3, -1, 2)
        
        return bm, weights
    
    def variates(self, dt, num_paths, num_steps):
        
        # We use the final positions of the Heston paths as the control variates to the payoffs, i.e. X = payoff(S_T), Y = S_T
        # Other controls are possible, but will have to be modified in code. This implementation is not flexible.
        # The paths are simulated twice, first to estimate the control variate coefficient then to do the actual thingy
        assert self.payoff is not None
        
        
        # Control variates scheme according to Shyamsundar P, et al.; https://scipost.org/SciPostPhysCodeb.28/pdf
        bm1, weights1 = self.brownian_motion(dt, 1000, 1000) # No need so many paths for this estimate, but this can be fine-tuned later
        Y1 = self.heston_model.multi_asset_path(bm1, dt)
        X1 = self.payoff(Y1)
        Y1 = Y1[:,:,-1,0] # Discard all path data
        X1 = X1 * weights1
        Y1 = Y1 * weights1
        X_mean = torch.mean(X1)
        Y_mean = torch.mean(Y1)
        
        X1 = X1 - X_mean
        Y1 = Y1 - Y_mean
        cov = X1.T @ Y1
        var = Y1.T @ Y1
        c = (-cov/var)
        
        # Second actual round now. Yay! :D
        bm2, weights2 = self.brownian_motion(dt, num_paths, num_steps)
        bm2 = torch.cat((bm2, -bm2), 0) # Antithetic variate
        Y2 = self.heston_model.multi_asset_path(bm2, dt)
        X2 = self.payoff(Y2)
        Y2 = Y2[:,:,-1,0] # Discard all path data
        X2 = X2 * weights2
        Y2 = Y2 * weights2
        
        return X2 + c * (Y2 - Y_mean)
        
    def set_payoff(self, payoff_fn):
        self.payoff = payoff_fn    
    
    def train(self, T, max_paths=10000, patience=10, tol=1e-2, max_epochs=1000000):
        
        assert self.payoff is not None
        
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        max_timesteps = int(T*365) # Smallest timestep is daily
        num_paths = min(1000, max_paths)
        num_steps = min(1000, max_timesteps)
        dt = T/num_steps
        self.sobol_engine = torch.quasirandom.SobolEngine(6 * num_steps, True)
        patience_count = 0
        
        with tqdm(total=max_epochs, desc="Training Progress") as pbar:
            for epoch in range(max_epochs):
                variates = self.variates(dt, num_paths, num_steps, self.heston_model.market['asset_cross_covariance'])
                mean = torch.mean(variates)
                loss = torch.sum((variates - mean)**2) / (2 * num_paths - 1) # AIS loss is variance
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
                pbar.set_postfix({'Loss': loss.item(), 
                  'Num Paths': num_paths, 
                  'Num Timesteps': num_steps})
                pbar.update(1)  
                
                # Adaptive training logic
                if epoch == 0:
                    best_loss = loss.item()
                    continue
                
                relative_improvement = (best_loss - loss.item()) / best_loss
                
                if relative_improvement < tol: 
                    patience_count += 1
                else: 
                    patience_count = 0
                
                if patience_count == patience:
                    
                    if num_steps == max_timesteps and num_paths == max_paths:
                            pbar.set_description("Model training stopped")
                    
                    patience_count = 0
                    num_paths = min(1.5 * num_paths, max_paths)
                    num_steps = min(2 * num_steps, max_timesteps)
                    dt = T / num_steps
                    self.sobol_engine = torch.quasirandom.SobolEngine(6 * num_steps, True)
                
                best_loss = min(best_loss, loss.item())
                
            pbar.close()