import torch
import torch.optim as optim
import normflows as nf
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from heston_model import HestonModel
from LSL import LSL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
standard_normal = torch.distributions.Normal(torch.tensor([0]).to(device), torch.tensor([1]).to(device))
eps = torch.finfo(torch.float).eps

class ClippedMaskedAffineFlow(nf.flows.Flow):
    
    # A modification to the RealNVP layer that provides stability in high dimensions
    # Details given in https://arxiv.org/pdf/2402.16408v1

    def __init__(self, b, t=None, s=None, a_pos=0.1, a_neg=2):
        
        super().__init__()
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer("b", self.b_cpu)

        if s is None:
            self.s = torch.zeros_like
        else:
            self.add_module("s", s)

        if t is None:
            self.t = torch.zeros_like
        else:
            self.add_module("t", t)
        
        self.a_pos = nn.Parameter(torch.tensor([a_pos], dtype=torch.float), requires_grad=True)
        self.a_neg = nn.Parameter(torch.tensor([a_neg], dtype=torch.float), requires_grad=True)

    def forward(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        scale = 2 / torch.pi * torch.where(scale >= 0, 
                                           self.a_pos * torch.arctan(scale / self.a_pos), 
                                           self.a_neg * torch.arctan(scale / self.a_neg)) # Thresholding as proposed in the paper
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det

    def inverse(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        scale = 2 / torch.pi * torch.where(scale >= 0, 
                                           self.a_pos * torch.arctan(scale / self.a_pos), 
                                           self.a_neg * torch.arctan(scale / self.a_neg)) # Thresholding as proposed in the paper
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det

class LOFTlayer(nf.flows.Flow):
    
    # Log soft extension layer, also proposed in https://arxiv.org/pdf/2402.16408v1

    def __init__(self, tau=100):
        
        super().__init__()
        self.tau = nn.Parameter(torch.tensor([tau], dtype=torch.float), requires_grad=True)

    def forward(self, z):
        log_det = torch.log(torch.clamp(torch.abs(z) - self.tau, 0) + 1)
        z_ = torch.sign(z) * (log_det + torch.clamp(torch.abs(z), max=self.tau))
        return z_, -log_det

    def inverse(self, z):
        a = torch.clamp(torch.abs(z) - self.tau, 0)
        z_ = torch.sign(z) * (torch.exp(a) - 1 + torch.clamp(torch.abs(z), max=self.tau))
        log_det = -torch.log(a + 1)
        return z_, log_det

class AIS():
    
    def __init__(self, num_timesteps, S0, r, LSLnetwork=None):
        
        # The model uses a modified Masked Affine Flow (MAF) layer that thresholds outputs to prevent
        # extremely high magnitudes. This allows it to be stable even at high dimensions and multiple
        # layers.
        
        # Simulate business daily for T timesteps
        
        self.device = device
        self.T = num_timesteps
        self.dt = 1 / 250
        self.latent_size = 6 * self.T
        
        flows = []
        b = torch.zeros(self.latent_size)
        b[::2] = 1
        for i in range(64):
            s = nf.nets.MLP([self.latent_size, 2 * self.latent_size, self.latent_size], init_zeros=True)
            t = nf.nets.MLP([self.latent_size, 2 * self.latent_size, self.latent_size], init_zeros=True)
            if i % 2 == 0:
                flows += [ClippedMaskedAffineFlow(b, t, s)]
            else:
                flows += [ClippedMaskedAffineFlow(1 - b, t, s)]
            
        self.AIS = nf.NormalizingFlow(nf.distributions.DiagGaussian(self.latent_size), flows).to(device)
        
        self.r = r
        
        self.AIS_optimizer = optim.Adam(self.AIS.parameters(), lr=1e-10, weight_decay=1e-5)
        self.heston_model = HestonModel()
        if LSLnetwork is not None:
            self.LSL = LSLnetwork
        else:
            self.LSL = LSL(1, 0.04)

    def train(self, params, corr, num_paths=254, max_epochs=4000):
        
        # We also implement control variates using the scheme detailed in Shyamsundar P, et al.; https://scipost.org/SciPostPhysCodeb.28/pdf
        # For the control variate, we check if the barrier is hit and if a stock closes below initial value.
        # Basically the same product minus callable and coupon. Coupon anyway will just be a constant difference
        
        sobol_engine = torch.quasirandom.SobolEngine(min(self.latent_size, 21201), True)
        remaining_steps = max(self.latent_size-21201, 0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.AIS_optimizer, 'min', 1.1**-1, 20)
    
        UHG_ratio = 2.0432
        Pfizer_ratio = 27.2777
        MC_ratio = 8.9847
        
        # Moving average approximation to control variate elements
        payoff_mean = torch.inf 
        control_mean = torch.inf 
        cov = torch.inf
        control_var = torch.inf
        with tqdm(total=max_epochs, desc="Training Progress") as pbar:
            for epoch in range(max_epochs):
                        
                steps = sobol_engine.draw(num_paths).reshape((num_paths, 3, -1, 2))
                remaining = torch.randn((num_paths, 3, remaining_steps, 2))
                steps = torch.cat((steps, remaining), dim=2)
                steps = steps * (1 - 2 * 1e-6) + 1e-6
                normal_steps = standard_normal.icdf(steps).reshape((-1, self.latent_size))
                
                # Since payoff can be negative, we instead minimise log(payoff**2 f(x) / g(x)), which also minimises the variance,
                # but the square term allows for a unique solution
                
                log_prob_latent = self.AIS.q0.log_prob(normal_steps)
                log_prob_flow = self.AIS.log_prob(normal_steps)
                log_weights = log_prob_latent - log_prob_flow
                
                normal_steps = normal_steps.reshape(num_paths, 3, -1, 2)
                paths = self.heston_model.multi_asset_path(normal_steps, steps, params, self.r, corr, dt=self.dt, verbose=False)
                payoff = self.LSL.evaluate_payoff(paths, self.r)
                sample_payoff_mean = torch.mean(payoff)
                
                final_worst_stock_idx = torch.min(paths[:,:,-1,0], dim=1)[1].reshape(-1,1) # This is already in log price
                worst_payout = torch.where(final_worst_stock_idx == 0,
                                           UHG_ratio,
                                           torch.where(final_worst_stock_idx == 1,
                                                       Pfizer_ratio,
                                                       MC_ratio))**-1
                breached = (paths[:,:,:,0] < np.log(0.59)).any(2).any(1).reshape(-1,1)
                control = torch.where(breached,
                                      worst_payout,
                                      1)
                sample_control_mean = torch.mean(control)
                
                if epoch == 0:                    
                    payoff_mean = sample_payoff_mean
                    control_mean = sample_control_mean
                    
                    sample_cov = torch.sum((control - control_mean) * (payoff - payoff_mean)) / (num_paths - 1)
                    sample_control_var = torch.sum((control - control_mean)**2) / (num_paths - 1)
                    
                    cov = sample_cov
                    control_var = sample_control_var
                else:
                    sample_cov = torch.sum((control - control_mean) * (payoff - payoff_mean)) / (num_paths - 1)
                    sample_control_var = torch.sum((control - control_mean)**2) / (num_paths - 1)
                
                    payoff_mean = (2 * payoff_mean + sample_payoff_mean) / 3
                    control_mean = (2 * control_mean + sample_control_mean) / 3
                    cov = (2 * cov + sample_cov) / 3
                    control_var = (2 * control_var + sample_control_var) / 3
                    
                c = -cov / control_var
                cv = payoff + c * (control - control_mean)
                
                loss = 2 * torch.log(torch.abs(cv)) + log_weights
                loss = torch.mean(loss) # This may be negative
                
                self.AIS_optimizer.zero_grad()
                loss.backward()
                self.AIS_optimizer.step()
                
                scheduler.step(loss)
                
                pbar.set_postfix({'Loss': loss.item()})
                pbar.update(1)
        
        return c, control_mean
          
    
if __name__ == "__main__":
    model = AIS(317, torch.tensor([20, 347, 2000]), 0.04)
        
    # [v0,theta,rho,kappa,sigma]
    params = torch.tensor([[1.0],
                            [0.0055],
                            [-0.2940],
                            [0.2],
                            [1.0]], dtype=torch.float64).to(device).reshape(1,-1).tile(3,1)
    
    c, control_mean = model.train(params, torch.eye(6), max_epochs=10)
    torch.save(model.AIS.state_dict(), f"AIS.pth")