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
    
    def __init__(self):
        
        # The model uses a modified Masked Affine Flow (MAF) layer that thresholds outputs to prevent
        # extremely high magnitudes. This allows it to be stable even at high dimensions and multiple
        # layers.
        
        # Simulate weekly for 1 year, i.e. 6 * 52 dimensional input
        
        self.device = device
        self.T = 1
        self.dt = 1 / 52
        self.latent_size = 6 * 52 * self.T
        
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
        
        self.market = {
            'S0_1': None,
            'S0_2': None,
            'S0_3': None,
            'r': None
        }
        
        self.payoff = None
        self.AIS_optimizer = optim.Adam(self.AIS.parameters(), lr=1e-10, weight_decay=1e-5)
        self.sobol_engine = None
        self.heston_model = HestonModel()
        self.LSL = LSL(1, 0.04)

    def train(self, num_paths=254, max_epochs=4000):
        
        # Simulate prices first
        sobol_steps = int(6 * (21201 / 6))
        sobol_engine = torch.quasirandom.SobolEngine(min(sobol_steps, self.latent_size), True)
        num_steps = int(self.T // self.dt) + 1
        losses = []
        with tqdm(total=max_epochs, desc="Training Progress") as pbar:
            for epoch in range(max_epochs):
                        
                steps = sobol_engine.draw(num_paths).reshape((num_paths, 3, -1, 2))
                if num_steps > sobol_steps:
                    steps = torch.cat((steps, torch.rand(num_paths, 3, num_steps-sobol_steps, 2)), dim=2)
                steps = steps * (1 - 2 * 1e-6) + 1e-6
                normal_steps = standard_normal.icdf(steps).reshape((-1, self.latent_size))
                
                # This method assumes h is positive but this idiot can actually lose money with the best outcome...
                
                # log_prob_latent = self.AIS.q0.log_prob(normal_steps)
                # bm, log_det_jacobian = self.AIS.forward_and_log_det(normal_steps)
                # log_prob_flow = self.AIS.q0.log_prob(bm)
                # bm = bm.reshape(steps.shape)
                # log_weights = log_prob_flow - log_prob_latent - log_det_jacobian
                
                log_prob_latent = self.AIS.q0.log_prob(normal_steps)
                log_prob_flow = self.AIS.log_prob(normal_steps)
                log_weights = log_prob_latent - log_prob_flow
                
                normal_steps = normal_steps.reshape(num_paths, 3, num_steps, 2)
                paths = self.heston_model.multi_asset_path(normal_steps, steps, self.dt)
                payoff = self.LSL.evaluate_payoff(paths)
                
                loss = 2 * torch.log(torch.abs(payoff)) + log_weights
                loss = torch.mean(loss**2)
                losses.append(loss.detach().numpy())
                
                self.AIS_optimizer.zero_grad()
                loss.backward()
                self.AIS_optimizer.step()
                
                pbar.set_postfix({'Loss': loss.item()})
                pbar.update(1)  
                
        return losses
          
    
if __name__ == "__main__":
    model = AIS()
    market = {
                'S0': 1,
                'r': 0.04,
                'asset_cross_correlation': 0.2
            }
    # For testing
    params = {
        'v0': torch.tensor([0.08], dtype=torch.float64).to(device),
        'theta': torch.tensor([0.1], dtype=torch.float64).to(device),
        'rho': torch.tensor([-0.8], dtype=torch.float64).to(device),
        'kappa': torch.tensor([3], dtype=torch.float64).to(device),
        'sigma': torch.tensor([0.25], dtype=torch.float64).to(device)
    }
    model.heston_model.set_market(market)
    model.heston_model.params = params

    # Initial fixing at 23/08/23. Simulate weekly for 1 year
    losses = model.train()
    torch.save(model.AIS.state_dict(), f"AIS.pth")

    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.show()