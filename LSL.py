import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from heston_model import HestonModel
from tqdm import tqdm

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
standard_normal = torch.distributions.Normal(torch.tensor([0]).to(device), torch.tensor([1]).to(device))
eps = torch.finfo(torch.float64).eps

class LSUnit(nn.Module):
    
    def __init__(self):
        super(LSUnit, self).__init__()
        
        # Input is (asset, stock_price, vol, barrier_breached).flatten() tuple, where barrier_breached is boolean
        self.hidden1 = nn.Linear(7, 16)
        init.kaiming_normal_(self.hidden1.weight, mode='fan_in', nonlinearity='relu')
        self.hidden2 = nn.Linear(16, 16)
        init.kaiming_normal_(self.hidden2.weight, mode='fan_in', nonlinearity='relu')
        self.output = nn.Linear(16, 1)
    
    def forward(self, x):
        d = x.clone()
        d = F.mish(self.hidden1(d))
        d = F.mish(self.hidden2(d))
        d = F.relu(self.output(d))
        
        return d

class LSL():
    
    # Stands for Longstaff-Schwartz Lookback
    
    def __init__(self, T, r, dt=1/52):
        
        # Default timestep taken to be weekly
        self.dt = dt
        self.quarter = int(0.25 / dt) # Number of timesteps in a quarter
        self.T = int(T / dt / self.quarter) * self.quarter # Time to maturity must be in quarters
        self.num_quarters = int(self.T / self.quarter)
        self.r = r 
        
        self.LSarray = [(lambda model: (model, optim.Adam(model.parameters(), lr=0.0001)))(LSUnit()) \
                         for _ in range(self.num_quarters - 1)] # One for each quarter, except the final
        self.heston_model = HestonModel()
        
    def load_models(self):
        for i in range(self.num_quarters - 1):
            self.LSarray[i][0].load_state_dict(torch.load(f'LSArray/unit_{i+1}.pth', weights_only=True))
            
    def train(self, num_paths=1000, max_epochs=500):
    
        UHG_ratio = 2.0432
        Pfizer_ratio = 27.2777
        MC_ratio = 8.9847
        
        # Simulate prices first
        sobol_steps = int(6 * (21201 / 6))
        sobol_engine = torch.quasirandom.SobolEngine(min(sobol_steps, 6 * self.T), True)
        
        with tqdm(total=max_epochs, desc="Training Progress") as pbar:
            for epoch in range(max_epochs):
                    
                batch_worst_loss = -torch.inf
                batch_best_loss = torch.inf
                
                steps = sobol_engine.draw(num_paths).reshape(num_paths, 3, -1, 2)
                if self.T > sobol_steps:
                    steps = torch.cat((steps, torch.rand(num_paths, 3, self.T-sobol_steps, 2)), dim=2)
                steps = steps * (1 - 2 * 1e-6) + 1e-6
                paths = self.heston_model.multi_asset_path(standard_normal.icdf(steps), steps, self.dt)
                
                # Paths is in log price
                breached = paths[:,:,:,0] <= np.log(0.59)
                final = paths[:,:,-1,:]
                
                # Payoff for terminating just before each quarter except last
                payoff = [-0.1025 * 0.25 * (quarter + 1) for quarter in range(self.num_quarters - 1)]
                
                # Payoff at maturity
                maturity_payoff = torch.where(breached.any((1,2)),
                                            torch.where(final[:,0,0] < torch.minimum(final[:,1,0], final[:,2,0]),
                                                        1 - 1/ UHG_ratio,
                                                        torch.where(final[:,1,0] < final[:,2,0],
                                                                    1 - 1 / Pfizer_ratio,
                                                                    1 - 1 / MC_ratio)),
                                            0) - 0.1025 * 0.25 * self.num_quarters
                
                breached = torch.where(breached,1,0)
                quarter_breached = torch.cummax(breached,2)[0][:,:,self.quarter-1::self.quarter].any(dim=1).unsqueeze(-1)
                quarter_path = paths[:,:,self.quarter-1::self.quarter,:].transpose(1,2).flatten(-2,-1)
                input = torch.cat((quarter_path, quarter_breached), dim=2) # Index is (path #, quarter #, inputs)
                discount = np.exp(-self.r * self.dt)
                next_payoff = discount * maturity_payoff
                for i in range(self.num_quarters - 1):
                    
                    unit, optim = self.LSarray[-(i+1)]
                    pred_payoff = unit(input[:,-(i+1),:])
                    loss = torch.mean((pred_payoff - next_payoff)**2)
                    next_payoff = discount * torch.clamp(next_payoff, payoff[-(i+1)])
                    
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    
                    if (epoch + 1) % 2000:
                        for params in optim.param_groups:
                            params['lr'] /= 1.1
                        
                    batch_worst_loss = max(batch_worst_loss, loss.item())
                    batch_best_loss = min(batch_best_loss, loss.item())
                
                
                pbar.set_postfix({'Worst loss': batch_worst_loss, 
                                  'Best Loss': batch_best_loss})
                pbar.update(1)  

    def evaluate_payoff(self, path):
    
        UHG_ratio = 2.0432
        Pfizer_ratio = 27.2777
        MC_ratio = 8.9847
                
        breached = path[:,:,:,0] <= np.log(0.59)
        final = path[:,:,-1,:]
        quarter_breached = torch.cummax(torch.where(path[...,0] <= np.log(0.59),1,0),2)[0][:,:,self.quarter-1::self.quarter].any(1).unsqueeze(-1)
        quarter_path = path[:,:,self.quarter-1::self.quarter,:].transpose(1,2).flatten(-2,-1)
        quarter_payoff = [-0.1025 * 0.25 * (quarter + 1) for quarter in range(self.num_quarters - 1)]
        maturity_payoff = torch.where(breached.any((1,2)),
                                    torch.where(final[:,0,0] < torch.minimum(final[:,1,0], final[:,2,0]),
                                                1 - 1 / UHG_ratio,
                                                torch.where(final[:,1,0] < final[:,2,0],
                                                            1 - 1 / Pfizer_ratio,
                                                            1 - 1 / MC_ratio)),
                                    0) - 0.1025 * 0.25 * self.num_quarters
        input = torch.cat((quarter_path, quarter_breached), dim=2)
        
        payoff = maturity_payoff.reshape(-1,1)
        discount = np.exp(-self.r * self.dt)
        
        for i in range(input.shape[1] - 1):
            current = quarter_payoff[-(i+1)]
            pred_payoff = self.LSarray[-(i+1)][0](input[:,-(i+1),:])
            payoff = torch.where(current >= pred_payoff,
                                 current + torch.zeros_like(payoff),
                                 discount * payoff)
            
        return payoff
        


if __name__ == "__main__":
    
    model = LSL(1,0.04)
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

    model.train()
    for i in range(len(model.LSarray)):
        torch.save(model.LSarray[i][0].state_dict(), f"unit_{i+1}.pth")
            
        