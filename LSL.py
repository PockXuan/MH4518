import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from heston_model import HestonModel
from tqdm import tqdm
import os

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
standard_normal = torch.distributions.Normal(torch.tensor([0]).to(device), torch.tensor([1]).to(device))
eps = torch.finfo(torch.float64).eps
cwd = os.getcwd()

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
        d = F.tanh(self.output(d))
                
        return d

class LSL():
    
    # Stands for Longstaff-Schwartz Lookback
    
    def __init__(self, model, time_elapsed, r):
                
        self.dt = 1/250
        self.model = model
        self.current_date = time_elapsed
        self.calling_dates = list(filter(lambda x:x>0, [i - time_elapsed for i in [129, 189, 254]]))
        self.payoff_dates = list(filter(lambda x:x>0, [i - time_elapsed for i in [70, 132, 193, 257, 321]]))
        
        if time_elapsed < 129:
            self.LSarray = [(lambda model: (model, optim.Adam(model.parameters(), lr=0.001)))(LSUnit()) for _ in range(3)]
        elif time_elapsed < 189:
            self.LSarray = [(lambda model: (model, optim.Adam(model.parameters(), lr=0.001)))(LSUnit()) for _ in range(2)]
        elif time_elapsed < 254:
            self.LSarray = [(lambda model: (model, optim.Adam(model.parameters(), lr=0.001)))(LSUnit()) for _ in range(1)]
        else:
            self.LSarray = []
            print("No future calls available. Product is now European.")
            
    def train(self, r, corr, params, num_paths=1000, max_epochs=200):
    
        UHG_ratio = 2.0432
        Pfizer_ratio = 27.2777
        MC_ratio = 8.9847
        
        # Simulate prices first
        end_date = 317
        num_steps = end_date - self.current_date
        sobol_engine = torch.quasirandom.SobolEngine(min(6 * num_steps, 21201), True)
        
        # Discounted coupon payouts for each model at the corresponding call date
        payout = -torch.exp(-r * self.dt * torch.tensor(self.payoff_dates)) * 0.1025 / 4
        payout = torch.cumsum(payout, dim=0)
        payout = torch.cat((payout[:-2], (payout[-2]+payout[-1]).reshape(1)))
        
        if self.current_date < 129:
            payout = payout
        elif self.current_date < 189:
            payout = payout[1:]
        elif self.current_date < 254:
            payout = payout[2:]
        else:
            print("No future calls available. Product is now European.")
            return
        
        with tqdm(total=max_epochs, desc="Training Progress") as pbar:
            for epoch in range(max_epochs):
                        
                uni = sobol_engine.draw(num_paths)
                remaining_steps = max(6 * num_steps - 21201, 0)
                remaining_path = torch.rand((num_paths, remaining_steps))
                uni = torch.cat((uni, remaining_path), dim=1).reshape(num_paths, 3, num_steps, 2)
                    
                batch_worst_loss = -torch.inf
                batch_best_loss = torch.inf
                    
                u = uni * (1 - 2 * 1e-6) + 1e-6
                paths = self.model.multi_asset_path(standard_normal.icdf(u), uni=uni, params=params, r=r, corr=corr, verbose=False)
                
                # Paths is in log price
                breached = paths[:,:,:,0] <= paths[:,:,0,0].unsqueeze(-1) + np.log(0.59)
                final = paths[:,:,-1,:]
                
                # Discounted final redemption amount
                discounted_final_redemption = torch.where(breached.any((1,2)),
                                                         torch.where(final[:,0,0] < torch.minimum(final[:,1,0], final[:,2,0]),
                                                                     1 - 1/ UHG_ratio,
                                                                     torch.where(final[:,1,0] < final[:,2,0],
                                                                                 1 - 1 / Pfizer_ratio,
                                                                                 1 - 1 / MC_ratio)),
                                                         0) * np.exp(-r * num_steps * self.dt)
                
                # Input to each model
                breached_at_call_date = torch.cummax(breached,2)[0][:,:,self.calling_dates].any(dim=1).unsqueeze(-1)
                stock_at_call_date = paths[:,:,self.calling_dates,:].transpose(1,2).flatten(-2,-1)
                unit_input = torch.cat((stock_at_call_date, breached_at_call_date), dim=2) # Index is (path #, unit #, inputs)
                
                # Total payoffs for each path for each model
                payoff = payout.reshape(1,-1).tile((num_paths,1))
                payoff[:,-1] = payoff[:,-1] + discounted_final_redemption
                next_payoff = payoff[:,-1]
                
                for i in range(-1, -1-len(self.LSarray), -1):
                    
                    unit, optim = self.LSarray[i]
                    pred_payoff = unit(unit_input[:,i,:])
                    loss = torch.mean((pred_payoff - next_payoff)**2)
                    
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    
                    next_payoff = torch.clamp(next_payoff, payoff[:,i-1])
                        
                    batch_worst_loss = max(batch_worst_loss, loss.item())
                    batch_best_loss = min(batch_best_loss, loss.item())
                
                
                pbar.set_postfix({'Worst loss': batch_worst_loss, 
                                  'Best Loss': batch_best_loss})
                pbar.update(1)  

    def evaluate_payoff(self, paths, r):
    
        UHG_ratio = 2.0432
        Pfizer_ratio = 27.2777
        MC_ratio = 8.9847
                
        payout = -torch.exp(-r * self.dt * torch.tensor(self.payoff_dates)) * 0.1025 / 4
        payout = torch.cumsum(payout, dim=0)
        payout = torch.cat((payout[:-2], payout[-1].reshape(1)))

        if self.current_date < 129:
            payout = payout
        elif self.current_date < 189:
            payout = payout[1:]
        elif self.current_date < 254:
            payout = payout[2:]
        else:
            payout = payout[-1]

        # Paths is in log price
        breached = paths[:,:,:,0] <= (paths[:,:,0,0].unsqueeze(-1) + np.log(0.59))
        final = paths[:,:,-1,:]

        # Input to each model
        breached_at_call_date = torch.cummax(breached,2)[0][:,:,self.calling_dates].any(dim=1).unsqueeze(-1)
        stock_at_call_date = paths[:,:,self.calling_dates,:].transpose(1,2).flatten(-2,-1)
        unit_input = torch.cat((stock_at_call_date, breached_at_call_date), dim=2) # Index is (path #, unit #, inputs)
        
        # Discounted final redemption amount
        worst_stock_idx = torch.min(final[:,:,0], dim=1)[1]
        worst_payout = torch.where(worst_stock_idx==0,
                                   1 / UHG_ratio * final[:,0,0],
                                   torch.where(worst_stock_idx==1,
                                               1 / Pfizer_ratio * final[:,1,0],
                                               1 / MC_ratio * final[:,2,0]))
        discounted_final_redemption = -worst_payout * np.exp(-r * 317 * self.dt)
        payoff = payout.reshape(1,-1).tile((paths.shape[0],1))
        payoff[:,-1] = payoff[:,-1] + discounted_final_redemption
        final_payoff = payoff[:,-1]
        
        for i in range(-1, -1-len(self.LSarray), -1):
            final_payoff = torch.where(self.LSarray[i][0](unit_input[:,i,:]).flatten() < payoff[:,i-1],
                                       payoff[:,i-1],
                                       final_payoff)
            
        # Don't forget that this payoff is solely from the issuer POV. The payout we want must be from investor POV.
            
        return -final_payoff
        


if __name__ == "__main__":
    
    model = LSL(0, 0.04)
        
    params = torch.tensor([[0.0041],
                            [0.0055],
                            [-0.2940],
                            [5.0],
                            [0.30470]], dtype=torch.float64).to(device).reshape(1,-1).tile(3,1)

    model.train(0.04, torch.eye(6), params, max_epochs=10)
    # for i in range(len(model.LSarray)):
    #     torch.save(model.LSarray[i][0].state_dict(), os.getcwd() + f'/unit_{i+1}.pth')
    uni = torch.rand((1000, 3, 317, 2))
    paths = model.model.multi_asset_path(standard_normal.icdf(uni), uni, params, 0.04, torch.eye(6))
    print(paths.shape)
    print(model.evaluate_payoff(paths, 0.04))
            
        