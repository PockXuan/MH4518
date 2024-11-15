import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from heston_model import HestonModel
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
        self.hidden1 = nn.Linear(4, 32)
        self.hidden2 = nn.Linear(32, 32) 
        self.hidden3 = nn.Linear(32, 4) 
        self.hidden4 = nn.Linear(4, 32) 
        self.hidden5 = nn.Linear(32, 32)  
        self.output = nn.Linear(32, 1)
    
    def forward(self, x):
        
        d = x.clone()
        d = F.mish(self.hidden1(d))
        d = F.mish(self.hidden2(d))
        d = F.mish(self.hidden3(d))
        d = F.mish(self.hidden4(d))
        d = F.mish(self.hidden5(d))
        d = F.softplus(self.output(d))
                
        return d


class CallablePayoff():
    
    # Stands for Longstaff-Schwartz Lookback
    
    def __init__(self, time_elapsed, r, S0):
                
        self.dt = 1/250
        self.r = r
        self.current_date = time_elapsed
        self.calling_dates = [129, 189, 254]
        self.payoff_dates = [70, 132, 193, 257, 321]
        self.final_fixing_date = 316
        self.S0 = torch.log(torch.tensor(S0)) # In order of UNH, PFE, MRK

        if time_elapsed <= 254:
            # Model inputs are discounted to initial fixing date and scaled to unity. 
            # Their outputs are already discounted, so we compare everything discounted
            # Model prediction also excludes coupon
            self.LSarray = []
            for _ in self.calling_dates:
                model = LSUnit()
                optimiser = optim.Adam(model.parameters())
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', 1.1**-1, 20)
                self.LSarray = [(model, optimiser, scheduler)] + self.LSarray
        else:
            print("No future calls available. Product is now European.")
    
    def evaluate_payoff(self, path, callable=True):

        # Path index of log payoff is in (path #, asset #, timestep #)

        path = torch.tensor(path)
        num_paths, _, num_timesteps = path.shape
        # path = path[torch.isfinite(path).all(dim=2).all(dim=1).reshape(-1,1,1).tile(1,3,num_timesteps)].reshape(-1,3,num_timesteps)
        # num_paths, _, num_timesteps = path.shape

        log_stock_change = path - self.S0.reshape(1,3,1).tile(num_paths, 1, num_timesteps)
        barrier_hit = torch.cummax(log_stock_change < np.log(0.59), dim=2)[0].any(dim=1)
        closed_below_initial = (log_stock_change[:,:,-1] < 0).any(dim=1).reshape(num_paths,1)
        final_smallest_log_stock_price_change, final_smallest_log_stock_price_idx = torch.min(log_stock_change[:,:,-1], dim=1, keepdim=True)
        input = torch.cat((path, barrier_hit.reshape(num_paths,1,num_timesteps)), dim=1)
        
        worst_case_payoff = torch.exp(final_smallest_log_stock_price_change) * 1000 * np.exp(-self.r * self.final_fixing_date / 250)
        
        payoff_at_maturity = torch.where(
            torch.logical_and(barrier_hit[:,-1].reshape(-1,1), closed_below_initial),
            worst_case_payoff,
            1000
        ) * np.exp(-self.r * self.final_fixing_date / 250)
        
        if callable and self.current_date < 254:
            return self.callable_payoff(input, payoff_at_maturity)
        
        coupon_payment = np.exp(-self.r * self.payoff_dates[-1] / 250) + np.exp(-self.r * self.payoff_dates[-2] / 250)
        if self.current_date <= self.payoff_dates[-3]: coupon_payment += np.exp(-self.r * self.payoff_dates[-3] / 250)
        if self.current_date <= self.payoff_dates[-4]: coupon_payment += np.exp(-self.r * self.payoff_dates[-4] / 250)
        if self.current_date <= self.payoff_dates[-5]: coupon_payment += np.exp(-self.r * self.payoff_dates[-5] / 250)
        coupon_payment *= 0.1025 * 1000 / 4
        
        return payoff_at_maturity + coupon_payment
    
    def callable_payoff(self, input, payoff_at_maturity):
        
        input_at_latest_call = input[:,:,self.calling_dates[-1] - self.current_date - 1]
        coupon_payment_after_latest_call = 0.1025 / 4 * (np.exp(-self.r * self.payoff_dates[-1] / 250) + np.exp(-self.r * self.payoff_dates[-2] / 250))
        call_revenue = 1000 * np.exp(-self.r * self.calling_dates[-1] / 250) # Value of exercising
        payoff_at_latest_call = torch.where(
            self.LSarray[-1][0](input_at_latest_call) * 1000 + coupon_payment_after_latest_call < call_revenue, # Choose lower of exercise and not
            payoff_at_maturity,
            call_revenue
        )
        
        if self.current_date > 193:
            return payoff_at_latest_call
        elif self.current_date > 189:
            return payoff_at_latest_call + 0.1025 / 4 * np.exp(-self.r * 193 / 250)

        input_at_second_latest_call = input[:,:,self.calling_dates[-2] - self.current_date]
        coupon_payment_after_second_latest_call = 0.1025 / 4 * np.exp(-self.r * self.payoff_dates[-3] /250)
        unexercised_payoff_at_second_latest_call = payoff_at_latest_call + coupon_payment_after_second_latest_call
        call_revenue = 1000 * np.exp(-self.r * self.calling_dates[-2] / 250)
        payoff_at_second_latest_call = torch.where(
            self.LSarray[-2][0](input_at_second_latest_call) * 1000 + coupon_payment_after_second_latest_call + coupon_payment_after_latest_call < call_revenue, # If expected exercised payoff is lower, investor will choose to not exercise
            unexercised_payoff_at_second_latest_call,
            call_revenue
        )

        if self.current_date > 132:
            return payoff_at_second_latest_call
        elif self.current_date > 129:
            return payoff_at_second_latest_call + 0.1025 / 4 * np.exp(-self.r * 132 / 250)

        input_at_earliest_call = input[:,:,self.calling_dates[-3] - self.current_date]
        coupon_payment_after_earliest_call = 0.1025 / 4 * np.exp(-self.r * self.payoff_dates[-4] / 250)
        unexercised_payoff_at_earliest_call = payoff_at_second_latest_call + coupon_payment_after_earliest_call
        call_revenue = 1000 * np.exp(-self.r * self.calling_dates[-3] / 250)
        payoff_at_earliest_call = torch.where(
            self.LSarray[-3][0](input_at_earliest_call) * 1000 + coupon_payment_after_earliest_call + coupon_payment_after_second_latest_call + coupon_payment_after_latest_call < call_revenue, # If expected exercised payoff is lower, investor will choose to not exercise
            unexercised_payoff_at_earliest_call,
            call_revenue
        )

        if self.current_date > 70:
            return payoff_at_earliest_call
        return payoff_at_earliest_call + 0.1025 / 4 * np.exp(-self.r * 70 / 250)

    def minimise_over_path(self, paths):

        # Path index of log payoff is in (path #, asset #, timestep #)

        paths = torch.tensor(paths)
        num_paths, _, num_timesteps = paths.shape
        paths = paths[torch.isfinite(paths).all(dim=2).all(dim=1).reshape(-1,1,1).tile(1,3,num_timesteps)].reshape(-1,3,num_timesteps)
        num_paths, _, num_timesteps = paths.shape

        log_stock_change = paths - self.S0.reshape(1,3,1)
        barrier_hit = torch.cummax(log_stock_change < np.log(0.59), dim=2)[0].any(dim=1)
        closed_below_initial = (log_stock_change[:,:,-1] < 0).any(dim=1).reshape(num_paths,1)
        final_smallest_log_stock_price_change, final_smallest_log_stock_price_idx = torch.min(log_stock_change[:,:,-1], dim=1, keepdim=True)
        input = torch.cat((paths, barrier_hit.reshape(num_paths,1,num_timesteps)), dim=1)
        
        worst_case_payoff = torch.exp(final_smallest_log_stock_price_change) * 1000 * np.exp(-self.r * self.final_fixing_date / 250)
        
        payoff_at_maturity = torch.where(
            torch.logical_and(barrier_hit[:,-1], closed_below_initial),
            worst_case_payoff,
            1000
        ) * np.exp(-self.r * self.final_fixing_date / 250)

        losses = []
            
        latest_model, latest_optimiser, latest_scheduler = self.LSarray[-1]
        latest_coupon_payment = np.exp(-self.r * self.payoff_dates[-1] / 250) + np.exp(-self.r * self.payoff_dates[-2] / 250)
        next_payoff_with_coupon = payoff_at_maturity + latest_coupon_payment
        predicted_payoff = latest_model(input[:,:,self.calling_dates[-1] - self.current_date]) * 1000 * np.exp(-self.r * self.calling_dates[-1] / 250) + latest_coupon_payment
        
        latest_loss = (predicted_payoff - next_payoff_with_coupon)**2
        latest_loss = torch.nanmean(latest_loss)
        losses = [latest_loss.item()] + losses
        
        latest_optimiser.zero_grad()
        latest_loss.backward()
        latest_optimiser.step()
        latest_scheduler.step(latest_loss)
        
        total_payoff = torch.where(predicted_payoff < 1000 * np.exp(-self.r * self.calling_dates[-1] / 250),
                                   next_payoff_with_coupon,
                                   1000 * np.exp(-self.r * self.calling_dates[-1] / 250))
            
        if self.current_date > self.calling_dates[-2]:
            return losses
            
        second_latest_model, second_latest_optimiser, second_latest_scheduler = self.LSarray[-2]
        second_latest_coupon_payment = np.exp(-self.r * self.payoff_dates[-3] / 250)
        next_payoff_with_coupon = total_payoff + second_latest_coupon_payment
        predicted_payoff = second_latest_model(input[:,:,self.calling_dates[-2] - self.current_date]) * 1000 * np.exp(-self.r * self.calling_dates[-2] / 250) + second_latest_coupon_payment
        
        second_latest_loss = (predicted_payoff - next_payoff_with_coupon)**2
        second_latest_loss = torch.nanmean(second_latest_loss)
        losses = [second_latest_loss.item()] + losses
        
        second_latest_optimiser.zero_grad()
        second_latest_loss.backward()
        second_latest_optimiser.step()
        second_latest_scheduler.step(second_latest_loss)
        
        total_payoff = torch.where(predicted_payoff < 1000 * np.exp(-self.r * self.calling_dates[-2] / 250),
                                   next_payoff_with_coupon,
                                   1000 * np.exp(-self.r * self.calling_dates[-2] / 250))
            
        if self.current_date > self.calling_dates[-3]:
            return losses
            
        earliest_model, earliest_optimiser, earliest_scheduler = self.LSarray[-3]
        earliest_coupon_payment = np.exp(-self.r * self.payoff_dates[-4] / 250)
        next_payoff_with_coupon = total_payoff + earliest_coupon_payment
        predicted_payoff = earliest_model(input[:,:,self.calling_dates[-3] - self.current_date]) * 1000 * np.exp(-self.r * self.calling_dates[-3] / 250) + earliest_coupon_payment
        
        earliest_loss = (predicted_payoff - next_payoff_with_coupon)**2
        earliest_loss = torch.nanmean(earliest_loss)
        losses = [earliest_loss.item()] + losses
        
        earliest_optimiser.zero_grad()
        earliest_loss.backward()
        earliest_optimiser.step()
        earliest_scheduler.step(earliest_loss)
            
        return losses

if __name__ == '__main__':
    from process_data import GARCH
    from heston_model import HestonModel

    # model = GARCH()
    # model.fit()

    model = HestonModel()
    s0 = torch.tensor([480.22924805, 34.61804962, 107.69082642])
    # [v0,theta,rho,kappa,sigma]
    params = torch.tensor([[0.0041],
                            [0.0055],
                            [-0.2940],
                            [5.0],
                            [0.30470]], dtype=torch.float64).to(device).to(device).reshape(1,-1).tile(3,1)

    thing = CallablePayoff(0, 0.0419, [480.22924805, 34.61804962, 107.69082642])

    epochs = 400
    with tqdm(total=epochs, desc="Training Progress") as pbar:
        for epoch in range(epochs):

            path = model.multi_asset_path(torch.randn((254,3,317,2)), torch.rand((254,3,317,2)), params, 0.0419, torch.eye(6), s0)
            losses = thing.minimise_over_path(path[:,:,:,0])

            # path = model.forecast(317, 256)
            # path = path[(path>0).all(axis=2).all(axis=1)]
            # losses = thing.minimise_over_path(np.log(path))
            
            pbar.set_postfix({'Worst loss': max(losses), 
                                'Best Loss': min(losses)})
            pbar.update(1)

    s0 = torch.tensor([480.22924805, 34.61804962, 107.69082642])
    path = model.multi_asset_path(torch.randn((254,3,317,2)), torch.rand((254,3,317,2)), params, 0.04, torch.eye(6), s0)

    # import matplotlib.pyplot as plt
    # plt.plot(path[:,0,:,0].T)
    # plt.show()

    x = thing.evaluate_payoff(path[:,:,:,0], True)
    print(x)
    # path = model.forecast(317, 256)
    # path = path[(path>0).all(axis=2).all(axis=1)]
    # payoff = thing.evaluate_payoff(np.log(path), False)
    # print(payoff)
    # torch.set_printoptions(profile="full")
    # print(torch.cat((thing.evaluate_payoff(path[:,:,:,0], False), torch.log(torch.tensor(path[:,:,:,0][:,:,-1])) - torch.log(s0).reshape(1,3), (torch.log(torch.tensor(path[:,:,:,0])) < (np.log(0.59) + torch.log(s0).reshape(1,3,1))).any(dim=2).any(dim=1).reshape(-1,1)), dim=1))
    # torch.set_printoptions(profile="default") # reset
    # print((payoff-1000)/10)