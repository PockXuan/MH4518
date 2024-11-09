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
        self.hidden1 = nn.Linear(4, 16)
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


class CallablePayoff():
    
    # Stands for Longstaff-Schwartz Lookback
    
    def __init__(self, time_elapsed, r, S0):
                
        self.dt = 1/250
        self.r = r
        self.current_date = time_elapsed
        self.calling_dates = [129, 189, 254]
        self.payoff_dates = [70, 132, 193, 257, 321]
        self.final_fixing_date = 316
        self.S0 = torch.tensor(S0) # In order of UNH, PFE, MRK

        if time_elapsed <= 254:
            # Model inputs are discounted to initial fixing date. Their outputs are also discounted, so we compare everything discounted
            # Model prediction also excludes coupon
            self.LSarray = [(lambda model: (model, optim.Adam(model.parameters(), lr=0.001)))(LSUnit()) for _ in range(sum(i >= time_elapsed for i in self.calling_dates))]
        else:
            print("No future calls available. Product is now European.")
    
    def evaluate_payoff(self, path):

        # Path index of log payoff is in (path #, asset #, timestep #)
    
        UHG_ratio = 2.0432
        Pfizer_ratio = 27.2777
        MC_ratio = 8.9847

        log_stock_change = path - self.S0.reshape(1,3,1)
        barrier_hit = torch.cummax(path < torch.log(0.59), dim=2).any(dim=1)
        closed_below_initial = (log_stock_change < 0).any(dim=1)
        final_smallest_log_stock_price, final_smallest_log_stock_price_idx = torch.min(path[:,:,-1], dim=1)

        worst_case_payoff = torch.where(
            final_smallest_log_stock_price_idx==0,
            1000 / UHG_ratio,
            torch.where(
                final_smallest_log_stock_price_idx==1,
                1000 / Pfizer_ratio,
                1000 / MC_ratio
            )
        ) * final_smallest_log_stock_price

        payoff_at_maturity = torch.where(
            barrier_hit[:,-1] and closed_below_initial,
            worst_case_payoff,
            1000
        )

        input = torch.cat((path, barrier_hit.reshape(-1,1,1)), dim=1)
        input_at_latest_call = input[:,:,self.calling_dates[-1] - self.current_date]
        coupon_payment_after_latest_call = 0.1025 / 4 * (np.exp(self.payoff_dates[-1]) + np.exp(self.payoff_dates[-2]))
        unexercised_payoff_at_latest_call = payoff_at_maturity * np.exp(-self.r * self.final_fixing_date)
        payoff_at_latest_call = torch.where(
            self.LSarray[-1](input_at_latest_call) + coupon_payment_after_latest_call < path[:,:,self.calling_dates[-1]] * np.exp(self.calling_dates[-1]), # If expected exercised payoff is lower, investor will choose to not exercise
            unexercised_payoff_at_latest_call,
            1000 * np.exp(self.calling_dates[-1])
        )

        if self.current_date > 193:
            return payoff_at_latest_call
        elif self.current_date > 189:
            return payoff_at_latest_call + 0.1025 / 4 * np.exp(193)

        input_at_second_latest_call = input[:,:,self.calling_dates[-2] - self.current_date]
        coupon_payment_after_second_latest_call = 0.1025 / 4 * np.exp(self.payoff_dates[-3])
        unexercised_payoff_at_second_latest_call = payoff_at_latest_call + coupon_payment_after_second_latest_call
        payoff_at_second_latest_call = torch.where(
            self.LSarray[-2](input_at_second_latest_call) + coupon_payment_after_second_latest_call + coupon_payment_after_latest_call < path[:,:,self.calling_dates[-2]] * np.exp(self.calling_dates[-2]), # If expected exercised payoff is lower, investor will choose to not exercise
            unexercised_payoff_at_second_latest_call,
            1000 * np.exp(self.calling_dates[-2])
        )

        if self.current_date > 132:
            return payoff_at_second_latest_call
        elif self.current_date > 129:
            return payoff_at_second_latest_call + 0.1025 / 4 * np.exp(132)

        input_at_earliest_call = input[:,:,self.calling_dates[-3] - self.current_date]
        coupon_payment_after_earliest_call = 0.1025 / 4 * np.exp(self.payoff_dates[-4])
        unexercised_payoff_at_earliest_call = payoff_at_second_latest_call + coupon_payment_after_earliest_call
        payoff_at_earliest_call = torch.where(
            self.LSarray[-3](input_at_earliest_call) + coupon_payment_after_earliest_call + coupon_payment_after_second_latest_call + coupon_payment_after_latest_call < path[:,:,self.calling_dates[-3]] * np.exp(self.calling_dates[-3]), # If expected exercised payoff is lower, investor will choose to not exercise
            unexercised_payoff_at_earliest_call,
            1000 * np.exp(self.calling_dates[-3])
        )

        if self.current_date > 70:
            return payoff_at_earliest_call
        return payoff_at_earliest_call + 0.1025 / 4 * np.exp(70)

    def minimise_over_path(self, paths):

        # Path index of log payoff is in (path #, asset #, timestep #)

        paths = torch.tensor(paths)
    
        UHG_ratio = 2.0432
        Pfizer_ratio = 27.2777
        MC_ratio = 8.9847

        log_stock_change = paths - self.S0.reshape(1,3,1)
        barrier_hit = torch.cummax(paths < np.log(0.59), dim=2)[0].any(dim=1)
        closed_below_initial = (log_stock_change < 0).any(dim=1)
        final_smallest_log_stock_price, final_smallest_log_stock_price_idx = torch.min(paths[:,:,-1], dim=1)
        input = torch.cat((paths, barrier_hit.reshape(-1,1,1)), dim=1)

        worst_case_payoff = torch.where(
            final_smallest_log_stock_price_idx==0,
            1000 / UHG_ratio,
            torch.where(
                final_smallest_log_stock_price_idx==1,
                1000 / Pfizer_ratio,
                1000 / MC_ratio
            )
        ) * final_smallest_log_stock_price

        payoff_at_maturity = torch.where(
            barrier_hit[:,-1] and closed_below_initial,
            worst_case_payoff,
            1000
        )
        payoff_at_next_calling_date = payoff_at_maturity

        losses = []
        for model_set, date in zip(reversed(self.LSarray), reversed(self.calling_dates)):
            model, optimiser = model_set
            predicted_payoff = model(input[:,:,date - self.current_date])
            loss = (predicted_payoff - payoff_at_next_calling_date)**2
            loss = torch.mean(loss)
            
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            losses = [loss] + losses

        return losses


from process_data import GARCH

# model = GARCH()
# model.fit()

thing = CallablePayoff(0, 0.419, [480.22924805, 34.61804962, 107.69082642])

for epoch in range(2):
    # path = model.forecast(250, 256)
    # losses = thing.minimise_over_path(path)
    losses = thing.minimise_over_path([[[0],[0],[0]]])
    print(losses)
