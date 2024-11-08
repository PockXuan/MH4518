import pandas as pd
import torch
import numpy as np
from pandas.tseries.offsets import BDay
from heston_model import HestonModel
from AIS import AIS
from corr_finder import construct_corr
import os

standard_normal = torch.distributions.Normal(torch.tensor([0]), torch.tensor([1]))

# For MRK
df = pd.read_csv(os.getcwd() + "/MH4518/datasets/MRK_calls.csv")
df = df[['lastTradeDate', 'strike', 'lastPrice', 'expiry']]

df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate']).dt.tz_localize(None).dt.date
df['expiry'] = pd.to_datetime(df['expiry']).dt.tz_localize(None).dt.date
df['business_days_to_expiry'] = df.apply(
    lambda row: len(pd.date_range(row['lastTradeDate'], row['expiry'], freq=BDay())), axis=1
)
df['business_days_to_expiry'] = df['business_days_to_expiry'] / 250

reference_date = pd.to_datetime("2023-08-23").date()
df['business_days_from_reference'] = df['lastTradeDate'].apply(
    lambda date: len(pd.date_range(reference_date, date, freq=BDay()))
)

MRK_dict = {}
for days_from_ref, group in df.groupby('business_days_from_reference'):
    tensor_group = torch.tensor(group[['strike', 'business_days_to_expiry', 'lastPrice']].values, dtype=torch.float32)
    MRK_dict[days_from_ref] = tensor_group

# For PFE
df = pd.read_csv(os.getcwd() + "/MH4518/datasets/PFE_calls.csv")
df = df[['lastTradeDate', 'strike', 'lastPrice', 'expiry']]

df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate']).dt.tz_localize(None).dt.date
df['expiry'] = pd.to_datetime(df['expiry']).dt.tz_localize(None).dt.date
df['business_days_to_expiry'] = df.apply(
    lambda row: len(pd.date_range(row['lastTradeDate'], row['expiry'], freq=BDay())), axis=1
)
df['business_days_to_expiry'] = df['business_days_to_expiry'] / 250

reference_date = pd.to_datetime("2023-08-23").date()
df['business_days_from_reference'] = df['lastTradeDate'].apply(
    lambda date: len(pd.date_range(reference_date, date, freq=BDay()))
)

PFE_dict = {}
for days_from_ref, group in df.groupby('business_days_from_reference'):
    tensor_group = torch.tensor(group[['strike', 'business_days_to_expiry', 'lastPrice']].values, dtype=torch.float32)
    PFE_dict[days_from_ref] = tensor_group
    
# For UNH
df = pd.read_csv(os.getcwd() + "/MH4518/datasets/UNH_calls.csv")
df = df[['lastTradeDate', 'strike', 'lastPrice', 'expiry']]

df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate']).dt.tz_localize(None).dt.date
df['expiry'] = pd.to_datetime(df['expiry']).dt.tz_localize(None).dt.date
df['business_days_to_expiry'] = df.apply(
    lambda row: len(pd.date_range(row['lastTradeDate'], row['expiry'], freq=BDay())), axis=1
)
df['business_days_to_expiry'] = df['business_days_to_expiry'] / 250

reference_date = pd.to_datetime("2023-08-23").date()
df['business_days_from_reference'] = df['lastTradeDate'].apply(
    lambda date: len(pd.date_range(reference_date, date, freq=BDay()))
)

UNH_dict = {}
for days_from_ref, group in df.groupby('business_days_from_reference'):
    tensor_group = torch.tensor(group[['strike', 'business_days_to_expiry', 'lastPrice']].values, dtype=torch.float32)
    UNH_dict[days_from_ref] = tensor_group
    
# For this we only do a projection from the latest data to the future
r = 0.0426 # Risk free interest today (05/11/2024)
heston = HestonModel()
ais = AIS(14, r)

# Get latest options data
UNH_data = UNH_dict[max(UNH_dict.keys())]
PFE_data = PFE_dict[max(PFE_dict.keys())]
MRK_data = MRK_dict[max(MRK_dict.keys())]

# Calibrate a set of Heston model parameters
UNH_params, a = heston.calibrate(UNH_data, 570.43, r)
PFE_params, b = heston.calibrate(PFE_data, 28, r)
MRK_params, c = heston.calibrate(MRK_data, 102.33, r)
params = torch.stack((UNH_params, PFE_params, MRK_params), dim=0).reshape(3,5)
corr = construct_corr(*params[:,2]) + torch.eye(6) * 0.00025 # To prevent numerical instability with LU factorisation

# Training the forward-looking model
c, control_mean = ais.train(params, corr, num_paths=254, max_epochs=3)

# Simulation
numpaths = 100
UHG_ratio = 2.0432
PFE_ratio = 27.2777
MRK_ratio = 8.9847

# corr and params must be the same as when training
# c and control_mean should be from the same training instance as train_AIS

sobol_engine = torch.quasirandom.SobolEngine(min(2 * 3 * 14, 21201), True)
u = sobol_engine.draw(numpaths)
remaining_steps = max(2 * 3 * 14 - 21201, 0)
remaining_path = torch.rand((numpaths, remaining_steps))
u = torch.cat((u, remaining_path), dim=1)

uniform = u
u = u * (1 - 2 * 1e-6) + 1e-6
u = standard_normal.icdf(u)

u, log_det = ais.AIS.forward_and_log_det(u)
u = u.reshape(numpaths, 3, -1, 2)
uniform = uniform.reshape(numpaths, 3, -1, 2)
simulation = heston.multi_asset_path(u, uniform, params, r=r, corr=corr, dt=1/250, verbose=True)

final_worst_stock_idx = torch.min(simulation[:,:,-1,0], dim=1)[1].reshape(-1,1) # This is already in log price
worst_payout = torch.where(final_worst_stock_idx == 0,
                            UHG_ratio,
                            torch.where(final_worst_stock_idx == 1,
                                        PFE_ratio,
                                        MRK_ratio))**-1
breached = (simulation[:,:,:,0] < np.log(0.59)).any(2).any(1).reshape(-1,1)

payoff = torch.where(breached,
                     worst_payout,
                     1)
control = worst_payout

cv = payoff + c * (control - control_mean)
weights = torch.exp(-log_det)
mean = torch.mean(cv * weights)
print(mean)
