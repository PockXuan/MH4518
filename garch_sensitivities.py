from process_data import GARCH
from payoff import CallablePayoff
from tqdm import tqdm
import torch
import numpy as np

import warnings
warnings.filterwarnings("ignore")

model = GARCH()
model.fit(True)
forecast = model.forecast(317, 5000)

s0 = np.array([489.44, 36.66, 111.30])
payoff = CallablePayoff(0, 0.0419, s0)

# GARCH sensitivities

# Model training
epochs = 1000
with tqdm(total=epochs, desc="Training Progress") as pbar:
    for epoch in range(epochs):

        paths = model.forecast(317, 1000)
        paths = torch.log(torch.tensor(paths))

        losses = payoff.minimise_over_path(paths)
            
        pbar.set_postfix({'Worst loss': max(losses), 
                            'Best Loss': min(losses)})
        pbar.update(1)

# Product delta sensitivities 
product_delta = torch.empty(3)
for i in range(3):

    s0_temp = s0
    s0_temp[i] *= 1.1
    up_paths = model.forecast(317, 10000, s0_temp)
    up_paths = torch.log(torch.tensor(up_paths))
    up_payoff = payoff.evaluate_payoff(up_paths)
    up_payoff = up_payoff[torch.isfinite(up_payoff)]

    s0_temp = s0
    s0_temp[i] *= 0.9
    down_paths = model.forecast(317, 10000, s0_temp)
    down_paths = torch.log(torch.tensor(down_paths))
    down_payoff = payoff.evaluate_payoff(down_paths)
    down_payoff = down_payoff[torch.isfinite(down_payoff)]

    product_delta[i] = (up_payoff.nanmean() - down_payoff.nanmean()) / (0.02 * s0[i])

print('Product delta:')
print(product_delta)

# Product gamma
product_gamma = torch.empty(3,3)
for i in range(3):
    for j in range(3):

        s0_temp = s0
        s0_temp[i] *= 1.1
        s0_temp[j] *= 1.1
        up1_paths = model.forecast(317, 10000, s0_temp)
        up1_paths = torch.log(torch.tensor(up1_paths))
        up1_payoff = payoff.evaluate_payoff(up1_paths)
        up1_payoff = up1_payoff[torch.isfinite(up1_payoff)]

        s0_temp = s0
        s0_temp[i] *= 1.1
        s0_temp[j] *= 0.9
        down1_paths = model.forecast(317, 10000, s0_temp)
        down1_paths = torch.log(torch.tensor(down1_paths))
        down1_payoff = payoff.evaluate_payoff(down1_paths)
        down1_payoff = down1_payoff[torch.isfinite(down1_payoff)]

        s0_temp = s0
        s0_temp[i] *= 0.9
        s0_temp[j] *= 1.1
        down2_paths = model.forecast(317, 10000, s0_temp)
        down2_paths = torch.log(torch.tensor(down2_paths))
        down2_payoff = payoff.evaluate_payoff(down2_paths)
        down2_payoff = down2_payoff[torch.isfinite(down2_payoff)]

        s0_temp = s0
        s0_temp[i] *= 0.9
        s0_temp[j] *= 0.9
        up2_paths = model.forecast(317, 10000, s0_temp)
        up2_paths = torch.log(torch.tensor(up2_paths))
        up2_payoff = payoff.evaluate_payoff(up2_paths)
        up2_payoff = up2_payoff[torch.isfinite(up2_payoff)]

        product_gamma[i][j] = (up1_payoff.nanmean() - down1_payoff.nanmean() - down2_payoff.nanmean() + up2_payoff.nanmean()) / (0.02**2 * s0[i] * s0[j])

print('Product gamma:')
print(product_gamma)

# Put delta sensitivities 
put_delta = torch.empty(3)
for i in range(3):

    s0_temp = s0
    s0_temp[i] *= 1.1
    up_paths = model.forecast(317, 10000, s0_temp)
    up_paths = torch.log(torch.tensor(up_paths))
    up_payoff = torch.clamp(s0[i] - up_paths[:,i,-1], 0)
    up_payoff = up_payoff[torch.isfinite(up_payoff)]

    s0_temp = s0
    s0_temp[i] *= 0.9
    down_paths = model.forecast(317, 10000, s0_temp)
    down_paths = torch.log(torch.tensor(down_paths))
    down_payoff = torch.clamp(s0[i] - down_paths[:,i,-1], 0)
    down_payoff = down_payoff[torch.isfinite(down_payoff)]

    put_delta[i] = (up_payoff.nanmean() - down_payoff.nanmean()) / (0.02 * s0[i])

print('Put delta:')
print(put_delta)

# Put gamma
put_gamma = torch.empty(3)
for i in range(3):

    s0_temp = s0
    s0_temp[i] *= 1.2
    up1_paths = model.forecast(317, 10000, s0_temp)
    up1_paths = torch.log(torch.tensor(up1_paths))
    up1_payoff = torch.clamp(s0[i] - up1_paths[:,i,-1], 0)
    up1_payoff = up1_payoff[torch.isfinite(up1_payoff)]

    s0_temp = s0
    up2_paths = model.forecast(317, 10000, s0_temp)
    up2_paths = torch.log(torch.tensor(up2_paths))
    up2_payoff = torch.clamp(s0[i] - up2_paths[:,i,-1], 0)
    up2_payoff = up2_payoff[torch.isfinite(up2_payoff)]

    s0_temp = s0
    s0_temp[i] *= 0.8
    down2_paths = model.forecast(317, 10000, s0_temp)
    down2_paths = torch.log(torch.tensor(down2_paths))
    down2_payoff = torch.clamp(s0[i] - down2_paths[:,i,-1], 0)
    down2_payoff = down2_payoff[torch.isfinite(down2_payoff)]

    put_gamma[i] = (up1_payoff.nanmean() - 2 * up2_payoff.nanmean() + down2_payoff.nanmean()) / (0.02**2 * s0[i]**2)

print('Put gamma:')
print(put_gamma)

# Stock delta  
stock_delta = torch.empty(3)
for i in range(3):

    s0_temp = s0
    s0_temp[i] *= 1.1
    up_paths = model.forecast(317, 10000, s0_temp)
    up_paths = torch.log(torch.tensor(up_paths))
    up_payoff = up_paths[:,i,-1] - s0[i]
    up_payoff = up_payoff[torch.isfinite(up_payoff)]

    s0_temp = s0
    s0_temp[i] *= 0.9
    down_paths = model.forecast(317, 10000, s0_temp)
    down_paths = torch.log(torch.tensor(down_paths))
    down_payoff = down_paths[:,i,-1] - s0[i]
    down_payoff = down_payoff[torch.isfinite(down_payoff)]

    stock_delta[i] = (up_payoff.nanmean() - down_payoff.nanmean()) / (0.02 * s0[i])

print('Stock delta:')
print(stock_delta)
