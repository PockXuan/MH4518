import torch
from heston_model import HestonModel
from payoff import CallablePayoff
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
standard_normal = torch.distributions.Normal(torch.tensor([0]).to(device), torch.tensor([1]).to(device))
s0 = torch.tensor([489.44, 36.65, 111])
r = 0.0419

sim = HestonModel()
model = CallablePayoff(0, 0.0419, s0)
sobol = torch.quasirandom.SobolEngine(3 * 2 * 317)

params, cov = sim.MLE()

# epochs = 3000
# with tqdm(total=epochs, desc="Training Progress") as pbar:
#     for epoch in range(epochs):
        
#         uni = sobol.draw(1024).reshape(1024,3,-1,2)
#         u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)

#         path = sim.multi_asset_path(u, uni, params, r, cov, s0)
#         losses = model.minimise_over_path(path[:,:,:,0])
        
#         pbar.set_postfix({'Worst loss': max(losses), 
#                             'Best Loss': min(losses)})
#         pbar.update(1)

# Find delta
delta = torch.empty(3)
for i in range(3):
        
    uni = sobol.draw(25000).reshape(25000,3,-1,2)
    av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
    uni = torch.cat((uni, av), dim=0)
    u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
    
    s0_temp = s0
    s0_temp[i] += 0.01 * s0[i]
    
    up_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
    up_payoff = model.evaluate_payoff(up_paths[:,:,:,0]).nanmean()
        
    uni = sobol.draw(25000).reshape(25000,3,-1,2)
    av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
    uni = torch.cat((uni, av), dim=0)
    u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
    
    s0_temp = s0
    s0_temp[i] -= 0.01 * s0[i]
    
    down_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
    down_payoff = model.evaluate_payoff(down_paths[:,:,:,0]).nanmean()
    
    delta[i] = (up_payoff - down_payoff) / 0.02 / s0[i]
print('Payoff delta:', delta.tolist())

# Find gamma
gamma = torch.empty((3,3))
for i in range(3):
    for j in range(3):
        
        uni = sobol.draw(25000).reshape(25000,3,-1,2)
        av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
        uni = torch.cat((uni, av), dim=0)
        u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
        
        s0_temp = s0
        s0_temp[i] += 0.01 * s0[i]
        s0_temp[j] += 0.01 * s0[j]
        
        up1_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
        up1_payoff = model.evaluate_payoff(up_paths[:,:,:,0]).nanmean()
            
        uni = sobol.draw(25000).reshape(25000,3,-1,2)
        av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
        uni = torch.cat((uni, av), dim=0)
        u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
        
        s0_temp = s0
        s0_temp[i] += 0.01 * s0[i]
        s0_temp[j] -= 0.01 * s0[j]
        
        down1_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
        down1_payoff = model.evaluate_payoff(down_paths[:,:,:,0]).nanmean()
            
        uni = sobol.draw(25000).reshape(25000,3,-1,2)
        av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
        uni = torch.cat((uni, av), dim=0)
        u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
        
        s0_temp = s0
        s0_temp[i] -= 0.01 * s0[i]
        s0_temp[j] += 0.01 * s0[j]
        
        down2_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
        down2_payoff = model.evaluate_payoff(down_paths[:,:,:,0]).nanmean()
        
        uni = sobol.draw(25000).reshape(25000,3,-1,2)
        av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
        uni = torch.cat((uni, av), dim=0)
        u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
        
        s0_temp = s0
        s0_temp[i] -= 0.01 * s0[i]
        s0_temp[j] -= 0.01 * s0[j]
        
        up2_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
        up2_payoff = model.evaluate_payoff(up_paths[:,:,:,0]).nanmean()
        
        gamma[i,j] = (up1_payoff - down1_payoff - down2_payoff + up2_payoff) / (4 * 0.01**2 * s0[i] * s0[j])
print('Payoff gamma:', gamma)

put_delta = torch.empty(3)
for i in range(3):
    
    uni = sobol.draw(25000).reshape(25000,3,-1,2)
    av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
    uni = torch.cat((uni, av), dim=0)
    u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
    
    s0_temp = s0
    s0_temp[i] += 0.01 * s0[i]
    
    up_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)[:,i,-1,0]
    up_put = torch.clamp(s0_temp[i]-torch.exp(up_paths), 0).nanmean()
    
    uni = sobol.draw(25000).reshape(25000,3,-1,2)
    av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
    uni = torch.cat((uni, av), dim=0)
    u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
    
    s0_temp = s0
    s0_temp[i] -= 0.01 * s0[i]
    
    down_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)[:,i,-1,0]
    down_put = torch.clamp(s0_temp[i]-torch.exp(down_paths), 0).nanmean()
    
    put_delta[i] = (up_put - down_put) / 0.02 / s0[i]
print('Put delta:', put_delta.tolist())

put_gamma = torch.empty(3)
for i in range(3):
    
    uni = sobol.draw(25000).reshape(25000,3,-1,2)
    av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
    uni = torch.cat((uni, av), dim=0)
    u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
    
    s0_temp = s0
    s0_temp[i] += 0.01 * s0[i]
    
    up_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)[:,i,-1,0]
    up_put = torch.clamp(s0_temp[i]-torch.exp(up_paths), 0).nanmean()
    
    uni = sobol.draw(25000).reshape(25000,3,-1,2)
    av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
    uni = torch.cat((uni, av), dim=0)
    u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
    
    s0_temp = s0
    
    mid_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)[:,i,-1,0]
    mid_put = torch.clamp(s0_temp[i]-torch.exp(mid_paths), 0).nanmean()
    
    uni = sobol.draw(25000).reshape(25000,3,-1,2)
    av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
    uni = torch.cat((uni, av), dim=0)
    u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
    
    s0_temp = s0
    s0_temp[i] -= 0.01 * s0[i]
    
    down_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)[:,i,-1,0]
    down_put = torch.clamp(s0_temp[i]-torch.exp(down_paths), 0).nanmean()
    
    put_gamma[i] = (up_put - 2 * mid_put + down_put) / (0.04 * s0[i]**2)
print('Put gamma:', put_gamma.tolist())

stock_delta = torch.empty(3)
for i in range(3):
    
    uni = sobol.draw(25000).reshape(25000,3,-1,2)
    av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
    uni = torch.cat((uni, av), dim=0)
    u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
    
    s0_temp = s0
    s0_temp[i] += 0.01 * s0[i]
    
    up_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)[:,i,-1,0]
    up_stock = torch.exp(up_paths).nanmean()
    
    uni = sobol.draw(25000).reshape(25000,3,-1,2)
    av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
    uni = torch.cat((uni, av), dim=0)
    u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
    
    s0_temp = s0
    s0_temp[i] -= 0.01 * s0[i]
    
    down_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)[:,i,-1,0]
    down_stock = torch.exp(down_paths).nanmean()
    
    stock_delta[i] = (up_stock - down_stock) / 0.02 / s0[i]
    
    # import matplotlib.pyplot as plt
    # plt.plot(up_paths.T.numpy(), color='blue', alpha=0.3)
    # plt.plot(down_paths.T.numpy(), color='red', alpha=0.3)
    # plt.show()
print('Stock delta:', stock_delta.tolist())

def construct_portfolio(product_delta, product_gamma, put_delta, put_gamma, stock_delta):
    product_gamma = torch.diag(product_gamma)
    
    # Gamma hedging
    number_of_puts = -product_gamma / put_gamma
    port1_delta = product_delta * number_of_puts * put_delta
    
    # Delta hedging
    number_of_stocks = -port1_delta / stock_delta
    
    return number_of_puts, number_of_stocks

puts, stocks = construct_portfolio(delta, gamma, put_delta, put_gamma, stock_delta)
print('Number of puts:')
print(puts.tolist())
print('Number of stocks:')
print(stocks.tolist())

def port_payoff(paths, puts, stocks):
    
    # Product payoff
    product_payoff = model.evaluate_payoff(paths)
    
    # Put and stocks payoff
    put_payoff = puts.reshape(1,3) * torch.clamp(s0.reshape(1,3) - torch.exp(paths[:,:,-1]), 0)
    put_payoff = put_payoff.mean(dim=1)
    
    stocks_payoff = stocks.reshape(1,3) * (s0.reshape(1,3) - torch.exp(paths[:,:,-1]))
    stocks_payoff = stocks_payoff.mean(dim=1)
    
    # Total payoff
    total_payoff = product_payoff.flatten() + put_payoff + stocks_payoff
    
    return total_payoff

# Portfolio delta
port_delta = torch.empty(3)
for i in range(3):
    
    uni = sobol.draw(25000).reshape(25000,3,-1,2)
    av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
    uni = torch.cat((uni, av), dim=0)
    u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
    
    s0_temp = s0
    s0_temp[i] += 0.01 * s0[i]
    
    up_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)[:,:,:,0]
    up_port = port_payoff(up_paths, puts, stocks).nanmean()
    
    uni = sobol.draw(25000).reshape(25000,3,-1,2)
    av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
    uni = torch.cat((uni, av), dim=0)
    u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
    
    s0_temp = s0
    s0_temp[i] -= 0.01 * s0[i]
    
    down_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)[:,:,:,0]
    down_port = port_payoff(down_paths, puts, stocks).nanmean()
    
    port_delta[i] = (up_port - down_port) / 0.02 / s0[i]
print('Portfolio delta:', port_delta.tolist())

# Portfolio gamma
portfolio_gamma = torch.empty((3,3))
for i in range(3):
    for j in range(3):
        
        uni = sobol.draw(25000).reshape(25000,3,-1,2)
        av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
        uni = torch.cat((uni, av), dim=0)
        u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
        
        s0_temp = s0
        s0_temp[i] += 0.01 * s0[i]
        s0_temp[j] += 0.01 * s0[j]
        
        up1_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)[:,:,:,0]
        up1_port = port_payoff(up1_paths, puts, stocks).nanmean()
            
        uni = sobol.draw(25000).reshape(25000,3,-1,2)
        av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
        uni = torch.cat((uni, av), dim=0)
        u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
        
        s0_temp = s0
        s0_temp[i] += 0.01 * s0[i]
        s0_temp[j] -= 0.01 * s0[j]
        
        down1_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)[:,:,:,0]
        down1_port = port_payoff(down1_paths, puts, stocks).nanmean()
            
        uni = sobol.draw(25000).reshape(25000,3,-1,2)
        av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
        uni = torch.cat((uni, av), dim=0)
        u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
        
        s0_temp = s0
        s0_temp[i] -= 0.01 * s0[i]
        s0_temp[j] += 0.01 * s0[j]
        
        down2_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)[:,:,:,0]
        down2_port = port_payoff(down2_paths, puts, stocks).nanmean()
        
        uni = sobol.draw(25000).reshape(25000,3,-1,2)
        av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
        uni = torch.cat((uni, av), dim=0)
        u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
        
        s0_temp = s0
        s0_temp[i] -= 0.01 * s0[i]
        s0_temp[j] -= 0.01 * s0[j]
        
        up2_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)[:,:,:,0]
        up2_port = port_payoff(up2_paths, puts, stocks).nanmean()
        
        portfolio_gamma[i,j] = (up1_port - down1_port - down2_port + up2_port) / (4 * 0.01**2 * s0[i] * s0[j])
print('Portfolio gamma:', portfolio_gamma)
