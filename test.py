import torch
import numpy as np
from heston_model import HestonModel
from payoff import CallablePayoff
from tqdm import tqdm
from scipy.optimize import minimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
standard_normal = torch.distributions.Normal(torch.tensor([0]).to(device), torch.tensor([1]).to(device))
s0 = torch.tensor([489.44, 36.65, 111])
r = 0.0419
# importance_drift = -r - 0.41 / 317
# importance_drift = torch.tensor([[importance_drift],
#                                  [0],
#                                  [importance_drift],
#                                  [0],
#                                  [importance_drift],
#                                  [0]])
paths = 50000

sim = HestonModel()
model = CallablePayoff(0, 0.0419, s0)
sobol = torch.quasirandom.SobolEngine(3 * 2 * 317)

params, cov = sim.MLE()

# # Regularise cov
# cov = 0.5 * (cov + cov.T)
# cov_eigvals, cov_eigvecs = torch.linalg.eigh(cov)
# cov = cov_eigvecs @ torch.diag(torch.clamp(cov_eigvals, 1e-6)) @ cov_eigvecs.T
        
# Common random numbers
paths //= 2
uni = sobol.draw(paths).reshape(paths,3,-1,2)
av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
uni = torch.cat((uni, av), dim=0)
u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
paths *= 2

def knock_in_min_basket_long_put(paths):

    assert len(paths.shape) == 3

    barrier_hit = ((paths - paths[:,:,0].unsqueeze(-1)) < np.log(0.59)).any(dim=2).any(dim=1).reshape(-1,1)
    worst_asset = paths[:,:,-1].argmin(dim=1)
    put_payoff = torch.clamp(s0.reshape(1,-1) - torch.exp(paths[:,:,-1]), max=0)
    put_payoff = torch.where(worst_asset==0,
                             put_payoff[:,0],
                             torch.where(worst_asset==1,
                                         put_payoff[:,1],
                                         put_payoff[:,2])).reshape(-1,1)
    final_payoff = 1000 * torch.where(barrier_hit, put_payoff, 0)

    return final_payoff

def train(paths_per_epochs, epochs):
    
    with tqdm(total=epochs, desc="Training Progress") as pbar:
        for epoch in range(epochs):
            
            uni = sobol.draw(paths_per_epochs).reshape(paths_per_epochs,3,-1,2)
            u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)

            path = sim.multi_asset_path(u, uni, params, r, cov, s0)
            losses = model.minimise_over_path(path[:,:,:,0])
            
            pbar.set_postfix({'Worst loss': max(losses), 
                                'Best Loss': min(losses)})
            pbar.update(1)

def find_delta(payoff_func, paths=None):
    
    # paths //= 2
    delta = torch.empty(3)
    for i in range(3):
            
        # uni = sobol.draw(paths).reshape(paths,3,-1,2)
        # av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
        # uni = torch.cat((uni, av), dim=0)
        # u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
        
        s0_temp = s0
        s0_temp[i] += 0.01 * s0[i]
        
        up_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
        up_payoff = payoff_func(up_paths[:,:,:,0])
            
        # uni = sobol.draw(paths).reshape(paths,3,-1,2)
        # av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
        # uni = torch.cat((uni, av), dim=0)
        # u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
        
        s0_temp = s0
        s0_temp[i] -= 0.01 * s0[i]
        
        down_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
        down_payoff = payoff_func(down_paths[:,:,:,0])
        
        delta_i_list = (up_payoff - down_payoff) / 0.02 / s0[i]
        delta[i] = delta_i_list.nanmean()
    
    return delta

def find_gamma_univariate(payoff_func, paths=None):
    
    gamma = torch.empty(3)
    for i in range(3):
            
        # uni = sobol.draw(paths).reshape(paths,3,-1,2)
        # av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
        # uni = torch.cat((uni, av), dim=0)
        # u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
        
        s0_temp = s0
        s0_temp[i] += 0.01 * s0[i]
        
        up_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
        up_payoff = payoff_func(up_paths[:,:,:,0])
            
        # uni = sobol.draw(paths).reshape(paths,3,-1,2)
        # av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
        # uni = torch.cat((uni, av), dim=0)
        # u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
        
        s0_temp = s0
        s0_temp[i] += 0.01 * s0[i]
        
        mid_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
        mid_payoff = payoff_func(mid_paths[:,:,:,0])
            
        # uni = sobol.draw(paths).reshape(paths,3,-1,2)
        # av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
        # uni = torch.cat((uni, av), dim=0)
        # u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
        
        s0_temp = s0
        s0_temp[i] -= 0.01 * s0[i]
        
        down_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
        down_payoff = payoff_func(down_paths[:,:,:,0])
        
        gamma_i_list = (up_payoff - 2 * mid_payoff + down_payoff) / (0.02**2 * s0[i]**2)
        gamma[i] = gamma_i_list.nanmean()
    
    return gamma

def find_gamma_cov(payoff_func, paths=None):
    
    # paths //= 2
    gamma = torch.empty(3,3)
    for i in range(3):
        for j in range(3):
            # uni = sobol.draw(paths).reshape(paths,3,-1,2)
            # av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
            # uni = torch.cat((uni, av), dim=0)
            # u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
            
            s0_temp = s0
            s0_temp[i] += 0.01 * s0[i]
            s0_temp[j] += 0.01 * s0[j]
            
            up1_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
            up1_payoff = payoff_func(up1_paths[:,:,:,0])
                
            # uni = sobol.draw(paths).reshape(paths,3,-1,2)
            # av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
            # uni = torch.cat((uni, av), dim=0)
            # u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
            
            s0_temp = s0
            s0_temp[i] += 0.01 * s0[i]
            s0_temp[j] -= 0.01 * s0[j]
            
            down1_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
            down1_payoff = payoff_func(down1_paths[:,:,:,0])
                
            # uni = sobol.draw(paths).reshape(paths,3,-1,2)
            # av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
            # uni = torch.cat((uni, av), dim=0)
            # u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
            
            s0_temp = s0
            s0_temp[i] -= 0.01 * s0[i]
            s0_temp[j] += 0.01 * s0[j]
            
            down2_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
            down2_payoff = payoff_func(down2_paths[:,:,:,0])
                
            # uni = sobol.draw(paths).reshape(paths,3,-1,2)
            # av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
            # uni = torch.cat((uni, av), dim=0)
            # u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
            
            s0_temp = s0
            s0_temp[i] -= 0.01 * s0[i]
            s0_temp[j] -= 0.01 * s0[j]
            
            up2_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
            up2_payoff = payoff_func(up2_paths[:,:,:,0])
            
            gamma_i_j_list = (up1_payoff - down1_payoff - down2_payoff + up2_payoff) / (4 * 0.01**2 * s0[i] * s0[j])
            gamma[i,j] = gamma_i_j_list.nanmean()
    
    return gamma

def hedge(product_delta, product_gamma, kimblp_delta, kimblp_gamma, stock_delta):

    # Gamma sensitivities
    def gamma_loss(number_of_baskets, product_gamma, kimblp_gamma):

        gamma = product_gamma + number_of_baskets[0] * kimblp_gamma

        return torch.sum(gamma**2).item()
    
    number_of_baskets = minimize(gamma_loss, 0, (product_gamma, kimblp_gamma)).x

    def delta_loss(number_of_stocks, number_of_baskets, product_delta, kimblp_delta, stock_delta):

        number_of_stocks = torch.tensor(number_of_stocks)

        delta = product_delta + number_of_baskets[0] * kimblp_delta + number_of_stocks * stock_delta

        return torch.sum(delta**2).sum()
    
    number_of_stocks = minimize(delta_loss, [0, 0, 0], (number_of_baskets, product_delta, kimblp_delta, stock_delta)).x

    return number_of_baskets[0], torch.tensor(number_of_stocks)

# Train model
train(2048, 1000)

# Find product sensitivities
product_delta = find_delta(model.evaluate_payoff, 100000)
product_gamma = find_gamma_cov(model.evaluate_payoff, 100000)
print('Product delta:')
print(product_delta)
print('Product gamma:')
print(product_gamma)

# Find gamma hedge sensitivities
kimblp_delta = find_delta(knock_in_min_basket_long_put, 100000)
kimblp_gamma = find_gamma_cov(knock_in_min_basket_long_put, 100000)
print('Knock in basket delta:')
print(kimblp_delta)
print('Knock in basket gamma:')
print(kimblp_gamma)

# Find stock sensitivities
stock_payoff = lambda paths: 1000*(torch.exp(paths[:,:,-1]) - s0.reshape(1,-1))
stock_delta = find_delta(stock_payoff, 100000)
stock_gamma = find_gamma_cov(stock_payoff, 100000)
print('Stock delta:')
print(stock_delta)
print('Stock gamma:')
print(stock_gamma)

# Optimise portfolio
number_of_baskets, number_of_stocks = hedge(product_delta, product_gamma, kimblp_delta, kimblp_gamma, stock_delta)
print('Number of baskets:')
print(number_of_baskets)
print('Number of stocks:')
print(number_of_stocks)

# Find portfolio sensitivities
portfolio_payoff = lambda paths: model.evaluate_payoff(paths) + number_of_baskets * knock_in_min_basket_long_put(paths) + number_of_stocks * stock_payoff(paths)
portfolio_delta = find_delta(portfolio_payoff, 100000)
portfolio_gamma = find_gamma_cov(portfolio_payoff, 100000)
print('Portfolio delta:')
print(portfolio_delta)
print('Portfolio gamma:')
print(portfolio_gamma)
