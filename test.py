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

print(params, cov)

# # Regularise cov
# cov = 0.5 * (cov + cov.T)
# cov_eigvals, cov_eigvecs = torch.linalg.eigh(cov)
# cov = cov_eigvecs @ torch.diag(torch.clamp(cov_eigvals, 1e-6)) @ cov_eigvecs.T

def get_brownian(paths, num_steps=317):
    sobol = torch.quasirandom.SobolEngine(3 * 2 * num_steps)
    
    # Common random numbers
    paths //= 2
    uni = sobol.draw(paths).reshape(paths,3,-1,2)[1:,:,:,:]
    av = torch.stack((1-uni[:,:,:,0], uni[:,:,:,1]), dim=-1)
    uni = torch.cat((uni, av), dim=0)
    u = standard_normal.icdf(uni * (1 - 2 * 1e-6) + 1e-6)
    
    return u, uni

def knock_in_min_basket_long_put(paths):

    assert len(paths.shape) == 3

    paths = torch.stack((paths[:,0,:], paths[:,2,:]), dim=1)
    barrier_hit = ((paths - paths[:,:,0].unsqueeze(-1)) < np.log(0.59)).any(dim=2).any(dim=1).reshape(-1,1)
    worst_asset = torch.where((paths[:,0,-1] - paths[:,0,0]) < (paths[:,1,-1] - paths[:,1,0]), 0, 1).reshape(-1,1)
    put_payoff = torch.clamp(1 - torch.exp(paths[:,:,-1] - paths[:,:,0]), 0)
    put_payoff = torch.where(worst_asset==0,
                             put_payoff[:,0].reshape(-1,1),
                             put_payoff[:,1].reshape(-1,1))
    final_payoff = 1000 * torch.where(barrier_hit, put_payoff, 0)

    return final_payoff

def knock_in_PFE_long_put(paths):

    assert len(paths.shape) == 3

    paths = paths[:,1,:]
    barrier_hit = ((paths - paths[:,0].unsqueeze(-1)) < np.log(0.59)).any(dim=1).reshape(-1,1)
    put_payoff = torch.clamp(1 - torch.exp(paths[:,-1] - paths[:,0]), 0).reshape(-1,1)
    final_payoff = 1000 * torch.where(barrier_hit, put_payoff, 0)

    return final_payoff

def stock_payoff(paths, stock):

    assert len(paths.shape) == 3
    assert stock in [0, 1, 2]

    paths = paths[:,stock,:]
    stock_payoff = torch.exp(paths[:,-1] - paths[:,0]).reshape(-1,1) - 1
    final_payoff = 1000 * stock_payoff

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

def find_delta(payoff_func, paths):
    
    delta = torch.empty(3)
    for i in range(3):
        
        u, uni = get_brownian(paths)
        
        s0_temp = s0.clone()
        s0_temp[i] += 0.01 * s0[i]
        
        up_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
        up_payoff = payoff_func(up_paths[:,:,:,0])
        
        u, uni = get_brownian(paths)
        
        s0_temp = s0.clone()
        s0_temp[i] -= 0.01 * s0[i]
        
        down_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
        down_payoff = payoff_func(down_paths[:,:,:,0])
        
        delta_i_list = (up_payoff - down_payoff) / 0.02 / s0[i]
        delta_i_list = delta_i_list[~delta_i_list.isnan()]
        delta[i] = delta_i_list.mean()
    
    return delta

def find_gamma_cov(payoff_func, paths):
    
    # paths //= 2
    gamma = torch.empty(3,3)
    for i in range(3):
        for j in range(3):
            
            u, uni = get_brownian(paths)
            
            s0_temp = s0.clone()
            s0_temp[i] += 0.01 * s0[i]
            s0_temp[j] += 0.01 * s0[j]
            
            up1_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
            up1_payoff = payoff_func(up1_paths[:,:,:,0])
        
            u, uni = get_brownian(paths)
            
            s0_temp = s0.clone()
            s0_temp[i] += 0.01 * s0[i]
            s0_temp[j] -= 0.01 * s0[j]
            
            down1_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
            down1_payoff = payoff_func(down1_paths[:,:,:,0])
        
            u, uni = get_brownian(paths)
            
            s0_temp = s0.clone()
            s0_temp[i] -= 0.01 * s0[i]
            s0_temp[j] += 0.01 * s0[j]
            
            down2_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
            down2_payoff = payoff_func(down2_paths[:,:,:,0])
        
            u, uni = get_brownian(paths)
            
            s0_temp = s0.clone()
            s0_temp[i] -= 0.01 * s0[i]
            s0_temp[j] -= 0.01 * s0[j]
            
            up2_paths = sim.multi_asset_path(u, uni, params, r, cov, s0_temp, verbose=False)
            up2_payoff = payoff_func(up2_paths[:,:,:,0])
            
            gamma_i_j_list = (up1_payoff - down1_payoff - down2_payoff + up2_payoff) / (4 * 0.01**2 * s0[i] * s0[j])
            gamma_i_j_list = gamma_i_j_list[~gamma_i_j_list.isnan()]
            gamma[i,j] = gamma_i_j_list.mean()
    
    return gamma

def hedge(product_delta, product_gamma, kimblp_delta, kimblp_gamma, PFE_delta, PFE_gamma, stock_delta):

    # Gamma sensitivities
    def gamma_loss(gamma_hedge, product_gamma, kimblp_gamma, PFE_gamma):

        gamma = product_gamma + gamma_hedge[0] * kimblp_gamma + gamma_hedge[0] * PFE_gamma

        # return torch.sum(gamma**2).item()
        
        # Spectral norm + L2 norm
        return torch.max(torch.abs(torch.linalg.eigvals(gamma.T @ gamma))).item() + torch.sum(gamma**2).item()
    
    result = minimize(gamma_loss, [0, 0], (product_gamma, kimblp_gamma, PFE_gamma)).x
    number_of_baskets = result[0]
    number_of_puts = result[1]
    
    print('Original gamma:')
    print(torch.max(torch.abs(torch.linalg.eigvals(product_gamma.T @ product_gamma))).item() + torch.sum(product_gamma**2).item())
    print('Hedged gamma:')
    new_gamma = product_gamma + result[0] * kimblp_gamma + result[0] * PFE_gamma
    print(torch.max(torch.abs(torch.linalg.eigvals(new_gamma.T @ new_gamma))).item() + torch.sum(new_gamma**2).item())
    print(new_gamma)

    def delta_loss(number_of_stocks, number_of_baskets, number_of_puts, product_delta, kimblp_delta, PFE_delta, stock_delta):

        number_of_stocks = torch.tensor(number_of_stocks)

        delta = product_delta + number_of_baskets * kimblp_delta + number_of_puts * PFE_delta \
                + number_of_stocks[0] * stock_delta[0] + number_of_stocks[1] * stock_delta[1] + number_of_stocks[2] * stock_delta[2]

        return torch.sum(delta**2).item()
    
    number_of_stocks = minimize(delta_loss, [0, 0, 0], (number_of_baskets, number_of_puts, product_delta, kimblp_delta, PFE_delta, stock_delta)).x
    
    print('Original delta:')
    print(torch.sum(product_delta**2).item())
    print('Gamma-hedged delta:')
    print(torch.sum((product_delta + number_of_baskets * kimblp_delta + number_of_puts * PFE_delta)**2).item())
    print('Hedged delta:')    
    print(torch.sum((product_delta + number_of_baskets * kimblp_delta + number_of_puts * PFE_delta + number_of_stocks[0] * stock_delta[0] + number_of_stocks[1] * stock_delta[1] + number_of_stocks[2] * stock_delta[2])**2).item())

    return number_of_baskets, number_of_puts, torch.tensor(number_of_stocks)


if __name__ == '__main__':
    # Train model
    train(2048, 1000)

    # # Find product sensitivities
    # product_delta = find_delta(model.evaluate_payoff, paths)
    # product_gamma = find_gamma_cov(model.evaluate_payoff, paths)

    product_delta = torch.tensor([0.2153, 4.1182, 0.6588])

    product_gamma = torch.tensor([[ 0.0020, -0.1705,  0.0459],
                                [ 0.1235,  0.7053, -0.2170],
                                [-0.0231,  0.4444,  0.1436]])

    print('Product delta:')
    print(product_delta)
    print('Product gamma:')
    print(product_gamma)

    # # Find basket sensitivities
    # kimblp_delta = find_delta(knock_in_min_basket_long_put, paths)
    # kimblp_gamma = find_gamma_cov(knock_in_min_basket_long_put, paths)

    kimblp_delta = torch.tensor([-0.0278, -0.5626,  0.0514])

    kimblp_gamma = torch.tensor([[ 0.0052, -0.0241, -0.0268],
                                [ 0.1589, -2.0160, -0.4437],
                                [-0.0276, -0.4629,  0.2484]])

    print('Knock in basket delta:')
    print(kimblp_delta)
    print('Knock in basket gamma:')
    print(kimblp_gamma)

    # # Find PFE long put sensitivities
    # PFE_delta = find_delta(knock_in_PFE_long_put, paths)
    # PFE_gamma = find_gamma_cov(knock_in_PFE_long_put, paths)

    PFE_delta = torch.tensor([0.0508, 0.5763, 0.0903])

    PFE_gamma = torch.tensor([[ 0.0025, -0.2873,  0.0404],
                            [ 0.0849, -1.0836, -0.2329],
                            [-0.0272,  0.4523,  0.0560]])
            
    print('PFE delta:')
    print(PFE_delta)
    print('PFE gamma:')
    print(PFE_gamma)

    # # Find stock sensitivities
    # stock_delta = []
    # stock_gamma = []
    # for stock in [0, 1, 2]:
    #     stock_delta.append(find_delta(lambda paths:stock_payoff(paths, stock), paths))
    #     stock_gamma.append(find_gamma_cov(lambda paths:stock_payoff(paths, stock), paths))

    #     print(f'Stock {stock} delta:')
    #     print(stock_delta[stock])
    #     print(f'Stock {stock} gamma:')
    #     print(stock_gamma[stock])

    stock_0_delta = torch.tensor([ 0.1040, -1.7954, -0.4333])

    stock_0_gamma = torch.tensor([[ 0.0164,  0.2444, -0.1134],
                                [ 0.4150, -5.2465, -0.7622],
                                [ 0.0554,  2.1684,  0.1792]])
            
    stock_1_delta = torch.tensor([ 0.0461,  0.6256, -0.1266])

    stock_1_gamma = torch.tensor([[ 0.0125, -0.3442,  0.0114],
                                [ 0.1474, -2.5728, -0.2533],
                                [-0.0126, -1.0182, -0.1964]])

    stock_2_delta = torch.tensor([-0.0244, -0.1774,  0.1016])

    stock_2_gamma = torch.tensor([[-1.1913e-02,  1.3465e-03,  1.0905e-01],
                                [-6.0174e-02, -2.2106e+00, -4.4639e-01],
                                [-1.0333e-01,  5.6564e-01, -4.3506e-01]])

    stock_delta = [stock_0_delta, stock_1_delta, stock_2_delta]
    stock_gamma = [stock_0_gamma, stock_1_gamma, stock_2_gamma]

    print(f'Stock deltas:')
    print(stock_delta)
    print(f'Stock gammas:')
    print(stock_gamma)

    # # Optimise portfolio
    # number_of_baskets, number_of_pfe, number_of_stocks = hedge(product_delta, product_gamma, kimblp_delta, kimblp_gamma, PFE_delta, PFE_gamma, stock_delta)

    # Original gamma:
    # 1.5564871520558712
    # Hedged gamma:
    # 0.7288721331382404
    # Original delta:
    # 17.439942770000002
    # Gamma-hedged delta:
    # 16.60861943852973
    # Hedged delta:
    # 5.588913652538903e-11

    number_of_baskets = 0.183953990102676
    number_of_pfe = 0
    number_of_stocks = [ -1.7143, -23.6223, -43.3234]

    print('Number of baskets:')
    print(number_of_baskets)
    print('Number of PFE:')
    print(number_of_pfe)
    print('Number of stocks:')
    print(number_of_stocks)

    def portfolio_payoff(paths, number_of_baskets, number_of_pfe, number_of_stocks):
        
        product_payoff = model.evaluate_payoff(paths)
        
        basket_payoff = knock_in_min_basket_long_put(paths)
        pfe_payoff = knock_in_PFE_long_put(paths)
        
        stocks_payoff = 1000 * (torch.exp(paths[:,:,-1] - paths[:,:,0]) - 1)
        
        # print(torch.cat((product_payoff, basket_payoff, pfe_payoff, number_of_stocks[0] * stocks_payoff[:,0].reshape(-1,1), number_of_stocks[1] * stocks_payoff[:,1].reshape(-1,1), number_of_stocks[2] * stocks_payoff[:,2].reshape(-1,1)), dim=1))
        
        number_of_stocks = torch.tensor(number_of_stocks).reshape(1,3)
        stocks_payoff = (stocks_payoff * number_of_stocks).sum(dim=1, keepdim=True)
        
        final_payoff = product_payoff + number_of_baskets * basket_payoff + number_of_pfe * pfe_payoff + stocks_payoff
        
        return final_payoff

    hedged_payoff = lambda paths: portfolio_payoff(paths, number_of_baskets, number_of_pfe, number_of_stocks)

    # Find portfolio sensitivities
    portfolio_delta = find_delta(hedged_payoff, paths)
    portfolio_gamma = find_gamma_cov(hedged_payoff, paths)

    # portfolio_delta = torch.tensor([   4.8150, -175.3534,   59.2100])

    # portfolio_gamma = torch.tensor([[-5.4726e-01,  1.5610e+01, -1.5366e+01],
    #                                 [-1.2436e+01, -7.3499e+02, -5.1564e+01],
    #                                 [ 4.0473e+00,  1.0471e+02,  2.6609e+01]])

    print('Portfolio delta:')
    print(portfolio_delta)
    print('Portfolio gamma:')
    print(portfolio_gamma)

