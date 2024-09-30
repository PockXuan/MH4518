import numpy as np
import torch
from torchimize.functions import lsq_lma
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 64 node Gaussian-Legendre quadrature as recommended
x64 = [0.0243502926634244325089558,0.0729931217877990394495429,0.1214628192961205544703765,0.1696444204239928180373136,0.2174236437400070841496487,0.2646871622087674163739642,0.3113228719902109561575127,0.3572201583376681159504426,0.4022701579639916036957668,0.4463660172534640879849477,0.4894031457070529574785263,0.5312794640198945456580139,0.5718956462026340342838781,0.6111553551723932502488530,0.6489654712546573398577612,0.6852363130542332425635584,0.7198818501716108268489402,0.7528199072605318966118638,0.7839723589433414076102205,0.8132653151227975597419233,0.8406292962525803627516915,0.8659993981540928197607834,0.8893154459951141058534040,0.9105221370785028057563807,0.9295691721319395758214902,0.9464113748584028160624815,0.9610087996520537189186141,0.9733268277899109637418535,0.9833362538846259569312993,0.9910133714767443207393824,0.9963401167719552793469245,0.9993050417357721394569056]
w64 = [0.0486909570091397203833654,0.0485754674415034269347991,0.0483447622348029571697695,0.0479993885964583077281262,0.0475401657148303086622822,0.0469681828162100173253263,0.0462847965813144172959532,0.0454916279274181444797710,0.0445905581637565630601347,0.0435837245293234533768279,0.0424735151236535890073398,0.0412625632426235286101563,0.0399537411327203413866569,0.0385501531786156291289625,0.0370551285402400460404151,0.0354722132568823838106931,0.0338051618371416093915655,0.0320579283548515535854675,0.0302346570724024788679741,0.0283396726142594832275113,0.0263774697150546586716918,0.0243527025687108733381776,0.0222701738083832541592983,0.0201348231535302093723403,0.0179517157756973430850453,0.0157260304760247193219660,0.0134630478967186425980608,0.0111681394601311288185905,0.0088467598263639477230309,0.0065044579689783628561174,0.0041470332605624676352875,0.0017832807216964329472961]

standard_normal = torch.distributions.Normal(torch.tensor([0]).to(device), torch.tensor([1]).to(device))

class HestonModel:
    def __init__(self):
        
        self.market = {
            'K': None,
            'T': None,
            'S0': None,
            'r': None,
            'asset_cross_covariance': None
        }
        
        # Initial values from the paper [V0,theta,rho,kappa,sigma]
        self.params = torch.tensor([1.2, 0.2, 0.3, -0.6, 0.2]).to(device)
    
    def pricing(self, parameters, market_data, u, w):
        
        num_nodes = u.shape[0]
        
        V0 = parameters[0]
        theta = parameters[1]
        rho = parameters[2]
        kappa = parameters[3]
        sigma = parameters[4]
        
        K = market_data['K']
        T = market_data['T']
        S0 = market_data['S0']
        r = market_data['r']
        
        # From the paper, their code uses 200 as an upper bound with 64 nodes
        u_max = 200
        u = 0.5 * u_max * (u + 1) # Rescaling to quadrature domain
        u = torch.cat((u,-u), dim=1)
        u = torch.cat((u-1j,u), dim=1)

        # Required constants for the characteristic function
        xi = kappa - sigma * rho * u * 1j
        d = torch.sqrt(xi**2 + sigma**2 * u * (u + 1j))
        
        A1 = u * (u + 1j) * torch.sinh(d * T/ 2)
        A2 = d * torch.cosh(d * T/ 2) + xi * torch.sinh(d * T/ 2)
        A = A1 / A2
        D = torch.log(d) + (kappa - d) * T/ 2 - torch.log((d + xi) / 2 + (d - xi) * torch.exp(-d * T) / 2)
        B = torch.exp(D)
        
        # Characteristic function
        char_func = torch.exp(1j * u * (torch.log(S0) + r * T) - T * kappa * theta  * rho * u * 1j / sigma - V0 * A + 2 * kappa * theta * D / sigma**2)
        
        offset = 0.5 * (market_data['S0'] - torch.exp(-market_data['r'] * market_data['T']) * market_data['K'])
        
        # Gauss-Legendre quadrature
        integrand = torch.real((K**(-u * 1j) / (u * 1j)).unsqueeze(-1) * char_func)
        integrand = integrand[:2*num_nodes,:] - K.unsqueeze(-1) * integrand[2*num_nodes:,:]
        integrand = w.reshape(-1,1) * 0.5 * u_max * (integrand[:num_nodes,:] + integrand[num_nodes:,:])
        integrand = torch.sum(integrand, dim=0) * torch.exp(-r * T.reshape(-1,1)) / np.pi
        
        return offset + integrand
        
    def jacobian(self, parameters, market_data, u, w):
        
        num_nodes = u.shape[0]
        
        V0 = parameters[0]
        theta = parameters[1]
        rho = parameters[2]
        kappa = parameters[3]
        sigma = parameters[4]
        
        K = market_data['K']
        T = market_data['T']
        S0 = market_data['S0']
        r = market_data['r']
        
        # From the paper, their code uses 200 as an upper bound with 64 nodes
        u_max = 200
        u = 0.5 * u_max * (u + 1) # Rescaling to quadrature domain
        u = torch.cat((u,-u), dim=1)
        u = torch.cat((u-1j,u), dim=1)

        # Required constants for the characteristic function
        xi = kappa - sigma * rho * u * 1j
        d = torch.sqrt(xi**2 + sigma**2 * u * (u + 1j))
        
        A1 = u * (u + 1j) * torch.sinh(d * T/ 2)
        A2 = d * torch.cosh(d * T/ 2) + xi * torch.sinh(d * T/ 2)
        A = A1 / A2
        D = torch.log(d) + (kappa - d) * T/ 2 - torch.log((d + xi) / 2 + (d - xi) * torch.exp(-d * T) / 2)
        B = torch.exp(D)
        
        # Characteristic function
        char_func = torch.exp(1j * u * (torch.log(S0) + r * T) - T * kappa * theta  * rho * u * 1j / sigma - V0 * A + 2 * kappa * theta * D / sigma**2)
        
        d.backward()
        dd_drho = rho.grad.clone()
        rho.grad.zero_()
        dd_dsigma = sigma.grad.clone()
        sigma.grad.zero_()
        
        A2.backward()
        dA2_drho = rho.grad.clone()
        rho.grad.zero_()
        dA2_dsigma = sigma.grad.clone()
        sigma.grad.zero_()
        
        A.backward()
        dA_drho = rho.grad.clone()
        rho.grad.zero_()
        dA_dsigma = sigma.grad.clone()
        sigma.grad.zero_()
        
        B.backward()
        dB_dkappa = kappa.grad.clone()
        kappa.grad.zero_()
        
        # Gradient vector of the characteristic function wrt [V0,theta,rho,kappa,sigma]
        h1 = -A
        h2 = 2 * kappa * D / sigma**2 - T * kappa * rho * u * 1j / sigma
        h3 = -V0 * dA_drho + 2 * kappa * theta * (dd_drho - d * dA2_drho / A2) / sigma**2 / d - T * kappa * theta * u * 1j / sigma
        h4 = V0 * dA_drho / sigma / u / 1j + 2 * theta * D / sigma**2 + 2 * kappa * theta * dB_dkappa / sigma**2 / B - T * theta * rho * u * 1j / sigma
        h5 = -V0 * dA_dsigma - 4 * kappa * theta * D / sigma**3 + 2 * kappa * theta * (dd_dsigma - d * dA2_dsigma / A2) / sigma**2 / d + T * kappa * theta * rho * u * 1j / sigma**2
        grad_vec = torch.stack([h1, h2, h3, h4, h5], dim=2)
        
        # Gauss-Legendre quadrature        
        integrand = torch.real((K**(-u * 1j) / (u * 1j)).unsqueeze(-1) * char_func * grad_vec)
        integrand = integrand[:2*num_nodes,:,:] - K.unsqueeze(-1) * integrand[2*num_nodes:,:,:]
        integrand = w.reshape(-1,1,1) * 0.5 * u_max * (integrand[:num_nodes,:,:] + integrand[num_nodes:,:,:])
        integrand = torch.sum(integrand, dim=0) * torch.exp(-r * T.reshape(-1,1)) / np.pi
        
        return integrand.T
    
    def train_params(self, K, T, S0, r):
        
        # Training method is based on the Levenberg-Marquardt method, but with the Jacobian calculated analytically
        # Paper is found at https://arxiv.org/pdf/1511.08718
        
        # K and T are lists of strike prices and corresponding times to maturity
        self.market['K'] = torch.tensor([K]).to(device).reshape(1,-1)
        self.market['T'] = torch.tensor([T]).to(device).reshape(1,-1)
        self.market['S0'] = torch.tensor([S0]).to(device)
        # Assumes no dividend. If there is, subtract from r for that asset. Since r is a scalar, it will have to be transformed into a tensor of the correct shape
        self.market['r'] = torch.tensor([r]).to(device)
        
        u = torch.tensor(x64).to(device).reshape(-1,1)
        w = torch.tensor(w64).to(device).reshape(-1,1)
        
        # The paper used a tolerance of 1e-10. Default is 1e-8 which should be good enough
        # They also used an initial damping factor equal to time to maturity, which is fucking huge; the default is 1e-3
        # Therefore I decided to use default. All other parameters as well
        # In particular, max_iter is 100, where in the paper the results are pretty good after less than 20
        return lsq_lma(self.params,
                       self.pricing,
                       self.jacobian,
                       (self.market, u, w))
        # This supposedly returns a list containing the parameters at each iteration

    def single_asset_path(self, u, dt, psic=1.5, gamma1=0.5, gamma2=0.5):
        
        # Brownian motion is generated separately and fed into the Heston model
        # This section is dedicated to path simulation by the quadratic exponential scheme
        # Paper is found at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=946405
        # An easy guide is available at https://medium.com/@alexander.tsoskounoglou/simulating-the-heston-model-with-quadratic-exponential-50cf2b1366b5
        
        V0 = self.params[0]
        theta = self.params[1]
        rho = self.params[2]
        kappa = self.params[3]
        sigma = self.params[4]
        
        S0 = self.market['S0']
        r = self.market['r']
        
        # u is an array of normally drawn values in the form (path #, timestep #, stock/vol)
        assert 1 <= psic and psic <= 2
        assert u.shape[2] == 2
        path = torch.empty((u.shape[0], u.shape[1]+1, 2), device=device)
        path[:,0,0] = torch.log(S0)
        path[:,0,1] = V0
        
        v1 = sigma**2 * torch.exp(-kappa * dt) * (1 - torch.exp(-kappa * dt)) / kappa
        v2 = theta * sigma**2 * (1 - torch.exp(-kappa * dt))**2 / 2 / kappa
        v3 = torch.exp(-kappa * dt)
        v4 = theta * (1 - torch.exp(-kappa * dt))
        
        # gamma values are somewhat arbitrary, but it is based on approximating the volatility integrated over the timestep
        # using a convex combination of V(t) and V(t+dt). The standard Euler simulation uses gamma1=1, gamma2=0, we use 0.5
        # for a more central approximation. The more sophisticated approach is by matching moments.
        x0 = r * dt - rho * kappa * theta * dt / sigma
        x1 = gamma1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
        x2 = gamma2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
        x3 = gamma1 * dt * (1 - rho**2)
        x4 = gamma2 * dt * (1 - rho**2)
        
        with tqdm(total=u.shape[1], desc="Simulation Progress") as pbar:
            for i in range(u.shape[1]):
                
                # Volatility update
                s2 = path[:,i,1] * v1 + v2
                m = path[:,i,1] * v3 + v4
                psi = s2 / m**2
                
                b2 = 2 / psi - 1 + torch.sqrt((2 / psi) * (2 / psi - 1))
                a = m / (1 + b2)
                
                p = (psi - 1) / (psi + 1)
                beta = (1 - p) / m
            
                path[:,i+1,1] = torch.where(psi <= psic, \
                                            a * (torch.sqrt(b2) + u[:,i,1])**2, \
                                            torch.where(standard_normal.icdf(u[:,i,1]) <= p, \
                                                        0, \
                                                        torch.log((1 - p) / (1 - standard_normal.icdf(u[:,i,1]))) / beta))
                
                # log stock price update
                path[:,i+1,0] = path[:,i,0] + x0 + x1 * path[:,i,1] + x2 * path[:,i+1,1] + torch.sqrt(x3 * path[:,i,1] + x4 * path[:,i+1,1]) * u[:,i,0]
            pbar.close()
            
        return path
    
    def multi_asset_path(self, u, dt, psic=1.5, gamma1=0.5, gamma2=0.5, n=3):
        
        # In this section we generalise the single asset path to multiple assets
        # This is based on the paper https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2729475
        # However, we make very strong assumptions;
        #   1. Every asset follows the same model parameters
        #   2. Every asset's stock price is related to every other by the same cross-covariance value
        #   3. Every asset's stock price is independent of other assets' volatility
        #   4. Every asset's volatility is independent of other assets' volatility
        # Therefore, the correlation between stock-volatility and volatility-volatility terms is
        # fully determined by the stock-stock covariance, with the exception of a stock and its 
        # corresponding volatility, which has correlation rho
        
        V0 = self.params[0]
        theta = self.params[1]
        rho = self.params[2]
        kappa = self.params[3]
        sigma = self.params[4]
        
        S0 = self.market['S0']
        r = self.market['r']
        
        # u is an array of normally drawn values in the form (path #, asset #, timestep #, stock/vol)
        assert 1 <= psic and psic <= 2
        assert u.shape[3] == 2 * n
        u = standard_normal.icdf(u) # However the decorrelator requires normal variables. To reuse code from earlier, this will be converted back promptly
        
        # Correlation matrix
        offdiag = torch.tensor([[self.heston_model.market['asset_cross_covariance'], 0], [0, 0]]).repeat(n,n)
        offdiag.diagonal().zero_()
        corr = torch.tensor([[1, rho], [rho, 1]]).unsqueeze(0).repeat(n,1,1)
        corr = torch.block_diag(*corr).to(device) + offdiag
        Q_inv = torch.inv(torch.tensor([[torch.sqrt(1 - rho**2), rho], [0, 1]])).unsqueeze(0).repeat(n,1,1).to(device)
        
        R = torch.cholesky(corr) # If this doesn't work then I guess we are fucked
        Q_inv = torch.block_diag(*Q_inv)
        # This worked experimentally, don't ask me
        num_paths, num_assets, num_timesteps, _ = u.shape
        u = Q_inv @ R @ u.transpose(2,3).flatten(1,2)
        u = u.reshape(num_paths, num_assets, num_timesteps, 2) # I just fucking pray to god this works LOL
        u = torch.exp(standard_normal.log_prob(u)) # u is now restored to a uniform distribution
        
        path = torch.empty((u.shape[0], u.shape[1]+1, 2), device=device)
        path[:,0,0] = torch.log(S0)
        path[:,0,1] = V0
        
        v1 = sigma**2 * torch.exp(-kappa * dt) * (1 - torch.exp(-kappa * dt)) / kappa
        v2 = theta * sigma**2 * (1 - torch.exp(-kappa * dt))**2 / 2 / kappa
        v3 = torch.exp(-kappa * dt)
        v4 = theta * (1 - torch.exp(-kappa * dt))
        
        # gamma values are somewhat arbitrary, but it is based on approximating the volatility integrated over the timestep
        # using a convex combination of V(t) and V(t+dt). The standard Euler simulation uses gamma1=1, gamma2=0, we use 0.5
        # for a more central approximation. The more sophisticated approach is by matching moments.
        x0 = r * dt - rho * kappa * theta * dt / sigma
        x1 = gamma1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
        x2 = gamma2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
        x3 = gamma1 * dt * (1 - rho**2)
        x4 = gamma2 * dt * (1 - rho**2)
        
        with tqdm(total=u.shape[1], desc="Simulation Progress") as pbar:
            for i in range(u.shape[1]):
                
                # Volatility update
                s2 = path[:,:,i,1] * v1 + v2
                m = path[:,:,i,1] * v3 + v4
                psi = s2 / m**2
                
                b2 = 2 / psi - 1 + torch.sqrt((2 / psi) * (2 / psi - 1))
                a = m / (1 + b2)
                
                p = (psi - 1) / (psi + 1)
                beta = (1 - p) / m
            
                path[:,:,i+1,1] = torch.where(psi <= psic, \
                                              a * (torch.sqrt(b2) + u[:,:,i,1])**2, \
                                              torch.where(standard_normal.icdf(u[:,:,i,1]) <= p, \
                                                          0, \
                                                          torch.log((1 - p) / (1 - standard_normal.icdf(u[:,:,i,1]))) / beta))
                
                # log stock price update
                path[:,:,i+1,0] = path[:,:,i,0] + x0 + x1 * path[:,:,i,1] + x2 * path[:,:,i+1,1] + torch.sqrt(x3 * path[:,:,i,1] + x4 * path[:,:,i+1,1]) * u[:,:,i,0]
            pbar.close()
        
        return path
        
        
    