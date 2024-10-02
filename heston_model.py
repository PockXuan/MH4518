import numpy as np
import torch
import torch.func as F
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
            'asset_cross_correlation': None
        }
        
        # Initial values from the paper [V0,theta,rho,kappa,sigma]
        self.params = torch.tensor([[1.2], 
                                    [0.2], 
                                    [-0.3], 
                                    [0.6], 
                                    [0.2]], requires_grad=True).to(device)
        
        self.quad = {'u': None,
                     'x': None}
    
    def pricing(self, parameters, market_data, u, w):
        
        num_nodes = int(u.shape[0] / 2)
        
        V0, theta, rho, kappa, sigma = parameters
        
        K = market_data['K']
        T = market_data['T']
        S0 = market_data['S0']
        r = market_data['r']

        # Required constants for the characteristic function
        xi = kappa - sigma * rho * u * 1j
        d = torch.sqrt(xi**2 + sigma**2 * u * (u + 1j))
        A1 = u * (u + 1j) * torch.sinh(d * T / 2)
        A2 = d * torch.cosh(d * T/ 2) + xi * torch.sinh(d * T/ 2)
        A = A1 / A2
        D = torch.log(d) + (kappa - d) * T/ 2 - torch.log((d + xi) / 2 + (d - xi) * torch.exp(-d * T) / 2)
        
        # Characteristic function
        char_func = torch.exp(1j * u * (torch.log(S0) + r * T) - T * kappa * theta  * rho * u * 1j / sigma - V0 * A + 2 * kappa * theta * D / sigma**2)
        
        offset = 0.5 * (market_data['S0'] - torch.exp(-market_data['r'] * market_data['T']) * market_data['K'])
        
        # Gauss-Legendre quadrature
        integrand = torch.real((K**(-u * 1j) / (u * 1j)) * char_func)
        integrand = integrand[:num_nodes,:] - K * integrand[num_nodes:,:]
        integrand = w.reshape(-1,1) * (integrand[:num_nodes//2,:] + integrand[num_nodes//2:,:])
        integrand = torch.sum(integrand, dim=0) * torch.exp(-r * T.reshape(-1,1)).reshape(1, -1) / np.pi
    
        return (offset + integrand).T
        
    def jacobian(self, parameters, market_data, u, w):
        
        num_nodes = int(u.shape[0] / 2)
        
        V0, theta, rho, kappa, sigma = parameters
        V0.retain_grad()
        theta.retain_grad()
        rho.retain_grad()
        kappa.retain_grad()
        sigma.retain_grad()
        
        K = market_data['K']
        T = market_data['T']
        S0 = market_data['S0']
        r = market_data['r']

        # Required constants for the characteristic function
        xi = kappa - sigma * rho * u * 1j
        d = torch.sqrt(xi**2 + sigma**2 * u * (u + 1j))
        A1 = u * (u + 1j) * torch.sinh(d * T / 2)
        A2 = d * torch.cosh(d * T / 2) + xi * torch.sinh(d * T / 2)
        A = A1 / A2
        D = torch.log(d) + (kappa - d) * T/ 2 - torch.log((d + xi) / 2 + (d - xi) * torch.exp(-d * T) / 2)
        B = torch.exp(D)
        
        # Characteristic function
        char_func = torch.exp(1j * u * (torch.log(S0) + r * T) - T * kappa * theta  * rho * u * 1j / sigma - V0 * A + 2 * kappa * theta * D / sigma**2)
        
        # Because why is there no easy way to do this...
        def d_func(rho, sigma):
            xi = kappa - sigma * rho * u * 1j
            return torch.sqrt(xi**2 + sigma**2 * u * (u + 1j))
        d_partial_real = F.jacrev(lambda x,y: d_func(x,y).real, (0,1))(rho, sigma)
        d_partial_imag = F.jacrev(lambda x,y: d_func(x,y).imag, (0,1))(rho, sigma)
        dd_drho = (d_partial_real[0] + 1j * d_partial_imag[0]).squeeze(-1)
        dd_dsigma = (d_partial_real[1] + 1j * d_partial_imag[1]).squeeze(-1)
        
        def A2_func(rho, sigma):
            d = d_func(rho, sigma)
            return d * torch.cosh(d * T / 2) + xi * torch.sinh(d * T / 2)
        A2_partial_real = F.jacrev(lambda x,y: A2_func(x,y).real, (0,1))(rho, sigma)
        A2_partial_imag = F.jacrev(lambda x,y: A2_func(x,y).imag, (0,1))(rho, sigma)
        dA2_drho = (A2_partial_real[0] + 1j * A2_partial_imag[0]).squeeze(-1)
        dA2_dsigma = (A2_partial_real[1] + 1j * A2_partial_imag[1]).squeeze(-1)
        
        def A_func(rho, sigma):
            A2 = A2_func(rho, sigma)
            return A1 / A2
        A_partial_real = F.jacrev(lambda x,y: A_func(x,y).real, (0,1))(rho, sigma)
        A_partial_imag = F.jacrev(lambda x,y: A_func(x,y).imag, (0,1))(rho, sigma)
        dA_drho = (A_partial_real[0] + 1j * A_partial_imag[0]).squeeze(-1)
        dA_dsigma = (A_partial_real[1] + 1j * A_partial_imag[1]).squeeze(-1)
            
        def B_func(kappa):
            xi = kappa - sigma * rho * u * 1j
            d = torch.sqrt(xi**2 + sigma**2 * u * (u + 1j))
            return torch.exp(torch.log(d) + (kappa - d) * T/ 2 - torch.log((d + xi) / 2 + (d - xi) * torch.exp(-d * T) / 2))
        B_partial_real = F.jacrev(lambda x: B_func(x).real)(kappa)
        B_partial_imag = F.jacrev(lambda x: B_func(x).imag)(kappa)
        dB_dkappa =(B_partial_real + 1j * B_partial_imag).squeeze(-1)
        
        # Gradient vector of the characteristic function wrt [V0,theta,rho,kappa,sigma]
        h1 = -A
        h2 = 2 * kappa * D / sigma**2 - T * kappa * rho * u * 1j / sigma
        h3 = -V0 * dA_drho + 2 * kappa * theta * (dd_drho - d * dA2_drho / A2) / sigma**2 / d - T * kappa * theta * u * 1j / sigma
        h4 = V0 * dA_drho / sigma / u / 1j + 2 * theta * D / sigma**2 + 2 * kappa * theta * dB_dkappa / sigma**2 / B - T * theta * rho * u * 1j / sigma
        h5 = -V0 * dA_dsigma - 4 * kappa * theta * D / sigma**3 + 2 * kappa * theta * (dd_dsigma - d * dA2_dsigma / A2) / sigma**2 / d + T * kappa * theta * rho * u * 1j / sigma**2
        grad_vec = torch.stack([h1, h2, h3, h4, h5], dim=2)
        
        # Gauss-Legendre quadrature
        integrand = torch.real((K**(-u * 1j) / (u * 1j)).unsqueeze(-1) * char_func.unsqueeze(-1) * grad_vec)
        integrand = integrand[:num_nodes,:,:] - K.unsqueeze(-1) * integrand[num_nodes:,:,:]
        integrand = w.reshape(-1,1,1) * (integrand[:num_nodes//2,:,:] + integrand[num_nodes//2:,:,:])
        integrand = torch.sum(integrand, dim=0) * torch.exp(-r * T.reshape(-1,1)) / np.pi
        
        return integrand
    
    def calibrate(self, K, T, S0, r):
        
        # Training method is based on the Levenberg-Marquardt method, but with the Jacobian calculated analytically
        # Paper is found at https://arxiv.org/pdf/1511.08718
        
        # K and T are lists of strike prices and corresponding times to maturity
        self.market['K'] = torch.tensor(K).to(device).reshape(1,-1)
        self.market['T'] = torch.tensor(T).to(device).reshape(1,-1)
        self.market['S0'] = torch.tensor([S0]).to(device)
        # Assumes no dividend. If there is, subtract from r for that asset. Since r is a scalar, it will have to be transformed into a tensor of the correct shape
        self.market['r'] = torch.tensor([r]).to(device)
        
        u = torch.tensor(x64).to(device).reshape(-1,1)
        w = torch.tensor(w64).to(device).reshape(-1,1)
        
        # From the paper, their code uses 200 as an upper bound with 64 nodes
        u_max = 200
        u = torch.cat((u,-u), dim=0)
        u = 0.5 * u_max * (u + 1) # Rescaling to quadrature domain
        u = torch.cat((u-1j,u), dim=0)
        
        w = 0.5 * u_max * w
        
        self.quad['u'] = u
        self.quad['w'] = w
        
        # The paper used a tolerance of 1e-10. Default is 1e-8 which should be good enough
        # They also used an initial damping factor equal to time to maturity
        # Therefore I decided to use default. All other parameters as well
        # In particular, max_iter is 100, where in the paper the results are pretty good after less than 20
        return self.lsq_lma(self.params,
                       self.pricing,
                       self.jacobian,
                       (self.market, u ,w),
                       tau=max(T))
        # This supposedly returns a list containing the parameters at each iteration

    def single_asset_path(self, u, dt, psic=1.5, gamma1=0.5, gamma2=0.5):
        
        # Brownian motion is generated separately and fed into the Heston model
        # This section is dedicated to path simulation by the quadratic exponential scheme
        # Paper is found at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=946405
        # An easy guide is available at https://medium.com/@alexander.tsoskounoglou/simulating-the-heston-model-with-quadratic-exponential-50cf2b1366b5
        
        # u is an array of normally drawn values in the form (path #, timestep #, stock/vol)
        assert 1 <= psic and psic <= 2
        assert u.shape[2] == 2
        
        numsteps = u.shape[1]
        
        V0 = self.params[0]
        theta = self.params[1]
        rho = self.params[2]
        kappa = self.params[3]
        sigma = self.params[4]
        
        S0 = self.market['S0']
        r = self.market['r']
        L = torch.linalg.cholesky(torch.tensor([[1, rho], [rho, 1]]))
        u = (L @ u.reshape(-1,2,1)).reshape(-1, numsteps, 2)
        uniform = torch.rand(u.shape)
        
        path = torch.empty((u.shape[0], numsteps + 1, 2), device=device)
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
            for i in range(numsteps):
                
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
                                            torch.where(uniform[:,i,1] <= p, \
                                                        0, \
                                                        torch.log((1 - p) / (1 - uniform[:,i,1])) / beta))
                
                # log stock price update
                path[:,i+1,0] = path[:,i,0] + x0 + x1 * path[:,i,1] + x2 * path[:,i+1,1] + torch.sqrt(x3 * path[:,i,1] + x4 * path[:,i+1,1]) * u[:,i,0]
                pbar.update()
                
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
        
        numsteps = u.shape[2]
        
        V0 = self.params[0]
        theta = self.params[1]
        rho = self.params[2]
        kappa = self.params[3]
        sigma = self.params[4]
        
        S0 = self.market['S0']
        r = self.market['r']
        ccc = self.market['asset_cross_correlation']
        
        # u is an array of normally drawn values in the form (path #, asset #, timestep #, stock/vol)
        assert 1 <= psic and psic <= 2
        assert u.shape[1] == n
        assert u.shape[3] == 2
        uniform = torch.exp(standard_normal.log_prob(u))
        
        # Correlation matrix
        offdiag = torch.tensor([[ccc, rho*ccc], [rho*ccc, rho*rho*ccc]])
        offdiag = offdiag.repeat(n,n) - torch.block_diag(*offdiag.repeat(n,1,1))
        corr = torch.tensor([[1, rho], [rho, 1]]).unsqueeze(0).repeat(n,1,1)
        corr = torch.block_diag(*corr).to(device) + offdiag
        Q_inv = torch.linalg.inv(torch.tensor([[torch.sqrt(1 - rho**2), rho], [0, 1]])).unsqueeze(0).repeat(n,1,1).to(device)
        
        R = torch.linalg.cholesky(corr)
        Q_inv = torch.block_diag(*Q_inv)
        # This worked experimentally, don't ask me
        num_paths, num_assets, num_timesteps, _ = u.shape
        u = Q_inv @ R @ u.transpose(2,3).flatten(1,2)
        L = torch.linalg.cholesky(torch.tensor([[1, rho], [rho, 1]]))
        u = (L @ u.reshape(-1,2,1)).reshape(-1, n, numsteps, 2)
        u = u.reshape(num_paths, num_assets, num_timesteps, 2)
        uniform = torch.rand(u.shape)
        
        path = torch.empty((u.shape[0], n, u.shape[2]+1, 2), device=device)
        path[:,:,0,0] = torch.log(S0)
        path[:,:,0,1] = V0
        
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
        
        with tqdm(total=u.shape[2], desc="Simulation Progress") as pbar:
            for i in range(u.shape[2]):
                
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
                                              torch.where(uniform[:,:,i,1] <= p, \
                                                          0, \
                                                          torch.log((1 - p) / (1 - uniform[:,:,i,1])) / beta))
                
                # log stock price update
                path[:,:,i+1,0] = path[:,:,i,0] + x0 + x1 * path[:,:,i,1] + x2 * path[:,:,i+1,1] + torch.sqrt(x3 * path[:,:,i,1] + x4 * path[:,:,i+1,1]) * u[:,:,i,0]
                pbar.update()
            pbar.close()
        
        return path

    def lsq_lma(self,
            p: torch.Tensor,
            function, 
            jac_function, 
            args, 
            ftol: float = 1e-8,
            ptol: float = 1e-8,
            gtol: float = 1e-8,
            tau: float = 1e-3,
            meth: str = 'lev',
            rho1: float = .25, 
            rho2: float = .75, 
            bet: float = 2,
            gam: float = 3,
            max_iter: int = 100,
        ):
        """
        Levenberg-Marquardt implementation for least-squares fitting of non-linear functions
        
        :param p: initial value(s)
        :param function: user-provided function which takes p (and additional arguments) as input
        :param jac_fun: user-provided Jacobian function which takes p (and additional arguments) as input
        :param args: optional arguments passed to function
        :param ftol: relative change in cost function as stop condition
        :param ptol: relative change in independant variables as stop condition
        :param gtol: maximum gradient tolerance as stop condition
        :param tau: factor to initialize damping parameter
        :param meth: method which is default 'lev' for Levenberg and otherwise Marquardt
        :param rho1: first gain factor threshold for damping parameter adjustment for Marquardt
        :param rho2: second gain factor threshold for damping parameter adjustment for Marquardt
        :param bet: multiplier for damping parameter adjustment for Marquardt
        :param gam: divisor for damping parameter adjustment for Marquardt
        :param max_iter: maximum number of iterations
        :return: list of results
        """
        
        # Ripped this off the torchimize pack because I couldn't get it to work

        fun = lambda p: function(p, *args)
        jac_fun = lambda p: jac_function(p, *args)
        
        f = fun(p)
        j = jac_fun(p)
        g = torch.matmul(j.T, f)
        H = torch.matmul(j.T, j)
        u = tau * torch.max(torch.diag(H))
        v = 2
        p_list = [p]
        while len(p_list) < max_iter:
            D = torch.eye(j.shape[1], device=j.device)
            D *= 1 if meth == 'lev' else torch.max(torch.maximum(H.diagonal(), D.diagonal()))
            try:
                h = -torch.linalg.lstsq(H+u*D, g, rcond=None, driver=None)[0]
            except:
                print("Error encountered")
                return p_list
            f_h = fun(p+h)
            rho_denom = torch.matmul(h.T, u*h-g)
            rho_nom = torch.matmul(f.T, f) - torch.matmul(f_h.T, f_h)
            rho = rho_nom / rho_denom if rho_denom > 0 else torch.inf if rho_nom > 0 else -torch.inf
            if rho > 0:
                p = p + h
                j = jac_fun(p)
                g = torch.matmul(j.T, fun(p))
                H = torch.matmul(j.T, j)
            p_list.append(p.clone())
            f_prev = f.clone()
            f = fun(p)
            if meth == 'lev':
                u, v = (u*torch.max(torch.tensor([1/3, 1-(2*rho-1)**3])), 2) if rho > 0 else (u*v, v*2)
            else:
                u = u*bet if rho < rho1 else u/gam if rho > rho2 else u

            # stop conditions
            gcon = max(abs(g)) < gtol
            pcon = (h**2).sum()**.5 < ptol*(ptol + (p**2).sum()**.5)
            fcon = ((f_prev-f)**2).sum() < ((ftol*f)**2).sum() if rho > 0 else False
            if gcon or pcon or fcon:
                break

        return p_list       

def calibration_test():
    
    karr = [0.9371, 0.8603, 0.8112, 0.7760, 0.7470, 0.7216, 0.6699, 0.6137,
            0.9956, 0.9868, 0.9728, 0.9588, 0.9464, 0.9358, 0.9175, 0.9025,
            1.0427, 1.0463, 1.0499, 1.0530, 1.0562, 1.0593, 1.0663, 1.0766,
            1.2287, 1.2399, 1.2485, 1.2659, 1.2646, 1.2715, 1.2859, 1.3046,
            1.3939, 1.4102, 1.4291, 1.4456, 1.4603, 1.4736, 1.5005, 1.5328]

    tarr = [0.119047619047619,  0.238095238095238,	0.357142857142857,  0.476190476190476,	0.595238095238095,  0.714285714285714,  1.07142857142857,  1.42857142857143,
            0.119047619047619,	0.238095238095238,  0.357142857142857,  0.476190476190476,  0.595238095238095,  0.714285714285714,  1.07142857142857,  1.42857142857143,
            0.119047619047619, 	0.238095238095238,	0.357142857142857,	0.476190476190476,	0.595238095238095,	0.714285714285714,	1.07142857142857,  1.42857142857143,
            0.119047619047619,	0.238095238095238,	0.357142857142857,	0.476190476190476,  0.595238095238095,	0.714285714285714,	1.07142857142857,  1.42857142857143,
            0.119047619047619,	0.238095238095238,	0.357142857142857,	0.476190476190476,	0.595238095238095,	0.714285714285714,	1.07142857142857,  1.42857142857143]

    S0 = 1.0
    r = 0.02

    model = HestonModel()
    print(model.calibrate(karr, tarr, S0, r))

def single_simulation_test(S0, r):
    
    numsteps = 10000
    
    sobol_engine = torch.quasirandom.SobolEngine(2 * numsteps, True)
    paths = sobol_engine.draw(20)
    paths = paths.reshape(-1, numsteps, 2)
    paths = standard_normal.icdf(paths)
    
    model = HestonModel()
    model.market['S0'] = torch.tensor([S0], device=device)
    model.market['r'] = torch.tensor([r], device=device)
    
    simulation = model.single_asset_path(paths, 1/365)
    
    plt.plot(simulation[:,:,0].T.detach().numpy())
    plt.title("Log price")
    plt.show()
    plt.clf()
    
    plt.plot(torch.exp(simulation[:,:,0]).T.detach().numpy())
    plt.title("Price")
    plt.show()
    plt.clf()
    
    plt.plot(simulation[:,:,1].T.detach().numpy())
    plt.title("Volatility")
    plt.show()
    plt.clf()
    
    return simulation
    
def multi_simulation_test(S0, r, asset_cross_covariance, n):
    
    numsteps = 1000
    
    sobol_engine = torch.quasirandom.SobolEngine(2 * n * numsteps, True)
    paths = sobol_engine.draw(20)
    paths = paths.reshape(-1, n, numsteps, 2)
    paths = standard_normal.icdf(paths)
    
    model = HestonModel()
    model.market['S0'] = torch.tensor([S0], device=device)
    model.market['r'] = torch.tensor([r], device=device)
    model.market['asset_cross_correlation'] = torch.tensor([asset_cross_covariance], device=device)
    
    simulation = model.multi_asset_path(paths, 1 / 365, n=n)
    
    for i in range(n):
        plt.plot(simulation[:,i,:,0].T.detach().numpy())
    plt.title("Log price")
    plt.show()
    plt.clf()
        
    for i in range(n):
        plt.plot(torch.exp(simulation[:,i,:,0]).T.detach().numpy())
    plt.title("Price")
    plt.show()
    plt.clf()
    
    for i in range(n):
        plt.plot(simulation[:,i,:,1].T.detach().numpy())
    plt.title("Volatility")
    plt.show()
    plt.clf()
    
    return simulation
    

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import numpy as np

    calibration_test()
    single_simulation_test(1.0, 0.02)
    multi_simulation_test(1.0, 0.02, 0.2, 3)
        
    