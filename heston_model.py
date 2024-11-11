import numpy as np
import torch
from tqdm import tqdm
import os
import pandas as pd
from scipy import optimize, special

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 64 node Gaussian-Legendre quadrature is recommended
# x64 = [0.0243502926634244325089558,0.0729931217877990394495429,0.1214628192961205544703765,0.1696444204239928180373136,0.2174236437400070841496487,0.2646871622087674163739642,0.3113228719902109561575127,0.3572201583376681159504426,0.4022701579639916036957668,0.4463660172534640879849477,0.4894031457070529574785263,0.5312794640198945456580139,0.5718956462026340342838781,0.6111553551723932502488530,0.6489654712546573398577612,0.6852363130542332425635584,0.7198818501716108268489402,0.7528199072605318966118638,0.7839723589433414076102205,0.8132653151227975597419233,0.8406292962525803627516915,0.8659993981540928197607834,0.8893154459951141058534040,0.9105221370785028057563807,0.9295691721319395758214902,0.9464113748584028160624815,0.9610087996520537189186141,0.9733268277899109637418535,0.9833362538846259569312993,0.9910133714767443207393824,0.9963401167719552793469245,0.9993050417357721394569056]
# w64 = [0.0486909570091397203833654,0.0485754674415034269347991,0.0483447622348029571697695,0.0479993885964583077281262,0.0475401657148303086622822,0.0469681828162100173253263,0.0462847965813144172959532,0.0454916279274181444797710,0.0445905581637565630601347,0.0435837245293234533768279,0.0424735151236535890073398,0.0412625632426235286101563,0.0399537411327203413866569,0.0385501531786156291289625,0.0370551285402400460404151,0.0354722132568823838106931,0.0338051618371416093915655,0.0320579283548515535854675,0.0302346570724024788679741,0.0283396726142594832275113,0.0263774697150546586716918,0.0243527025687108733381776,0.0222701738083832541592983,0.0201348231535302093723403,0.0179517157756973430850453,0.0157260304760247193219660,0.0134630478967186425980608,0.0111681394601311288185905,0.0088467598263639477230309,0.0065044579689783628561174,0.0041470332605624676352875,0.0017832807216964329472961]

# # 96 nodes
# x96 = [0.0162767448496029695791346,0.0488129851360497311119582,0.0812974954644255589944713,0.1136958501106659209112081,0.1459737146548969419891073,0.1780968823676186027594026,0.2100313104605672036028472,0.2417431561638400123279319,0.2731988125910491414872722,0.3043649443544963530239298,0.3352085228926254226163256,0.3656968614723136350308956,0.3957976498289086032850002,0.4254789884073005453648192,0.4547094221677430086356761,0.4834579739205963597684056,0.5116941771546676735855097,0.5393881083243574362268026,0.5665104185613971684042502,0.5930323647775720806835558,0.6189258401254685703863693,0.6441634037849671067984124,0.6687183100439161539525572,0.6925645366421715613442458,0.7156768123489676262251441,0.7380306437444001328511657,0.7596023411766474987029704,0.7803690438674332176036045,0.8003087441391408172287961,0.8194003107379316755389996,0.8376235112281871214943028,0.8549590334346014554627870,0.8713885059092965028737748,0.8868945174024204160568774,0.9014606353158523413192327,0.9150714231208980742058845,0.9277124567223086909646905,0.9393703397527552169318574,0.9500327177844376357560989,0.9596882914487425393000680,0.9683268284632642121736594,0.9759391745851364664526010,0.9825172635630146774470458,0.9880541263296237994807628,0.9925439003237626245718923,0.9959818429872092906503991,0.9983643758631816777241494,0.9996895038832307668276901]
# w96 = [0.0325506144923631662419614,0.0325161187138688359872055,0.0324471637140642693640128,0.0323438225685759284287748,0.0322062047940302506686671,0.0320344562319926632181390,0.0318287588944110065347537,0.0315893307707271685580207,0.0313164255968613558127843,0.0310103325863138374232498,0.0306713761236691490142288,0.0302999154208275937940888,0.0298963441363283859843881,0.0294610899581679059704363,0.0289946141505552365426788,0.0284974110650853856455995,0.0279700076168483344398186,0.0274129627260292428234211,0.0268268667255917621980567,0.0262123407356724139134580,0.0255700360053493614987972,0.0249006332224836102883822,0.0242048417923646912822673,0.0234833990859262198422359,0.0227370696583293740013478,0.0219666444387443491947564,0.0211729398921912989876739,0.0203567971543333245952452,0.0195190811401450224100852,0.0186606796274114673851568,0.0177825023160452608376142,0.0168854798642451724504775,0.0159705629025622913806165,0.0150387210269949380058763,0.0140909417723148609158616,0.0131282295669615726370637,0.0121516046710883196351814,0.0111621020998384985912133,0.0101607705350084157575876,0.0091486712307833866325846,0.0081268769256987592173824,0.0070964707911538652691442,0.0060585455042359616833167,0.0050142027429275176924702,0.0039645543384446866737334,0.0029107318179349464084106,0.0018539607889469217323359,0.0007967920655520124294381]

# 128 nodes
x128 = [0.0122236989606157641980521,0.0366637909687334933302153,0.0610819696041395681037870,0.0854636405045154986364980,0.1097942311276437466729747,0.1340591994611877851175753,0.1582440427142249339974755,0.1823343059853371824103826,0.2063155909020792171540580,0.2301735642266599864109866,0.2538939664226943208556180,0.2774626201779044028062316,0.3008654388776772026671541,0.3240884350244133751832523,0.3471177285976355084261628,0.3699395553498590266165917,0.3925402750332674427356482,0.4149063795522750154922739,0.4370245010371041629370429,0.4588814198335521954490891,0.4804640724041720258582757,0.5017595591361444642896063,0.5227551520511754784539479,0.5434383024128103634441936,0.5637966482266180839144308,0.5838180216287630895500389,0.6034904561585486242035732,0.6228021939105849107615396,0.6417416925623075571535249,0.6602976322726460521059468,0.6784589224477192593677557,0.6962147083695143323850866,0.7135543776835874133438599,0.7304675667419088064717369,0.7469441667970619811698824,0.7629743300440947227797691,0.7785484755064119668504941,0.7936572947621932902433329,0.8082917575079136601196422,0.8224431169556438424645942,0.8361029150609068471168753,0.8492629875779689691636001,0.8619154689395484605906323,0.8740527969580317986954180,0.8856677173453972174082924,0.8967532880491581843864474,0.9073028834017568139214859,0.9173101980809605370364836,0.9267692508789478433346245,0.9356743882779163757831268,0.9440202878302201821211114,0.9518019613412643862177963,0.9590147578536999280989185,0.9656543664319652686458290,0.9717168187471365809043384,0.9771984914639073871653744,0.9820961084357185360247656,0.9864067427245862088712355,0.9901278184917343833379303,0.9932571129002129353034372,0.9957927585349811868641612,0.9977332486255140198821574,0.9990774599773758950119878,0.9998248879471319144736081]
w128 = [0.0244461801962625182113259,0.0244315690978500450548486,0.0244023556338495820932980,0.0243585572646906258532685,0.0243002001679718653234426,0.0242273192228152481200933,0.0241399579890192849977167,0.0240381686810240526375873,0.0239220121367034556724504,0.0237915577810034006387807,0.0236468835844476151436514,0.0234880760165359131530253,0.0233152299940627601224157,0.0231284488243870278792979,0.0229278441436868469204110,0.0227135358502364613097126,0.0224856520327449668718246,0.0222443288937997651046291,0.0219897106684604914341221,0.0217219495380520753752610,0.0214412055392084601371119,0.0211476464682213485370195,0.0208414477807511491135839,0.0205227924869600694322850,0.0201918710421300411806732,0.0198488812328308622199444,0.0194940280587066028230219,0.0191275236099509454865185,0.0187495869405447086509195,0.0183604439373313432212893,0.0179603271850086859401969,0.0175494758271177046487069,0.0171281354231113768306810,0.0166965578015892045890915,0.0162550009097851870516575,0.0158037286593993468589656,0.0153430107688651440859909,0.0148731226021473142523855,0.0143943450041668461768239,0.0139069641329519852442880,0.0134112712886163323144890,0.0129075627392673472204428,0.0123961395439509229688217,0.0118773073727402795758911,0.0113513763240804166932817,0.0108186607395030762476596,0.0102794790158321571332153,0.0097341534150068058635483,0.0091830098716608743344787,0.0086263777986167497049788,0.0080645898904860579729286,0.0074979819256347286876720,0.0069268925668988135634267,0.0063516631617071887872143,0.0057726375428656985893346,0.0051901618326763302050708,0.0046045842567029551182905,0.0040162549837386423131943,0.0034255260409102157743378,0.0028327514714579910952857,0.0022382884309626187436221,0.0016425030186690295387909,0.0010458126793403487793129,0.0004493809602920903763943]

standard_normal = torch.distributions.Normal(torch.tensor([0]).to(device), torch.tensor([1]).to(device))
dt = 1/250


class HestonModel:
    
    def __init__(self):
         
        u = torch.tensor(x128, dtype=torch.float64).to(device).reshape(-1,1)
        w = torch.tensor(w128, dtype=torch.float64).to(device).reshape(-1,1)
        
        self.quad = {'u': u,
                     'w': w}
    
    def pricing(self, KT, S0, r, params):
        
        v0, theta, rho, kappa, sigma = params
        K = KT[:,0].reshape(1,-1)
        T = KT[:,1].reshape(1,-1)
        S0 = torch.tensor(S0)
        u, w = self.quad.values()
        num_nodes = u.shape[0]
            
        # From the paper, their code uses 200 as an upper bound with 64 nodes
        u_max = 200
        u = torch.cat((u,-u), dim=0)
        u = 0.5 * u_max * (u + 1) # Rescaling to quadrature domain
        u = torch.cat((u-1j,u), dim=0)
        w = 0.5 * u_max * w

        # Required constants for the characteristic function
        xi = kappa - sigma * rho * u * 1j
        d = torch.sqrt(xi**2 + sigma**2 * u * (u + 1j))
        A1 = u * (u + 1j) * torch.sinh(d * T / 2)
        A2 = d * torch.cosh(d * T/ 2) + xi * torch.sinh(d * T/ 2)
        A = A1 / A2
        D = torch.log(d) + (kappa - d) * T / 2 - torch.log((d + xi) / 2 + (d - xi) * torch.exp(-d * T) / 2)
        
        # Characteristic function
        char_func = torch.exp(1j * u * (torch.log(S0) + r * T) - T * kappa * theta * rho * u * 1j / sigma - v0 * A + 2 * kappa * theta * D / sigma**2)
        
        offset = 0.5 * (S0 - torch.exp(-r * T) * K)
        
        # Gauss-Legendre quadrature
        integrand = torch.real((K**(-u * 1j) / (u * 1j)) * char_func)
        integrand = integrand[:2*num_nodes,:] - K * integrand[2*num_nodes:,:]
        integrand = w.reshape(-1,1) * (integrand[:num_nodes,:] + integrand[num_nodes:,:])
        integrand = torch.sum(integrand, dim=0) * torch.exp(-r * T.reshape(-1,1)).reshape(1, -1) / np.pi
        
        return (offset + integrand).T
    
    def jacobian(self, KT, S0, r, params):
        
        v0, theta, rho, kappa, sigma = params
        K = KT[:,0].reshape(1,-1)
        T = KT[:,1].reshape(1,-1)
        S0 = torch.tensor(S0)
        u, w = self.quad.values()
        num_nodes = u.shape[0]
            
        # From the paper, their code uses 200 as an upper bound with 64 nodes
        u_max = 200
        u = torch.cat((u,-u), dim=0)
        u = 0.5 * u_max * (u + 1) # Rescaling to quadrature domain
        u = torch.cat((u-1j,u), dim=0)
        w = 0.5 * u_max * w

        # Required constants for the characteristic function
        xi = kappa - sigma * rho * u * 1j
        d = torch.sqrt(xi**2 + sigma**2 * u * (u + 1j))
        cosh = torch.cosh(d * T / 2)
        sinh = torch.sinh(d * T / 2)
        A1 = u * (u + 1j) * sinh
        A2 = d * cosh + xi * sinh
        A = A1 / A2
        D = torch.log(d) + (kappa - d) * T / 2 - torch.log((d + xi) / 2 + (d - xi) * torch.exp(-d * T) / 2)
        B = torch.exp(D)
        
        # Characteristic function
        char_func = torch.exp(1j * u * (torch.log(S0) + r * T) - T * kappa * theta * rho * u * 1j / sigma - v0 * A + 2 * kappa * theta * D / sigma**2)
        
        # Analytical gradient components
        dd_drho = -xi * sigma * u * 1j / d
        dA2_drho = -sigma * (2 + T * xi) * u * 1j / 2 / d * (xi * cosh + d * sinh)
        dB_drho = torch.exp(kappa * T / 2) * (dd_drho - d * dA2_drho / A2) / A2
        dA1_drho = -1j * u**2 * (u + 1j) * T * xi * sigma * cosh / 2 / d
        dA_drho = (dA1_drho - A * dA2_drho) / A2
        dB_dkappa = 1j * dB_drho / sigma / u + T * B / 2
        dd_dsigma = (rho / sigma - 1 / xi) * dd_drho + sigma * u**2 / d
        dA1_dsigma = u * (u + 1j) * T * dd_dsigma * cosh / 2
        dA2_dsigma = rho * dA2_drho / sigma - (2 + T * xi) * dA1_drho / (1j * u * T * xi) + sigma * T * A1 / 2
        dA_dsigma = (dA1_dsigma - A * dA2_dsigma) / A2
        
        # Gradient vector of the characteristic function wrt [v0,theta,rho,kappa,sigma]
        h1 = -A
        h2 = 2 * kappa * D / sigma**2 - T * kappa * rho * u * 1j / sigma
        h3 = -v0 * dA_drho + 2 * kappa * theta * (dd_drho - d * dA2_drho / A2) / sigma**2 / d - T * kappa * theta * u * 1j / sigma
        h4 = v0 * dA_drho / sigma / u / 1j + 2 * theta * D / sigma**2 + 2 * kappa * theta * dB_dkappa / sigma**2 / B - T * theta * rho * u * 1j / sigma
        h5 = -v0 * dA_dsigma - 4 * kappa * theta * D / sigma**3 + 2 * kappa * theta * (dd_dsigma - d * dA2_dsigma / A2) / sigma**2 / d + T * kappa * theta * rho * u * 1j / sigma**2
        grad_vec = torch.stack([h1, h2, h3, h4, h5], dim=2)
        
        # Gauss-Legendre quadrature
        integrand = torch.real((K**(-u * 1j) / (u * 1j)).unsqueeze(-1) * char_func.unsqueeze(-1) * grad_vec)
        integrand = integrand[:2*num_nodes,:,:] - K.unsqueeze(-1) * integrand[2*num_nodes:,:,:]
        integrand = w.reshape(-1,1,1) * (integrand[:num_nodes,:,:] + integrand[num_nodes:,:,:])
        integrand = torch.sum(integrand, dim=0) * torch.exp(-r * T.reshape(-1,1)) / np.pi
        
        return integrand

    def levmarq(self, KT_price, S0, rate, max_iter=1000):
        
        # Order is [v0,theta,rho,kappa,sigma]
        params = torch.tensor([[0.5],
                                [0.5],
                                [-0.6],
                                [2.0],
                                [0.5]], dtype=torch.float64).to(device)
        
        KT = KT_price[:,:2]
        T = KT_price[:,1].reshape(1,-1)
        price = KT_price[:,2].reshape(-1,1)
        fun = lambda p: self.pricing(KT, S0, rate, p) - price
        r = fun(params)
        r_norm = 0.5 * r.T @ r
        j = self.jacobian(KT, S0, rate, params)
        hess = j.T @ j
        mu = torch.max(T) * torch.max(torch.diag(hess))
        differential = j.T @ r
        D = torch.eye(j.shape[1]).to(device)
        count = 0
        
        for _ in range(max_iter):
            
            h = -torch.linalg.solve(hess + mu * D, differential)
            r_h = fun(params + h)
            r_h_norm = 0.5 * r_h.T @ r_h
            dL = h.T @ (mu * h - differential)
            dF = r_norm - r_h_norm
            
            if dL > 0 and dF > 0:
                params = params + h
                r = r_h
                j = self.jacobian(KT, S0, rate, params)
                differential = j.T @ r_h
                hess = j.T @ j
                mu = mu / 2
                count += 1
            else:
                mu = mu * 2
            
            if torch.norm(r) < 1e-10 \
                or torch.max(torch.abs(differential)) < 1e-10 \
                or torch.norm(h) / torch.norm(params - h) < 1e-10:
                    break
        
        return params, count
  
    def calibrate(self, KT_price, S0, rate):
        
        # Training method is based on the Levenberg-Marquardt method, but with the Jacobian calculated analytically
        # Paper is found at https://arxiv.org/pdf/1511.08718
        
        # Dataset
        KT_price = KT_price.to(device).reshape(-1,3)
        
        return self.levmarq(KT_price, S0, rate)

    def MLE(self, starting_date=0):

        starting_date += 411

        def preprocess_stock_data(df):
            """Convert dates, calculate Garman-Klass variance, and take log of stock prices."""
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.date
            return df[['Date', 'Open', 'High', 'Low', 'Close', 'log_return']]

        """Load stock data for UNH, PFE, MRK, and interest rates, preprocess, and merge."""
        # Load stock data
        df1 = preprocess_stock_data(pd.read_csv(os.getcwd() + "/datasets/UNH.csv"))
        df2 = preprocess_stock_data(pd.read_csv(os.getcwd() + "/datasets/PFE.csv"))
        df3 = preprocess_stock_data(pd.read_csv(os.getcwd() + "/datasets/MRK.csv"))

        # Rename columns for clarity
        df1.columns = ['Date', 'Open_UNH', 'High_UNH', 'Low_UNH', 'Close_UNH', 'log_return_UNH']
        df2.columns = ['Date', 'Open_PFE', 'High_PFE', 'Low_PFE', 'Close_PFE', 'log_return_PFE']
        df3.columns = ['Date', 'Open_MRK', 'High_MRK', 'Low_MRK', 'Close_MRK', 'log_return_MRK']

        # Merge stock data by Date
        stock_df = df1.merge(df2, on='Date').merge(df3, on='Date').iloc[:starting_date]

        # Load and preprocess interest rates
        rates = pd.read_csv(os.getcwd() + "/datasets/DGS10.csv").replace('.', np.nan).ffill()
        rates.columns = ['Date', 'true_rate']
        rates['true_rate'] = pd.to_numeric(rates['true_rate']) / 100  # Convert to decimal
        rates['Date'] = pd.to_datetime(rates['Date']).dt.tz_localize(None).dt.date

        # Merge stock and rate data
        stock_df = stock_df.merge(rates, on='Date')
        
        # Reconstruct previous volatility using Yang-Zhang volatility over 20 days
        # Yang, D., & Zhang, Q. (2000). Drift‐Independent Volatility Estimation Based on High, Low, Open, and Close Prices. The Journal of Business, 73(3), 477–492. https://doi.org/10.1086/209650
        overnight_temp_UNH = np.log(stock_df['Open_UNH'] / stock_df['Close_UNH'].shift(1))
        overnight_temp_PFE = np.log(stock_df['Open_PFE'] / stock_df['Close_PFE'].shift(1))
        overnight_temp_MRK = np.log(stock_df['Open_MRK'] / stock_df['Close_MRK'].shift(1))

        overnight_UNH = (overnight_temp_UNH - overnight_temp_UNH.rolling(window=20).mean())**2
        overnight_PFE = (overnight_temp_PFE - overnight_temp_PFE.rolling(window=20).mean())**2
        overnight_MRK = (overnight_temp_MRK - overnight_temp_MRK.rolling(window=20).mean())**2
        
        open_close_temp_UNH = np.log(stock_df['Close_UNH'] / stock_df['Open_UNH'])
        open_close_temp_PFE = np.log(stock_df['Close_PFE'] / stock_df['Open_PFE'])
        open_close_temp_MRK = np.log(stock_df['Close_MRK'] / stock_df['Open_MRK'])

        open_close_UNH = (open_close_temp_UNH - open_close_temp_UNH.rolling(window=20).mean())**2
        open_close_PFE = (open_close_temp_PFE - open_close_temp_PFE.rolling(window=20).mean())**2
        open_close_MRK = (open_close_temp_MRK - open_close_temp_MRK.rolling(window=20).mean())**2

        rs_var_UNH = np.log(stock_df['High_UNH'] / stock_df['Open_UNH']) * np.log(stock_df['High_UNH'] / stock_df['Close_UNH']) + np.log(stock_df['Low_UNH'] / stock_df['Open_UNH']) * np.log(stock_df['Low_UNH'] / stock_df['Close_UNH'])
        rs_var_PFE = np.log(stock_df['High_PFE'] / stock_df['Open_PFE']) * np.log(stock_df['High_PFE'] / stock_df['Close_PFE']) + np.log(stock_df['Low_PFE'] / stock_df['Open_PFE']) * np.log(stock_df['Low_PFE'] / stock_df['Close_PFE'])
        rs_var_MRK = np.log(stock_df['High_MRK'] / stock_df['Open_MRK']) * np.log(stock_df['High_MRK'] / stock_df['Close_MRK']) + np.log(stock_df['Low_MRK'] / stock_df['Open_MRK']) * np.log(stock_df['Low_MRK'] / stock_df['Close_MRK'])

        k = 0.34 / (1.34 + 21 / 19)
        stock_df['yz_var_UNH'] = np.sqrt(overnight_UNH.rolling(window=20).mean() + k * open_close_UNH.rolling(window=20).mean() + (1 - k) * np.sqrt(rs_var_UNH.rolling(window=20).mean()))
        stock_df['yz_var_PFE'] = np.sqrt(overnight_PFE.rolling(window=20).mean() + k * open_close_PFE.rolling(window=20).mean() + (1 - k) * np.sqrt(rs_var_PFE.rolling(window=20).mean()))
        stock_df['yz_var_MRK'] = np.sqrt(overnight_MRK.rolling(window=20).mean() + k * open_close_MRK.rolling(window=20).mean() + (1 - k) * np.sqrt(rs_var_MRK.rolling(window=20).mean()))
        
        # MLE for CIR process (Kladivk, 2007) MAXIMUM LIKELIHOOD ESTIMATION OF THE COX-INGERSOLL-ROSS PROCESS: THE MATLAB IMPLEMENTATION
        data = stock_df[[f'yz_var_{stock}' for stock in ['UNH', 'PFE', 'MRK']] + [f'log_return_{stock}' for stock in ['UNH', 'PFE', 'MRK']] + ['true_rate']]
        data['vol_gain_UNH'] = data[f'yz_var_UNH'].diff()
        data['vol_gain_PFE'] = data[f'yz_var_PFE'].diff()
        data['vol_gain_MRK'] = data[f'yz_var_MRK'].diff()
        data = data.iloc[40:]
        
        # [v0,theta,rho,kappa,sigma]
        # Since all kappa are negative we estimate those separately, using eqn 22 of Dimitroff et al. (2011)
        # A Parsimonious Multi-Asset Heston Model: Calibration and Derivative Pricing
        kappa_UNH = ((dt * (len(data) - 1))**-1 * data['log_return_UNH']**2).sum()
        theta_UNH, sigma_UNH = self.get_params(data, kappa_UNH, 'UNH')
        theta_UNH, sigma_UNH = self.minimise(theta_UNH, kappa_UNH, sigma_UNH, data, 'UNH')
        rho_UNH, return_shock_UNH = self.fit_rho(theta_UNH, kappa_UNH, sigma_UNH, data, 'UNH')
        v0_UNH = data[f'yz_var_UNH'].iloc[-1]
        params_UNH = torch.tensor([v0_UNH, theta_UNH, rho_UNH, kappa_UNH, sigma_UNH])
        
        kappa_PFE = ((dt * (len(data) - 1))**-1 * data['log_return_PFE']**2).sum()
        theta_PFE, sigma_PFE = self.get_params(data, kappa_PFE, 'PFE')
        theta_PFE, sigma_PFE = self.minimise(theta_PFE, kappa_PFE, sigma_PFE, data, 'PFE')
        rho_PFE, return_shock_PFE = self.fit_rho(theta_PFE, kappa_PFE, sigma_PFE, data, 'UNH')
        v0_PFE = data[f'yz_var_PFE'].iloc[-1]
        params_PFE = torch.tensor([v0_PFE, theta_PFE, rho_PFE, kappa_PFE, sigma_PFE])
        
        kappa_MRK = ((dt * (len(data) - 1))**-1 * data['log_return_MRK']**2).sum()
        theta_MRK, sigma_MRK = self.get_params(data, kappa_PFE, 'MRK')
        theta_MRK, sigma_MRK = self.minimise(theta_MRK, kappa_MRK, sigma_MRK, data, 'MRK')
        rho_MRK, return_shock_MRK = self.fit_rho(theta_MRK, kappa_MRK, sigma_MRK, data, 'UNH')
        v0_MRK = data[f'yz_var_MRK'].iloc[-1]
        params_MRK = torch.tensor([v0_MRK, theta_MRK, rho_MRK, kappa_MRK, sigma_MRK])
        
        params = torch.stack((params_UNH, params_PFE, params_MRK), dim=0)
        
        # Assume covariances in volatility are only through the respective stock prices
        log_return_shocks_cov = torch.stack((return_shock_UNH, return_shock_PFE, return_shock_MRK), dim=0).cov()
        s12 = log_return_shocks_cov[0,1]
        s13 = log_return_shocks_cov[0,2]
        s23 = log_return_shocks_cov[1,2]
        r1 = rho_UNH
        r2 = rho_PFE
        r3 = rho_MRK
        covariance_matrix = torch.tensor([
                                          [1, r1,                   s12     , s12 * r2     , s13     , s13 * r3     ],
                                          [r1, 1,                   s12 * r1, s12 * r1 * r2, s13 * r1, s13 * r1 * r3],
                                          [s12     , s12 * r1     , 1, r2,                   s23     , s23 * r3     ],
                                          [s12 * r2, s12 * r1 * r2, r2, 1,                   s23 * r2, s23 * r2 * r3],
                                          [s13     , s13 * r1     , s23     , s23 * r2     , 1, r3                  ],
                                          [s13 * r3, s13 * r1 * r3, s23 * r3, s23 * r2 * r3, r3, 1                  ]
                                          ])
        
        # Regularise cov
        covariance_matrix = 0.5 * (covariance_matrix + covariance_matrix.T)
        v, q = torch.linalg.eigh(covariance_matrix)
        covariance_matrix = q @ torch.diag(torch.clamp(v, 1e-16)) @ q.T
        
        return params, covariance_matrix
        
    def get_params(self, data, kappa, stock):

        sqrt_var = np.sqrt(data[f'yz_var_{stock}'])
        observations = data[f'vol_gain_{stock}'] / sqrt_var / np.sqrt(dt)
        observations = observations + kappa * sqrt_var * np.sqrt(dt)
        left = np.sqrt(dt) / sqrt_var
        
        observations = torch.tensor(observations.values).reshape(-1,1)
        left = torch.tensor(left.values).reshape(-1,1)
        kt = left.T @ observations / torch.sum(left**2)
        theta0 = kt / kappa
        
        shocks = observations - kt * left
        sigma0 = shocks.std()
        
        return theta0.item(), sigma0.item()
        
    def minimise(self, theta0, kappa, sigma0, data, stock):
        
        var_data = data[f'yz_var_{stock}'].to_numpy()
        
        tks = optimize.minimize(self.loss, [theta0, sigma0], (kappa, var_data), 'SLSQP', bounds=[(0, None), (0, None)]).x
        
        return tks
        
    def loss(self, x, kappa, var_data):
        
        theta = x[0]
        sigma = x[1]
        
        c = 2 * kappa / (sigma**2 * (1 - np.exp(-kappa * dt)))
        u = c * var_data[:-1] * np.exp(-kappa * dt).item()
        v = c * var_data[1:]
        q = 2 * kappa * theta / sigma**2 - 1
        
        loss = u + v - 0.5 * q * np.log(v / u) - np.log(special.iv(q, 2 * np.sqrt(u * v))) # Removed constant terms
        loss = loss.sum()
        
        return loss
   
    def fit_rho(self, theta, kappa, sigma, data, stock):
        
        vol_shock = data[f'yz_var_{stock}'].diff() - kappa * (theta - data[f'yz_var_{stock}'].shift(1)) * dt
        vol_shock = vol_shock / sigma / np.sqrt(data[f'yz_var_{stock}'].shift(1)) / np.sqrt(dt)
        vol_shock = torch.tensor(vol_shock.iloc[40:].values)
        
        return_shock = data[f'log_return_{stock}'] - (data['true_rate'] - data[f'yz_var_{stock}'] / 2) * dt
        return_shock = return_shock / np.sqrt(data[f'yz_var_{stock}'].shift(1)) / np.sqrt(dt)
        return_shock = torch.tensor(return_shock.iloc[40:].values)
        
        rho = torch.stack((return_shock, vol_shock), dim=0).cov()[0,1]
        
        return rho.item(), return_shock
   
    def single_asset_path(self, u, uni, params, r, S0=1, dt=1/250, psic=1.5, gamma1=0.5, gamma2=0.5, verbose=False):
        
        # Brownian motion is generated separately and fed into the Heston model
        # This section is dedicated to path simulation by the quadratic exponential scheme
        # Paper is found at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=946405
        # An easy guide is available at https://medium.com/@alexander.tsoskounoglou/simulating-the-heston-model-with-quadratic-exponential-50cf2b1366b5
        
        # u is an array of normally drawn values in the form (path #, timestep #, stock/vol)
        assert 1 <= psic and psic <= 2
        assert u.shape[2] == 2
        assert u.shape == uni.shape
        
        numsteps = u.shape[1]
        v0, theta, rho, kappa, sigma = params
        S0 = torch.tensor(S0).to(device)
        r = torch.tensor(r).to(device)
        
        u = u.to(torch.float64)
        
        path = torch.empty((u.shape[0], numsteps + 1, 2), device=device)
        path[:,0,0] = torch.log(S0)
        path[:,0,1] = v0
        
        exp = torch.exp(-kappa * dt)
        v1 = sigma**2 * exp * (1 - exp) / kappa
        v2 = theta * sigma**2 * (1 - exp)**2 / (2 * kappa)
        v3 = exp
        v4 = theta * (1 - exp)
        
        # gamma values are somewhat arbitrary, but it is based on approximating the volatility integrated over the timestep
        # using a convex combination of V(t) and V(t+dt). The standard Euler simulation uses gamma1=1, gamma2=0, we use 0.5
        # for a more central approximation. The more sophisticated approach is by matching moments.
        x1 = gamma1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
        x2 = gamma2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
        x3 = gamma1 * dt * (1 - rho**2)
        x4 = gamma2 * dt * (1 - rho**2)
        
        A = x2 + 0.5 * x4
        
        
        with tqdm(total=u.shape[1], desc="Simulation Progress", disable = not verbose) as pbar:
            for i in range(numsteps):
                
                # Volatility update
                s2 = path[:,i,1] * v1 + v2
                m = path[:,i,1] * v3 + v4
                psi = s2 / m**2
                
                b2 = (2 / psi) - 1 + torch.sqrt((2 / psi) * ((2 / psi) - 1))
                a = m / (1 + b2)
                
                p = (psi - 1) / (psi + 1)
                beta = (1 - p) / m
            
                path[:,i+1,1] = torch.where(psi <= psic, \
                                            a * (torch.sqrt(b2) + u[:,i,1])**2, \
                                            torch.where(uni[:,i,1] <= p, \
                                                        0, \
                                                        torch.log((1 - p) / (1 - uni[:,i,1])) / beta))
                
                # log stock price update
                x0 = torch.where(psi <= psic,
                                 -A * b2 * a / (1 - 2 * A * a) + 0.5 * torch.log(1 - 2 * A *a),
                                 -torch.log(p + beta * (1 - p) / (beta - A))) - (x1 + 0.5 * x3) * path[:,i,1]
                
                path[:,i+1,0] = path[:,i,0] + r * dt + x0 + x1 * path[:,i,1] + x2 * path[:,i+1,1] + torch.sqrt(x3 * path[:,i,1] + x4 * path[:,i+1,1]) * u[:,i,0]
                
                pbar.update()
            
            pbar.close()
            
        return path
    
    def multi_asset_path(self, u, uni, params, r, corr, S0=None, dt=1/250, psic=1.5, gamma1=0.5, gamma2=0.5, n=3, verbose=False):
        
        # In this section we generalise the single asset path to multiple assets
        # This is based on the paper https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2729475
        # However, we make very strong assumptions;
        #   1. Every asset's stock price is related to every other by the same cross-covariance value
        #   2. Every asset's stock price is independent of other assets' volatility
        #   3. Every asset's volatility is independent of other assets' volatility
        # Therefore, the correlation between stock-volatility and volatility-volatility terms is
        # fully determined by the stock-stock covariance, with the exception of a stock and its 
        # corresponding volatility, which has correlation rho
        
        if S0 is None:
            S0 = torch.ones((n,1)).to(device)
        else:
            S0 = S0.to(device)
        r = torch.tensor(r).to(device)
        
        # u is an array of normally drawn values in the form (path #, asset #, timestep #, stock/vol)
        assert 1 <= psic and psic <= 2
        assert u.shape[1] == n
        assert u.shape[3] == 2
        assert u.shape == uni.shape
        assert params.shape == (n, 5)
        
        num_paths, num_assets, num_timesteps, _ = u.shape
        u = u.to(torch.float64)
        
        rho_arr = params[:,2].reshape(-1,1)
        s1 = torch.cat((torch.sqrt(1-rho_arr**2), rho_arr), dim=1)
        s2 = torch.cat((torch.zeros_like(rho_arr), torch.ones_like(rho_arr)), dim=1)
        Q_inv = torch.block_diag(*torch.stack((s1,s2), dim=1)).to(device)
        R = torch.linalg.cholesky(corr)
        decorrelator = Q_inv @ R
        u = u.transpose(1,2).reshape(num_paths, num_timesteps, -1, 1) # So that last 2 dimensions is a column vector of normal variables
        u = decorrelator @ u
        u = u.reshape(num_paths, num_timesteps, num_assets, 2).transpose(1,2)
        
        v0_arr = params[:,0].reshape(-1)
        theta_arr = params[:,1].reshape(-1)
        rho_arr = params[:,2].reshape(-1)
        kappa_arr = params[:,3].reshape(-1)
        sigma_arr = params[:,4].reshape(-1)
        
        path = torch.empty((u.shape[0], n, u.shape[2]+1, 2), device=device)
        path[:,:,0,0] = torch.log(S0).reshape((1,-1)).tile(num_paths, 1)
        path[:,:,0,1] = v0_arr.reshape((1,-1)).tile(num_paths, 1)
        
        exp = torch.exp(-kappa_arr * dt)
        v1 = sigma_arr**2 * exp * (1 - exp) / kappa_arr
        v2 = theta_arr * sigma_arr**2 * (1 - exp)**2 / 2 / kappa_arr
        v3 = exp
        v4 = theta_arr * (1 - exp)
        
        # gamma values are somewhat arbitrary, but it is based on approximating the volatility integrated over the timestep
        # using a convex combination of V(t) and V(t+dt). The standard Euler simulation uses gamma1=1, gamma2=0, we use 0.5
        # for a more central approximation. The more sophisticated approach is by matching moments.
        x1 = gamma1 * dt * (kappa_arr * rho_arr / sigma_arr - 0.5) - rho_arr / sigma_arr
        x2 = gamma2 * dt * (kappa_arr * rho_arr / sigma_arr - 0.5) + rho_arr / sigma_arr
        x3 = gamma1 * dt * (1 - rho_arr**2)
        x4 = gamma2 * dt * (1 - rho_arr**2)
        
        A = x2 + 0.5 * x4
        
        with tqdm(total=u.shape[2], desc="Simulation Progress", disable = not verbose) as pbar:
            for i in range(u.shape[2]):
                
                # Volatility update
                s2 = path[:,:,i,1] * v1 + v2
                m = path[:,:,i,1] * v3 + v4
                psi = s2 / m**2
                
                b2 = (2 / psi) - 1 + torch.sqrt((2 / psi) * ((2 / psi) - 1))
                a = m / (1 + b2)
                
                p = (psi - 1) / (psi + 1)
                beta = (1 - p) / m
            
                path[:,:,i+1,1] = torch.where(psi <= psic, \
                                              a * (torch.sqrt(b2) + u[:,:,i,1])**2, \
                                              torch.where(uni[:,:,i,1] <= p, \
                                                          0, \
                                                          torch.log((1 - p) / (1 - uni[:,:,i,1])) / beta))
                
                # log stock price update
                x0 = torch.where(psi <= psic,
                                 -A * b2 * a / (1 - 2 * A * a) + 0.5 * torch.log(1 - 2 * A *a),
                                 -torch.log(p + beta * (1 - p) / (beta - A))) - (x1 + 0.5 * x3) * path[:,:,i,1] + r * dt
                
                path[:,:,i+1,0] = path[:,:,i,0] + r * dt + x0 + x1 * path[:,:,i,1] + x2 * path[:,:,i+1,1] + torch.sqrt(x3 * path[:,:,i,1] + x4 * path[:,:,i+1,1]) * u[:,:,i,0]
                                
                pbar.update()
            pbar.close()
        
        return path
  
def calibration_test(S0, r):
    
    model = HestonModel()
    
    # True values    
    true_params = torch.tensor([[0.08],
                                [0.1],
                                [-0.8],
                                [3],
                                [0.25]], dtype=torch.float64).to(device)
    
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

    K = torch.tensor(karr, dtype=torch.float64).to(device).reshape(-1,1)
    T = torch.tensor(tarr, dtype=torch.float64).to(device).reshape(-1,1)
    KT = torch.cat((K,T), dim=1)
    
    true_price = model.pricing(KT, S0, r, true_params)
    KT_price = torch.cat((KT, true_price), dim=1)
                                
    # [v0,theta,rho,kappa,sigma]
    result, count = model.calibrate(KT_price, S0, r)
    print('Initial values: [0.5, 0.5, -0.6, 2.0, 0.5]')
    print('Found values:', result.flatten())
    print('Correct values: [0.08, 0.1, -0.8, 3, 0.25]')
    print('Number of iterations:', count)
    
    return result

def single_simulation_test():
    
    # [v0,theta,rho,kappa,sigma]
    params = torch.tensor([[0.0041],
                            [0.0055],
                            [-0.2940],
                            [5.0],
                            [0.30470]], dtype=torch.float64).to(device)
    
    numsteps = 100
    
    sobol_engine = torch.quasirandom.SobolEngine(2 * numsteps, True)
    paths = sobol_engine.draw(1000)
    # paths = torch.rand((10000,2*numsteps))
    paths = paths.reshape(-1, numsteps, 2)
    uniform = paths
    paths = paths * (1 - 2 * 1e-6) + 1e-6
    paths = standard_normal.icdf(paths)
    
    model = HestonModel()
    
    simulation = model.single_asset_path(paths, uniform, params, r=0.04, S0=1, dt=1/100, verbose=True)
    
    print(torch.mean(simulation[:,-1,0]))
    m = torch.mean(simulation[:,-1,1])
    print(m)
    print(torch.mean((simulation[:,-1,1])**2))
    print(torch.mean((simulation[:,-1,1])**3))
    print(torch.mean((simulation[:,-1,1])**4))
    
    plt.plot(simulation[:,:,0].T.detach().numpy())
    plt.title("Log price")
    plt.show()
    plt.clf()
    
    plt.plot(torch.exp(simulation[:,:,0]).T.detach().numpy())
    plt.title("Price")
    plt.show()
    plt.clf()
    
    plt.plot(torch.exp(simulation[:,:,1]).T.detach().numpy())
    plt.title("Volatility")
    plt.show()
    plt.clf()
    
    plt.hist(torch.sort(simulation[:,-1,1])[0].detach().numpy(), bins=100) 
    plt.axvline(torch.sort(simulation[:,-1,1])[0].mean().detach().numpy(), color='tab:orange', linestyle='dashed', linewidth=1, label=f'Mean: {torch.sort(simulation[:,-1,1])[0].mean().detach().numpy():.4f}')
    plt.legend()
    plt.show()
    
    return simulation
    
def multi_simulation_test(S0):
    
    numsteps = 100
    n=3
    a=10000
    
    if 2 * n * numsteps <= 21201:
        sobol_engine = torch.quasirandom.SobolEngine(2 * n * numsteps, True)
        paths = sobol_engine.draw(a)
        paths = paths.reshape(-1, n, numsteps, 2)
    else:
        paths = torch.rand((a, n, numsteps, 2))
    uniform = paths
    paths = paths * (1 - 2 * 1e-6) + 1e-6
    paths = standard_normal.icdf(paths)
    
    model = HestonModel()
    
    params = torch.tensor([[0.0041],
                            [0.0055],
                            [-0.2940],
                            [5.0],
                            [0.30470]], dtype=torch.float64).to(device).reshape(1,-1).tile(3,1)
    
    S0 = torch.tensor(S0)
    
    simulation = model.multi_asset_path(paths, uniform, params, S0=S0, r=0, corr=torch.eye(n*2), n=n, dt=1/100, verbose=True)
    
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
    
    s0 = torch.tensor([489.44, 36.65, 111])
    
    # S0 = 1000
    # r = 0.04
    # params = calibration_test(S0, r)
    # single_simulation_test()
    # multi_simulation_test([20, 347, 2000])
    
    model = HestonModel()
    params, cov = model.MLE()
    paths = model.multi_asset_path(torch.randn(256,3,250,2), torch.rand(256,3,250,2), params, 0.04, cov, s0, verbose=True)
    
    plt.plot(paths[:,0,:,0].T.detach().numpy())
    plt.title("Log price")
    plt.show()
    plt.clf()
    
    plt.plot(paths[:,0,:,1].T.detach().numpy())
    plt.title("Volatility")
    plt.show()
    plt.clf()
    
    