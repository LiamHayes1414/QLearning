from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import math
from scipy.optimize import root_scalar
from scipy.optimize import fsolve

#initialize
DemandFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

#Settings data (variables)
@dataclass(slots=True)
class Config:
    #Market Characteristics
    lags: int = 1
    firms: int = 2
    mrktsz: int = 1000
    gamelen: int = 10**8
    #Demand features
    mc: int = 1
    a: float = 0.1
    b: float = 0.5
    demand: Optional[DemandFn] = None
    K: float = 50  # Chance for no innovation to occur
    delta: float = 0.95
    #Learning parameters
    epsilon_decay: float = -1/(gamelen*0.75)
    learningrate = 0.25
    #State variables
    prices_count = 15
    investments_count = 5
    price_interval_margin = 0.02
    investment_interval_margin = 0.1
    position_options: list = field(default_factory=lambda: [1,0])  # 1=Leader, 0=Follower

    #Holder variables
    MonopolyP: float = None
    FollowerP:float = None
    LeaderP:float=None
    MonopolyX: float = None
    FollowerX:float = None
    LeaderX:float=None
    MonopolyProfit: float = None
    FollowerProfit:float = None
    LeaderProfit:float=None

    # Initialize as empty lists (filled on init)
    invest_options: list = field(default_factory=list)
    price_options: list = field(default_factory=list)

    #store key variables/info
    Details = {}

    def __post_init__(self):
        if self.demand is None:
            self.demand = self.mult_nomial

        
        num_followers = self.firms - 1 


        """PRICING_____________________________________________________________________________"""
        #Equilibrium

  
        def price_equations(vars):
            P, p = vars  # P is Leader, p is Follower
                    
            D = (num_followers * math.exp(-self.a * p)) + ((1 + self.b) * math.exp(-self.a * P)) + 1
                    
            #market shares
            s_L = ((1 + self.b) * math.exp(-self.a * P)) / D
            s_f = math.exp(-self.a * p) / D
                    
            #Follower and leader formula's equal to zero (proved on paper)
            eq_leader = self.mc + (1 / (self.a * (1 - s_L))) - P
            eq_follower = self.mc + (1 / (self.a * (1 - s_f))) - p
                    
            return [eq_leader, eq_follower]

        initial_guess = [self.mc + 10, self.mc + 10]
                
        # Run the numerical solver
        solution, info, ier, mesg = fsolve(price_equations, initial_guess, full_output=True)
                
        if ier == 1:
            leader_price, follower_price = solution

        #calculate price options
        def MonopolyPrice(p):
            #solve for p in given function (formula derived on paper)
            formula = ((self.mc*self.a) + (1+self.b)*math.exp(-self.a * p) + 1) / self.a 

            """
            Monopoly Price 
                 mc*a + (1+b)e^(-a*p) + 1
            p =  ------------------------
                            a
                            
            """
                 
            return formula - p
        
        #Dynamically find on monopoly price
        monopoly_solution = root_scalar(MonopolyPrice, bracket=[0, 100], method="brentq")
        monopoly_price = monopoly_solution.root


        self.price_options = np.linspace(follower_price * (1-self.price_interval_margin),
                                         monopoly_price*(1+self.price_interval_margin),
                                         self.prices_count).tolist()
        

        """INVESTMENT_____________________________________________________________________________ """

      
        D = (num_followers * math.exp(-self.a * follower_price)) + ((1 + self.b) * math.exp(-self.a * leader_price)) + 1
        Rl = (leader_price - self.mc) * ((1+self.b)*math.exp(-self.a * leader_price))/D
        Rf = (follower_price - self.mc) * (math.exp(-self.a*follower_price))/D
        #Scale revenue by fixed market size
        Rl = Rl * self.mrktsz
        Rf = Rf * self.mrktsz

        C = (1+
            self.delta*(
                (num_followers-1)/(4*num_followers)if num_followers != 0 else 0
                +(2*num_followers)
                +1
                )
            )
        def investment_equations(vars):
            xl, xf = vars
            N = self.delta*(Rl - xl - Rf + xf + self.K)

            LeaderEq = N/(4*C) -self.K - xl
            FollowerEq = N/(4*num_followers*C) -xf

            return [LeaderEq, FollowerEq]
            
        initial_guess = [self.K, self.K]
                
        # Run the numerical solver
        solution, info, ier, mesg = fsolve(investment_equations, initial_guess, full_output=True)
                
        if ier == 1:
            leader_investment, follower_investment = solution

        monopoly_investment = 0

        self.invest_options = np.linspace(monopoly_investment,
                                         follower_investment*(1+self.investment_interval_margin),
                                         self.investments_count).tolist()
        
        """PROFITS_____________________________________________________________________________ """
        #using variables from investment calculatinos above,Rl,Rf
        D = ((1 + self.b) * math.exp(-self.a * monopoly_price)) + 1
        monopoly_profits = (((monopoly_price - self.mc)*((1+self.b)*math.exp(-self.a * monopoly_price))/D)*self.mrktsz) - monopoly_investment

   
        leader_profits = Rl - leader_investment
        follower_profits = Rf - follower_investment
        
        #save all info
        #Prices
        self.Details['Monopoly Price'] = monopoly_price
        self.Details['Price Interval'] = self.price_options
        self.MonopolyP = monopoly_price
        self.Details['Leader Price'] = leader_price
        self.Details['Follower Price'] = follower_price
        self.LeaderP =  leader_price
        self.FollowerP = follower_price
        
        #Investment
        self.Details['Monopoly Investment'] = monopoly_investment
        self.Details['Investment Interval'] = self.invest_options
        self.MonopolyX = monopoly_investment
        self.Details['Leader Investment'] = leader_investment
        self.Details['Follower Investment'] = follower_investment
        self.LeaderX = leader_investment
        self.FollowerX = follower_investment
        
        #Profit
        self.Details['Monopoly Profit'] = monopoly_profits
        self.MonopolyProfit = monopoly_profits
        self.Details['Leader Profit'] = leader_profits
        self.Details['Follower Profit'] = follower_profits
        self.LeaderProfit = leader_profits
        self.FollowerProfit = follower_profits

    def mult_nomial(self, prices:np.ndarray, Leader:np.ndarray):
        Prod_Attractiveness = np.exp(-self.a*prices)

        Leader_Multiplier = 1+ (self.b* Leader)
        Prod_Attractiveness = Prod_Attractiveness * Leader_Multiplier

        MarketDemand = np.sum(Prod_Attractiveness) + 1
    
        MarketShares = Prod_Attractiveness / MarketDemand

        return MarketShares
