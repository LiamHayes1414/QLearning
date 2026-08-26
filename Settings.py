from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import math
from scipy.optimize import root_scalar,fsolve
 
#initialize
DemandFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

#Settings data (variables)
@dataclass(slots=True)
class Config:
    #Market Characteristics
    lags: int = 1
    firms: int = 2
    mrktsz: int = 1000
    caplen:int = 10**9 #hard cap at 1B iterations
    startingM:int=1
    #Demand features
    mc: int = 1
    a: float = 0.1
    b: float = 0.5
    demand: Optional[DemandFn] = None
    K: float = 100  # Chance for no innovation to occur
    delta: float = 0.95
    #Learning parameters
    learningrate: float = 0.25
    #State variables
    prices_count = 15
    investments_count = 7
    price_interval_margin = 0.02
    investment_interval_margin = 0.1
    #based on above values     _States_                      _actions_             visit each ~X times
    explorationlen: int = (prices_count**(firms*lags)) * (prices_count*investments_count) * 100
    epsilon_decay: float = -1/(explorationlen)

    #Holder variables
    MonopolyP: float = None
    FollowerP:float = None
    LeaderP:float=None
    MonopolyX: float = None
    FollowerX:float = None
    LeaderX:float=None
    MonopolyLeaderProfit: float = None
    MonopolyFollowerProfit: float = None
    FollowerProfit:float = None
    LeaderProfit:float=None
    position_options:list = None
    CS_Theory:float=0
    CS_Real:float=0
    LeaderMrktShr: float = None
    FollowerMrktShr: float = None
    MonopolyFollowerMrktShr: float = None
    MonopolyLeaderMrktShr: float = None

    # Initialize as empty lists (filled on init)
    invest_options: list = field(default_factory=list)
    price_options: list = field(default_factory=list)

    #store key variables/info
    Details = {}

    def __post_init__(self):
        if self.demand is None:
            self.demand = self.mult_nomial

        if self.firms >1:
            self.position_options: list = [1,0]  # 1=Leader, 0=Follower
        elif self.firms == 1:
            self.position_options: list =  [1] #monopoly always leader
        num_followers = self.firms - 1 


        """PRICING_____________________________________________________________________________"""
        #Equilibrium

        if self.firms > 1:
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
            else:
                return ValueError("Price Solution Not found")

        #calculate price options
        def MonopolyPrice(p):
            #solve for p in given function (formula derived on paper)
            formula = ((1+ (num_followers+1+self.b)*math.exp(-self.a * p))/self.a) + self.mc

            """
            Monopoly Price 
                 1+ (n+1+b)e^(-a*p)
            p =  ------------------ + mc
                         a       
            """
                 
            return formula - p
        
        #Dynamically find on monopoly price
        monopoly_solution = root_scalar(MonopolyPrice, bracket=[0, 100], method="brentq")
        monopoly_price = monopoly_solution.root

        if self.firms >1:
            self.price_options = np.linspace(follower_price * (1-self.price_interval_margin),monopoly_price*(1+self.price_interval_margin),self.prices_count).tolist()
        elif self.firms ==1:
            margin = self.price_interval_margin *5
            self.price_options = np.linspace(monopoly_price * (1-margin),monopoly_price*(1+margin),self.prices_count).tolist()

        #Round options
        self.price_options = np.round(self.price_options,2)

        
        """INVESTMENT_____________________________________________________________________________ """

        if self.firms >1:
            D = (num_followers * math.exp(-self.a * follower_price)) + ((1 + self.b) * math.exp(-self.a * leader_price)) + 1
            

            s_L = (((1 + self.b) * math.exp(-self.a * leader_price)) / D) *self.mrktsz
            s_f = (math.exp(-self.a * follower_price) / D) *self.mrktsz
            #Profit gap
            Profit_Gap = ((leader_price - self.mc)*s_L) - ((follower_price - self.mc)*s_f) 

            leader_investment = max(((Profit_Gap + self.K)*(self.delta * num_followers)/(num_followers+1)**2) - self.K,0)
            follower_investment = max(((Profit_Gap + self.K)*(self.delta * num_followers)/(num_followers+1)**2),0)

        monopoly_investment = 0
        if self.firms >1:
            self.invest_options = np.linspace(monopoly_investment,follower_investment*(1+self.investment_interval_margin),self.investments_count).tolist()
        elif self.firms == 1:
            self.invest_options = np.linspace(monopoly_investment,self.K*(1+self.investment_interval_margin),self.investments_count).tolist()

        if self.investments_count == 1: self.invest_options = [0]

        #Round options
        self.invest_options = np.round(self.invest_options,2)

        """PROFITS_____________________________________________________________________________ """
        if self.firms>1:
            D = (num_followers * math.exp(-self.a * follower_price)) + ((1 + self.b) * math.exp(-self.a * leader_price)) + 1
            MonopolyD = (num_followers * math.exp(-self.a * monopoly_price)) + ((1 + self.b) * math.exp(-self.a * monopoly_price)) + 1 

            #Market shares
            monopoly_follower_marketshare = (math.exp(-self.a * monopoly_price)/MonopolyD)*self.mrktsz
            leader_marketshare = ((1+self.b)*math.exp(-self.a * leader_price)/D)*self.mrktsz
            follower_marketshare = (math.exp(-self.a * follower_price)/D)*self.mrktsz

            monopoly_follower_profits = ((monopoly_price - self.mc)*monopoly_follower_marketshare) - monopoly_investment

        else:
            MonopolyD =  ((1 + self.b) * math.exp(-self.a * monopoly_price)) + 1

        #Market Shares - monopoly leader
        monopoly_leader_marketshare = ((1+self.b)*math.exp(-self.a * monopoly_price)/MonopolyD)*self.mrktsz
        
        monopoly_leader_profits = ((monopoly_price - self.mc)*monopoly_leader_marketshare) - monopoly_investment
        

        if self.firms>1:
            if self.investments_count == 1:
                leader_profits = (leader_price - self.mc)*leader_marketshare
                follower_profits = (follower_price - self.mc)*follower_marketshare
            else:
                leader_profits = ((leader_price - self.mc)*leader_marketshare) - leader_investment
                follower_profits = ((follower_price - self.mc)*follower_marketshare) - follower_investment
        
            #save all info
            #Prices
            self.Details['Leader Price'] = leader_price
            self.Details['Follower Price'] = follower_price
            self.LeaderP =  leader_price
            self.FollowerP = follower_price
            #Investment
            self.Details['Leader Investment'] = leader_investment
            self.Details['Follower Investment'] = follower_investment
            self.LeaderX = leader_investment
            self.FollowerX = follower_investment
            #Profit
            self.Details['Leader Profit'] = leader_profits
            self.Details['Follower Profit'] = follower_profits
            self.LeaderProfit = leader_profits
            self.FollowerProfit = follower_profits
            self.Details['Monopoly Follower Profit'] = monopoly_follower_profits
            self.MonopolyFollowerProfit = monopoly_follower_profits
            #Market Shares
            self.Details['Leader MarketShr'] = leader_marketshare
            self.Details['Follower Marketshr'] = follower_marketshare
            self.LeaderMrktShr = leader_marketshare
            self.FollowerMrktShr = follower_marketshare
            self.Details['Monopoly Follower Marketshr'] = monopoly_follower_marketshare
            self.MonopolyFollowerMrktShr = monopoly_follower_marketshare

        #Prices
        self.Details['Monopoly Price'] = monopoly_price
        self.Details['Price Interval'] = self.price_options
        self.MonopolyP = monopoly_price
        #Investment
        self.Details['Monopoly Investment'] = monopoly_investment
        self.Details['Investment Interval'] = self.invest_options
        self.MonopolyX = monopoly_investment
        #Profit
        self.Details['Monopoly Leader Profit'] = monopoly_leader_profits
        self.MonopolyLeaderProfit = monopoly_leader_profits
        #Market Shares
        self.Details['Monopoly Leader Marketshr'] = monopoly_leader_marketshare
        self.MonopolyLeaderMrktShr = monopoly_leader_marketshare

    def mult_nomial(self, prices:np.ndarray, Leader:np.ndarray):
        Prod_Attractiveness = np.exp(-self.a*prices)

        Leader_Multiplier = 1+ (self.b* Leader)
        Prod_Attractiveness = Prod_Attractiveness * Leader_Multiplier

        MarketDemand = np.sum(Prod_Attractiveness) + 1
    
        MarketShares = Prod_Attractiveness / MarketDemand

        return MarketShares
    
    def ConsumerSurplus(self,prices:np.ndarray,round:int,IndustryM:int,LeaderIndx):
        num_followers = self.firms - 1
        #check if there is a leader currently
        LeaderExists = np.any(LeaderIndx == 1)
        LeaderGamePrice = None
        if LeaderExists:
            LeaderGamePrice = prices[np.argmax(LeaderIndx)]

        FollowerGamePrices = prices[LeaderIndx ==0]  

        if self.firms>1:
            mu = (self.LeaderX + num_followers*self.FollowerX)/(self.LeaderX + num_followers*self.FollowerX + self.K)
            ExpectedM = self.startingM + mu * round  

            CS_static_theory = (
                math.log(
                    1
                    + (1 + self.b) * math.exp(-self.a * self.LeaderP)
                    + num_followers * math.exp(-self.a * self.FollowerP)
                ) / self.a
            )

            CS_static_real = (
                math.log(
                    1
                    + ((1 + self.b) * math.exp(-self.a * LeaderGamePrice) if LeaderExists else 0) 
                    + np.sum(np.exp(-self.a * FollowerGamePrices))
                ) / self.a
            )
            
        else: #monopoly
            ExpectedM = 0

            CS_static_theory = (
                math.log(
                    1
                    + (1 + self.b) * math.exp(-self.a * self.MonopolyP)
                ) / self.a
            )

            CS_static_real = (math.log(1+ (1 + self.b) * math.exp(-self.a * LeaderGamePrice)) / self.a)
        
        return CS_static_theory,CS_static_real,ExpectedM
