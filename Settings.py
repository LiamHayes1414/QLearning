from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import math
from scipy.optimize import root_scalar

#initialize
DemandFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

#Settings data (variables)
@dataclass(slots=True)
class Config:
    lags: int = 2
    firms: int = 2
    mrktsz: int = 1000
    position_options: list = field(default_factory=lambda: [1, 0])  # 1=Leader, 0=Follower
    mc: int = 0
    a: float = 0.1
    b: float = 0.5
    demand: Optional[DemandFn] = None
    gamelen: int = 100000000
    K: float = 1.0  # Chance for no innovation to occur
    delta: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9999999
    alpha = 0.35
    prices_count = 15

    # Initialize as empty lists (filled on init)
    invest_options: list = field(default_factory=list)
    price_options: list = field(default_factory=list)

    def __post_init__(self):
        if self.demand is None:
            self.demand = self.mult_nomial

        #calculate price options
        def target_function(p):
            formula_right_side = ((1 + self.b) * math.exp(-self.a * p) + 1) / self.a
            return formula_right_side - p

        #Dynamically find price interval based on monopoly price
        solution = root_scalar(target_function, bracket=[0, 100], method="brentq")

        self.price_options = np.linspace(0,
                                         solution.root*(1+(1/(self.prices_count-1))),
                                         self.prices_count).tolist()


        self.invest_options = np.linspace(0, 500, 5).tolist()

    def mult_nomial(self, prices:np.ndarray, Leader:np.ndarray):
        Prod_Attractiveness = np.exp(-self.a*prices)

        Leader_Multiplier = 1+ (self.b* Leader)
        Prod_Attractiveness = Prod_Attractiveness * Leader_Multiplier

        MarketDemand = np.sum(Prod_Attractiveness) + 1
    
        MarketShares = Prod_Attractiveness / MarketDemand

        return MarketShares
