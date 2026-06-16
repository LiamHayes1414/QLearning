from itertools import product
from Firm import Firm as firm
from collections import deque
from Settings import Config
import numpy as np
import time
from tqdm import tqdm
from Visualize import plotting,plot_visit_counts_3d
import json
import random


#load settings
config = Config()
prices = config.price_options
firms = config.firms
lags = config.lags
positions = config.position_options
investments = config.invest_options
Demand = config.demand
mc = config.mc
GameLen = config.gamelen
K = config.K
delta = config.delta
mrktsz = config.mrktsz
epsilon_start = config.epsilon_start
epsilon_min = config.epsilon_min
epsilon_decay = config.epsilon_decay
learning_rate = config.learningrate


#Possible state and action vectors
Possible_States = list(product(*( [prices] * (firms * lags) + [positions] )))
Possible_Actions = list(product(prices, investments))

#Create heterogeneous firm instances
"""Manual update when more firms"""
Firm1 = firm(Possible_States,Possible_Actions,config)
Firm2 = firm(Possible_States,Possible_Actions,config)

Firms = [Firm1,Firm2]

#State log
State_log = deque(maxlen=lags)
#Append initial state window
for lag in range(lags):
    State_log.append([1]*firms)

Industry_m = 1
Profits_log = np.empty((GameLen, 2))
Price_log = np.empty((GameLen, 2))
Invest_log = np.empty((GameLen, 2))

for round in tqdm(range(GameLen)):

    epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** round))

    Prices = []
    Investments = []
    Leadership = []
    for f in Firms:
        Action = f.Action(State_log, epsilon)
        Prices.append(Action[0])
        Investments.append(Action[1])
        Leadership.append(f.Leader)

    #Save current state for updating the Q-Matrix
    CurrentState = list(State_log)

    #Save actioned prices into state log
    State_log.append(Prices)

    #Store in np arrays for vectorization
    Price_Actions = np.array(Prices)
    Investment_Actions = np.array(Investments)
    Leader = np.array(Leadership)

    MarketShares = Demand(Price_Actions,Leader) * mrktsz #for readability

    Profit = (Price_Actions - mc)*MarketShares - Investment_Actions

    #Calculate best possible outcomes next period 
    Leader_Best = []
    Follower_Best = []
    for f in Firms:
        L_Best,F_Best = f.ProfitExpectations(State_log)

        Leader_Best.append(L_Best)
        Follower_Best.append(F_Best)

    #Innovation success probabilities
    MarketInnovation = np.sum(Investment_Actions) + K
    Firm_Probabilities = Investment_Actions/MarketInnovation

    LeaderBest = np.array(Leader_Best)
    FollowerBest = np.array(Follower_Best)

    ValueExpectations = (1-learning_rate)*Profit + learning_rate*delta*(Firm_Probabilities*LeaderBest + (1-Firm_Probabilities)*FollowerBest)

    #Update Q-Matrices given state and action choices
    for i, f in enumerate(Firms, start=0):
        f.UpdateQ(CurrentState,(Prices[i],Investments[i]),ValueExpectations[i])


    #Draw next innovation leader
    NoWinnerProb = 1- np.sum(Firm_Probabilities)
    probabilities = np.append(Firm_Probabilities, NoWinnerProb)

    outcomes = list(range(0, firms)) + [None]
  
    winner = random.choices(outcomes, weights=probabilities, k=1)[0]
   
    if winner != None:
        Industry_m+=1
        for i, f in enumerate(Firms, start=0):
            f.Leader = 0
            if int(winner) == i:
                f.Leader = 1

    #Save profits for graphing 
    Profits_log[round] = Profit
    Price_log[round] = Price_Actions
    Invest_log[round] = Investment_Actions

#Process results
"""Manual update when more firms"""
plot_start = time.perf_counter()
plotting(Profits_log, Price_log, Invest_log,config)
plot_visit_counts_3d(Firm1.visit_counts,Firm2.visit_counts)
plot_elapsed = time.perf_counter() - plot_start

print(f"Plotting completed in {plot_elapsed:.2f} seconds")
print(f"Firm 1 Visits| Min:{np.min(Firm1.visit_counts)}, Max:{np.max(Firm1.visit_counts)}, Avg:{np.mean(Firm1.visit_counts)}")
print(f"Firm 2 Visits| Min:{np.min(Firm2.visit_counts)}, Max:{np.max(Firm2.visit_counts)}, Avg:{np.mean(Firm2.visit_counts)}")


config.Details['Firm1 Visits'] = {"Min":np.min(Firm1.visit_counts),"Max":np.max(Firm1.visit_counts),"Avg":np.mean(Firm1.visit_counts)}
config.Details['Firm2 Visits'] = {"Min":np.min(Firm2.visit_counts),"Max":np.max(Firm2.visit_counts),"Avg":np.mean(Firm2.visit_counts)}
   

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
with open("TrainingResults/Details.json", "w") as json_file:
    json.dump(config.Details, json_file, indent=4, cls=NumpyEncoder)
   

