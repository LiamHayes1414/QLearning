from itertools import product
from Firm import Firm as firm
from collections import deque
from Settings import Config
import numpy as np
import time
from tqdm import tqdm
from Visualize import plotting, plot_visit_counts_3d, leaderplots
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
GameLen = config.explorationlen
GameCap = config.caplen
K = config.K
delta = config.delta
mrktsz = config.mrktsz
epsilon_decay = config.epsilon_decay
learning_rate = config.learningrate


#Possible state and action vectors
Possible_States = list(product(*( [prices] * (firms * lags) + [positions] )))
Possible_Actions = list(product(prices, investments))

#Create firm instances
Firms = []
for f in range(firms):
    Firm = firm(Possible_States,Possible_Actions,config)
    Firms.append(Firm)

#State log
State_log = deque(maxlen=lags)
#Append initial state window
for lag in range(lags):
    State_log.append([1]*firms)

Industry_m = 1
Profits_log = np.empty((GameCap, firms+1))
Price_log = np.empty((GameCap, firms+1))
Invest_log = np.empty((GameCap, firms+1))
round = 0
Firm_Stationarity = [0]
Stationarity_Target = 100000
with tqdm(total=Stationarity_Target, desc="Tracking Stationarity") as pbar:
    while min(Firm_Stationarity) < Stationarity_Target:
        if round == GameCap:
            break

        epsilon = max((epsilon_decay*round) + 1,0)

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
            if config.firms >1:
                L_Best,F_Best = f.ProfitExpectations(State_log)

                Leader_Best.append(L_Best)
                Follower_Best.append(F_Best)
            elif firms == 1:
                L_Best = f.ProfitExpectations(State_log)
                Leader_Best.append(L_Best)


        #Innovation success probabilities
        if config.firms >1:
            MarketInnovation = np.sum(Investment_Actions) + K
            Firm_Probabilities = Investment_Actions/MarketInnovation

            LeaderBest = np.array(Leader_Best)
            FollowerBest = np.array(Follower_Best)

            ValueExpectations = (1-learning_rate)*Profit + learning_rate*delta*(Firm_Probabilities*LeaderBest + (1-Firm_Probabilities)*FollowerBest)
        elif config.firms == 1:
            LeaderBest = np.array(Leader_Best)
            #In monopoly firm is guaranteed to remain leader
            ValueExpectations = (1-learning_rate)*Profit + learning_rate*delta*LeaderBest

        #Update Q-Matrices given state and action choices
        Firm_Stationarity = []
        for i, f in enumerate(Firms, start=0):
            f.UpdateQ(CurrentState,(Prices[i],Investments[i]),ValueExpectations[i])
            Firm_Stationarity.append(f.Stationarity_Counter)



        #Draw next innovation leader - only applies when 2+ firms
        if config.firms >1:
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
        if 1 in Leadership: #no leader on first round(and maybe more if no one wins innovation)
            Profits_log[round] = np.append(Profit,Leadership.index(1))
            Price_log[round] = np.append(Price_Actions,Leadership.index(1))
            Invest_log[round] = np.append(Investment_Actions,Leadership.index(1))
        else:
            Profits_log[round] = np.append(Profit,None)
            Price_log[round] = np.append(Price_Actions,None)
            Invest_log[round] = np.append(Investment_Actions,None)
        pbar.n = int(min(Firm_Stationarity))
        pbar.refresh()
        round+=1

#Process results
plot_start = time.perf_counter()
plotting(Profits_log, Price_log, Invest_log,config)
leaderplots(Profits_log, Price_log, Invest_log, config)
plot_visit_counts_3d(Firms)
plot_elapsed = time.perf_counter() - plot_start

print(f"Plotting completed in {plot_elapsed:.2f} seconds")
print(f"Rounds: {round}")

for i, f in enumerate(Firms):
    config.Details[f'Firm{i+1} Visits'] = {"Min":np.min(f.visit_counts),"Max":np.max(f.visit_counts),"Avg":np.mean(f.visit_counts)}
config.Details['Rounds'] = round
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
   

