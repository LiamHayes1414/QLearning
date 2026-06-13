from itertools import product
from Firm import Firm as firm
from collections import deque
from Settings import Config
import numpy as np
import time
from tqdm import tqdm
from Visualize import plotting,plot_visit_counts_3d

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
learning_rate = config.alpha

#Possible state and action vectors
Possible_States = list(product(*( [prices] * (firms * lags) + [positions] )))
Possible_Actions = list(product(prices, investments))

#Create heterogeneous firm instances
Firm1 = firm(Possible_States,Possible_Actions)
Firm2 = firm(Possible_States,Possible_Actions)

#State log
State_log = deque(maxlen=lags)
State_log.append([1, 1])
#State_log.append([3, 4])

Industry_m = 1
Profits_log = np.empty((GameLen, 2))
Price_log = np.empty((GameLen, 2))
Invest_log = np.empty((GameLen, 2))

for round in tqdm(range(GameLen)):

    epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** round))

    Price1,Invest1 = Firm1.Action(State_log, epsilon)
    Price2,Invest2 = Firm2.Action(State_log, epsilon)

    #Save current state for updating the Q-Matrix
    CurrentState = list(State_log)

    #Save actioned prices into state log
    State_log.append([Price1, Price2])

    #Store in np arrays for vectorization
    Price_Actions = np.array([Price1,Price2])
    Investment_Actions = np.array([Invest1,Invest2])
    Leader = np.array([Firm1.Leader,Firm2.Leader])

    MarketShares = Demand(Price_Actions,Leader) * mrktsz #for readability

    Profit = (Price_Actions - mc)*MarketShares - Investment_Actions

    #Calculate best possible outcomes next period 
    Leader_Best1,Follower_Best1 = Firm1.ProfitExpectations(State_log)
    Leader_Best2,Follower_Best2 = Firm2.ProfitExpectations(State_log)

    #Innovation success probabilities
    MarketInnovation = np.sum(Investment_Actions) + K
    Firm_Probabilities = Investment_Actions/MarketInnovation

    LeaderBest = np.array([Leader_Best1,Leader_Best2])
    FollowerBest = np.array([Follower_Best1,Follower_Best2])

    ValueExpectations = (1-learning_rate)*Profit + learning_rate*delta*(Firm_Probabilities*LeaderBest + (1-Firm_Probabilities)*FollowerBest)

    #Update Q-Matrices given state and action choices
    Firm1.UpdateQ(CurrentState,(Price1,Invest1),ValueExpectations[0])
    Firm2.UpdateQ(CurrentState,(Price2,Invest2),ValueExpectations[1])

    #Draw next innovation leader
    NoWinnerProb = 1- np.sum(Firm_Probabilities)
    probabilities = np.append(Firm_Probabilities, NoWinnerProb)
    outcomes = ["1","2","None"]

    winner = np.random.choice(outcomes, p=probabilities)

    if winner == "1" or winner == "2":
        Industry_m+=1
        if winner == "1":
            Firm1.Leader = 1
            Firm2.Leader = 0
        if winner == "2":
            Firm1.Leader = 0
            Firm2.Leader=1

    #Save profits for graphing 
    Profits_log[round] = Profit
    Price_log[round] = Price_Actions
    Invest_log[round] = Investment_Actions

plot_start = time.perf_counter()
plotting(Profits_log, Price_log, Invest_log)
plot_visit_counts_3d(Firm1.visit_counts,Firm2.visit_counts)
plot_elapsed = time.perf_counter() - plot_start

print(f"Plotting completed in {plot_elapsed:.2f} seconds")
print(f"Firm 1 Visits| Min:{np.min(Firm1.visit_counts)}, Max:{np.max(Firm1.visit_counts)}, Avg:{np.mean(Firm1.visit_counts)}")
print(f"Firm 2 Visits| Min:{np.min(Firm2.visit_counts)}, Max:{np.max(Firm2.visit_counts)}, Avg:{np.mean(Firm2.visit_counts)}")


   

   

