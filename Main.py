from itertools import product
from Firm import Firm as firm
from collections import deque
from Settings import Config
import numpy as np
import time
from tqdm import tqdm
from Visualize import plotting, plot_visit_counts_3d, leaderplots,strategy
import json
import random
from Helper import format_eta,NumpyEncoder

#load settings
config = Config()
prices = config.price_options
firms = config.firms
lags = config.lags
positions = config.position_options
investments = config.invest_options
Demand = config.demand
mc = config.mc
ExpLen = config.explorationlen
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
    #assign a leader if no r&d case
    if config.investments_count == 1 and f==0:
        Firm.Leader=1
    Firms.append(Firm)

#State log
State_log = deque(maxlen=lags)
#Append initial state window
for lag in range(lags):
    State_log.append([1]*firms)

Industry_m = 1
#initialize memory (for speed up)
Downsample_len = 100
    #fixed experimentation log
Profits_explog = []
Price_explog = []
Invest_explog = []
    #stationarity log
Profits_statlog = []
Price_statlog = []
Invest_statlog = []

round = 0
Firm_Stationarity = [0]
Stationarity_Target = 250000
training_start = time.perf_counter()

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
        if round % Downsample_len == 0: #downsample within exploration length
            if 1 in Leadership: #no leader on first round(and maybe more if no one wins innovation)
                Profits_explog.append(np.append(Profit,Leadership.index(1)))
                Price_explog.append(np.append(Price_Actions,Leadership.index(1)))
                Invest_explog.append(np.append(Investment_Actions,Leadership.index(1)))
            else:
                Profits_explog.append(np.append(Profit,None))
                Price_explog.append(np.append(Price_Actions,None))
                Invest_explog.append(np.append(Investment_Actions,None))

        elif round>ExpLen: #experimentation is over
            
            Profits_statlog.append(np.append(Profit,Leadership.index(1)))
            Price_statlog.append(np.append(Price_Actions,Leadership.index(1)))
            Invest_statlog.append(np.append(Investment_Actions,Leadership.index(1)))

            if min(Firm_Stationarity) == 0:
                print("dumped")
                #if Q matrix changes dump stat log
                Profits_statlog = []
                Price_statlog = []
                Invest_statlog = []

                print(Price_statlog)



        #Progress bar - only updates every X rounds
        if round % 20000 == 0:
            completed_rounds = round + 1
            rounds_left_in_exp = max(ExpLen - completed_rounds, 0)
            elapsed = time.perf_counter() - training_start
            rounds_per_second = completed_rounds / elapsed if elapsed > 0 else 0

            if rounds_left_in_exp > 0 and rounds_per_second > 0:
                # Seconds remaining until epsilon finishes decaying to zero.
                seconds_left = rounds_left_in_exp / rounds_per_second
                exp_eta = format_eta(seconds_left)
            else:
                exp_eta = "Done"
            pbar.n = int(min(Firm_Stationarity))
            pbar.set_postfix(rounds=f"{completed_rounds / 1e6:.2f}M/{ExpLen / 1e6:.2f}M",Exp_ETA=exp_eta)
            pbar.refresh()
        round+=1

#Process results
plot_start = time.perf_counter()
plotting((Profits_explog,Profits_statlog), (Price_explog,Price_statlog), (Invest_explog,Invest_statlog),config,Downsample_len)
if config.investments_count>1:leaderplots((Profits_explog,Profits_statlog), (Price_explog,Price_statlog), (Invest_explog,Invest_statlog), config, Downsample_len)
plot_visit_counts_3d(Firms)
price_pattern,invest_pattern = strategy(Price_statlog, Invest_statlog, config)
plot_elapsed = time.perf_counter() - plot_start

print(f"Plotting completed in {plot_elapsed:.2f} seconds")
print(f"Rounds: {round}")

for i, f in enumerate(Firms):
    zeros_count = np.sum(f.visit_counts == 0)
    total_size = f.visit_counts.size
    config.Details[f'Firm{i+1} Visits'] = {
        "Min_F":np.min(f.visit_counts[..., 0, :]),
        "Max_F":np.max(f.visit_counts[..., 0, :]),
        "Avg_F":np.mean(f.visit_counts[..., 0, :]),
        'Pct_Missed_F':f'{np.mean(f.visit_counts[..., 0, :] == 0) * 100}%'
        }
    if firms>1:
        config.Details[f'Firm{i+1} Visits']["Min_L"] = np.min(f.visit_counts[..., 1, :])
        config.Details[f'Firm{i+1} Visits']["Max_L"] = np.max(f.visit_counts[..., 1, :])
        config.Details[f'Firm{i+1} Visits']["Avg_L"] = np.mean(f.visit_counts[..., 1, :])
        config.Details[f'Firm{i+1} Visits']['Pct_Missed_L'] = f'{np.mean(f.visit_counts[..., 1, :] == 0) * 100}%'

config.Details['Rounds'] = round
config.Details['Final Prices'] = Price_Actions
config.Details['Price Patterns'] = price_pattern
config.Details['Invest Patterns'] = invest_pattern
    
with open("TrainingResults/Details.json", "w") as json_file:
    json.dump(config.Details, json_file, indent=4, cls=NumpyEncoder)
   

