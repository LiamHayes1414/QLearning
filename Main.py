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
    State_log.append([prices[0]]*firms)

Industry_m = 1
Downsample_len = 10
    #fixed experimentation log
Profits_explog = []
Price_explog = []
Invest_explog = []
    #stationarity log
Profits_statlog = []
Price_statlog = []
Invest_statlog = []

Round = 0
Stat_log_Counter = 0
Firm_Stationarity = [0]
Stationarity_Target = 100000
training_start = time.perf_counter()

with tqdm(total=Stationarity_Target, desc="Tracking Stationarity") as pbar:
    while min(Firm_Stationarity) < Stationarity_Target:
        if Round == GameCap:
            break

        epsilon = max((epsilon_decay*Round) + 1,0)

        Prices = []
        Investments = []
        Leadership = []
        Firm_Stationarity = []
        for f in Firms:
            Action = f.Action(State_log, epsilon)
            Prices.append(Action[0])
            Investments.append(Action[1])
            Leadership.append(f.Leader)
            

        #Save current state for updating the Q-Matrix
        CurrentState = [prices[:] for prices in State_log]

        #Save actioned prices into state log
        State_log.append(list(Prices))

        #Store in np arrays for vectorization
        Price_Actions = np.array(Prices)
        Investment_Actions = np.array(Investments)
        Leader = np.array(Leadership)

        MarketShares = Demand(Price_Actions,Leader) * mrktsz #for readability

        Profit = (Price_Actions - mc)*MarketShares - Investment_Actions
    
        #Calculate best possible outcomes next period 
        Leader_Best = []
        Follower_Best = []
        Current_Values = []

        for i, f in enumerate(Firms, start=0):

            price_indices = f.decodelog(CurrentState)

            if firms >1:
                L_Best,F_Best = f.ProfitExpectations(State_log)
                Leader_Best.append(L_Best)
                Follower_Best.append(F_Best)
                #Add position indicator to price indices 
                price_indices.append(f.Leader)

            elif firms == 1: #monopoly
                L_Best = f.ProfitExpectations(State_log)
                Leader_Best.append(L_Best)
                #Add position indicator to price indices 
                price_indices.append(0)

            state_index = tuple(price_indices)

            action_index = Possible_Actions.index((Prices[i],Investments[i]))
            Current_Val = f.Q_matrix[state_index][action_index]

            Current_Values.append(Current_Val)

        #convert to numpy arrays
        LeaderBest = np.array(Leader_Best)
        Current_Values = np.array(Current_Values)
    
        #Innovation success probabilities
        if firms >1:
            MarketInnovation = np.sum(Investment_Actions) + K
            Firm_Probabilities = Investment_Actions/MarketInnovation

            FollowerBest = np.array(Follower_Best)#Only exists in non monopoly case

            ValueExpectations = (1-learning_rate)*Current_Values + learning_rate*(Profit + delta*(Firm_Probabilities*LeaderBest + (1-Firm_Probabilities)*FollowerBest))
        elif firms == 1:
            #In monopoly firm is guaranteed to remain leader
            ValueExpectations = (1-learning_rate)*Current_Values + learning_rate*(Profit + delta*LeaderBest )

        #Make sure i don't get any floating point errors
        cleaned_expectations = np.round(ValueExpectations, 4)

        #Update Q-Matrices given state and action choices
        for i, f in enumerate(Firms, start=0):
            f.UpdateQ(CurrentState,(Prices[i],Investments[i]),cleaned_expectations[i])
            Firm_Stationarity.append(f.Stationarity_Counter) #Stat counter updates int Firm.Action()

        #Draw next innovation leader - only applies when 2+ firms
        if firms >1:
            NoWinnerProb = 1- np.sum(Firm_Probabilities)
            probabilities = np.append(Firm_Probabilities, NoWinnerProb)

            outcomes = list(range(0, firms)) + [None]
        
            winner = random.choices(outcomes, weights=probabilities, k=1)[0]
        
            if winner is not None:
                Industry_m+=1
                for i, f in enumerate(Firms, start=0):
                    f.Leader = 0
                    if int(winner) == i:
                        f.Leader = 1
        #Save profits for graphing
        if Round % Downsample_len == 0 or Round>ExpLen: #downsample within exploration length
            if 1 in Leadership: #no leader on first round(and maybe more if no one wins innovation)
                Profits_explog.append(np.append(Profit,Leadership.index(1)))
                Price_explog.append(np.append(Price_Actions,Leadership.index(1)))
                Invest_explog.append(np.append(Investment_Actions,Leadership.index(1)))
            else:
                Profits_explog.append(np.append(Profit,None))
                Price_explog.append(np.append(Price_Actions,None))
                Invest_explog.append(np.append(Investment_Actions,None))

            #See how many entries are recorded past experimentation (not downsampled)
            if Round>ExpLen:Stat_log_Counter+=1

        if Round>ExpLen: #experimentation is over
            if min(Firm_Stationarity) == 0:
                #if Q matrix changes dump stat log
                Profits_statlog = []
                Price_statlog = []
                Invest_statlog = []
            else:
                Profits_statlog.append(np.append(Profit,Leadership.index(1)))
                Price_statlog.append(np.append(Price_Actions,Leadership.index(1)))
                Invest_statlog.append(np.append(Investment_Actions,Leadership.index(1)))
                
        #Progress bar - only updates every X rounds
        if Round % 20000 == 0:
            completed_rounds = Round + 1
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
        Round+=1

#Process results
plot_start = time.perf_counter()
plotting((Profits_explog,Profits_statlog), (Price_explog,Price_statlog), (Invest_explog,Invest_statlog),config,Downsample_len,Stat_log_Counter)
if config.investments_count>1 and firms>1:leaderplots((Profits_explog,Profits_statlog), (Price_explog,Price_statlog), (Invest_explog,Invest_statlog), config, Downsample_len)
#plot_visit_counts_3d(Firms)
strategy(Price_statlog, Invest_statlog, config)
plot_elapsed = time.perf_counter() - plot_start

print(f"Plotting completed in {plot_elapsed:.2f} seconds")
print(f"Rounds: {Round}")

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

    #print out duplicate values in matrices
    for state in np.ndindex(f.state_shape):
        # Extract the action vector for this specific state
        action_values = f.Q_matrix[state]
        
        # Find unique action values and their counts
        max_val = np.max(action_values)
    
        # 2. Count how many actions share this maximum value
        max_count = np.sum(action_values == max_val)
        
        if max_count > 1 and max_val>0:
            print(f"State {state}: Tie detected! Max value {max_val} appears {max_count} times.")

config.Details['Rounds'] = Round

with open("TrainingResults/Details.json", "w") as json_file:
    json.dump(config.Details, json_file, indent=4, cls=NumpyEncoder)
   

