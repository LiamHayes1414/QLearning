import numpy as np
import itertools

class Firm:

    def __init__(self,Possible_States,Possible_Actions,SettingsConfig):

        #Initialize Q-Matrix
        state_shape = tuple(len(set(index_elements)) for index_elements in zip(*Possible_States))
        action_shape = (len(Possible_Actions),) #flat action vector
        self.state_shape = state_shape #to use in anlysis

        self.Q_matrix = np.zeros(state_shape + action_shape)
        self.visit_counts = np.zeros(state_shape + action_shape, dtype=np.int32) 
        #^store how many times each state action part is visited to make the model aware of its confidence

        self.config = SettingsConfig

        #Initialize settings obj
        self.price_options = self.config.price_options
        self.possible_actions = Possible_Actions

        self.Leader = 1 if self.config.firms == 1 else 0 #binary indicator if firm is leader or not (initialized at 1 if monopoly)
        self.Stationarity_Counter = 0 #log how many consecutive updates are stationary (for solution convergence)
        self.Stat_Responses = {}
    def decodelog(self,log):
        flat_arr = np.array(log).ravel()
        clean_prices = np.round(flat_arr, 4)
        price_indices = np.searchsorted(self.price_options, clean_prices)
        return list(price_indices)

    def Action(self, log, epsilon=0):

        price_indices = self.decodelog(log)

        #add position indicator (currently leader or follower)
        if self.config.firms>1:
            price_indices.append(int(self.Leader))
        elif self.config.firms == 1:
            #only 1 leadership option which is at index 0
            price_indices.append(0)
    
        state_index = tuple(price_indices)
        state_action_values = self.Q_matrix[state_index]
        state_action_visits = self.visit_counts[state_index]

        Optimal = True
        rounded_values = np.round(state_action_values, 4)

        if np.random.random() < epsilon : 
            #within probability of epsilon, choose least visited action
            min_visits = np.min(state_action_visits)
            min_indices = np.flatnonzero(state_action_visits == min_visits)
            #action_index = np.random.choice(min_indices)
            action_index = min_indices[0]
            Optimal = False
        else:
            #optimal action selection with random choise on tie
            max_value = np.max(rounded_values)
            max_indices = np.flatnonzero(rounded_values == max_value)
            action_index = np.random.choice(max_indices)

            if epsilon==0 and len(max_indices) >1:
                print("True")
        price, invest = self.possible_actions[action_index]

        if Optimal:
            key = tuple(price_indices)
            new_val = (price, invest)

            # Check if key (State) already exists in dictionary
            if key in self.Stat_Responses:
                # Add the rounded action to the set
                self.Stat_Responses[key].add(new_val)
                
                # CATCH IT: If the set size is greater than 1, a genuine shift occurred
                if len(self.Stat_Responses[key]) > 1:
                    
                    sorted_indices = np.argsort(rounded_values)

                    top_1_idx = sorted_indices[-1]
                    top_2_idx = sorted_indices[-2]

                    top_1_val = rounded_values[top_1_idx]
                    top_1_act = self.possible_actions[top_1_idx]

                    top_2_val = rounded_values[top_2_idx]
                    top_2_act = self.possible_actions[top_2_idx]
                    """if abs(top_1_val - top_2_val) <0.001:

                        print(f" Max 1 (Index {top_1_idx}) Action {top_1_act}: Q = {top_1_val:.15f}")
                        print(f" Max 2 (Index {top_2_idx}) Action {top_2_act}: Q = {top_2_val:.15f}")
                        print("__")
                    """
                    

                    self.Stationarity_Counter = 0 
                    self.Stat_Responses[key].clear()
                    self.Stat_Responses[key].add(new_val)
                else:
                    # FIX 2: If size is 1, the new action successfully matched the existing one
                    self.Stationarity_Counter += 1
            else: 
                # Initialize tracking with a SET instead of a single tuple
                self.Stat_Responses[key] = {new_val}
            
        return price, invest
    
    def ProfitExpectations(self,log):
        price_indices = self.decodelog(log)

        #Generate both states if leader or follower next round
        LeaderState =  price_indices + [0 if self.config.firms == 1 else 1] #if monoply leader index at 0 (only 1 option)
        Leader_index = tuple(LeaderState)
        Leader_values = np.round(self.Q_matrix[Leader_index], 4)
        Leader_best = np.max(Leader_values)

        if self.config.firms>1:
            #follower only exists if 2+ firms
            FollowerState = price_indices + [0]
            Follower_index = tuple(FollowerState)
            Follower_values = np.round(self.Q_matrix[Follower_index], 4)
            Follower_best = np.max(Follower_values)

            return Leader_best,Follower_best
        
        return Leader_best
    
    def UpdateQ(self,log,actions:tuple,Value):

        price_indices = self.decodelog(log)

        #add position indicator (currently leader or follower)
        price_indices.append(0 if self.config.firms == 1 else self.Leader) #for monopoly leader index is 0 even though leader==1
    

        state_index = tuple(price_indices)
        action_index = self.possible_actions.index(actions)
        Prev_Val = self.Q_matrix[state_index][action_index]

        #Update value in place
        self.Q_matrix[state_index][action_index] = Value        

        #Update visit count matrix
        self.visit_counts[state_index][action_index] += 1

        #Safety to make sure Q matrix updates are also consistent (along with stat_dict)
        if np.round(Prev_Val, 4) != Value: self.Stationarity_Counter = 0





















