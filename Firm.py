import numpy as np
import itertools


class Firm:

    def __init__(self,Possible_States,Possible_Actions,SettingsConfig):

        #Initialize Q-Matrix
        state_shape = tuple(len(set(index_elements)) for index_elements in zip(*Possible_States))
        action_shape = (len(Possible_Actions),) #flat action vector

        self.Q_matrix = np.zeros(state_shape + action_shape)
        self.visit_counts = np.zeros(state_shape + action_shape, dtype=np.int32) 
        #^store how many times each state action part is visited to make the model aware of its confidence

        self.config = SettingsConfig

        #Initialize settings obj
        self.price_options = self.config.price_options
        self.possible_actions = Possible_Actions

        self.Leader = 1 if self.config.firms == 1 else 0 #binary indicator if firm is leader or not (initialized at 1 if monopoly)

    def decodelog(self,log):
        flat_list = list(itertools.chain.from_iterable(log))

        price_indices = [int(np.searchsorted(self.price_options, p)) for p in flat_list]

        return price_indices


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

        if np.random.random() < epsilon : 
            #within probability of epsilon, choose least visited action
            min_visits = np.min(state_action_visits)
            min_indices = np.flatnonzero(state_action_visits == min_visits)
            action_index = np.random.choice(min_indices)
        else:
            #optimal action selection with random choise on tie
            max_value = np.max(state_action_values)
            max_indices = np.flatnonzero(state_action_values == max_value)
            action_index = np.random.choice(max_indices)
        price, invest = self.possible_actions[action_index]

        return price, invest
    
    def ProfitExpectations(self,log):
        price_indices = self.decodelog(log)

        #Generate both states if leader or follower next round
        LeaderState =  price_indices + [0 if self.config.firms == 1 else 1] #if monoply leader index at 0 (only 1 option)
        Leader_index = tuple(LeaderState)
        Leader_values = self.Q_matrix[Leader_index]
        Leader_best = np.max(Leader_values)

        if self.config.firms>1:
            #follower only exists if 2+ firms
            FollowerState = price_indices + [0]
            Follower_index = tuple(FollowerState)
            Follower_values = self.Q_matrix[Follower_index]
            Follower_best = np.max(Follower_values)

            return Leader_best,Follower_best
        
        return Leader_best
    
    def UpdateQ(self,log,actions:tuple,Value):

        price_indices = self.decodelog(log)

        #add position indicator (currently leader or follower)
        price_indices.append(0 if self.config.firms == 1 else self.Leader) #for monopoly leader index is 0 even though leader==1
    

        state_index = tuple(price_indices)
        action_index = self.possible_actions.index(actions)

        #Update value in place
        self.Q_matrix[state_index][action_index] = Value

        #Update visit count matrix
        self.visit_counts[state_index][action_index] += 1





















