Stat_Responses = {}
Stationarity_Counter = 0

price_indices = [10,12,1]
price = 10
invest = 5

StatesandAction = [
    ([10,12,1], (10,5)),
    ([10,12,1], (10,5)),
    ([10,12,1], (11,6)), 
    ([10,12,1], (11,6))  
]

for key,new_val in StatesandAction:

    key = tuple(key)

    # Check if key (State) already exists in dictionary
    if key in Stat_Responses:
        # Add the rounded action to the set
                    
        #If more than one value in current key
        if Stat_Responses[key] != new_val:
            Stationarity_Counter = 0 
            Stat_Responses[key] = new_val
        else:
            Stationarity_Counter += 1
    else: 
        # Initialize tracking with a SET instead of a single tuple
        Stat_Responses[key] = new_val
    print(Stationarity_Counter)

print(Stat_Responses)