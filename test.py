import numpy as np
from Helper import find_pattern

actual = [1,0,1,0,1,0,1,0,1,0,8]
prices = actual * (100000//len(actual))

arr = np.array(prices)

print(find_pattern(arr))