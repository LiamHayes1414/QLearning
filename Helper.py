import json
import numpy as np
def format_eta(seconds):
    if seconds < 3600:
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}:{mins:02d}:{secs:02d}"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    

def block_average_2d(matrix, block_size):
    """
    Averages blocks of size (block_size x block_size) in a 2D matrix.
    Truncates edges if they don't fit perfectly.
    """
    #Calculate new dimensions that perfectly fit the block size
    n_rows = (matrix.shape[0] // block_size) * block_size
    n_cols = (matrix.shape[1] // block_size) * block_size
    trimmed = matrix[:n_rows, :n_cols]
    
    #Reshape into 4D and average across the block axes
    reshaped = trimmed.reshape(n_rows // block_size, block_size, 
                               n_cols // block_size, block_size)
    return reshaped.mean(axis=(1, 3))

def find_pattern(array:np.ndarray):
    unique_states, unique_indexes = np.unique(array, axis=0, return_inverse=True)

    State_Indexes = unique_indexes[:-1]
    Result_Indexes = unique_indexes[1:]

    States_Results = np.column_stack((State_Indexes, Result_Indexes))

    unique_pairs, counts = np.unique(States_Results, axis=0, return_counts=True)

    #Store state action relationships in dict for easy access
    relationships = []
    for (left, right), count in zip(unique_pairs, counts):
        relationships.append((left,right,count))

    return relationships,unique_states