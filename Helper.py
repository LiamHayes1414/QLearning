import json
import numpy as np
from tqdm import tqdm
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

    target_cols = array.shape[1]-1
    Search_range = 1000
    patterns = []
    for col in range(target_cols):
        col_data = array[:,col]
        detected_length = None
        # Only checking up to maximum search range
        for lag in tqdm(range(1, Search_range)):
            #Compare array against shifts of itself
            if np.array_equal(col_data[:-lag], col_data[lag:]):
                detected_length = lag
                break

        if detected_length:
            extracted_pattern = col_data[:detected_length]
            patterns.append(extracted_pattern)
        else:
            patterns.append(None)

    return patterns
