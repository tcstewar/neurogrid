import numpy as np

def convert_weights(w, max_weight=None):
    if max_weight is None: 
        max_weight = np.max(w)
        
    p = np.abs(w)/max_weight
    w = np.where(w>=0, max_weight, -max_weight)
    
    return p, w