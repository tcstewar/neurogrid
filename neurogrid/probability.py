import numpy as np

def convert_weights(w, max_weight=None, bits=None):
    if max_weight is None: 
        max_weight = np.max(np.abs(w))
    p = np.abs(w)/max_weight
    w = np.where(w>=0, max_weight, -max_weight)
    
    if bits is not None:
        p = np.round(p*(2**bits))/(2**bits)
        
    return p, w