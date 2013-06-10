import numpy as np

from . import activity
from . import decoders

from . import cache

def weights(a, b, fan_in, rng, name='X', func=None, adjust_target=True):
    item = cache.Item(a_n=a.cache_neurons, a_e=a.cache_encoders, adjust_target=adjust_target,
                    b=b.cache_encoders, name=name, fan_in=fan_in, seed=rng.get_state())
    w = item.get()
    if w is None:


        X, A = activity.classic(a.neurons, a.encoders, rng)
        N_A, S = A.shape
        N_B = len(b.encoders)
        
        connect = np.array([rng.permutation(N_A)[:fan_in] for i in range(N_B)])        
            
        target = X        
        if func is not None: target=func(target)    
        
        error = np.zeros(target.shape, dtype='f')
        w = np.zeros((N_A, N_B), dtype='f')
        for i in range(N_B):
            A_sub = A[connect[i]]
            
            d = decoders.classic(A_sub, target-error, rng)
            
            if adjust_target:
                Xhat = np.dot(A_sub.T, d)               
                
                error += Xhat - target
                 
                #target -= Xhat/(N_B-i)
                
            
            w[connect[i],i] = np.dot(d, b.encoders[i])
            
        item.set(w)    
    return w    
        
        


