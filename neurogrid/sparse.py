import numpy as np

from . import activity
from . import decoders

from . import cache

def weights(a, b, fan_in, rng, name='X', func=None, adjust_target=False, fc=500, fr=500, input_noise=0):
    item = cache.Item(a_n=a.cache_neurons, a_e=a.cache_encoders, adjust_target=adjust_target,
                    b=b.cache_encoders, name=name, fan_in=fan_in, seed=rng.get_state(), input_noise=input_noise)
    w = item.get()
    if w is None:


        X, A = activity.classic(a.neurons, a.encoders, rng, fc=fc, fr=fr, input_noise=input_noise)
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
        
        
def dual_weights(a, b, fan_in, rng, name='X', func=None, fc_in=500, fr_in=500, fc_out=500, fr_out=500, input_noise=0, activity_noise=0.1):
    item = cache.Item(a_n=a.cache_neurons, a_e=a.cache_encoders, 
                    b=b.cache_encoders, name=name, fan_in=fan_in, seed=rng.get_state(), input_noise=input_noise, activity_noise=activity_noise)
    w = item.get()
    if w is None:

        X, A = activity.classic(a.neurons, a.encoders, rng, fc=fc_in, fr=fr_in, input_noise=input_noise)
        N_A, S = A.shape
        N_B = len(b.encoders)
        
        connect = np.array([rng.permutation(N_A)[:fan_in] for i in range(N_B)])        
            
        target = X        
        if func is not None: target=func(target)    
        
        target_e = fc_out + fr_out * target       
        target_i = fc_out - fr_out * target       
        
        w_e = np.zeros((N_A, N_B), dtype='f')
        w_i = np.zeros((N_A, N_B), dtype='f')
        for i in range(N_B):
            A_sub = A[connect[i]]
            
            d_e = decoders.nonnegative(A_sub, target_e, rng, noise=activity_noise)
            d_i = decoders.nonnegative(A_sub, target_i, rng, noise=activity_noise)
            
            w_e[connect[i],i] = np.dot(d_e, np.where(b.encoders[i].T>0, b.encoders[i].T, 0)) + np.dot(d_i, np.where(b.encoders[i].T<0, -b.encoders[i].T, 0))
            w_i[connect[i],i] = np.dot(d_i, np.where(b.encoders[i].T>0, b.encoders[i].T, 0)) + np.dot(d_e, np.where(b.encoders[i].T<0, -b.encoders[i].T, 0))

        w = w_e, w_i
            
        item.set(w)    
    return w    


    