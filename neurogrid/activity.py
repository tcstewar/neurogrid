import numpy as np
import samples


def classic(neurons, encoders, rng, sample_count=500, use_spikes=False, X=None):
    N, D = encoders.shape
    if X is None:
        X = samples.random(sample_count, D, rng)

    f_max = 1000
    inputs = np.dot(encoders, X.T)*f_max
    
    e_input = np.where(inputs>0, inputs, 0)
    i_input = np.where(inputs<0, -inputs, 0)
    
    if not use_spikes:
        A = neurons.rate(e_input, i_input)
    else:        
        A=[]
        for i in range(sample_count):
            AA = neurons.accumulate(e_input[:,i], i_input[:,i])
            A.append(AA)
        A = np.array(A).T    
            
    return X, A
    


