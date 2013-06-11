import numpy as np
import samples


def create_stimulus(X, encoders, fr, fc):
    input = np.dot(encoders, X.T)
    e_input = fc + input * fr
    i_input = fc - input * fr 
    
    return e_input, i_input




def classic(neurons, encoders, rng, sample_count=500, use_spikes=False, X=None, fr=500, fc=500):
    N, D = encoders.shape
    if X is None:
        X = samples.random(sample_count, D, rng)


    e_input, i_input = create_stimulus(X, encoders, fr, fc)

    
    
    if not use_spikes:
        A = neurons.rate(e_input, i_input)
    else:        
        A=[]
        for i in range(sample_count):
            AA = neurons.accumulate(e_input[:,i], i_input[:,i])
            A.append(AA)
        A = np.array(A).T    
            
    return X, A
    


