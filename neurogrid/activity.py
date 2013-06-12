import numpy as np
import samples


def create_stimulus(X, encoders, fr, fc):
    fp = fc + X * fr
    fm = fc - X * fr
    
    
    e_input = np.dot(fp, np.where(encoders.T>0, encoders.T, 0)) + np.dot(fm, np.where(encoders.T<0, -encoders.T, 0))
    i_input = np.dot(fm, np.where(encoders.T>0, encoders.T, 0)) + np.dot(fp, np.where(encoders.T<0, -encoders.T, 0))

    return e_input.T, i_input.T

    # Note: This simplistic approach does not work: 
    #  it misses a factor of (np.where(encoders.T<0, -encoders.T, 0)+np.where(encoders.T<0, -encoders.T, 0)) on fc
    #input = np.dot(encoders, X.T)
    #e_input = fc + input * fr
    #i_input = fc - input * fr 
    
    #return e_input, i_input




def classic(neurons, encoders, rng, sample_count=500, use_spikes=False, X=None, fr=500, fc=500, input_noise=0):
    N, D = encoders.shape
    if X is None:
        X = samples.random(sample_count, D, rng)


    e_input, i_input = create_stimulus(X, encoders, fr, fc)

    # TODO: jitter orthogonally, not everywhere
    if input_noise>0:
        noise = rng.randn(*e_input.shape)*input_noise
        e_input += noise
        i_input += noise
    
    
    if not use_spikes:
        A = neurons.rate(e_input, i_input)
    else:        
        A=[]
        for i in range(sample_count):
            AA = neurons.accumulate(e_input[:,i], i_input[:,i])
            A.append(AA)
        A = np.array(A).T    
            
    return X, A
    


