import numpy as np
import samples


def classic(neurons, encoders, rng, sample_count=10):
    N, D = encoders.shape
    X = samples.random(sample_count, D, rng)
    
    f_max=1000
    inputs = np.dot(encoders, X.T)*f_max
    
    e_input = np.where(inputs>0, inputs, 0)
    i_input = np.where(inputs<0, -inputs, 0)
    
    
    
    A = neurons.rate(e_input, i_input)
    return X, A
    

if __name__=='__main__':
    import neurons
    import encoders
    
    rng = np.random.RandomState()
    n = neurons.RateNeuron(20, rng, balanced=True, nonlinear=0)
    e = encoders.random(20, 1, rng)
    
    X, A = classic(n, e, rng)
    
    import matplotlib.pyplot as plt
    
    plt.plot(X[:,0], A.T)
    plt.show()
    
