import numpy as np

class RateNeuron:
    def __init__(self, N, rng, bias=1):
        self.bias = rng.randn(N) * bias
        self.e_gain = rng.uniform(0.5, 2, N)
        self.i_gain = rng.uniform(0.5, 2, N)
        self.nonlinear = rng.uniform(-0.001, 0.001, N)
        
    def rate(self, e_input, i_input):
        input = np.outer(e_input, self.e_gain) - \
                np.outer(i_input, self.i_gain) + \
                np.outer(e_input * i_input, self.nonlinear) 
                        
        rate = np.where(input>0, input/10, 0)
        
        return rate                
        
                
