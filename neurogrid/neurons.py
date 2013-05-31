import numpy as np

class RateNeuron:
    def __init__(self, N, rng, bias=200, nonlinear=1, balanced=False):
        self.bias = rng.randn(N) * bias
        self.e_gain = rng.uniform(0.5, 2, N)
        if balanced:
            self.i_gain = self.e_gain
        else:    
            self.i_gain = rng.uniform(0.5, 2, N)
        self.nonlinear = rng.uniform(-0.001*nonlinear, 0.001*nonlinear, N)
        
    def rate(self, e_input, i_input):
        
        input = e_input*self.e_gain[:,None] - i_input*self.i_gain[:,None] + \
                (e_input*i_input)*self.nonlinear[:,None] + self.bias[:, None]
         
        # thresholded linear neuron                        
        rate = np.where(input>0, input/10, 0)
        
        return rate                
        
                
