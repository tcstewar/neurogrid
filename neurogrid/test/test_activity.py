import numpy as np
import unittest

import neurogrid as ng

class TestActivity(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState()
        
    def test_classic(self):

        N = 20
        n = ng.neurons.RateNeuron(N, self.rng, balanced=True, nonlinear=0)
        e = ng.encoders.random(N, 1, self.rng)
    
        X, A = ng.activity.classic(n, e, self.rng)
    
        import matplotlib.pyplot as plt
        
        plt.plot(X[:,0], A.T)
        plt.show()
    
        
            
    
    
if __name__=='__main__':
    unittest.main()
