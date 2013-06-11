import numpy as np
import unittest

import neurogrid as ng

class TestActivity(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState()
        
    def test_classic(self):

        N = 20
        S = 10
        n = ng.neurons.SpikeNeuron(N, self.rng, balanced=True, nonlinear=0)
        e = ng.encoders.random(N, 1, self.rng)
    
        X, A = ng.activity.classic(n, e, self.rng, use_spikes=True, sample_count=S)
        X, A2 = ng.activity.classic(n, e, self.rng, use_spikes=False, sample_count=S)
    
        mse = np.sum((A-A2)**2)/(N * S)
        
    
        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.plot(X[:,0], A2.T-A.T)
        #plt.show()
    
        self.assertLess(mse, 3)
        
            
    
    
if __name__=='__main__':
    unittest.main()
