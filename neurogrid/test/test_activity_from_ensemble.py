import numpy as np
import unittest

import neurogrid as ng

class TestActivity(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState()
        
    def test_classic(self):
    
        source = ng.ensemble.Ensemble(10, 10, 1, seed=1)

        N = 20
        n = ng.neurons.SpikeNeuron(N, self.rng, balanced=True, nonlinear=0)
        e = ng.encoders.random(N, 1, self.rng)
    
        X, A = ng.activity.classic(n, e, self.rng)

        X, A2 = ng.activity.generate_from_ensemble(n, e, self.rng, source, 0.01)

        
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.plot(X[:,0], A.T)
        plt.figure()
        plt.plot(X[:,0], A2.T)
        plt.show()
    
        
            
    
    
if __name__=='__main__':
    unittest.main()
