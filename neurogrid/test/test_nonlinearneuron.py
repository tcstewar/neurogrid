import numpy as np
import unittest

import neurogrid as ng

class TestNonlinearNeuron(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(seed=5)
        
    def test_nonlinearneuron(self):
        N1 = 30
        N2 = 31
        D = 1
        S = 100
        T = 0.5
        dt = 0.001
        pstc = 0.04
        
        
        decay = np.exp(-dt/pstc)
        
        a = ng.ensemble.Ensemble(N1, N1, D, seed=1, nonlinear=1, balanced=False)
        
        X = ng.samples.random(S, D, self.rng)
        
        d_e, d_i = a.get_dual_decoder(fr_in=500, fc_in=500, fr_out=250, fc_out=500)
        
        X, A = ng.activity.classic(a.neurons, a.encoders, self.rng, X=X, fr=500, fc=500)
        
        X_hat_e = np.dot(A.T, d_e)
        X_hat_i = np.dot(A.T, d_i)
        
        
        import matplotlib.pyplot as plt
        plt.scatter(X, X_hat_e, color='b')
        plt.scatter(X, X_hat_i, color='r')        
        
        plt.show()
                
        return
        
        
        

            
    
    
if __name__=='__main__':
    unittest.main()
