import numpy as np
import unittest

import neurogrid as ng

class TestCommunicate(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(seed=3)

    def test_standard(self):
        N1 = 10
        N2 = 11
        D = 2
        
        a = ng.ensemble.Ensemble(N1, N1, D, seed=1)
        b = ng.ensemble.Ensemble(N2, N2, D, seed=2)
        
        X, A = ng.activity.classic(a.neurons, a.encoders, rng=self.rng)
        d1 = a.get_decoder()
        
        Xhat = np.dot(A.T, d1)
        
        Y, B = ng.activity.classic(b.neurons, b.encoders, rng=self.rng, X=Xhat)
        d2 = b.get_decoder()
        Yhat = np.dot(B.T, d2)
        
        rmse = np.sqrt(np.sum((X-Yhat)**2)/len(X))
        self.assertLess(rmse, 0.02)
        
        #import matplotlib.pyplot as plt
        #plt.scatter(X, Yhat)
        #plt.show()
            
    def test_weights(self):
        N1 = 10
        N2 = 11
        D = 2
        
        a = ng.ensemble.Ensemble(N1, N1, D, seed=1)
        b = ng.ensemble.Ensemble(N2, N2, D, seed=2)
        d1 = a.get_decoder()
        
        w = np.dot(d1, b.encoders.T)
        
        X, A = ng.activity.classic(a.neurons, a.encoders, rng=self.rng)
        
        input = np.dot(A.T, w).T
        
        B = b.neurons.rate(input*1000, 0)
        d2 = b.get_decoder()
        Yhat = np.dot(B.T, d2)
        
        #import matplotlib.pyplot as plt
        #plt.scatter(X, Yhat)
        #plt.show()

        rmse = np.sqrt(np.sum((X-Yhat)**2)/len(X))
        self.assertLess(rmse, 0.02)
        

    def test_sparse(self):
        N1 = 10
        N2 = 11
        D = 2
        
        a = ng.ensemble.Ensemble(N1, N1, D, seed=1)
        b = ng.ensemble.Ensemble(N2, N2, D, seed=2)
                
        w = ng.sparse.weights(a, b, 10, rng=self.rng)
        
        X, A = ng.activity.classic(a.neurons, a.encoders, rng=self.rng)
        
        input = np.dot(A.T, w).T
        
        B = b.neurons.rate(input*1000, 0)
        d2 = b.get_decoder()
        Yhat = np.dot(B.T, d2)
        
        #import matplotlib.pyplot as plt
        #plt.scatter(X, Yhat)
        #plt.show()

        rmse = np.sqrt(np.sum((X-Yhat)**2)/len(X))
        self.assertLess(rmse, 0.15)
            
    
    
if __name__=='__main__':
    unittest.main()
