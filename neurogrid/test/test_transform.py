import numpy as np
import unittest

import neurogrid as ng

class TestTransform(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(seed=6)

    def test_linear(self):
        N1 = 10
        N2 = 11
        D = 2
        
        a = ng.ensemble.Ensemble(N1, N1, D, seed=1)
        b = ng.ensemble.Ensemble(N2, N2, D, seed=2)
        
        X, A = ng.activity.classic(a.neurons, a.encoders, rng=self.rng)
        d1 = a.get_decoder()
        
        Xhat = np.dot(A.T, d1)
        
        T = -1
        
        Y, B = ng.activity.classic(b.neurons, b.encoders, rng=self.rng, X=Xhat*T)
        d2 = b.get_decoder()
        Yhat = np.dot(B.T, d2)
        
        rmse = np.sqrt(np.sum((X-Yhat*T)**2)/len(X))
        self.assertLess(rmse, 0.02)
        
        #import matplotlib.pyplot as plt
        #plt.scatter(X, Yhat)
        #plt.show()
   
    def test_square(self):
        N1 = 10
        N2 = 11
        D = 2
        
        a = ng.ensemble.Ensemble(N1, N1, D, seed=1)
        b = ng.ensemble.Ensemble(N2, N2, D, seed=2)
        
        def square(x):
            return x*x
        
        X, A = ng.activity.classic(a.neurons, a.encoders, rng=self.rng)
        d1 = a.get_decoder(name='square', func=square)
        
        Xhat = np.dot(A.T, d1)
        
        Y, B = ng.activity.classic(b.neurons, b.encoders, rng=self.rng, X=Xhat)
        d2 = b.get_decoder()
        Yhat = np.dot(B.T, d2)
        
        rmse = np.sqrt(np.sum((square(X)-Yhat)**2)/len(X))
        self.assertLess(rmse, 0.05)
        
        #import matplotlib.pyplot as plt
        #plt.scatter(X, Yhat)
        #plt.show()
            
        
    def test_sparse_square(self):
        N1 = 32
        N2 = 17
        S = 32
        D = 2
        
        a = ng.ensemble.Ensemble(N1, N1, D, seed=7)
        b = ng.ensemble.Ensemble(N2, N2, D, seed=8)

        def square(x):
            return x*x
                
        w = ng.sparse.weights(a, b, S, rng=self.rng, func=square, name='square', adjust_target=True)
        
        X, A = ng.activity.classic(a.neurons, a.encoders, rng=self.rng)
        
        input = np.dot(A.T, w).T
        
        B = b.neurons.rate(input*1000, 0)
        d2 = b.get_decoder()
        Yhat = np.dot(B.T, d2)
        
        import matplotlib.pyplot as plt
        plt.scatter(X, Yhat)
        plt.show()

        rmse = np.sqrt(np.sum((square(X)-Yhat)**2)/len(X))
        self.assertLess(rmse, 0.1)
        
        print rmse
        
        

    
        
            
    
    
if __name__=='__main__':
    unittest.main()
