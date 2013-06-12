import numpy as np
import unittest

import neurogrid as ng

class TestDecoders(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState()
        
    def test_classic(self):
        N = 30
        n = ng.neurons.RateNeuron(N, self.rng, balanced=True, nonlinear=0)
        e = ng.encoders.random(N, 1, self.rng)
    
        X, A = ng.activity.classic(n, e, self.rng)    
        d = ng.decoders.classic(A, X, self.rng)
                    
        Xhat = np.dot(A.T, d)
        
        mse = np.sum((Xhat - X)**2)/len(X)
        
        self.assertAlmostEqual(mse, 0, 3)
        
        print mse
        #import matplotlib.pyplot as plt    
        #plt.plot(X[:,0], Xhat[:,0])
        #plt.show()

    def test_nonnegative(self):
        N = 900
        n = ng.neurons.RateNeuron(N, self.rng, balanced=False, nonlinear=10, bias=500)
        e = ng.encoders.random(N, 2, self.rng)
    
        X, A = ng.activity.classic(n, e, self.rng, fc=500, fr=400)    
        
        
        X = 200 + 150*X
        
        d = ng.decoders.nonnegative(A, X, self.rng)
                    
        Xhat = np.dot(A.T, d)
        
        mse = np.sum((Xhat - X)**2)/len(X)

        print mse
        import matplotlib.pyplot as plt    
        plt.scatter(X[:,0], Xhat[:,0], color='b')
        plt.scatter(X[:,1], Xhat[:,1], color='r')
        plt.show()

        
        self.assertLess(mse, 0.01)

        
            
    
    
if __name__=='__main__':
    unittest.main()
