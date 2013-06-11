import numpy as np
import unittest

import neurogrid as ng

class TestActivity(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState()
        
    def test_standard(self):
        N1 = 50
        N2 = 51
        D = 2
        S = 10
        T = 0.5
        dt = 0.001
        pstc = 0.01
        
        
        decay = np.exp(-dt/pstc)
        
        a = ng.ensemble.Ensemble(N1, N1, D, seed=1)
        b = ng.ensemble.Ensemble(N2, N2, D, seed=2)
        
        X = ng.samples.random(S, D, self.rng)
        
        f_max = 1000
        inputs = np.dot(a.encoders, X.T)*f_max    
        e_input = np.where(inputs>0, inputs, 0)
        i_input = np.where(inputs<0, -inputs, 0)
        
        fs = np.zeros(N1*N1, dtype='f')
        fs2 = np.zeros(N2*N2, dtype='f')

        d1 = a.get_decoder()
        d2 = b.get_decoder()

        w = np.dot(d1, b.encoders.T)
        

        Y = []
        X_vals = []
        
        
        for i in range(S):
            e_in = e_input[:,i]
            i_in = i_input[:,i]
            
            for j in range(int(T/dt)):
                s = a.neurons.tick(e_in, i_in, dt)
                fs = fs * decay + s * (1-decay)
                
                in2 = np.dot(fs, w)*f_max/dt
                
                s2 = b.neurons.tick(in2, 0, dt)
                fs2 = fs2 * decay + s2 * (1-decay)
                
                Yhat = np.dot(fs2, d2)/dt
                
                Y.append(Yhat)
                X_vals.append(X[i])
                

        import matplotlib.pyplot as plt
        plt.plot(Y)
        plt.plot(X_vals)
        plt.show()

        
        return
        
        
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
    
        
            
    
    
if __name__=='__main__':
    unittest.main()
