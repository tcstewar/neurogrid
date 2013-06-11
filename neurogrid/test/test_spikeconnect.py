import numpy as np
import unittest

import neurogrid as ng

class TestActivity(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(seed=5)
        
    def test_standard(self):
        N1 = 30
        N2 = 31
        D = 2
        S = 10
        T = 0.5
        dt = 0.001
        pstc = 0.04
        
        
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

        #w = np.dot(d1, b.encoders.T)
        w = ng.sparse.weights(a, b, 30, self.rng)
        
        

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
        Y = np.array(Y)
        X_vals = np.array(X_vals)
        
        indices = [(i+1)*int(T/dt)-1 for i in range(S)]



        #import matplotlib.pyplot as plt
        #plt.plot(Y)
        #plt.plot(X_vals)
        
        #plt.figure()
        #plt.imshow(w[:30,:30])
        
        #plt.show()
        
        rmse = np.sqrt(np.sum((X_vals[indices]-Y[indices])**2)/S)
        print rmse        
        self.assertLess(rmse, 0.05)
        

        
        
            
    
    
if __name__=='__main__':
    unittest.main()
