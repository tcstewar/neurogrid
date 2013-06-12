import numpy as np
import unittest

import neurogrid as ng

class TestNonlinearNeuron(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(seed=5)
        
    def test_communication(self):
        N1 = 30
        N2 = 31
        D = 2
        S = 10
        T = 0.5
        dt = 0.001
        pstc = 0.04
        
        
        decay = np.exp(-dt/pstc)
        
        a = ng.ensemble.Ensemble(N1, N1, D, seed=1, nonlinear=1, balanced=False)
        b = ng.ensemble.Ensemble(N2, N2, D, seed=3, nonlinear=1, balanced=False)
        
        d_e, d_i = a.get_dual_decoder(fr_in=500, fc_in=500, fr_out=250, fc_out=500)

        d = b.get_decoder(fr=250, fc=500)
        
        X = ng.samples.random(S, D, self.rng)                
        X, A = ng.activity.classic(a.neurons, a.encoders, self.rng, X=X, fr=500, fc=500)
        
        X_hat_e = np.dot(A.T, d_e)
        X_hat_i = np.dot(A.T, d_i)


        
        w_e = np.dot(d_e, np.where(b.encoders.T>0, b.encoders.T, 0)) + np.dot(d_i, np.where(b.encoders.T<0, -b.encoders.T, 0))

        w_i = np.dot(d_i, np.where(b.encoders.T>0, b.encoders.T, 0)) + np.dot(d_e, np.where(b.encoders.T<0, -b.encoders.T, 0))

        fs = np.zeros(N1*N1, dtype='f')
        fs2 = np.zeros(N2*N2, dtype='f')
        
        Y = []
        X_vals = []
        
        e_input, i_input = ng.activity.create_stimulus(X, a.encoders, fc=500, fr=500)
        
        for i in range(S):
            e_in = e_input[:,i]
            i_in = i_input[:,i]
            
            for j in range(int(T/dt)):
                s = a.neurons.tick(e_in, i_in, dt)
                fs = fs * decay + s * (1-decay)
                                
                in_e = np.dot(fs, w_e)/dt
                in_i = np.dot(fs, w_i)/dt
                
                s2 = b.neurons.tick(in_e, in_i, dt)
                fs2 = fs2 * decay + s2 * (1-decay)

               
                Yhat = np.dot(fs2, d)/dt
            
                Y.append(Yhat)
                X_vals.append(X[i])
            
        import matplotlib.pyplot as plt
        plt.plot(Y)
        plt.plot(X_vals)
        plt.show()

        
        
                
        return
        
        
        

            
    
    
if __name__=='__main__':
    unittest.main()
