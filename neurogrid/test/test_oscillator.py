import numpy as np
import unittest

import neurogrid as ng

class TestOscillator(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(seed=5)
        
    def test_oscillator(self):
        N1 = 30
        
        D = 2
        
        def func_feedback(x):
            return np.dot(x,np.array([[1.1, 1], [-1, 1.1]]))
            
        T = 6
        dt = 0.001
        pstc = 0.1
        
        nonlinear = 10
        balanced = False
        
        decay = np.exp(-dt/pstc)
        
        A = ng.ensemble.Ensemble(N1, N1, D, seed=1, nonlinear=nonlinear, balanced=balanced, encoder_type='random')        
        Ad_e, Ad_i = A.get_dual_decoder(fr_in=400, fc_in=500, fr_out=400, fc_out=500, func=func_feedback, name='feedback', input_noise=200)        
        Ad = A.get_decoder(fr=400, fc=500)                
        
        AAw_e = np.dot(Ad_e, np.where(A.encoders.T>0, A.encoders.T, 0)) + np.dot(Ad_i, np.where(A.encoders.T<0, -A.encoders.T, 0))
        AAw_i = np.dot(Ad_i, np.where(A.encoders.T>0, A.encoders.T, 0)) + np.dot(Ad_e, np.where(A.encoders.T<0, -A.encoders.T, 0))
        
        Afs = np.zeros(N1*N1, dtype='f')
        
        output = []
        
        spikes = []
        
        t = []
        now = 0
        for j in range(int(T/dt)):

                in_e = np.dot(Afs, AAw_e)/dt
                in_i = np.dot(Afs, AAw_i)/dt

                As = A.neurons.tick(in_e, in_i, dt)
                Afs = Afs * decay + As * (1-decay)

                out = np.dot(Afs, Ad)/dt            
                output.append(out)
                
                t.append(now)
                spikes.append(np.sum(As))
                now += dt
                
        output = np.array(output)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(t, output)
        
        plt.figure()
        plt.scatter(output[:,0], output[:,1])
        
        #plt.figure()
        #plt.plot(t, spikes)
        plt.show()
        
                
        return
        
        
        

            
    
    
if __name__=='__main__':
    unittest.main()
