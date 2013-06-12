import numpy as np
import unittest

import neurogrid as ng

class TestOscillator(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(seed=5)
        
    def test_oscillator(self):
        N1 = 30
        N2 = 31
        
        D = 2  
        S = 80
        T = 0.06
        dt = 0.001
        pstc = 0.1
        
        nonlinear = 10
        
        decay = np.exp(-dt/pstc)
        
        def func_feedback(x):
            return np.dot(x,np.array([[1.1, 1], [-1, 1.1]]))        
        
        A = ng.ensemble.Ensemble(N1, N1, D, seed=1, nonlinear=nonlinear, balanced=False, encoder_type='random')
        B = ng.ensemble.Ensemble(N2, N2, D, seed=3, nonlinear=nonlinear, balanced=False, encoder_type='random')
        
        Ad_e, Ad_i = A.get_dual_decoder(fr_in=400, fc_in=500, fr_out=200, fc_out=250)
        Bd_e, Bd_i = B.get_dual_decoder(fr_in=200, fc_in=500, fr_out=200, fc_out=250, func=func_feedback, name='feedback', input_noise=100)
        
        Bd = B.get_decoder(fr=200, fc=500)                
        
        XA = np.zeros((S,2), dtype='f')
        XA[0,0] = 0
        XA[1,0] = 0
        
        ABw_e = np.dot(Ad_e, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Ad_i, np.where(B.encoders.T<0, -B.encoders.T, 0))
        ABw_i = np.dot(Ad_i, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Ad_e, np.where(B.encoders.T<0, -B.encoders.T, 0))

        BBw_e = np.dot(Bd_e, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Bd_i, np.where(B.encoders.T<0, -B.encoders.T, 0))
        BBw_i = np.dot(Bd_i, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Bd_e, np.where(B.encoders.T<0, -B.encoders.T, 0))
        
        Afs = np.zeros(N1*N1, dtype='f')
        Bfs = np.zeros(N2*N2, dtype='f')
        
        output = []
        input = []
        spikes = []
        
        Ae_input, Ai_input = ng.activity.create_stimulus(XA, A.encoders, fc=500, fr=400)
                
        t = []
        now = 0
        for i in range(S):
            print i, S
            for j in range(int(T/dt)):
                
                As = A.neurons.tick(Ae_input[:,i], Ai_input[:,i], dt)
                Afs = Afs * decay + As * (1-decay)
                
                in_e = np.dot(Afs, ABw_e)/dt + np.dot(Bfs, BBw_e)/dt
                in_i = np.dot(Afs, ABw_i)/dt + np.dot(Bfs, BBw_i)/dt

                Bs = B.neurons.tick(in_e, in_i, dt)
                Bfs = Bfs * decay + Bs * (1-decay)

                out = np.dot(Bfs, Bd)/dt            
                output.append(out)
                input.append(XA[i])
                
                t.append(now)
                now += dt
                
                spikes.append(np.sum(Bs)/500)
                

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(t, output)
        plt.plot(t, input)
        
        plt.plot(t, spikes)
        plt.show()
        
        
        
                
        return
        
        
        

            
    
    
if __name__=='__main__':
    unittest.main()
