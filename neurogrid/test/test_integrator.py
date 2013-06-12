import numpy as np
import unittest

import neurogrid as ng

class TestIntegrator(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(seed=5)
        
    def test_integrator(self):
        N1 = 30
        N2 = 31
        
        D = 2  
        S = 10
        T = 0.5
        dt = 0.001
        pstc = 0.04
        
        nonlinear = 10
        
        decay = np.exp(-dt/pstc)
        
        A = ng.ensemble.Ensemble(N1, N1, D, seed=1, nonlinear=nonlinear, balanced=False, encoder_type='diamond')
        B = ng.ensemble.Ensemble(N2, N2, D, seed=3, nonlinear=nonlinear, balanced=False, encoder_type='diamond')
        
        Ad_e, Ad_i = A.get_dual_decoder(fr_in=400, fc_in=500, fr_out=180, fc_out=200)
        Bd_e, Bd_i = B.get_dual_decoder(fr_in=180, fc_in=400, fr_out=180, fc_out=200)
        
        Bd = B.get_decoder(fr=180, fc=400)                
        
        XA = ng.samples.random(S, D, self.rng)                
        XA[[i*2 for i in range(S/2)],:] = 0   # zero out every other input to show steady response
        
        ABw_e = np.dot(Ad_e, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Ad_i, np.where(B.encoders.T<0, -C.encoders.T, 0))
        ABw_i = np.dot(Ad_i, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Ad_e, np.where(B.encoders.T<0, -C.encoders.T, 0))

        BBw_e = np.dot(Bd_e, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Bd_i, np.where(B.encoders.T<0, -C.encoders.T, 0))
        BBw_i = np.dot(Bd_i, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Bd_e, np.where(B.encoders.T<0, -C.encoders.T, 0))
        
        Afs = np.zeros(N1*N1, dtype='f')
        Bfs = np.zeros(N2*N2, dtype='f')
        
        output = []
        
        Ae_input, Ai_input = ng.activity.create_stimulus(XA, A.encoders, fc=500, fr=400)
        
        
        for i in range(S):
            for j in range(int(T/dt)):
                
                As = A.neurons.tick(Ae_input[:,i], Ai_input[:,i], dt)
                Afs = Afs * decay + As * (1-decay)
                
                in_e = np.dot(Afs, ACw_e)/dt + np.dot(Bfs, BCw_e)/dt
                in_i = np.dot(Afs, ACw_i)/dt + np.dot(Bfs, BCw_i)/dt
                
                
                Bs = B.neurons.tick(in_e, in_i, dt)
                Bfs = Bfs * decay + Cs * (1-decay)

                out = np.dot(Bfs, Bd)/dt            
                output.append(out)


        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(output)
        plt.show()
        
                
        return
        
        
        

            
    
    
if __name__=='__main__':
    unittest.main()
