import numpy as np
import unittest

import neurogrid as ng

class TestIntegrator(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(seed=8)
        
    def test_integrator(self):
        N1 = 20
        N2 = 21
        
        D = 1  
        S = 20
        T = 0.5
        dt = 0.001
        pstc = 0.1
        
        nonlinear = 10
        balanced = False
        
        decay = np.exp(-dt/pstc)
        
        A = ng.ensemble.Ensemble(N1, N1, D, seed=6, nonlinear=nonlinear, balanced=balanced, encoder_type='random')
        B = ng.ensemble.Ensemble(N2, N2, D, seed=7, nonlinear=nonlinear, balanced=balanced, encoder_type='random')
        
        Ad_e, Ad_i = A.get_dual_decoder(fr_in=400, fc_in=500, fr_out=50, fc_out=250, input_noise=50, sample_count=500)
        Bd_e, Bd_i = B.get_dual_decoder(fr_in=250, fc_in=500, fr_out=250, fc_out=250, input_noise=200, sample_count=500)
        
        Bd = B.get_decoder(fr=250, fc=500)                
        
        XA = ng.samples.random(S, D, self.rng)             
        self.rng.shuffle(XA)
        XA[[i*2+1 for i in range(S/2)],:] = 0   # zero out every other input to show steady response
        
        #print XA.shape
        #XA = np.array([[np.cos(t/(50.0)*2*np.pi)] for t in range(200)])
        #print XA.shape
        #T = 0.01
        #S = 200

        
        ABw_e = np.dot(Ad_e, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Ad_i, np.where(B.encoders.T<0, -B.encoders.T, 0))
        ABw_i = np.dot(Ad_i, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Ad_e, np.where(B.encoders.T<0, -B.encoders.T, 0))

        BBw_e = np.dot(Bd_e, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Bd_i, np.where(B.encoders.T<0, -B.encoders.T, 0))
        BBw_i = np.dot(Bd_i, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Bd_e, np.where(B.encoders.T<0, -B.encoders.T, 0))
        
        Afs = np.zeros(N1*N1, dtype='f')
        Bfs = np.zeros(N2*N2, dtype='f')
        
        output = []
        input = []
        
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

                scale = 1.0
                
                #in_e = 200 + np.dot(Bfs, BBw_e)/dt
                #in_i = 200 + np.dot(Bfs, BBw_i)/dt
                
                #print now, (in_e+in_i)[0]*scale#, in_e[0], in_i[0], Bfs[10]/dt, Bfs[15]/dt
                
                Bs = B.neurons.tick(in_e*scale, in_i*scale, dt)
                Bfs = Bfs * decay + Bs * (1-decay)

                out = np.dot(Bfs, Bd)/dt            
                output.append(out)
                input.append(XA[i])
                
                t.append(now)
                now += dt

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(t, output)
        plt.plot(t, input)
        plt.show()
        
                
        return
        
        
        

            
    
    
if __name__=='__main__':
    unittest.main()
