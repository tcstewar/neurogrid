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
        S = 200
        T = 0.05
        dt = 0.001
        pstc_e = 0.1
        pstc_i = 0.11
        
        nonlinear = 10
        balanced = False
        
        decay_e = np.exp(-dt/pstc_e)
        decay_i = np.exp(-dt/pstc_i)
        
        A = ng.ensemble.Ensemble(N1, N1, D, seed=6, nonlinear=nonlinear, balanced=balanced, encoder_type='random')
        B = ng.ensemble.Ensemble(N2, N2, D, seed=7, nonlinear=nonlinear, balanced=balanced, encoder_type='random')
        
        Ad_e, Ad_i = A.get_dual_decoder(fr_in=400, fc_in=500, fr_out=250, fc_out=250, input_noise=50)
        Bd_e, Bd_i = B.get_dual_decoder(fr_in=250, fc_in=500, fr_out=250, fc_out=250, input_noise=200)



        
        Bd = B.get_decoder(fr=250, fc=500)                
        
        spacing = 20
        pts = ng.samples.random(S/spacing, D, self.rng)             
        self.rng.shuffle(pts)
        
        XA = np.zeros((S, D), dtype='f')
        
        
        XA[[i*spacing for i in range(S/spacing)],:] = pts
        
        #print XA.shape
        #XA = np.array([[np.cos(t/(50.0)*2*np.pi)] for t in range(200)])
        #print XA.shape
        #T = 0.01
        #S = 200

        
        ABw_e = np.dot(Ad_e, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Ad_i, np.where(B.encoders.T<0, -B.encoders.T, 0))
        ABw_i = np.dot(Ad_i, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Ad_e, np.where(B.encoders.T<0, -B.encoders.T, 0))

        BBw_e = np.dot(Bd_e, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Bd_i, np.where(B.encoders.T<0, -B.encoders.T, 0))
        BBw_i = np.dot(Bd_i, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Bd_e, np.where(B.encoders.T<0, -B.encoders.T, 0))
        
        ABp_e, ABw_e = ng.probability.convert_weights(ABw_e)
        ABp_i, ABw_i = ng.probability.convert_weights(ABw_i)
        
        BBp_e, BBw_e = ng.probability.convert_weights(BBw_e)
        BBp_i, BBw_i = ng.probability.convert_weights(BBw_i)
        
        
        As = np.zeros(N1*N1, dtype='f')
        Bs = np.zeros(N2*N2, dtype='f')
        Afs = np.zeros(N1*N1, dtype='f')
        Bfs = np.zeros(N2*N2, dtype='f')
        
        Bf_in_e = np.zeros(N2*N2, dtype='f')
        Bf_in_i = np.zeros(N2*N2, dtype='f')
        
        
        output = []
        input = []
        
        Ae_input, Ai_input = ng.activity.create_stimulus(XA, A.encoders, fc=500, fr=400)

                
        t = []
        now = 0
        for i in range(S):
            print i, S
            for j in range(int(T/dt)):
                
                As = A.neurons.tick(Ae_input[:,i], Ai_input[:,i], dt)
                Afs = Afs * decay_e + As * (1-decay_e)
                
                Bf_in_e *= decay_e
                Bf_in_i *= decay_i

                Asp = np.where(As>0)[0]                
                if len(Asp)>0:
                    r = self.rng.random_sample((len(Asp), ABw_e.shape[1]))
                    w_e = np.where(r<ABp_e[Asp], ABw_e[Asp], 0)
                    in_e = np.sum(w_e, axis=0)/dt

                    r = self.rng.random_sample((len(Asp), ABw_i.shape[1]))
                    w_i = np.where(r<ABp_i[Asp], ABw_i[Asp], 0)
                    in_i = np.sum(w_i, axis=0)/dt

                    Bf_in_e += in_e * (1-decay_e)
                    Bf_in_i += in_i * (1-decay_i)
                
                Bsp = np.where(Bs>0)[0]                
                if len(Bsp)>0:
                    r = self.rng.random_sample((len(Bsp), BBw_e.shape[1]))
                    w_e = np.where(r<BBp_e[Bsp], BBw_e[Bsp], 0)
                    in_e = np.sum(w_e, axis=0)/dt

                    r = self.rng.random_sample((len(Bsp), BBw_i.shape[1]))
                    w_i = np.where(r<BBp_i[Bsp], BBw_i[Bsp], 0)
                    in_i = np.sum(w_i, axis=0)/dt

                    Bf_in_e += in_e * (1-decay_e)
                    Bf_in_i += in_i * (1-decay_i)

                
                Bs = B.neurons.tick(Bf_in_e, Bf_in_i, dt)
                Bfs = Bfs * decay_e + Bs * (1-decay_e)

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
