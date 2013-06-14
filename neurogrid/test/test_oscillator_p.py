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
            
        T = 3
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
        
        AAp_e, AAw_e = ng.probability.convert_weights(AAw_e)
        AAp_i, AAw_i = ng.probability.convert_weights(AAw_i)
        
        As = np.zeros(N1*N1, dtype='f')
        Afs = np.zeros(N1*N1, dtype='f')
        f_in_e = np.zeros(N1*N1, dtype='f')
        f_in_i = np.zeros(N1*N1, dtype='f')
        
        output = []
        
        spikes = []
        
        t = []
        now = 0
        for j in range(int(T/dt)):
                sp = np.where(As>0)[0]

                f_in_e *= decay
                f_in_i *= decay
                
                if len(sp)>0:
                    r = self.rng.random_sample((len(sp), AAw_e.shape[1]))
                    w_e = np.where(r<AAp_e[sp], AAw_e[sp], 0)
                    in_e = np.sum(w_e, axis=0)/dt

                    r = self.rng.random_sample((len(sp), AAw_i.shape[1]))
                    w_i = np.where(r<AAp_i[sp], AAw_i[sp], 0)
                    in_i = np.sum(w_i, axis=0)/dt

                    f_in_e += in_e * (1-decay)
                    f_in_i += in_i * (1-decay)
                
                As = A.neurons.tick(f_in_e, f_in_i, dt)
                

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
