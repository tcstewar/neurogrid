import numpy as np
import unittest

import neurogrid as ng

class TestNonlinearNeuron(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(seed=5)
        
    def test_communication(self):
        N1 = 30
        N2 = 31
        N3 = 32
        
        D = 2  
        S = 10
        T = 0.5
        dt = 0.001
        pstc = 0.04
        
        nonlinear = 10
        
        decay = np.exp(-dt/pstc)
        
        A = ng.ensemble.Ensemble(N1, N1, D, seed=1, nonlinear=nonlinear, balanced=False, encoder_type='diamond')
        B = ng.ensemble.Ensemble(N2, N2, D, seed=3, nonlinear=nonlinear, balanced=False, encoder_type='diamond')
        C = ng.ensemble.Ensemble(N3, N3, D, seed=4, nonlinear=nonlinear, balanced=False, encoder_type='diamond')
        
        Ad_e, Ad_i = A.get_dual_decoder(fr_in=400, fc_in=500, fr_out=180, fc_out=200)
        Bd_e, Bd_i = B.get_dual_decoder(fr_in=400, fc_in=500, fr_out=180, fc_out=200)
        
        
        #X, activity = ng.activity.classic(A.neurons, A.encoders, self.rng, fc=500, fr=400)
        #Ae = np.dot(activity.T, Ad_e)    
        #Ai = np.dot(activity.T, Ad_i)
        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.scatter(X, Ae, color='b')
        #plt.scatter(X, Ai, color='r')
        
            
        

        Cd = C.get_decoder(fr=180, fc=400)     # add
        Cd2 = C.get_decoder(fr=180*2, fc=400)  # average
        
        
        
        XA = ng.samples.random(S, D, self.rng)                
        XB = ng.samples.random(S, D, self.rng)                
        np.random.shuffle(XB)
        
        
        
        ACw_e = np.dot(Ad_e, np.where(C.encoders.T>0, C.encoders.T, 0)) + np.dot(Ad_i, np.where(C.encoders.T<0, -C.encoders.T, 0))
        ACw_i = np.dot(Ad_i, np.where(C.encoders.T>0, C.encoders.T, 0)) + np.dot(Ad_e, np.where(C.encoders.T<0, -C.encoders.T, 0))

        BCw_e = np.dot(Bd_e, np.where(C.encoders.T>0, C.encoders.T, 0)) + np.dot(Bd_i, np.where(C.encoders.T<0, -C.encoders.T, 0))
        BCw_i = np.dot(Bd_i, np.where(C.encoders.T>0, C.encoders.T, 0)) + np.dot(Bd_e, np.where(C.encoders.T<0, -C.encoders.T, 0))
        
        Afs = np.zeros(N1*N1, dtype='f')
        Bfs = np.zeros(N2*N2, dtype='f')
        Cfs = np.zeros(N3*N3, dtype='f')
        
        output = []
        output2 = []
        output_rate = []
        correct = []
        
        Ae_input, Ai_input = ng.activity.create_stimulus(XA, A.encoders, fc=500, fr=400)
        Be_input, Bi_input = ng.activity.create_stimulus(XB, B.encoders, fc=500, fr=400)
        
        
        
        Arate = A.neurons.rate(Ae_input, Ai_input)
        Brate = B.neurons.rate(Be_input, Bi_input)
        
        in_e = np.dot(Arate.T, ACw_e) + np.dot(Brate.T, BCw_e)
        in_i = np.dot(Arate.T, ACw_i) + np.dot(Brate.T, BCw_i)
        
        """
        print in_e.shape, in_i.shape
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(C.encoders[:,0], C.encoders[:,1])
        
        plt.figure()
        plt.scatter(Ae_input, Ai_input)
        #plt.figure()
        #plt.plot(XA, Arate.T)
        plt.figure()
        #plt.plot((in_e+in_i).T)
        #plt.scatter(in_e[:,2], in_i[:,2])
        index = 2
        plt.scatter(in_e[index,:], in_i[index,:], color='r')
        plt.scatter(in_e[index+1,:]+20, in_i[index+1,:], color='b')
        plt.scatter(in_e[index+2,:]+40, in_i[index+2,:], color='m')
        print XA[index:index+3]
        plt.show()
        1/0
        """
        
        
        
        Crate = C.neurons.rate(in_e.T, in_i.T)
        
        rate_out = np.dot(Crate.T, Cd)
        
        
        
        for i in range(S):
            for j in range(int(T/dt)):
                
                As = A.neurons.tick(Ae_input[:,i], Ai_input[:,i], dt)
                Afs = Afs * decay + As * (1-decay)

                Bs = B.neurons.tick(Be_input[:,i], Bi_input[:,i], dt)
                Bfs = Bfs * decay + Bs * (1-decay)
                
                in_e = np.dot(Afs, ACw_e)/dt + np.dot(Bfs, BCw_e)/dt
                in_i = np.dot(Afs, ACw_i)/dt + np.dot(Bfs, BCw_i)/dt

                Cs = C.neurons.tick(in_e, in_i, dt)
                Cfs = Cfs * decay + Cs * (1-decay)

               
                out = np.dot(Cfs, Cd)/dt            
                output.append(out)

                out2 = np.dot(Cfs, Cd2)/dt            
                output2.append(out2)
                
                correct.append(XA[i]+XB[i])
                
                
                output_rate.append(rate_out[i])
                

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(output_rate)
        plt.plot(correct)
        plt.figure()
        plt.plot(output)
        plt.plot(correct)
        plt.figure()
        plt.plot(output2)
        plt.plot(np.array(correct)/2)
        plt.show()

        
        
                
        return
        
        
        

            
    
    
if __name__=='__main__':
    unittest.main()
