import numpy as np
import unittest

import neurogrid as ng

class TestNonlinearCommunication(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(seed=5)
        
    def test_communication(self):
        N1 = 30
        N2 = 31
        
        D = 2 
        S = 20
        
        nonlinear = 10
        balanced = False
        
        encoder_type = 'random'
        
        A = ng.ensemble.Ensemble(N1, N1, D, seed=1, nonlinear=nonlinear, 
                                    balanced=balanced, encoder_type=encoder_type)
        B = ng.ensemble.Ensemble(N2, N2, D, seed=3, nonlinear=nonlinear, 
                                    balanced=balanced, encoder_type=encoder_type)
        
        Ad_e, Ad_i = A.get_dual_decoder(fr_in=400, fc_in=500, fr_out=400, fc_out=500)
        Bd = B.get_decoder(fr=400, fc=500)
        Ad = A.get_decoder(fr=400, fc=500)
        
        XA = ng.samples.random(S, D, self.rng)                        
        
        enc = B.encoders
        #norm = np.sum(np.abs(enc), axis=1)
        #enc = enc/norm[:,None]
        
        ABw_e = np.dot(Ad_e, np.where(enc.T>0, enc.T, 0)) + np.dot(Ad_i, np.where(enc.T<0, -enc.T, 0))
        ABw_i = np.dot(Ad_i, np.where(enc.T>0, enc.T, 0)) + np.dot(Ad_e, np.where(enc.T<0, -enc.T, 0))

        Afs = np.zeros(N1*N1, dtype='f')
        Bfs = np.zeros(N2*N2, dtype='f')
        
        
        Ae_input, Ai_input = ng.activity.create_stimulus(XA, A.encoders, fc=500, fr=400)
        Be_input, Bi_input = ng.activity.create_stimulus(XA, B.encoders, fc=500, fr=400)
        
        #print Ae_input.shape, Ai_input.shape
        Arate = A.neurons.rate(Ae_input, Ai_input)
        
        in_e = np.dot(Arate.T, ABw_e)
        in_i = np.dot(Arate.T, ABw_i)
        #print in_e.shape, in_i.shape

        Brate = B.neurons.rate(in_e.T, in_i.T)
        
        output = np.dot(Brate.T, Bd)
        outputA = np.dot(Arate.T, Ad)


        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(output, linewidth=3)
        plt.plot(XA)

        plt.figure()
        plt.plot(outputA, linewidth=3)
        plt.plot(XA)
        
        
        plt.figure()
        plt.plot(output-XA)
        
        #plt.figure()
        #plt.scatter(Ae_input[:,0], Ai_input[:,0])

        #plt.figure()
        #plt.scatter(in_e[0,:], in_i[0,:])
        #plt.scatter(Be_input[:,0], Bi_input[:,0], color='r')
        
        
        plt.show()

        
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
        
                
        """
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
        """
        
        
                
        return
        
        
        

            
    
    
if __name__=='__main__':
    unittest.main()
