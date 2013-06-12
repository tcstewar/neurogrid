import numpy as np
import unittest

import neurogrid as ng


class TestTuningCurve(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(seed=5)
        
    def test_curves(self):
        N = 3
        D = 2 
        S = 100
        
        XA = ng.samples.random(S, D, self.rng)                

        A = ng.ensemble.Ensemble(N, N, D, seed=1, nonlinear=10, balanced=False)
        
        
        J_e, J_i = ng.activity.create_stimulus(XA, A.encoders, fc=500, fr=200)
        
        rate = A.neurons.rate(J_e, J_i)
        
        
        
        
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        for i in range(N*N):
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(XA[:,0], XA[:,1], rate[i,:])
                
        plt.show()
        
        

            
    
    
if __name__=='__main__':
    unittest.main()
