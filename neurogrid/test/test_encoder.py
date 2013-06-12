import numpy as np
import scipy.signal
import unittest

import neurogrid as ng

class TestEncoders(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState()
        
    def _test_random(self):
        for D in range(1, 10):
            e = ng.encoders.random(10, D, self.rng)
            self.assertEqual(e.shape, (10,D))
            self.assertAlmostEqual(np.sum(np.sum(e*e, axis=1) - np.ones((10,))), 0)

    def test_diamond(self):
        #for D in range(1, 10):
        #    e = ng.encoders.random(10, D, self.rng)
        #    self.assertEqual(e.shape, (10,D))
        #    self.assertAlmostEqual(np.sum(np.abs(e), axis=1)) - np.ones((10,))), 0)

        e = ng.encoders.diamond(100, 2, self.rng)
        import pylab
        pylab.scatter(e[:,0], e[:,1])
        pylab.show()

            
    def _test_swapped(self):    
        k = np.array([0, 1, 0, 1, 4, 1, 0, 1, 0], dtype='f')/8
        k.shape = 3,3        

        for D in range(1,4):

            e = ng.encoders.swapped(100, D, self.rng, rows=10, cols=10)        
            e = e[:,0]
            e.shape = 10,10        
            e2 = scipy.signal.convolve2d(e, k, mode='same')        
            diff = np.sum((e-e2)**2)/100
            
            #print 'diff_swapped',diff
            self.assertLess(diff, 0.1)


            e = ng.encoders.random(100, D, self.rng)        
            e = e[:,0]
            e.shape = 10,10        
            e2 = scipy.signal.convolve2d(e, k, mode='same')        
            diff = np.sum((e-e2)**2)/100
            #print 'diff_random',diff
            
            self.assertGreater(diff, 0.2/D)        
            
    
    
if __name__=='__main__':
    unittest.main()
