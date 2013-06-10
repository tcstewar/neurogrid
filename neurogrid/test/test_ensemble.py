import numpy as np
import unittest

import neurogrid as ng

class TestActivity(unittest.TestCase):
    def test_classic(self):
        N = 10
        D = 5
    
        e = ng.ensemble.Ensemble(N, N, D, seed=1)
        
        print e.evaluate_decoder()
        

    
        
            
    
    
if __name__=='__main__':
    unittest.main()
