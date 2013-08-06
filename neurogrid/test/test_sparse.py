import sys
sys.path.append('.')

import numpy as np
import unittest

import neurogrid as ng

class TestSparse(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState()
        
    def test_classic(self):

        N = 200
        fc = 500
        fr = 400
        input_noise = 100
        balanced = False
        nonlinear = 10
        decoder_noise = 0.1
        D = 1
        show_plot = False
        decoder_bits = 8
        target_sparsity = 0.8
        sparsity_algorithm = 'greedy'
        
        def function(x):
            return x
        
        n = ng.neurons.RateNeuron(N, self.rng, balanced=balanced, nonlinear=nonlinear)
        e = ng.encoders.random(N, D, self.rng)
    
        X, A = ng.activity.classic(n, e, self.rng, fc=fc, fr=fr, input_noise=input_noise)
        
        fX = function(X)
        
        
        if sparsity_algorithm=='random':
            d = ng.decoders.sparse_random(A, fX, self.rng, noise=decoder_noise, sparsity=target_sparsity)
        elif sparsity_algorithm=='greedy':    
            d = ng.decoders.sparse_greedy(A, fX, self.rng, noise=decoder_noise, sparsity=target_sparsity)            
        else:
            d = ng.decoders.classic(A, fX, self.rng, noise=decoder_noise)
        
        
        
        if decoder_bits is not None:
            p, w = ng.probability.convert_weights(d, bits=decoder_bits)
            d = p*w
        
        
        # compute sparsity        
        items = np.product(d.shape)
        sparsity = float(items-len(np.nonzero(d)[0]))/items
        print 'zeros:', sparsity
        
        
        
        # generate a new set of data for rmse computation
        X2, A2 = ng.activity.classic(n, e, self.rng, fc=fc, fr=fr, input_noise=input_noise)        
        
        fX2 = function(X)
        
        fX2hat = np.dot(A2.T, d)        
        rmse = np.sqrt(np.sum((fX2-fX2hat)**2)/len(fX2))
        print 'rmse:', rmse
        
    
        if show_plot:
            import matplotlib.pyplot as plt
            plt.scatter(X2, fX2hat)
            plt.plot([-1, 1], [-1, 1])
            plt.show()
    
        
            
    
    
if __name__=='__main__':
    unittest.main()
