from . import neurons
from . import encoders
from . import cache
from . import samples

import numpy as np

class Ensemble:
    def __init__(self, rows, cols, dimensions, seed, encoder_type='random'):
        rng = np.random.RandomState(seed)        
        self.rngs = [np.random.RandomState(rng.randint(0,0x7fffffff)) for i in range(10)]
        
        
        with cache.Item(name='neurons', N=rows*cols, seed=seed) as item:
            self.neurons = item.get()
            if self.neurons is None:
                self.neurons = neurons.RateNeuron(rows*cols, self.rngs[0])
                item.set(self.neurons)
        
        with cache.Item(name='encoders', encoder_type=encoder_type, 
                        rows=rows, cols=cols, dimensions=dimensions, 
                        seed=seed) as item:
            self.encoders = item.get()
            if self.encoders is None: 
                if encoder_type=='random':
                    self.encoders = encoders.random(rows*cols, dimensions, self.rngs[1])
                elif encoder_type=='swapped':   
                    self.encoders = encoders.swapped(rows*cols, dimensions, self.rngs[1], rows, cols, iterations=200)
                elif encoder_type=='kohonen':
                    self.encoders = encoders.kohonen(rows*cols, dimensions, self.rngs[1], rows, cols, iterations=200)
                item.set(self.encoders)    
 
            
             
