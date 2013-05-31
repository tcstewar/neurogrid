from . import neurons
from . import encoders
from . import cache
from . import samples

import numpy as np

class Ensemble:
    def __init__(self, rows, cols, dimensions, seed, encoder_type='random'):
        rng = np.random.RandomState(seed)        
        self.rngs = [np.random.RandomState(rng.randint(0,0x7fffffff)) for i in range(10)]
        
        
        self.neurons = cache.get(name='neurons', N=rows*cols, seed=seed)
        if self.neurons is None: 
            self.neurons = neurons.RateNeuron(rows*cols, self.rngs[0])
            cache.set(self.neurons, name='neurons', N=rows*cols, seed=seed)
        
        self.encoders = cache.get(name='encoders', encoder_type=encoder_type, 
                                    rows=rows, cols=cols, dimensions=dimensions, seed=seed)
        if self.encoders is None: 
            if encoder_type=='random':
                self.encoders = encoders.random(rows*cols, dimensions, self.rngs[1])
            elif encoder_type=='swapped':   
                self.encoders = encoders.swapped(rows*cols, dimensions, self.rngs[1], rows, cols, iterations=200)
            elif encoder_type=='kohonen':
                self.encoders = encoders.kohonen(rows*cols, dimensions, self.rngs[1], rows, cols, iterations=200)
            cache.set(self.encoders, name='encoders', encoder_type=encoder_type, rows=rows, cols=cols, dimensions=dimensions, seed=seed)
                
        
        
