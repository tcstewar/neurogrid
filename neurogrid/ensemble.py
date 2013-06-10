from . import neurons
from . import encoders
from . import cache
from . import samples
from . import activity
from . import decoders


import numpy as np

class Ensemble:
    def __init__(self, rows, cols, dimensions, seed, encoder_type='random'):
        self.seed = seed
        rng = np.random.RandomState(seed)        
        self.rngs = [np.random.RandomState(rng.randint(0,0x7fffffff)) for i in range(10)]
        
        
        self.cache_neurons = cache.Item(name='neurons', N=rows*cols, seed=seed)
        self.neurons = self.cache_neurons.get()
        if self.neurons is None:
            self.neurons = neurons.RateNeuron(rows*cols, self.rngs[0])
            self.cache_neurons.set(self.neurons)

        
        self.cache_encoders = cache.Item(name='encoders', encoder_type=encoder_type, 
                        rows=rows, cols=cols, dimensions=dimensions, 
                        seed=seed)
        self.encoders = self.cache_encoders.get()
        if self.encoders is None: 
            if encoder_type=='random':
                self.encoders = encoders.random(rows*cols, dimensions, self.rngs[1])
            elif encoder_type=='swapped':   
                self.encoders = encoders.swapped(rows*cols, dimensions, self.rngs[1], rows, cols, iterations=200)
            elif encoder_type=='kohonen':
                self.encoders = encoders.kohonen(rows*cols, dimensions, self.rngs[1], rows, cols, iterations=200)
            self.cache_encoders.set(self.encoders)    
            
    def get_decoder(self, name='X', func=None):
        item = cache.Item(name='decoder', decoder_name=name, seed=self.seed, neurons=self.cache_neurons, encoders=self.cache_encoders)
        d = item.get()
        if d is None:
            X, A = activity.classic(self.neurons, self.encoders, self.rngs[2])    
            if func is not None:
                X = func(X)
            d = decoders.classic(A, X, self.rngs[3])
            item.set(d)        
        return d
        
    def evaluate_decoder(self, name='X', func=None):  
        d = self.get_decoder(name=name, func=func)
        
        X_train, A_train = activity.classic(self.neurons, self.encoders, self.rngs[2])    
        X_test, A_test = activity.classic(self.neurons, self.encoders, self.rngs[4])    
        
        Xhat_train = np.dot(A_train.T, d)
        Xhat_test = np.dot(A_test.T, d)
        
        mse_train = np.sum((X_train-Xhat_train)**2)/len(X_train)
        mse_test = np.sum((X_test-Xhat_test)**2)/len(X_test)
        
        return mse_train, mse_test
            
             
