from . import neurons
from . import encoders
from . import cache
from . import samples
from . import activity
from . import decoders
from . import dendrites

import numpy as np

class Ensemble:
    def __init__(self, rows, cols, dimensions, seed, encoder_type='random', bias=500, balanced=True, nonlinear=0, dendrite_width=None, input_scale=0.005):
        self.seed = seed
        rng = np.random.RandomState(seed)        
        self.rngs = [np.random.RandomState(rng.randint(0,0x7fffffff)) for i in range(10)]

        self.cache_encoders = cache.Item(name='encoders', encoder_type=encoder_type, 
                        rows=rows, cols=cols, dimensions=dimensions, 
                        seed=seed)
        self.encoders = self.cache_encoders.get()
        if self.encoders is None: 
            if encoder_type=='random':
                self.encoders = encoders.random(rows*cols, dimensions, self.rngs[1])
            elif encoder_type=='diamond':
                self.encoders = encoders.diamond(rows*cols, dimensions, self.rngs[1])
            elif encoder_type=='swapped':   
                self.encoders = encoders.swapped(rows*cols, dimensions, self.rngs[1], rows, cols, iterations=200)
            elif encoder_type=='kohonen':
                self.encoders = encoders.kohonen(rows*cols, dimensions, self.rngs[1], rows, cols, iterations=200)
            self.cache_encoders.set(self.encoders)    

        
        self.cache_neurons = cache.Item(name='neurons', N=rows*cols, seed=seed, bias=bias, balanced=balanced, nonlinear=nonlinear, dendrite_width=dendrite_width, input_scale=input_scale)
        self.neurons = self.cache_neurons.get()
        if self.neurons is None:
            dendrite = None
            if dendrite_width is not None:
                dendrite = dendrites.blur_kernel(rows, cols, dendrite_width)
        
            self.neurons = neurons.SpikeNeuron(rows*cols, self.rngs[0], bias=bias, balanced=balanced, nonlinear=nonlinear, dendrite=dendrite, input_scale=input_scale)
            
            self.cache_neurons.set(self.neurons)

    def get_average_rate(self, fc=500, fr=500):
        X, A = activity.classic(self.neurons, self.encoders, self.rngs[6], fc=fc, fr=fr) 
        return np.sum(A)/(A.shape[0]*A.shape[1])
                
    def get_decoder(self, name='X', func=None, mode='classic', fc=500, fr=500, input_noise=0):
        item = cache.Item(name='decoder', decoder_name=name, seed=self.seed, neurons=self.cache_neurons, encoders=self.cache_encoders, mode=mode, fc=fc, fr=fr, input_noise=input_noise)
        d = item.get()
        if d is None:
            X, A = activity.classic(self.neurons, self.encoders, self.rngs[2], fc=fc, fr=fr, input_noise=input_noise) 

            #import pylab
            #pylab.plot(X, A.T)
            #coverage = np.sum(np.where(A.T>0, 1.0, 0.0))/(A.shape[0]*A.shape[1])
            #max_rate = np.max(A)
            #avg_rate = np.sum(A)/(A.shape[0]*A.shape[1])
            #pylab.title('Coverage: %1.3f  Max:%1.3f  Avg:%1.3f'%(coverage, max_rate, avg_rate))
            #pylab.show()
            
            if func is not None:
                X = func(X)
            dfunc = {'classic':decoders.classic, 
                    'nonnegative':decoders.nonnegative}[mode]
            d = dfunc(A, X, self.rngs[3])
            item.set(d)        
        return d

    def get_dual_decoder(self, name='X', func=None, mode='nonnegative', fc_in=500, fr_in=500, fc_out=500, fr_out=500, input_noise=0, activity_noise=0.1):
        item = cache.Item(name='decoder', decoder_name=name, seed=self.seed, 
                          neurons=self.cache_neurons, encoders=self.cache_encoders, 
                          mode=mode, fc_in=fc_in, fr_in=fr_in, fc_out=fc_out, fr_out=fr_out, input_noise=input_noise, activity_noise=activity_noise)
        d = item.get()
        if d is None:
            X, A = activity.classic(self.neurons, self.encoders, self.rngs[2], fc=fc_in, fr=fr_in, input_noise=input_noise)    
            if func is not None:
                X = func(X)
            dfunc = {'classic':decoders.classic, 
                    'nonnegative':decoders.nonnegative}[mode]
            X_e = fc_out + fr_out * X       
            X_i = fc_out - fr_out * X       
                        
            d_e = dfunc(A, X_e, self.rngs[3], noise=activity_noise)
            d_i = dfunc(A, X_i, self.rngs[3], noise=activity_noise)
            d = d_e, d_i
            
            #X_e_hat = np.dot(A.T, d_e)
            #mse = np.sum((X_e-X_e_hat)**2)/len(X_e)
            #print 'rmse',np.sqrt(mse)
            
            item.set(d)        
        return d


        
    def evaluate_decoder(self, name='X', func=None):  
        d = self.get_decoder(name=name, func=func)
        
        # TODO: the rng needs to be reset if we want to recompute the mse_train
        #X_train, A_train = activity.classic(self.neurons, self.encoders, self.rngs[2])    
        X_test, A_test = activity.classic(self.neurons, self.encoders, self.rngs[4])    
        
        #Xhat_train = np.dot(A_train.T, d)
        Xhat_test = np.dot(A_test.T, d)
        
        #mse_train = np.sum((X_train-Xhat_train)**2)/len(X_train)
        mse_test = np.sum((X_test-Xhat_test)**2)/len(X_test)
        
        return mse_test
            
             
