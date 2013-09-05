"""A simple neuron model that can have some NeuroGrid-like effects.

This is basically an LIF model with some pre-processing of its data.
The idea is that excitatory and inhibitory inputs can have different
gains, and there can be a non-linear interaction term (product of
excitatory and inhibitory inputs) that can also be added.  After this
pre-processing, a standard LIF model is used, giving both a rate and
a spiking implementation.
"""

import numpy as np

from . import dendrites


class RateNeuron:
    def __init__(self, N, rng, bias=500, nonlinear=1, balanced=False, 
                 tau_ref=0.002, tau_rc=0.02, input_scale=0.005, 
                 dendrite=None):
        """Create a set of rate neurons.
        
        These are LIF neurons with some preprocessing.  Inputs are
        spread using an optional dendrite matrix: `dot(e_in, dendrite)`
        and then combined in the following manner: 
        `e_in*e_gain + i_in*i_gain + e_in*i_in*nonlinear + bias`
        Each neuron gets its own randomly generated `e_gain`, `i_gain`,
        `nonlinear`, and `bias`.  The ranges for these random choices
        have been vaguely hand-fit to look like Neurogrid neurons.

        :param integer N: number of neurons
        :param numpy.random.RandomState rng: randon number generator
        :param float bias: scaling factor for bias current
        :param float nonlinear: amount of nonlinearity (0 is none, 
                                10 is close to that seen in neurogrid)
        :param float balanced: whether or not the gain on the inhibitory
                               input is the same as the excitatory gain
        :param float tau_ref: refractory time (in seconds)
        :param float tau_rc: membrane time constant (in seconds)
        :param matrix dendrite: NxN matrix for spreading activation
        :param float input_scale: overall scaling factor for the input      
        """
        self.input_scale = input_scale         
        self.tau_ref = tau_ref
        self.tau_rc = tau_rc         
        self.dendrite = dendrite
        
        # compute parameters for individual neurons:
        
        # gain on excitatory inputs
        self.e_gain = (rng.uniform(0.5, 4, N)**2) * input_scale 
        # background current
        self.bias = rng.uniform(-1.5, 2.5, N) * bias * input_scale * 5
        # gain on inhibitory inputs
        if balanced:
            self.i_gain = self.e_gain
        else:    
            self.i_gain = (rng.uniform(0.5, 4, N)**2) * input_scale
        # nonlinearity    
        self.nonlinear = rng.uniform(-0.001*nonlinear, 0.001*nonlinear, N) * input_scale
        
    def _compute_current(self, e_input, i_input):
        """Helper function to combine inputs into a single current.
        
        This handles either vector inputs or matrix inputs
        """
        if len(e_input.shape)==1:
            J = e_input*self.e_gain - i_input*self.i_gain + \
                    (e_input*i_input)*self.nonlinear
            if self.dendrite is not None:
                J = dendrites.apply_vector(J, self.dendrite)
            J = J + self.bias    
        else:    
            J = e_input*self.e_gain[:,None] - i_input*self.i_gain[:,None] + \
                    (e_input*i_input)*self.nonlinear[:,None]
            if self.dendrite is not None:
                J = dendrites.apply_matrix(J, self.dendrite)
            J = J + self.bias[:, None]    
        return J
        
    def rate(self, e_input, i_input):
        J = self._compute_current(e_input, i_input)
        np.seterr(divide='ignore', invalid='ignore') 
        isi = self.tau_ref - self.tau_rc * np.log(
            1 - 1.0 / np.maximum(J, 0))
        np.seterr(divide='warn', invalid='warn') 
        
        rate = np.where(J > 1, 1 / isi, 0)
        
        return rate                
        

class SpikeNeuron(RateNeuron):
    def __init__(self, N, rng, **args):
        RateNeuron.__init__(self, N, rng, **args)
        self.voltage = np.zeros(N, dtype='f')
        self.refractory_time = np.zeros(N, dtype='f')
        
    def tick(self, e_input, i_input, dt):
        J = self._compute_current(e_input, i_input)
        
        # Euler's method
        dV = dt / self.tau_rc * (J - self.voltage)

        # increase the voltage, ignore values below 0
        v = np.maximum(self.voltage + dV, 0)  
        
        # handle refractory period        
        post_ref = 1.0 - (self.refractory_time - dt) / dt

        # set any post_ref elements < 0 = 0, and > 1 = 1
        v *= np.clip(post_ref, 0, 1)
        
        # determine which neurons spike
        # if v > 1 set spiked = 1, else 0
        spiked = np.where(v > 1, 1.0, 0.0)
        
        # adjust refractory time (neurons that spike get
        # a new refractory time set, all others get it reduced by dt)

        # linearly approximate time since neuron crossed spike threshold
        overshoot = (v - 1) / dV 
        spiketime = dt * (1.0 - overshoot)

        # adjust refractory time (neurons that spike get a new
        # refractory time set, all others get it reduced by dt)
        self.refractory_time = np.where(
            spiked, spiketime + self.tau_ref, self.refractory_time - dt)

        self.voltage = v * (1 - spiked)
        
        return spiked

    def accumulate(self, e_input, i_input, dt=0.001, T=1, T0=0.1):

        # initialize by running neurons for a while
        for i in range(int(T0/dt)):
            s = self.tick(e_input, i_input, dt)

        total = None
        
        for i in range(int(T/dt)):
            s = self.tick(e_input, i_input, dt)
            if i == 0:
                total = s
            else:
                total += s
        return total / T
               
