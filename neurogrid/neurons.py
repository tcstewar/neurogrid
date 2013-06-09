import numpy as np


class RateNeuron:
    def __init__(self, N, rng, bias=200, nonlinear=1, balanced=False, 
                 tau_ref=0.002, tau_rc=0.02, input_scale = 0.005):
                 
        self.input_scale = input_scale         
        self.tau_ref = tau_ref
        self.tau_rc = tau_rc         
        self.bias = rng.randn(N) * bias * input_scale
        self.e_gain = rng.uniform(0.5, 2, N) * input_scale
        if balanced:
            self.i_gain = self.e_gain
        else:    
            self.i_gain = rng.uniform(0.5, 2, N) * input_scale
        self.nonlinear = rng.uniform(-0.001*nonlinear, 0.001*nonlinear, N) * input_scale
        
    def compute_current(self, e_input, i_input):
        J = e_input*self.e_gain[:,None] - i_input*self.i_gain[:,None] + \
                (e_input*i_input)*self.nonlinear[:,None] + self.bias[:, None]
        return J
        
    def rate(self, e_input, i_input):
        J = self.compute_current(e_input, i_input)
         
        isi = self.tau_ref - self.tau_rc * np.log(
            1 - 1.0 / np.maximum(J, 0))
        
        rate = np.where(J > 1, 1 / isi, 0)
        
        return rate                
        

class SpikeNeuron(RateNeuron):
    def __init__(self, N, rng, **args):
        RateNeuron.__init__(self, N, rng, **args)
        self.voltage = np.zeros(N, dtype='f')
        self.refractory_time = np.zeros(N, dtype='f')
        
    def tick(self, e_input, i_input, dt):

        J = e_input*self.e_gain - i_input*self.i_gain + \
                (e_input*i_input)*self.nonlinear + self.bias

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
               
