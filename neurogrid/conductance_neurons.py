import numpy as np

from . import dendrites


class ConductanceNeuron:
    def __init__(self, N, rng, bias=0.5,
                 tau_ref=0.002, tau_rc=0.02, 
                 e_e=3.0, e_i=0.1, sigma=0.1,
                 gain_e=1.0, gain_i=1.0):
        self.tau_ref = tau_ref
        self.tau_rc = tau_rc         
        
        e_e = np.log(e_e)-(sigma)**2/2        
        self.e_e = np.abs(rng.lognormal(e_e, sigma, N))
        e_i = np.log(e_i)-(sigma)**2/2        
        self.e_i = np.abs(rng.lognormal(e_i, sigma, N))
        
        gain_e = np.log(gain_e)-(sigma)**2/2        
        self.gain_e = np.abs(rng.lognormal(gain_e, sigma, N))
        gain_i = np.log(gain_i)-(sigma)**2/2        
        self.gain_i = np.abs(rng.lognormal(gain_i, sigma, N))
        
        
        bias = np.log(bias)-(sigma)**2/2        
        self.bias = rng.lognormal(bias, sigma, N) 
                
        self.voltage = np.zeros(N, dtype='f')
        self.refractory_time = np.zeros(N, dtype='f')
        
        
    def tick(self, g_e, g_i, dt):
    
    
        threshold = 10.0
        
        vv = self.voltage        
        dV = dt / self.tau_rc * (-vv + 
                      vv**2/2 + self.gain_e*g_e*(self.e_e-vv) + 
                      self.gain_i*g_i*(self.e_i-vv) + self.bias)           
        
        # increase the voltage, ignore values below 0
        v = np.maximum(self.voltage + dV, 0)  
        
        # handle refractory period        
        post_ref = 1.0 - (self.refractory_time - dt) / dt

        # set any post_ref elements < 0 = 0, and > 1 = 1
        v *= np.clip(post_ref, 0, 1)
        
        # determine which neurons spike
        # if v > 1 set spiked = 1, else 0
        spiked = np.where(v > threshold, 1.0, 0.0)
        
        # adjust refractory time (neurons that spike get
        # a new refractory time set, all others get it reduced by dt)

        # linearly approximate time since neuron crossed spike threshold
        overshoot = (v - threshold) / dV 
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
               
