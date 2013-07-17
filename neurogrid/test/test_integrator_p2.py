import numpy as np
import unittest

import neurogrid as ng

class TestIntegrator(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(seed=8)
        
    def test_integrator(self):
        N1 = 10   # number of neurons in the first pool is N1*N1
        N2 = 11   # number of neurons in the second pool is N2*N2
        
        D = 1           # dimensionality of the representation
        S = 300         # number of steps to simulate
        S_limit = 60    # stop at this step (to speed up testing parameters)
        T = 0.10        # length of a step (in seconds)
        dt = 0.001      # timestep for simulation
        pstc_e = 0.1    # excitatory post-synaptic time constant
        pstc_i = 0.1    # inhibitory post-synaptic time constant
        dendrite_width = 0  # amount of spreading from the dendrites (None is no dendrites, 0 is no spreading but hits neighbours)
        activity_noise = 0.1  # noise level to add to A before finding decoder
        decoder_mode = 'nonnegative'    # decoder solving method: 'classic' or 'nonnegative'
        
        rate_mode = False          # rate or spiking neurons
        probabilistic_bits = None  # number of bits to use for probabilistic spikes, or None for non-probabilistic
        
        nonlinear = 0      # amount of nonlinearity in the neurons (0 is none)
        balanced = True    # whether or not the excitatory and inhibitory synapses are perfectly balanced
        
        decay_e = np.exp(-dt/pstc_e)
        decay_i = np.exp(-dt/pstc_i)
        
        A = ng.ensemble.Ensemble(N1, N1, D, seed=None, nonlinear=nonlinear, balanced=balanced, encoder_type='random', dendrite_width=dendrite_width)
        B = ng.ensemble.Ensemble(N2, N2, D, seed=None, nonlinear=nonlinear, balanced=balanced, encoder_type='random', dendrite_width=dendrite_width)
        
        # compute decoders:  fr_in and fc_in are the firing rates expected as inputs to the neuron
        #                    and fr_out and fc_out are the outputs.  input_noise is common mode noise.
        #                    fc is the firing rate (+ or -) to represent zero, and fc+fr is the excitatory
        #                    rate for 1.
        # A dual decoder is the pair of decoders needed for decoding + and - separately (positive-only)
        Ad_e, Ad_i = A.get_dual_decoder(fr_in=400, fc_in=500, fr_out=250, fc_out=250, input_noise=50, mode=decoder_mode, activity_noise=activity_noise)
        Bd_e, Bd_i = B.get_dual_decoder(fr_in=250, fc_in=500, fr_out=250, fc_out=250, input_noise=200, mode=decoder_mode, activity_noise=activity_noise)

        # This is a standard NEF decoder (unconstrained)
        Bd = B.get_decoder(fr=250, fc=500)                

        # generate random input pulses every 20 steps        
        # pts = ng.samples.random(S/spacing, D, self.rng)             
        # self.rng.shuffle(pts)
        
        XA = np.zeros((S, D), dtype='f')
        XA[0:200:20] = np.linspace(1,0,10).reshape(10,D)
        XA[10:200:20] = np.linspace(-1,0,10).reshape(10,D)
        # spacing = 10
        # pt_squence = np.linspace(1,0,n_stim/2)
        # 
        # XA[[i*spacing for i in range(S/spacing)],:] = pts
        # print 'stim sequence:'
        # print XA
        

        # compute the connection weight matrices
        ABw_e = np.dot(Ad_e, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Ad_i, np.where(B.encoders.T<0, -B.encoders.T, 0))
        ABw_i = np.dot(Ad_i, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Ad_e, np.where(B.encoders.T<0, -B.encoders.T, 0))

        BBw_e = np.dot(Bd_e, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Bd_i, np.where(B.encoders.T<0, -B.encoders.T, 0))
        BBw_i = np.dot(Bd_i, np.where(B.encoders.T>0, B.encoders.T, 0)) + np.dot(Bd_e, np.where(B.encoders.T<0, -B.encoders.T, 0))
        
        # convert to probabilities
        if probabilistic_bits != None:
            ABp_e, ABw_e = ng.probability.convert_weights(ABw_e, bits=probabilistic_bits)
            ABp_i, ABw_i = ng.probability.convert_weights(ABw_i, bits=probabilistic_bits)
        
            BBp_e, BBw_e = ng.probability.convert_weights(BBw_e, bits=probabilistic_bits)
            BBp_i, BBw_i = ng.probability.convert_weights(BBw_i, bits=probabilistic_bits)
        
        
        As = np.zeros(N1*N1, dtype='f')     # spikes from population A
        Bs = np.zeros(N2*N2, dtype='f')     # spikes from population B
        Afs = np.zeros(N1*N1, dtype='f')    # filtered spikes
        Bfs = np.zeros(N2*N2, dtype='f')    # filtered spikes
        
        Bf_in_e = np.zeros(N2*N2, dtype='f')  # filtered excitatory input to B
        Bf_in_i = np.zeros(N2*N2, dtype='f')  # filtered inhibitory input to B
        
        
        output = []
        input = []
        
        # generate the excitatory and inhibitory stimuli for the given input sequence
        Ae_input, Ai_input = ng.activity.create_stimulus(XA, A.encoders, fc=500, fr=400)

                
        t = []
        now = 0
        for i in range(S)[:S_limit]:
            print i,'/', S
            for j in range(int(T/dt)):
                
                # run the A neurons
                if rate_mode:
                    As = A.neurons.rate(Ae_input[:,i], Ai_input[:,i])*dt
                else:
                    As = A.neurons.tick(Ae_input[:,i], Ai_input[:,i], dt)
                # create filtered output
                Afs = Afs * decay_e + As * (1-decay_e)
                
                # compute input to B
                
                Bf_in_e *= decay_e   # decay the existing input
                Bf_in_i *= decay_i

                if probabilistic_bits == None:
                    Bf_in_e += np.dot(As, ABw_e) * (1-decay_e)/dt
                    Bf_in_i += np.dot(As, ABw_i) * (1-decay_i)/dt
                    Bf_in_e += np.dot(Bs, BBw_e) * (1-decay_e)/dt
                    Bf_in_i += np.dot(Bs, BBw_i) * (1-decay_i)/dt
                else:    
                
                    Asp = np.where(As>0)[0]     # handle incoming spikes from A           
                    if len(Asp)>0:
                        # incoming excitation
                        r = self.rng.random_sample((len(Asp), ABw_e.shape[1]))
                        w_e = np.where(r<ABp_e[Asp], ABw_e[Asp], 0)
                        in_e = np.sum(w_e, axis=0)/dt
                        Bf_in_e += in_e * (1-decay_e)

                        # incoming inhibition
                        r = self.rng.random_sample((len(Asp), ABw_i.shape[1]))
                        w_i = np.where(r<ABp_i[Asp], ABw_i[Asp], 0)
                        in_i = np.sum(w_i, axis=0)/dt
                        Bf_in_i += in_i * (1-decay_i)
                    
                    Bsp = np.where(Bs>0)[0]    # handle recurrent spikes from B            
                    if len(Bsp)>0:
                        # recurrent excitation
                        r = self.rng.random_sample((len(Bsp), BBw_e.shape[1]))
                        w_e = np.where(r<BBp_e[Bsp], BBw_e[Bsp], 0)
                        in_e = np.sum(w_e, axis=0)/dt
                        Bf_in_e += in_e * (1-decay_e)

                        # recurrent inhibition
                        r = self.rng.random_sample((len(Bsp), BBw_i.shape[1]))
                        w_i = np.where(r<BBp_i[Bsp], BBw_i[Bsp], 0)
                        in_i = np.sum(w_i, axis=0)/dt
                        Bf_in_i += in_i * (1-decay_i)

                # run the B population
                
                if rate_mode:
                    Bs = B.neurons.rate(Bf_in_e, Bf_in_i)*dt
                else:    
                    Bs = B.neurons.tick(Bf_in_e, Bf_in_i, dt)
                Bfs = Bfs * decay_e + Bs * (1-decay_e)

                # decoder the output from B
                out = np.dot(Bfs, Bd)/dt
                output.append(out)
                input.append(XA[i])
                
                t.append(now)
                now += dt

        # plot results        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(t, output)
        plt.plot(t, input)
        plt.show()
        
                
        return
        
        
        

            
    
    
if __name__=='__main__':
    unittest.main()
