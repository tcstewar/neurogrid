"""Generate the activity matrix for a group of neurons.

This is an NxS matrix where N is the number of neurons and S is the
number of sample points, and each value is the measure of the
steady-state firing rate of that neurons for that stimulus.
"""

import numpy as np
import samples

def create_stimulus(X, encoders, fr, fc):
    """Convert from X (represented space) to firing rates.
    
    This does the mapping from represented data to actual neural input.
    For standard NEF, this was just `dot(encoders, X.T)`.  However,
    for neurogrid we want to separate excitatory and inhibitory inputs
    since the neuron does not respond the same to (excite=2, inhibit=1)
    as (excite=3, inhibit=2).
    
    The idea is to map X=1 to (excite=fc+fr, inhibit=fc-fr) for a
    neuron with encoder=1, and to (excite=fc-fr, inhibit=fc+fr) for
    a neuron with encoder=-1.  For multidimensional encoders, we
    form some combination of these.
    
    Note that with this system, a neuron with encoder=(0,1) will always
    get a total firing rate of 2*fc, but a neuron with encoder=(0.7, 0.7)
    will get a total firing rate of 2*1.4*fc.  We might want to look at
    other schemes (but this will change how we compute the connection
    weight matrix from the encoders/decoders).

    :param matrix X: An SxD matrix of S represented values to convert
    :param matrix encoders: An NxD matrix of encoders to use
    :param float fr: The increase in firing rate from 0 to +1 (or -1)
    :param float fc: The firing rate (excitatory and inhibitory) for 0    
    :returns: Two NxS matrices (excitatory and inhibitory)
    """

    assert len(X.shape) == 2
    assert len(encoders.shape) == 2
    assert X.shape[1] == encoders.shape[1]
    assert fc >= fr
    
    fp = fc + X * fr
    fm = fc - X * fr
    
    e_input = np.dot(fp, np.where(encoders.T>0, encoders.T, 0)) + \
              np.dot(fm, np.where(encoders.T<0, -encoders.T, 0))
    i_input = np.dot(fm, np.where(encoders.T>0, encoders.T, 0)) + \
              np.dot(fp, np.where(encoders.T<0, -encoders.T, 0))

    return e_input.T, i_input.T

    # Note: This simplistic approach does not work: 
    #  it misses a factor of (np.where(encoders.T<0, -encoders.T, 0)+
    #                     np.where(encoders.T<0, -encoders.T, 0)) on fc
    #
    #input = np.dot(encoders, X.T)
    #e_input = fc + input * fr
    #i_input = fc - input * fr 




def classic(neurons, encoders, rng, sample_count=500, 
            use_spikes=False, X=None, fr=500, fc=500, input_noise=0):
    """Generate the A matrix using constant inputs.

    To do this, we take our X values, convert them into neuron
    input, pass that into the neurons, and determine the resulting
    firing rates.  For this classic version, we give a constant input
    into each neuron.
    
    :param Neuron neurons: the neurons to use
    :param matrix encoders: an NxD encoder matrix
    :param numpy.random.RandomState rng: random number generator
    :param integer sample_count: number of samples for X
    :param boolean use_spikes: if True, simulate spiking behaviour of
                               neurons; otherwise use rate mode
    :param X: SxD matrix of values to sample over (if None, generate
              these randomly using *sample_count* for S)    
    :type X: matrix or None    
    :param float fr: range to use when converting X to neural inputs
    :param float fc: mean to use when converting X to neural inputs
    :param float input_noise: standard deviation of common mode noise
                              to add to the neural inputs
    :returns: A tuple of an SxD matrix (the sample values used) and
              an NxS matrix (the steady-state firing rate for each 
              neuron for each sample point)     
    """                           
    N, D = encoders.shape
    
    if X is None:
        # generate some samples
        X = samples.random(sample_count, D, rng)

    # convert from represented space to stimulus for each neuron
    e_input, i_input = create_stimulus(X, encoders, fr, fc)

    # add common mode noise to the input.  This is noise that
    #  increases excitation and inhibition equally, and is meant
    #  to make the system more robust to variability
    if input_noise>0:
        noise = rng.randn(*e_input.shape)*input_noise
        e_input += noise
        i_input += noise
    
    
    if not use_spikes:
        # if we're in rate mode, just compute everything all at
        #  once, since neurons.rate() accepts matrices (not just
        #  vectors)
        A = neurons.rate(e_input, i_input)
    else:        
        # if we're in spike mode, run the neurons for a while with
        #  each input and record the average spike rate
        A=[]
        for i in range(sample_count):
            # simulate for 0.1 seconds, then average over 1.0 seconds
            AA = neurons.accumulate(e_input[:,i], i_input[:,i], 
                                    T=1.0, T0=0.1)
            A.append(AA)
        A = np.array(A).T    
            
    return X, A
    
def generate_from_ensemble(neurons, encoders, rng, 
                           source_ensemble, source_pstc, 
                           simulation_time=None, startup_time=None,
                           sample_count=500, X=None, 
                           fr_source=400, fc_source=500, 
                           fr=400, fc=500, input_noise=0):
    """Generate the A matrix using input coming from another ensemble.

    Instead of driving the neurons with a steady-state input, here we
    provide input by giving a steady-state input into some *other*
    group of neurons, decoding their output using the dual-decoder
    approach, generating spikes, passing that through a weight matrix,
    and finally giving that resulting data into the neurons we're
    actually interested in.
    
    :param Neuron neurons: the neurons to use
    :param matrix encoders: an NxD encoder matrix
    :param numpy.random.RandomState rng: random number generator
    :param Ensemble source_ensemble: the ensemble of neurons to get
                                     stimulation from
    :param float source_pstc: the post-synaptic time constant between
                              the source_ensemble and these neurons    
    :param float simulation_time: the amount of time to run the neurons
                                  for to find an average (if None, use
                                  100*source_pstc)
    :param float startup_time: the amount of time to run the neurons
                               for before finding an average, so as to
                               avoid startup transients (if None, use
                               10*source_pstc)                                  
    :param integer sample_count: number of samples for X
    :param boolean use_spikes: if True, simulate spiking behaviour of
                               neurons; otherwise use rate mode
    :param X: SxD matrix of values to sample over (if None, generate
              these randomly using *sample_count* for S)    
    :type X: matrix or None    
    :param float fr_source: range to use when converting X to neural
                        inputs for the source_ensemble
    :param float fc_source: mean to use when converting X to neural
                        inputs for the source_ensemble
    :param float fr: range to use for the target inputs to the neurons
    :param float fc: mean to use for the target inputs to the neurons
                        
    :param float input_noise: standard deviation of common mode noise
                              to add to the source_ensemble
    """

    N, D = encoders.shape
    if X is None:
        # generate some samples
        X = samples.random(sample_count, D, rng)

    # get decoders for the given source ensemble    
    d_e, d_i = source_ensemble.get_dual_decoder(fr_in=fr_source, 
                  fc_in=fc_source, fr_out=fr, fc_out=fc, 
                  input_noise=input_noise)                
        
        
    # convert from represented space to stimulus for each neuron in
    #  the source ensemble
    e_input, i_input = create_stimulus(X, source_ensemble.encoders, 
                                       fr_source, fc_source)
    
    # create weight matrices
    w_e = np.dot(d_e, np.where(encoders.T>0, encoders.T, 0)) + \
          np.dot(d_i, np.where(encoders.T<0, -encoders.T, 0))
    w_i = np.dot(d_i, np.where(encoders.T>0, encoders.T, 0)) + \
          np.dot(d_e, np.where(encoders.T<0, -encoders.T, 0))
    
    dt = 0.001
    
    # how much time to run the neuron model before recording data
    #   (this avoids transients from startup or changing the stimulus)
    if startup_time is None: startup_time = 10*source_pstc
    # how much time to record data for
    if simulation_time is None: simulation_time = 100*source_pstc

    # scaling factor for implementing pstc filter
    decay = np.exp(-dt/source_pstc)
    
    # filtered spikes coming out of the source_ensemble
    fs = np.zeros(source_ensemble.encoders.shape[1], dtype='f')
    
    A=[]
    for i in range(sample_count):
        print '%d/%d'%(i,sample_count)
        
        # get the steady-state input to present
        e_in = e_input[:,i]
        i_in = i_input[:,i]
        
        # simulate for this input, with t going from -startup_time
        #  to +simulation_time, adding up spikes when t>0
        total = None
        t = -startup_time
        while t<simulation_time:
            # feed the steady-state input into the source_ensemble
            s = source_ensemble.neurons.tick(e_in, i_in, dt)

            # TODO: Add probabilistic spiking?
            
            # update the filtered spikes
            fs = fs * decay + s * (1-decay)

            # multiply output by connection weight matrix
            in_e = np.dot(fs, w_e)/dt
            in_i = np.dot(fs, w_i)/dt
            
            # feed resulting input into neurons            
            s = neurons.tick(in_e, in_i, dt)
            
            if t>=0.0: # if we should be recording this data
                # accumulate spikes
                if total is None: total = s
                else: total += s
            t += dt    
        # record the average spike rate    
        A.append(total / simulation_time)
    A = np.array(A).T            
            
    return X, A

                           
    
	
	


