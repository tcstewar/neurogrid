import numpy as np
import samples


def create_stimulus(X, encoders, fr, fc):
    fp = fc + X * fr
    fm = fc - X * fr
    
    
    e_input = np.dot(fp, np.where(encoders.T>0, encoders.T, 0)) + np.dot(fm, np.where(encoders.T<0, -encoders.T, 0))
    i_input = np.dot(fm, np.where(encoders.T>0, encoders.T, 0)) + np.dot(fp, np.where(encoders.T<0, -encoders.T, 0))

    return e_input.T, i_input.T

    # Note: This simplistic approach does not work: 
    #  it misses a factor of (np.where(encoders.T<0, -encoders.T, 0)+np.where(encoders.T<0, -encoders.T, 0)) on fc
    #input = np.dot(encoders, X.T)
    #e_input = fc + input * fr
    #i_input = fc - input * fr 
    
    #return e_input, i_input




def classic(neurons, encoders, rng, sample_count=500, use_spikes=False, X=None, fr=500, fc=500, input_noise=0):
    N, D = encoders.shape
    if X is None:
        X = samples.random(sample_count, D, rng)


    e_input, i_input = create_stimulus(X, encoders, fr, fc)

    # TODO: jitter orthogonally, not everywhere
    if input_noise>0:
        noise = rng.randn(*e_input.shape)*input_noise
        e_input += noise
        i_input += noise
    
    
    if not use_spikes:
        A = neurons.rate(e_input, i_input)
    else:        
        A=[]
        for i in range(sample_count):
            AA = neurons.accumulate(e_input[:,i], i_input[:,i])
            A.append(AA)
        A = np.array(A).T    
        
            
    return X, A
    
def generate_from_ensemble(neurons, encoders, rng, source_ensemble, source_pstc, 
                           sample_count=500, X=None, 
                           fr_in=400, fc_in=500, fr_out=400, fc_out=500, input_noise=0):

    N, D = encoders.shape
    if X is None:
        X = samples.random(sample_count, D, rng)

    d_e, d_i = source_ensemble.get_dual_decoder(fr_in=fr_in, fc_in=fc_in, fr_out=fr_out, fc_out=fr_out, input_noise=input_noise)                
        
    e_input, i_input = create_stimulus(X, source_ensemble.encoders, fr_in, fc_in)

    # Do we need this?
    #if input_noise>0:
    #    noise = rng.randn(*e_input.shape)*input_noise
    #    e_input += noise
    #    i_input += noise
    
    w_e = np.dot(d_e, np.where(encoders.T>0, encoders.T, 0)) + np.dot(d_i, np.where(encoders.T<0, -encoders.T, 0))
    w_i = np.dot(d_i, np.where(encoders.T>0, encoders.T, 0)) + np.dot(d_e, np.where(encoders.T<0, -encoders.T, 0))

    
    dt = 0.001
    T0 = 10*source_pstc
    T = 100*source_pstc
    decay = np.exp(-dt/source_pstc)
    A=[]
    for i in range(sample_count):
        print '%d/%d'%(i,sample_count)
        e_in = e_input[:,i]
        i_in = i_input[:,i]
        
        fs = np.zeros(source_ensemble.encoders.shape[1], dtype='f')
        
        total = None
        t = -T0
        while t<T:
            s = source_ensemble.neurons.tick(e_in, i_in, dt)
            fs = fs * decay + s * (1-decay)

            in_e = np.dot(fs, w_e)/dt
            in_i = np.dot(fs, w_i)/dt
                
            s = neurons.tick(in_e, in_i, dt)
            if t>=0.0:
                if total is None: total = s
                else: total += s
            t += dt    
        A.append(total/T)
    A = np.array(A).T            
            
    return X, A

                           
    
	
	


