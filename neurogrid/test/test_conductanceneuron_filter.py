import numpy as np
import neurogrid as ng
import time

import neurogrid.filter
reload(neurogrid.filter)

rng = np.random.RandomState(seed=1)

N = 5
T = 0.200


n1 = ng.conductance_neurons.ConductanceNeuron(N, rng, bias=0.5, 
               gain_e=50.0, gain_i=10.0)

rng = np.random.RandomState(seed=1)
n2 = ng.conductance_neurons.ConductanceNeuron(N, rng, bias=0.5, 
               gain_e=50.0, gain_i=10.0)
              
               
dt = 0.001
scale = 0.001/dt

fr_e = 500
fr_i = 200

pstc = 0.01

fs_e = np.zeros(N, dtype='f')
fs_i = np.zeros(N, dtype='f')


def make_spike_train(T, rate):
    times = []
    if rate<=0: return times
    t = 0.0
    while t<T:
        interval = rng.exponential(1.0/rate)
        t += interval
        if t<T:
            times.append(t)
    return times

spike_times_e = make_spike_train(T, fr_e)    
spike_times_i = make_spike_train(T, fr_i)

train1 = []
train2 = []
data_fs_e = []
data_fs_i = []
data_s_e = []
data_s_i = []
t = 0

filter_e = neurogrid.filter.Filter(g_sat=1.0, tau=0.01)
filter_i = neurogrid.filter.Filter(g_sat=1.0, tau=0.01)

input_e = []
input_i = []

pstc = 0.01

fs_e = np.zeros(N, dtype='f')
fs_i = np.zeros(N, dtype='f')

decay = np.exp(-dt/pstc)

time1 = 0
time2 = 0
spike_index_e = 0
spike_index_i = 0
while t<T:
    
    spikes_e = []
    while spike_index_e<len(spike_times_e) and t+dt>spike_times_e[spike_index_e]:
        spikes_e.append(spike_times_e[spike_index_e]-t)
        spike_index_e += 1
    spikes_i = []
    while spike_index_i<len(spike_times_i) and t+dt>spike_times_i[spike_index_i]:
        spikes_i.append(spike_times_i[spike_index_i]-t)
        spike_index_i += 1
        
    start = time.time()    
    values_e = filter_e.tick(spikes_e, dt)    
    values_i = filter_i.tick(spikes_i, dt)
    time1 += time.time()-start    
    input_e.extend(values_e)        
    input_i.extend(values_i)        
    s1 = n1.tick(np.ones(N)*values_e[-1], np.ones(N)*values_i[-1], dt)
    train1.append(s1)
    
    
    if len(spikes_e)>0:
        s_e = np.ones(N)
    else:    
        s_e = np.zeros(N)
    if len(spikes_i)>0:
        s_i = np.ones(N)
    else:    
        s_i = np.zeros(N)
       
    scale = 0.023/0.3885   
    
    start = time.time()
    fs_e = fs_e * decay + s_e * (1-decay)
    fs_i = fs_i * decay + s_i * (1-decay)
    time2 += time.time()-start
    
    s2 = n2.tick(fs_e*scale, fs_i*scale, dt)
    train2.append(s2)
    
    data_fs_e.extend(fs_e)
    
    
    t += dt

    
train1 = np.array(train1)    
train2 = np.array(train2)    
import pylab

print time1, time2

pylab.figure()
pylab.subplot(211)
if len(spike_times_e)>0:
    pylab.plot(np.linspace(0, T, len(input_e)), input_e, color='b')
    pylab.stem(spike_times_e, np.ones_like(spike_times_e)*np.max(input_e), color='b')

pylab.subplot(212)
if len(spike_times_i)>0:
    pylab.plot(np.linspace(0, T, len(input_i)), input_i, color='r')
    pylab.stem(spike_times_i, np.ones_like(spike_times_i)*np.max(input_i), color='r')


pylab.figure()

t = np.arange(len(train1))*dt
train = np.array(train1)
for i in range(N):
    pylab.subplot(N, 1, i+1)
    pylab.plot(t, train1[:,i], color='b')
    pylab.plot(t, train2[:,i], color='g')



pylab.show()

