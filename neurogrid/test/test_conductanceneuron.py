import numpy as np
import neurogrid as ng

rng = np.random.RandomState(seed=1)

N = 5
T = 0.200


n = ng.conductance_neurons.ConductanceNeuron(N, rng, bias=0.5, 
               gain_e=50.0, gain_i=10.0)

dt = 0.0001
scale = 0.001/dt

fr_e = 500
fr_i = 200

pstc = 0.01

fs_e = np.zeros(N, dtype='f')
fs_i = np.zeros(N, dtype='f')

spike_times_e = []
spike_times_i = []

dts = 0.00001
t = 0
while t<T:
    if rng.uniform(0,1)<fr_e*dts:
        spike_times_e.append(t)
    if rng.uniform(0,1)<fr_i*dts:
        spike_times_i.append(t)
    t += dts

train = []
data_fs_e = []
data_fs_i = []
data_s_e = []
data_s_i = []
t = 0

spike_index_e=0
spike_index_i=0
decay = np.exp(-dt/pstc)

while t<T:
    if spike_index_e<len(spike_times_e) and t>spike_times_e[spike_index_e]:
        s_e = np.ones(N)
        spike_index_e += 1
    else:
        s_e = np.zeros(N)
    if spike_index_i<len(spike_times_i) and t>spike_times_i[spike_index_i]:
        s_i = np.ones(N)
        spike_index_i += 1
    else:
        s_i = np.zeros(N)
        
    #s_e = np.where(rng.uniform(0,1,N)<fr_e*dt, 1.0, 0.0)
    #s_i = np.where(rng.uniform(0,1,N)<fr_i*dt, 1.0, 0.0)
    
    fs_e = fs_e * decay + s_e * (1-decay)
    fs_i = fs_i * decay + s_i * (1-decay)
    
    s = n.tick(fs_e*scale, fs_i*scale, dt)
    
    train.append(s)
    data_fs_e.append(fs_e)
    data_fs_i.append(fs_i)
    data_s_e.append(s_e)
    data_s_i.append(s_i)
    
    t += dt

import pylab

t = np.arange(len(train))*dt
train = np.array(train)
for i in range(N):
    pylab.subplot(N, 1, i+1)
    pylab.plot(t, train[:,i])

"""
pylab.figure()
pylab.subplot(211)
pylab.plot(data_fs_e, 'b.-')
pylab.stem(data_s_e, 'b')
pylab.ylim((0,np.max(data_fs_e)))
pylab.subplot(212)
pylab.plot(data_fs_i, 'g.-')
pylab.stem(data_s_i, 'g')
pylab.ylim((0,np.max(data_fs_i)))
"""
pylab.show()

print np.max(data_fs_e)