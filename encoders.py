import numpy as np

def random(N, D, rng):
    samples = rng.randn(N, D)
    norm = np.sqrt(np.sum(samples*samples, axis=1))
    return samples/norm[:,None]

def score(encoders, index, rows, cols):
    i = index % cols
    j = index / cols
    
    sim = 0
    if i>0: sim += np.dot(encoders[j*cols+i], encoders[j*cols+i-1])
    if i<cols-1: sim += np.dot(encoders[j*cols+i], encoders[j*cols+i+1])
    if j>0: sim += np.dot(encoders[j*cols+i], encoders[(j-1)*cols+i])
    if j<rows-1: sim += np.dot(encoders[j*cols+i], encoders[(j+1)*cols+i])
    return sim

def swapped(N, D, rng, rows, cols, iterations=100):
    assert rows*cols == N
    
    encoders = random(N, D, rng)
    
    for k in range(iterations):
        target = rng.randint(0, N, N)
        for i in range(N):
            j = target[i]
            if i != j:
                sim1 = score(encoders, i, rows, cols) + score(encoders, j, rows, cols)
                encoders[[i,j],:] = encoders[[j,i],:]
                sim2 = score(encoders, i, rows, cols) + score(encoders, j, rows, cols)
                if sim1 > sim2:
                    encoders[[i,j],:] = encoders[[j,i],:]
    
    return encoders            
            
            
    
    
    
