import numpy as np

def random(N, D, rng):
    if D==1:
        samples = np.linspace(-1, 1, N)
        samples.shape = N,1
        return samples
    else:
        samples = rng.randn(N, D)
        norm = np.sqrt(np.sum(samples*samples, axis=1))
        radius = rng.uniform(0,1,N)**(1.0/D)
        scale = radius / norm
        return samples*scale[:,None]


