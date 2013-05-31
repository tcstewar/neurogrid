import numpy as np


def classic(activity, targets, rng, noise=0.1):
    A = activity
    X = targets
    
    A += rng.randn(*A.shape)*noise
    
    G = np.dot(A, A.T)
    U = np.dot(A, X)
    
    Ginv = np.linalg.pinv(G)
    return np.dot(Ginv, U)
    

def lstsq(activity, targets, rng, noise=0.1):
    A = activity
    X = targets
    
    A += rng.randn(*A.shape)*noise
    
    return np.linalg.lstsq(A.T, X)[0]
    
    


if __name__=='__main__':
    import neurons
    import encoders
    import activity
    
    N = 30
    rng = np.random.RandomState()
    n = neurons.RateNeuron(N, rng, balanced=True, nonlinear=0)
    e = encoders.random(N, 1, rng)
    
    X, A = activity.classic(n, e, rng)
    
    
    
    d = classic(A, X*X, rng)
    d = lstsq(A, X*X, rng)
    
    
    Xhat = np.dot(A.T, d)
    
    import matplotlib.pyplot as plt    
    plt.plot(X[:,0], Xhat[:,0])
    plt.show()
            
