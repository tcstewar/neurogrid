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
    
    


