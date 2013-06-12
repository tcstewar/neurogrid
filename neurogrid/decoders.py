import numpy as np
import scipy.optimize

def classic(activity, targets, rng, noise=0.1):
    A = activity
    X = targets
    
    A += rng.randn(*A.shape)*(noise*np.max(A))
    
    G = np.dot(A, A.T)
    U = np.dot(A, X)
    
    Ginv = np.linalg.pinv(G)
    d = np.dot(Ginv, U)
    
    return d

def lstsq(activity, targets, rng, noise=0.1):
    A = activity
    X = targets
    
    A += rng.randn(*A.shape)*(noise*np.max(A))
    
    return np.linalg.lstsq(A.T, X)[0]
    
def nonnegative(activity, targets, rng, noise=0.1):
    A = activity
    X = targets
    
    A += rng.randn(*A.shape)*(noise*np.max(A))
    
    d = []
    for i in range(X.shape[1]):
        dd = scipy.optimize.nnls(A.T, X[:,i])[0]            
        d.append(dd)
        
    return np.array(d).T    
            
   
    


