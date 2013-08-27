import numpy as np
import scipy.optimize

def classic(activity, targets, rng, noise=0.1, minSV=0.01):
    A = activity
    X = targets
    
    A = A + rng.randn(*A.shape)*(noise*np.max(A))
    
    G = np.dot(A, A.T)
    U = np.dot(A, X)
    
    Ginv = np.linalg.pinv(G, rcond=minSV)
    d = np.dot(Ginv, U)
    
    return d
    
def classic2(activity, targets, rng, noise=0.1, minSV=0.0001):
    A = activity
    X = targets
    
    #A = A + rng.randn(*A.shape)*(noise*np.max(A))
    G = np.dot(A, A.T)
    G = G + np.eye(A.shape[0])*(noise*np.max(A))**2
    U = np.dot(A, X)
    
    Ginv = np.linalg.pinv(G, rcond=minSV)
    d = np.dot(Ginv, U)
    
    return d

def sparse_random(activity, targets, rng, noise=0.1, sparsity=0.8, minSV=0.01):
    d = classic(activity, targets, rng, noise, minSV=minSV)
    
    N = len(activity)
    for i in range(N):
        if rng.random_sample()<sparsity:
            d[i]=0
        else:
            d[i]*=1.0/(1.0-sparsity)
    
    return d


def sparse_greedy(activity, targets, rng, noise=0.1, sparsity=0.8, minSV=0.01):
    d = classic(activity, targets, rng, noise)
    
    weight = np.sum(d*d, axis=1)
    
    index = weight.argsort()
    N = int(len(weight)*sparsity)
    
    keep_index = index[N:]
    
    d2 = classic(activity[keep_index], targets, rng, noise, minSV=minSV)
    
    d = d*0
    d[keep_index] = d2
    
    return d



def lstsq(activity, targets, rng, noise=0.1, minSV=0.01):
    A = activity
    X = targets
    
    A = A + rng.randn(*A.shape)*(noise*np.max(A))
    
    return np.linalg.lstsq(A.T, X, rcond=minSV)[0]
    
def nonnegative(activity, targets, rng, noise=0.1):
    A = activity
    X = targets
    
    A = A + rng.randn(*A.shape)*(noise*np.max(A))
    
    d = []
    for i in range(X.shape[1]):
        dd = scipy.optimize.nnls(A.T, X[:,i])[0]            
        d.append(dd)
        
    return np.array(d).T    
            
   
    


