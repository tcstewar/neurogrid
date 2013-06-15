import numpy as np
import math

def blur_kernel(rows, cols, width, surround=True):
    kernel = np.zeros((rows, cols), dtype='f')
    if width == 0:
        kernel[0, 0] = 1
    else:
        for i in range(rows):
            for j in range(cols):
                dx = i
                if dx>rows/2: dx-=rows
                dy = j
                if dy>cols/2: dy-=cols
            
                distance = float(abs(dx)+abs(dy))#math.sqrt(dx*dx + dy*dy)
                kernel[i, j] = math.exp(-distance/width)

        kernel /= np.sum(kernel)
    kernel = np.roll(kernel, (rows+1)/2-1, axis=0)
    kernel = np.roll(kernel, (cols+1)/2-1, axis=1)
    
    if surround:
        kernel = scipy.signal.convolve2d(kernel, surround_kernel(rows, cols), mode='same')
    
    return kernel    
    
def surround_kernel(rows, cols):
    kernel = np.zeros((rows, cols), dtype='f')
    kernel[1, -1] = 1
    kernel[1, 0] = 1
    kernel[1, 1] = 1
    
    kernel[-1, -1] = 1
    kernel[-1, 0] = 1
    kernel[-1, 1] = 1
    
    kernel[0, -1] = 1
    kernel[0, 1] = 1
    kernel /= np.sum(kernel)
    kernel = np.roll(kernel, (rows+1)/2-1, axis=0)
    kernel = np.roll(kernel, (cols+1)/2-1, axis=1)
    return kernel    
        
    
    
import scipy.signal    
def apply_matrix(matrix, kernel):
    N, D = matrix.shape
    assert N == kernel.shape[0] * kernel.shape[1]
    
    r =[]
    for i in range(D):
        m = matrix[:,i]
        m.shape = kernel.shape
        row = scipy.signal.convolve2d(m, kernel, mode='same')
        row.shape = (N,)
        r.append(row)
    return np.array(r).T    

def apply_vector(vector, kernel):
    N, = vector.shape
    assert N == kernel.shape[0] * kernel.shape[1]
    v = vector[:]
    v.shape = kernel.shape
    row = scipy.signal.convolve2d(v, kernel, mode='same')
    row.shape = (N,)
    return row    
    
    