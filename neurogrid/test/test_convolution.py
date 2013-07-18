import neurogrid as ng
import numpy as np
import math
import unittest


# DxD discrete fourier transform matrix            
def discrete_fourier_transform(D):
    m=[]
    for i in range(D):
        row=[]
        for j in range(D):            
            row.append(complex_exp((-2*math.pi*1.0j/D)*(i*j)))
        m.append(row)
    return m

# DxD discrete inverse fourier transform matrix            
def discrete_fourier_transform_inverse(D):
    m=[]
    for i in range(D):
        row=[]
        for j in range(D):            
            row.append(complex_exp((2*math.pi*1.0j/D)*(i*j))/D)
        m.append(row)
    return m

# formula for e^z for complex z
def complex_exp(z):
    a=z.real
    b=z.imag
    return math.exp(a)*(math.cos(b)+1.0j*math.sin(b))

def product(x):
    return x[0]*x[1]


def output_transform(dimensions):
    ifft=np.array(discrete_fourier_transform_inverse(dimensions))

    def makeifftrow(D,i):
        if i==0 or i*2==D: return ifft[i]
        if i<=D/2: return ifft[i]+ifft[-i].real-ifft[-i].imag*1j
        return np.zeros(dimensions)
    ifftm=np.array([makeifftrow(dimensions,i) for i in range(dimensions/2+1)])
    
    ifftm2=[]
    for i in range(dimensions/2+1):
        ifftm2.append(ifftm[i].real)
        ifftm2.append(-ifftm[i].real)
        ifftm2.append(-ifftm[i].imag)
        ifftm2.append(-ifftm[i].imag)
    ifftm2=np.array(ifftm2)

    return ifftm2.T

def input_transform(dimensions,first,invert=False):
    fft=np.array(discrete_fourier_transform(dimensions))

    M=[]
    for i in range((dimensions/2+1)*4):
        if invert: row=fft[-(i/4)]
        else: row=fft[i/4]
        if first:
            if i%2==0:
                row2=np.array([row.real,np.zeros(dimensions)])
            else:
                row2=np.array([row.imag,np.zeros(dimensions)])
        else:
            if i%4==0 or i%4==3:
                row2=np.array([np.zeros(dimensions),row.real])
            else:    
                row2=np.array([np.zeros(dimensions),row.imag])
        M.extend(row2)
    return M
    


class TestDecoders(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState()
        
    def test_conv(self):
        dim = 32
        subdim = 16
        array_size = dim/subdim
        N1 = 7
        N2 = 15
        
        A = [ng.ensemble.Ensemble(N1, N1, subdim, seed=1) for i in range(array_size)]
        Ad = [a.get_decoder() for a in A]
        B = [ng.ensemble.Ensemble(N1, N1, subdim, seed=1) for i in range(array_size)]
        Bd = [b.get_decoder() for b in B]

        C = [ng.ensemble.Ensemble(N2, N2, 2, seed=1) for i in range(array_size)]
        Cd = [c.get_decoder(name='product', func=lambda x: (x[:,0]*x[:,1])[:,None]) for c in C]

        D = [ng.ensemble.Ensemble(N1, N1, subdim, seed=1) for i in range(array_size)]
        Dd = [d.get_decoder() for d in D]
        
        
        A_transform = input_transform(dim,True)  # transform from A to C
        B_transform = input_transform(dim,False) # transform from B to C
        
        D_transform = output_transform(dim)  # transform from C to D
        
        
        #print A_transform
            
        
        
        


    
        
"""               
def make_convolution(net, name, A, B, C, dimensions, neurons, 
                      encoders=[[1,1],[1,-1],[-1,1],[-1,-1]], radius=3,
                      pstc_out=0.01, pstc_in=0.01, pstc_gate=0.01,
                      invert_first=False, invert_second=False, output_scale=1):

    net.make_array(name, neurons, (dimensions/2+1)*4, dimensions=2, 
                   encoders=encoders, radius=radius)

    A2=input_transform(dimensions,True,invert_first)
    B2=input_transform(dimensions,False,invert_second)
    
    net.connect(A, name, transform=A2, pstc=pstc_in)
    net.connect(B, name, transform=B2, pstc=pstc_in)
    
    ifftm2=output_transform(dimensions)*output_scale
    
    net.connect(name, C, func=product, transform=ifftm2, pstc=pstc_out)
"""
    
if __name__=='__main__':
    unittest.main()    
