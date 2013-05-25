import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def encoder_flat(e, rows, cols):
    N, D = e.shape
    assert N == rows * cols
    
    plt.figure()
    for i in range(D):
        plt.subplot(1, D, i+1)
        
        img = e[:,i]
        img.shape = rows, cols
        plt.imshow(img, interpolation='none')


def encoder_3d(e):
    fig = plt.figure()
    ax = Axes3D(fig)
    
    ax.scatter(e[:,0], e[:,1], e[:,2])
        

