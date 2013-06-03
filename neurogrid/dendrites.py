import numpy as np
import math

def blur_kernel(rows, cols, width):
    kernel = np.zeros((rows, cols))
    if width == 0:
        kernel[0, 0] = 1
    else:
        for i in range(rows):
            for j in range(cols):
                dx = i
                if dx>rows/2: dx-=rows
                dy = j
                if dy>cols/2: dy-=cols
            
                distance = math.sqrt(dx*dx + dy*dy)
                kernel[i, j] = math.exp(-distance/width)

        kernel /= np.sum(kernel)
    return kernel    
    
