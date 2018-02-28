get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import signal

# load the image as grayscale
f=mpimg.imread("../data/lena.png")         

# "Gaussian" kernel defined in Lecture 3b. Page3
g = 1.0/256 * np.array([[1, 4, 6, 4, 1], 
                   [2, 8, 12, 8, 2], 
                   [6, 24, 36, 24, 6],  
                   [2, 8, 12, 8, 2], 
                   [1, 4, 6, 4, 1]]) ;
# show image
plt.subplot(1, 2, 1)
plt.imshow(f, cmap='gray')      
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(g, cmap='gray')



h1 = signal.convolve2d(f, g, mode='full')
H1 = np.fft.fft2(h1)


# padding zeros in the end of f and g
padf = np.pad(f, ((0, 4), (0,4)), 'constant');
padg = np.pad(g, ((0, 255), (0, 255)), 'constant');

# compute the Fourier transforms of f and g 
F = np.fft.fft2(padf)
G = np.fft.fft2(padg)

# compute the product 
H2 = np.multiply(F, G)

# inverse Fourier transform
h2 = np.fft.ifft2(H2);


# In[91]:

mse1=(np.abs(H1-H2) ** 2).mean()
mse2=(np.abs(h1-h2)** 2 ).mean()
print('difference between H1 and H2', mse1)
print('difference between h1 and h2', mse2)

