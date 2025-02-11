from numba import cuda
import numpy as np

from math import exp

@cuda.jit
def gaussian_gpu(sigma, kernel_size,kernel):
    m=kernel//2
    n=kernel_size//2

    x=cuda.threadIdx.x
    y=cuda.threadIdx.y

    kernel[x,y]=exp(-(x-m)**2 + (y-n)**2 / (2*sigma**2)) # The amount of smoothing and the amount of blur means the higher the kernel and sigma size higher the blur

sigma=5.3
kernel_size=30

kernel=np.zeros((kernel_size, kernel_size), np.float32) # float32 matrix (commonly used dataType to respesernt images in arrays)

d_kernel=cuda.to_device(kernel) # transfering data from cpu to gpu 

@cuda.jit

def convolve(result,mask,image): # convloution function for CUDA
    i,j=cuda.grid(2) #2d cords
    image_rows, image_cols =image.shape
    if( i>=image_rows) or (j>=image_cols):
        return 
    
    delta_rows= mask.shape[0] //2
    delta_cols= mask.shape[1] //2

    s=0
    for k in range(mask.shape[0]):
        for l in range(mask.shape[1]):
            i_k=i-k+delta_rows
            j_l=j-l+delta_cols
            if (i_k>=0) and (i_k<image_rows) and (j_l>=0) and (j_l<=image_cols):
                s+=mask[k,l] * image[i_k,j_l]
    result[i,j]=s

from PIL import Image, ImageOps # used to work with images in python

image=np.array(ImageOps.grayscale(Image.open('ahmad.jpeg'))) # converting image to nparray

d_image= cuda.to_device(image) # transfering image from host to device

d_result = cuda.device_array_like(image)

gaussian_gpu[(1,), (kernel_size, kernel_size) ]( sigma, kernel_size, d_kernel)

blockdim=(32,32)

griddim=(image.shape[0]// blockdim[0]+ 1, image.shape[1] // blockdim[1]+1)

import time
start= time.process_time()
convolve[griddim,blockdim](d_result,d_kernel,d_image)
print(time.process_time()- start)
result = d_result.copy_to_host()

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(image, cmap='gray')
plt.title("Before convolution:")
plt.figure()
plt.imshow(result, cmap='gray')
plt.title("After convolution:")
plt.show()

