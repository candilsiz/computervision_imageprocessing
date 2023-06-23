#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 22:37:28 2023

@author: candilsiz
"""


from PIL import Image
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt

###################################

# Dilation

def Dilation(img):

    # kernel shape
    x = 5
    se=np.ones((x,x))
    
    rows,cols=img.shape
    erode_img=np.zeros((rows,cols))

    newimg=np.zeros((img.shape[0]+2,img.shape[1]+2))
    newimg[1:-1,1:-1]=img
    for i in range(0,rows-2):
        for j in range(0,cols-2):
            # print(newimg[i:i+3,j:j+3])
            erode_img[i,j]=np.min(se * newimg[i:i+x,j:j+x])
            
    return erode_img


image = Image.open("Figure1.png").convert('L')
im = np.array(image)

dilated_image = Dilation(im)


plt.imshow(dilated_image, cmap = "gray")
plt.title("Dilated Image")
plt.axis("off")
plt.show()


# Erosion

def Erosion(image, kernel):
  

    # Get the dimensions (row,colm) of the input image and kernel
    image_row, image_colm = image.shape[0], image.shape[1]
    kernel_row, kernel_colm = kernel.shape[0], kernel.shape[1]

    # Create a padded version of the input image with zeros around the edges
    padded_image = [[0 for j in range(image_colm + kernel_colm - 1)] for i in range(image_row + kernel_row - 1)]
    for i in range(image_row):
        for j in range(image_colm):
            padded_image[i + kernel_row // 2][j + kernel_colm // 2] = image[i][j]

    # Apply dilation to the padded image
    erosed_image = [[0 for j in range(image_colm)] for i in range(image_row)]
    for i in range(image_row):
        for j in range(image_colm):
            fulled = 0
            for ki in range(kernel_row):
                for kj in range(kernel_colm):
                    if kernel[ki][kj] == 1:
                        ii, jj = i + ki - kernel_row // 2, j + kj - kernel_colm // 2
                        if ii >= 0 and ii < image_row and jj >= 0 and jj < image_colm:
                            # Return the max value of the kernel window
                            fulled = max(fulled, padded_image[ii + kernel_row // 2][jj + kernel_colm // 2])
            erosed_image[i][j] = fulled

    return erosed_image


image = Image.open("Figure1.png").convert('L')
image_np = np.array(image)

kernel_size = 9
kernel = [[1]*kernel_size for _ in range(kernel_size)]
arr_kernel = np.array(kernel)

# returns list as output_image
output_image = Erosion(image_np, arr_kernel)

plt.imshow(output_image, cmap = "gray")
plt.title("Erosed Image")
plt.axis('off')
plt.show()

input_to_dilation = np.array(output_image)

opened = Dilation(input_to_dilation)

plt.title("Opened İmage (First Erosion than Dilation")
plt.imshow(opened, cmap = "gray")
plt.axis('off')
plt.show()
    
###################################

# Histogram

def histogram(image):
  
    # Initialize an array to store the histogram values
    frequency = np.zeros(256, dtype=int)
    scale = range(256)

    # Loop over each pixel in the image though images rows and colomns
    for row in range(imArray.shape[0]):
        for col in range(imArray.shape[1]):
            # pixel value will be whatever the (row,col) is pointing
            pixelVal = int(imArray[row,col])
            # increments the specific pixels values count location @ (row,col)
            frequency[pixelVal] += 1

    # Plot the histogram (colomn chart)
    plt.bar(scale,frequency, color='blue')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Grayscale Image Histogram')
    plt.show()
    

imList = ["Figure2_a.jpg","Figure2_b.jpg"]

for i in range(2):
    im = Image.open(imList[i]).convert("L")
    imArray = np.array(im)
    histogram(imArray)
    

###########################

# 2-D Convolution

def convolution2D(image, kernel):
    # returns the array's number of rows and colomns
    row, col = kernel.shape
    # Assume 3x3 kernel ,then  m = n = 3
    if (row == col):
        # Assume 500 x 500 image x = y = 500
        y, x = image.shape
        y = y - row + 1
        x = x - row + 1
        # conv_image will take the same dimensions and elemtents as original image
        conv_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                conv_image[i][j] = np.sum(image[ i:i+row , j:j+row ]*kernel)
                
    return conv_image


img = Image.open("Figure4.jpg")

arrayIm = np.array(img)

kerneldict = dict()

kerneldict[0] = [[-1,0,1],[-2,0,2],[-1,0,1]]   #sobel_x 
kerneldict[1] = [[-1,-2,-1],[0,0,0],[1,2,1]]   #sobel_y
kerneldict[2] = [[-1,0,1],[-1,0,1],[-1,0,1]]   #prewitt_x
kerneldict[3] = [[-1,-1,-1],[0,0,0],[1,1,1]]   #prewitt_y

kernels = ["Sobel_x","Sobel_y", "Prewitt_x", "Prewitt_y" ]

for i in range(4):
    kernel = np.array(kerneldict[i])
    convolved_Image = convolution2D(arrayIm, kernel)
    plt.subplot(2,2,i+1)
    plt.imshow(convolved_Image, cmap = "gray")
    plt.axis('off')
    plt.title(kernels[i] + "Filter")
    
    if i == 3:
        plt.figure(figsize=(15,15))
        plt.show()


##########################


#Otsu's Tresholding        


def OtsuTreshold(img_array):
    hist, bins = np.histogram(img_array, bins = 256, range = [0, 256])
    
    image_size = img_array.shape[0] * img_array.shape[1]
    cum_sum = np.cumsum(hist)
    # Initiate the treshold value
    initiate = -1
    threshold = -1
  
    for t in range(256):
        # Calculating the weight using cumulative summation for all pixels from index [0] to to index [t] till t = [0 1 2 ... 256]
        weight0 = cum_sum[t]
        weight1 = image_size - weight0
        # To handle division while calculating Mean
        if weight0 == 0 or weight1 == 0:
            pass
        # Calculating mean sum(x * fx)
        mean0 = np.sum(np.arange(0, t) * hist[:t]) / weight0
        mean1 = np.sum(np.arange(t, 256) * hist[t:]) / weight1
        # Calculating the Variance
        var_between = weight0 * weight1 * (mean0 - mean1) ** 2
        
        # updating the treshold value and finding the optimal treshold 
        if var_between > initiate:
            initiate = var_between
            threshold = t
            
    # Constructing the image due to the optimal Treshold value found        
    binary_img = np.zeros_like(img_array)
    binary_img[img_array > threshold] = 255
    binary_img[img_array <= threshold] = 0
    
    return binary_img
   
   
imList = ["Figure3_a.jpg","Figure3_b.png"]

for i in range(2):

    src_img = Image.open(imList[i]).convert("L")
    Im_array = np.array(src_img)  
    
    thresholded_img = OtsuTreshold(Im_array)
    binary_im = Image.fromarray(thresholded_img)
    
    display(binary_im)
    
   









        
        
    