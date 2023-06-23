# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon May 15 21:01:29 2023

# @author: candilsiz
# """


               #### HW3 Image Segmentation ####
               

from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.segmentation import slic
from sklearn.cluster import KMeans
from skimage.segmentation import mark_boundaries


def obtain_superpixels(image, superpixel_type, n_segments,sigma):
    
    if superpixel_type == 'slic':
        labels = slic(image, n_segments = n_segments, sigma = sigma)
        
        plt.imshow(mark_boundaries(image, labels))
        plt.axis("off")
        plt.show()
    
    return labels

def compute_gabor_texture_features(image, labels, scales, orientations):
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize the filter bank and feature vector
    filter_bank = []
    feature_vector = []
    
    for scale in scales:
        for theta in np.arange(np.pi/orientations, np.pi + np.pi/orientations , np.pi/orientations):
            print('Scale: ',scale, 'Oriantation: ',theta)
            
            # Gabor kernel parameters
            sigma = 2.0
            lambd = 3.0
            gamma = 2.5
            
            kernel = cv2.getGaborKernel((scale, scale), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)

            filtered_image = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            
            normalized_image = cv2.normalize(filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
            
            # Increasing Brightness for better Gabor Texture Features Display
            forEnhance = Image.fromarray(normalized_image)
            enhancer = ImageEnhance.Brightness(forEnhance)
            enhancedImage = enhancer.enhance(2.5)
            
            plt.imshow(enhancedImage, cmap ="gray")
            plt.axis("off")
            plt.show()
            
            filter_bank.append(normalized_image)
    
 
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        
        # Find pixels inside the superpixel
        mask = np.zeros_like(labels)
        mask[labels == label] = 1
        
        # Compute the average Gabor feature for the superpixel
        superpixel_features = []
        
        for filtered_image in filter_bank:
            superpixel_filtered_image = filtered_image * mask
            superpixel_filtered_image = superpixel_filtered_image[superpixel_filtered_image != 0]
            
            if len(superpixel_filtered_image) > 0:
                superpixel_average = np.mean(superpixel_filtered_image)
            else:
                superpixel_average = 0.0
            
            superpixel_features.append(superpixel_average)
        
        
        feature_vector.append(superpixel_features)
    
    return feature_vector

# Images used in the assignment
imageList = ['1.jpg','2.jpg','3.jpg','5.jpg','7.jpg','8.jpg',"9.jpg","10.jpg"]

# Part 1: Obtain superpixels for each image

# Super Pixel Labels for Each Image
superpixel_labels = []

for image_filename in imageList:
    
    #   Parameters
    superpixel_type = 'slic'
    n_segments = 60   
    sigma = 3  
    image = cv2.imread(image_filename)
    
    labels = obtain_superpixels(image, superpixel_type, n_segments, sigma)
    superpixel_labels.append(labels)

# Part 2: Compute Gabor texture features for all images

# Gabor Filter Banks
#scales = [1, 5, 10, 25]  
scales = [1, 5, 15, 55]  
orientations = 4  


feature_vectors = []

for i, image_filename in enumerate(imageList):
    image = cv2.imread(image_filename)
    labels = superpixel_labels[i]
    features = compute_gabor_texture_features(image, labels, scales, orientations)
    feature_vectors.append(features)
    
    
data = np.concatenate(feature_vectors)


n_clusters = 5 # Number of clusters
kmeans = KMeans(n_clusters=n_clusters)
labels = kmeans.fit_predict(data)

label_idx = 0
colored_labels = np.zeros_like(superpixel_labels[0])


for i, image_labels in enumerate(superpixel_labels):
        
    unique_labels = np.unique(image_labels)
        
    for label in unique_labels:
        
        colored_labels[image_labels == label] = labels[label_idx]
        label_idx += 1        


# pseudo_color_image = cv2.cvtColor(colored_labels, cv2.COLOR_GRAY2BGR)
    plt.imshow(colored_labels)
    plt.axis("off")
    plt.show()


# Displaying Images with their segmentation back to back.
image_path1 = "10.jpg"  
image1 = cv2.imread(image_path1)


image_path2 = "10_0.png"
image2 = cv2.imread(image_path2)

# Match the Image's Size
image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))


blended_image = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)

# Display the result
plt.imshow(image2)
plt.imshow(blended_image)
plt.axis("off")
plt.show()






 
 





