#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 20:08:06 2023

@author: candilsiz
"""
from skimage.transform import (hough_line, hough_line_peaks)
from PIL import Image
from IPython.display import display
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import math
  



def OriantationHist(image):

    # Canny Edge Detector
    
    # t_lower = 450 # Lower Threshold
    # t_upper = 700  # Upper threshold
    # aperture_size = 5  # Aperture size
    
    t_lower = 200 # Lower Threshold
    t_upper = 500  # Upper threshold
    aperture_size = 5  # Aperture size
    
      
    edge = cv2.Canny(image, t_lower, t_upper, apertureSize = aperture_size)
    
    imageEdges = Image.fromarray(edge)
    
    display(imageEdges)
    
    # Hough Transform
    
    tested_angles = np.linspace(-np.pi , np.pi, 360)
    
    
    hspace, theta, dist = hough_line(edge, tested_angles)
    
    #plt.figure(figsize = (10,10))
    
    #plt.imshow(hspace)
    
    h, q, distance = hough_line_peaks(hspace, theta, dist)
    
    
    angle_list = []
    
    # Radians to degree
    for rad in q:
        deg = rad * 180 / np.pi
        angle_list.append(deg)
        
    # Now we have angles list and lenght list
    
    distance_list = [abs(int(i)) for i in list(distance)]
    
    angle_list = [round(x) for x in angle_list]
    
    combined_list = list(zip(angle_list, distance_list))
    
    #print(combined_list)
    
    # define the range of angles
    angle_min = -360.0
    angle_max = 360.0
    
    # define the number of bins  6 9 11 21
    num_bins = 11
    
    # calculate the width of each bin
    bin_width = (angle_max - angle_min) / num_bins
    
    
    histogram = np.zeros(num_bins)
    
    # iterate over the data and update the histogram
    for data in combined_list:
        angle, length = data
        
        bin_index = int(np.floor((angle - angle_min) / bin_width))
        
        # update the histogram by adding the length to the corresponding bin
        histogram[bin_index] += length
        
    #print(list(histogram))
    
    
    
    # DISPLAY HOUGHTRANSFORM LINES
    

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    
    ax[0].imshow(np.log(1 + hspace),
                 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), dist[-1], dist[0]],
                 cmap='gray', aspect=1/1.5)
    ax[0].set_title('Hough transform')
    ax[0].set_xlabel('Angles (degrees)')
    ax[0].set_ylabel('Distance (pixels)')
    ax[0].axis('image')
    
    ax[1].imshow(image, cmap='gray')
    
    origin = np.array((0, image.shape[1]))
    
    for _, angle, dist in zip(*hough_line_peaks(hspace, theta, dist)):
        angle_list.append(angle) #Not for plotting but later calculation of angles
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        ax[1].plot(origin, (y0, y1), '-r')
    ax[1].set_xlim(origin)
    ax[1].set_ylim((image.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Detected lines')
    
    plt.tight_layout()
    plt.show()
    
    
    return list(histogram)
    


# Image Reading Part
bookList = ["bitmemisoykulerR.png", "chessR.png","cinaliR.png", "cppR.png", "dataminingR.png",
            "harrypotterR.png", "heidiR.png","stephenkingR.png","AlgorithmsR.png","kpssR.png",
            "lordofringsR.png", "patternrecognitionR.png","sefillerR.png","shawshankR.png", 
            "tersR.png"  ]

#bookList = ["AlgorithmsR.png","lordofringsR.png", "patternrecognitionR.png","sefillerR.png","tersR.png"]

oriatation_List = []


for i in range(len(bookList)):
    img = Image.open(bookList[i]).convert('L')
    image = np.array(img)
    result = OriantationHist(image)
    
    ###
    # Oriatation Histogram as visually
    
    fig = plt.figure(figsize = (10, 5))
    
    #bins = ["-π","-3π/5","-π/5","π/5","3π/5","π"]
    bins = ["-π","-4π/5","-3π/5","-2π/5","-π/5","0","π/5","2π/5","3π/5","4π/5","π"]
    
    #bins = ["-π","-9π/10","-8π/10","-7π/10","-6π/10","-5π/10","-4π/10","-3π/10",
    #        "-2π/10","-π/10","0","π/10","2π/10","3π/10","4π/10","5π/10","6π/10",
    #        "7π/10","8π/10","9π/10","π"]
    

    # creating the bar plot
    plt.bar(bins, result, color ='maroon', width = 0.4)
        
    plt.xlabel("Oriantation Angles")
    plt.ylabel("Accumalted Lines Lenghts")
    plt.title("Rotated Image")
    plt.show()
    
    
    ###
    oriatation_List.append(result)
shift = random.randint(1,10)



# Selected Book Rotated Oriantation Histogram
selectedBookR = Image.open("chessR.png").convert('L')

selectedBookR = np.array(selectedBookR)

rotatedHist = OriantationHist(selectedBookR)


def circular_shift(lst, n):
    # Shifts a list by n positions to propriately represent ciruclar shift
    return lst[-n:] + lst[:-n]

# Selected Book Oriantation Histogram
selectedBook = Image.open("chess.png").convert('L')

selectedBook = np.array(selectedBook)

orginalHist = OriantationHist(selectedBook)

print("Original Histogram: ",orginalHist, "\nRotated Histogram: ",rotatedHist)

# myImage = np.array(orginalHist)
rotatedHist = circular_shift(rotatedHist,shift)
# myrotatedImageList = [np.array(lst) for lst in oriatation_List]



def euclidean_distance(lst1, lst2):
    # Computes the Euclidean distance between two lists
    distance = math.sqrt(sum([(lst1[i] - lst2[i])**2 for i in range(len(lst1))]))
    return abs(distance)


def most_similar_list(my_list, other_lists):
    # Returns the most similar list in bookList
    least_similarity = float("inf")
    most_similar_list = None
    
    for i in range(len(my_list)):
        shifted_list = circular_shift(my_list, i)
        
        for j in range(len(other_lists)):
            similarity = euclidean_distance(shifted_list, other_lists[j])
                
                
            if similarity <= least_similarity:
                least_similarity = similarity
                most_similar_list = other_lists[j]
                #print(similarity)

                
    return most_similar_list


result = most_similar_list(orginalHist, oriatation_List)
#result = most_similar_list(myImage, myrotatedImageList)

result = rotatedHist


# Display Part

# result = Image.fromarray(edge)

# display(result)



print("Most Similar Match is:", rotatedHist)


