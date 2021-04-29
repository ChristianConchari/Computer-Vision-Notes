from typing import final
import cv2
import os
import numpy as np
from sklearn.decomposition import PCA

#==================================================================================================
#To solve this exercise we used a dataset called "Bacteria detection with darkfield microscopy"
#uploaded by Long Nguyen on the Kaggle platform, the link is as follows:
#(https://www.kaggle.com/longnguyen2306/bacteria-detection-with-darkfield-microscopy?select=images).
#From this dataset we extracted the images that in my opinion could be processed in a better way
#with the methods learned so far in this evaluation. Additionally, 30% of the images in the dataset
#were edited so that the number of bodies present in the image was kept to a minimum and we could
#emulate a relatively clean microscopic view of water. In total, I will use a dataset
#consisting of 60 images in total, which, given that no machine learning techniques are being used,
#is considered sufficient to test the performance of the developed code.
#===================================================================================================

def detect_contamination(image,flag):
    final_image = np.copy(image)
    w = final_image.shape[0]
    h = final_image.shape[1]
    if (flag):
        cv2.rectangle(final_image, (0, 0), (h, w), (255, 0, 0), 35)
        final_image = cv2.putText(final_image, 'Contaminated water', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(final_image, (0, 0), (h, w), (0, 255, 0), 35)
        final_image = cv2.putText(final_image, 'Clean water', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    return cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

def find_contours(image,preprocessed):
    # copy the image to avoid overwriting.
    image_copy = np.copy(image)
    # find the contours in the image
    # use cv2.RETR_EXTERNAL to return only extreme outer flags
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
    # draw the found contours in the copied image
    cv2.drawContours(image_copy, contours, -1, (0, 0, 255), 5)  
    return image_copy, contours

def morphological_operations(image):
    # Create a 4x4 kernel for morphological process
    s = np.ones((2,2),np.uint8)

    # Apply erotion to reduce noise and small parts
    morph = cv2.erode(image, s, iterations = 7)
    
    # Apply closing to fill contours
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, s, 25)

    # Apply CannyEdge detection
    morph = cv2.Canny(morph, 30, 90)

    # Apply adaptative threshold to highlight contours
    morph = cv2.adaptiveThreshold (morph, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 1)

    # Dilate the contours found
    return cv2.dilate(morph, s, iterations = 2)

def PCA_compression(image):
    # define the custom compression rate for PCA
    compression_percentage = 0.3

    # define the number of components to mantain in order to a compression percentage
    n_comp = int(compression_percentage*image.shape[0])
    # Initialize PCA object
    pca = PCA(n_components=n_comp)

    # Standardize the data, so all instances have 0 as the center
    pca.fit(image) 

    # Find the (n_comp) number of principal components and remove the less important 
    # Theres's also another function that joins fit and tranform: pca.Fit_transform()
    principal_components = pca.transform(image) 

    # Since PCA reduces the number of columns, we will need to transform the results 
    # to the original space to display the compressed image
    return pca.inverse_transform(principal_components) 

def run():

    # recommended images to test:
    # For clean water
    # pure_water2.png
    # pure_water3.png
    # pure_water4.png
    # pure_water7.png
    # pure_water8.png
    # pure_water13.png
    # pure_water15.png
    # For contaminated water
    # 301.png
    # 170.png
    # 113.png
    # 001.png
    # 043.png
    # 122.png
    # 168.png
    
    # Read an image from the dataset 
    image = cv2.imread("dataset_bacteria"+"\\"+"001.png")
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compression Stage
    pca_compressed = np.uint8(PCA_compression(image_gray))
    # Morphological operations stage
    preprocessed = morphological_operations(pca_compressed)
    # Find Contours stage
    image_with_contours, contours = find_contours(image,preprocessed,)

    contour = len(contours)
    
    if contour >= 8:
        print(f'In the image were found {contour}, therefore the water is contaminated.')
        flag = True
    else:
        print(f'In the image were found {contour}, therefore the water is clean.')
        flag = False

    # Make decision stage
    result = detect_contamination(image_with_contours,flag)

    
    cv2.imshow('Parasites detection in water', result)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

if __name__=='__main__':
    run()