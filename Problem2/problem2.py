from typing import final
import cv2
import os
import numpy as np

#==================================================================================================
#To solve this exercise through the methods learned in the course up to the point of
#this evaluation. Of the methods learned, the Oriented FAST and Rotated BRIEF (ORB)
#were implemented, with the Brute Force Matcher. It is necessary to mention how the
#algorithm to be implemented works, to understand the problems that have to be dealt with
#in its implementation.

#The main objective of ORB is to find the most important features in the objects within
#an image, through keyPoints and Descriptors. Once we obtain them, we will use a method known
#as brute force matching to compare them and if they are very similar, classify them as a match.
#The problem can arise when features are confusing, i.e. features that do not match the pattern
#we want to detect are matched, which would be a false match. When working with images that
#contain many elements, such as background environments or other objects that can be confuse,
#the probability of finding a false match increases, and we may not even find matches if the object
#we want to detect has many differences from our target object.

#For the code to work, it is necessary to work with images that meet certain characteristics,
#so that the programme can be generalised to more than one image. That is why it is recommended
#to use some of the following images to test the code.
#===================================================================================================
def make_decision(good, query_image):
    good_matches = len(good)

    if (good_matches>34):
        print(f'A mask has been detected in the input image, with {good_matches} keypoints.')
        mask = True
    else:
        print(f'No mask detected in the input image, with {good_matches} keypoints.')
        mask = False

    final_image = np.copy(query_image)

    w = final_image.shape[0]
    h = final_image.shape[1]

    if (mask):
        cv2.rectangle(final_image, (0, 0), (h, w), (0, 255, 0), 35)
        final_image = cv2.putText(final_image, 'Mask Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(final_image, (0, 0), (h, w), (255, 0, 0), 35)
        final_image = cv2.putText(final_image, 'No Mask Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

    return cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)


def BF_matching(kp_template, dp_template, kp_query, dp_query, template_mask, query_image):
    # Define the accuracy level to accept a match as true
    accuracy_percentage = 0.6
    # Declare the Brute Force Matcher Object
    bf = cv2.BFMatcher()
    # Perform the matching between the ORB descriptors of the template mask image and the query image with mask
    matches = bf.knnMatch(dp_template, dp_query,k=2)

    good = []

    for m,n in matches:
        if m.distance < accuracy_percentage*n.distance:
            good.append([m])
        

    # Connect the keypoints in the template mask image with their best matching keypoints in the query image with mask
    result = cv2.drawMatchesKnn(template_mask, kp_template, query_image, kp_query, good, None, flags = 2)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result, good

def ORB_detection(contours):
    # Create the ORB object parameters
    orb = cv2.ORB_create(nfeatures=50000, scaleFactor = 1.1, edgeThreshold = 25,  nlevels = 8) 
    # Find the keypoints and descriptors for the template mask image
    kp, dp = orb.detectAndCompute(contours, None)
    return kp, dp

def load_images(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Apply median blur to reduce the image noise
    image = cv2.medianBlur(image,5)
    return image

def pre_processing(image):
    # Define the morphological operator kernel to be used
    s = np.ones((2,2),np.uint8)
    # Use Canny edge detection method
    contours = cv2.Canny(image, 60, 120)
    # Apply Dilation to highlight the edges previously found
    dilation = cv2.dilate(contours, s, iterations = 1)
    # Apply Closing to reduce remaining parts
    return cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, s,5)

def run():
    ## --------------------
    ## Loading images stage
    ## --------------------
    # Paths to switch between input images
    query_image_path = 'mask_image.jpg' #Image 0 with mask
    #query_image_path = 'mask\mask1.jpg' #Image 1 with mask
    #query_image_path = 'mask\mask2.jpg' #Image 2 with mask
    #query_image_path = 'mask\mask3.jpg' #Image 3 with mask
    #query_image_path = 'mask\mask4.jpg' #Image 4 with mask
    #query_image_path = 'no_mask_image.jpg' #Image 0 without mask
    #query_image_path = 'no_mask'+'\\'+'no_mask1.jpg' #Image 1 without mask
    #query_image_path = 'no_mask'+'\\'+'no_mask2.jpg' #Image 2 without mask
    #query_image_path = 'no_mask'+'\\'+'no_mask3.jpg' #Image 3 without mask
    #query_image_path = 'no_mask'+'\\'+'no_mask4.jpg' #Image 4 without mask

    # Don't change this path: for mask template image
    mask_template_path = 'mask.jpg'
    # Load the template mask image
    template_mask = load_images(mask_template_path)
    # Load the query image for mask
    query_image = load_images(query_image_path)

    ## --------------------
    ## Pre-processing stage
    ## --------------------

    #Apply the preprocessing stage to the template mask image
    preprocessed_template = pre_processing(template_mask)
    preprocessed_query = pre_processing(query_image)

    ## -------------------------------------
    ## Processing stage: Detection using ORB
    ## -------------------------------------
    kp_template, dp_template = ORB_detection(preprocessed_template)
    kp_query, dp_query = ORB_detection(preprocessed_query)

    result, good = BF_matching(kp_template, dp_template, kp_query, dp_query, template_mask, query_image)
    
    ##===============
    ## Decision stage
    ##===============

    final_image = make_decision(good, query_image)
    
    cv2.imshow('ORB keypoints matches',result)
    cv2.imshow('Image Mask Detection Using ORB',final_image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 


if __name__=='__main__':
    run()