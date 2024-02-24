#!/usr/bin/env python
# coding: utf-8

# ## ORB and Brute Force matcher

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt


# In[4]:


def compute_similarity_score(image1_path, image2_path):
    # Load images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors of both images
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate similarity score
    similarity_score = len(matches) / min(len(keypoints1), len(keypoints2))

    return similarity_score


# In[5]:


image1_path = 'cattle_4.jpg'
image2_path = 'cattle_5.jpg'


# In[6]:


similarity_score = compute_similarity_score(image1_path, image2_path)
print("Probability of Similarity Score:", similarity_score)


# In[ ]:




