#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[7]:


import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image


# In[8]:


# Load pre-trained InceptionV3 model
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')


# In[9]:


# Function to preprocess the image and convert into grey scale(modified)
def preprocess_image(image_path):
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    # Resize the image to match InceptionV3 input size
    img = img.resize((299, 299))
    # Convert image to array
    img_array = np.array(img)
    # Convert grayscale image to RGB by repeating the single channel 3 times
    img_array = np.stack((img_array,) * 3, axis=-1)
    # Add an extra dimension to represent batch size
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess input based on InceptionV3 requirements
    img_array = preprocess_input(img_array)
    return img_array


# Function to extract features using InceptionV3
def extract_features(image_path):
    img = preprocess_image(image_path)
    features = inception_model.predict(img)
    return features

# Function to generate embedding using features
def generate_embedding(features):
    embedding = features / np.linalg.norm(features)
    return embedding

# Function to calculate similarity between two embeddings using cosine similarity
def calculate_similarity(embedding1, embedding2):
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]


# In[10]:


def cosine_similarity(image1_embedding, image2_embedding) -> bool:
    # Handle case where one or both embeddings are None
    if image1_embedding is None or image2_embedding is None:
        return 0.0
    
    # Calculate dot product of the embeddings
    dot_product = np.dot(image1_embedding.flatten(), image2_embedding.flatten())
    # Calculate norms of the embeddings
    norm1 = np.linalg.norm(image1_embedding)
    norm2 = np.linalg.norm(image2_embedding)
    # Calculate cosine similarity
    similarity = dot_product / (norm1 * norm2)
    return similarity


# In[11]:


# Example usage
image1_path = 'cattle_5.jpg'
image2_path = 'cattle_4.jpg'


# In[ ]:





# In[12]:


# Extract features using InceptionV3
features1 = extract_features(image1_path)
features2 = extract_features(image2_path)

# Generate embeddings using features
embedding1 = generate_embedding(features1)
embedding2 = generate_embedding(features2)


# In[13]:


embedding1


# In[14]:


embedding2


# In[15]:


calculate_similarity_score = cosine_similarity(embedding1, embedding2)


# In[16]:


calculate_similarity_score


# In[ ]:




