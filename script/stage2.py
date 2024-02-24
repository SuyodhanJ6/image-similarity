import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import sys
sys.path.append('/home/suyodhan/Documents/Internship/image-similarity')


import logging
import os
from datetime import datetime

# Creating logs directory to store log in files
LOG_DIR = "logs"
LOG_DIR = os.path.join(os.getcwd(), LOG_DIR)

# Creating LOG_DIR if it does not exists.
os.makedirs(LOG_DIR, exist_ok=True)


# Creating file name for log file based on current timestamp
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
file_name = f"log_{CURRENT_TIME_STAMP}.log"

# Creating file path for projects.
log_file_path = os.path.join(LOG_DIR, file_name)


logging.basicConfig(
    filename=log_file_path,
    filemode="w",
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)







# Load pre-trained InceptionV3 model
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
# Log the model name
logging.info("Model name: Inceptionv3")

# Function to preprocess the image for InceptionV3
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
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


# # Example usage
image1_path = '/home/suyodhan/Documents/Internship/image-similarity/Images/newCattle.png'
# image2_path = '/home/suyodhan/Documents/Internship/Image Similarity/cattle2.jpeg'


# # Extract features using InceptionV3
# features1 = extract_features(image1_path)
# features2 = extract_features(image2_path)

# # Generate embeddings using features
# embedding1 = generate_embedding(features1)
# embedding2 = generate_embedding(features2)

# calculate_similarity_score = cosine_similarity(embedding1, embedding2)


# print(calculate_similarity_score)

import os

# Path to the 'claim1' folder
claim1_folder = '/home/suyodhan/Documents/Internship/image-similarity/New/bg removed/bg/claim1'

# List all image files in the 'claim1' folder
claim1_images = [os.path.join(claim1_folder, filename) for filename in os.listdir(claim1_folder) if filename.endswith('.png')]

# Extract features and generate embeddings for 'image1'
features1 = extract_features(image1_path)
embedding1 = generate_embedding(features1)

# List to store similarity scores
similarity_scores = []

# Calculate similarity score for each image in 'claim1'
for claim1_image_path in claim1_images:
    # Extract features and generate embeddings for the current 'claim1' image
    features2 = extract_features(claim1_image_path)
    embedding2 = generate_embedding(features2)
    
    # Calculate similarity score
    similarity_score = cosine_similarity(embedding1, embedding2)    
    
    # Store similarity score
    similarity_scores.append((claim1_image_path, similarity_score))

# print(similarity_scores)

# Print or return similarity scores
for image_path, score in similarity_scores:
    logging.info('Similarity score between {} and {}: {}'.format(image1_path, image_path, score))
