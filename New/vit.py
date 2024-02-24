# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DD0MP0oyeZ076QU84Vg5leNG0-c6N2di
"""

!pip install transformers
!pip install torch
!pip install requests
!pip install pillow

import numpy as np
import warnings
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore")

# Load image processor and model
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

# Load images and process them in batches
batch_size = 1  # Adjust batch size based on available memory
image_paths = ["cattle_4.jpg", "cattle_4.jpg"]
num_images = len(image_paths)
all_embeddings = []

for i in range(0, num_images, batch_size):
    batch_image_paths = image_paths[i:i+batch_size]
    batch_embeddings = []

    for image_path in batch_image_paths:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.detach().numpy().squeeze()
        batch_embeddings.append(embedding)

    all_embeddings.extend(batch_embeddings)

# Calculate cosine similarity
def cosine_similarity(embeddings1, embeddings2):
    norm1 = np.linalg.norm(embeddings1)
    norm2 = np.linalg.norm(embeddings2)
    similarity = np.dot(embeddings1, embeddings2) / (norm1 * norm2)
    return similarity

# Get embeddings for the first two images
embeddings = np.array(all_embeddings[:2])

# Calculate similarity score
score = cosine_similarity(embeddings[0].flatten(), embeddings[1].flatten())
score

"""GREY SCALE"""

# Load images and process them in batches
batch_size = 1  # Adjust batch size based on available memory
image_paths = ["cattle_4.jpg", "cattle_4.jpg"]
num_images = len(image_paths)
all_embeddings = []

for i in range(0, num_images, batch_size):
    batch_image_paths = image_paths[i:i+batch_size]
    batch_embeddings = []

    for image_path in batch_image_paths:
        # Convert image to grayscale
        image = Image.open(image_path).convert('L')
        # Convert grayscale image to RGB by repeating the single channel 3 times
        image = np.stack((image,) * 3, axis=-1)
        image = Image.fromarray(image)

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.detach().numpy().squeeze()
        batch_embeddings.append(embedding)

    all_embeddings.extend(batch_embeddings)

# Calculate cosine similarity
def cosine_similarity(embeddings1, embeddings2):
    norm1 = np.linalg.norm(embeddings1)
    norm2 = np.linalg.norm(embeddings2)
    similarity = np.dot(embeddings1, embeddings2) / (norm1 * norm2)
    return similarity

# Get embeddings for the first two images
embeddings = np.array(all_embeddings[:2])

# Calculate similarity score
score = cosine_similarity(embeddings[0].flatten(), embeddings[1].flatten())
score