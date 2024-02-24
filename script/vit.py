import os
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import numpy as np

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



# Load the pre-trained image processor and model
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

logging.info("Model name: VIT")
logging.info("Image Format: BR")

def cosine_similarity(image1_embedding, current_embedding) -> bool:
    # Handle case where one or both embeddings are None
    # if image1_embedding is None or current_embedding is None:
    #     return 0.0
    
    return np.dot(image1_embedding, current_embedding) / (np.linalg.norm(image1_embedding) * np.linalg.norm(current_embedding))

# Load and process image1
image1_path = "/home/suyodhan/Documents/Internship/image-similarity/New/grayscale_applied/grayscale_image/main/grayscale_1-removebg-preview.png"
image1 = Image.open(image1_path)
inputs1 = processor(images=image1, return_tensors="pt")
outputs1 = model(**inputs1)
last_hidden_states_image1 = outputs1.last_hidden_state
array_image1 = last_hidden_states_image1.detach().numpy()
array_image1_flat = array_image1.flatten()

# Path to the 'claim1' folder
claim1_folder = '/home/suyodhan/Documents/Internship/image-similarity/New/grayscale_applied/grayscale_image/claim2'

# List all image files in the 'claim1' folder
claim1_images = [os.path.join(claim1_folder, filename) for filename in os.listdir(claim1_folder) if filename.endswith('.png')]

# List to store similarity scores
similarity_scores = []

# Calculate similarity score for each image in 'claim1'
for claim1_image_path in claim1_images:
    # Load and process the current 'claim1' image
    image2 = Image.open(claim1_image_path)
    inputs2 = processor(images=image2, return_tensors="pt")
    outputs2 = model(**inputs2)
    last_hidden_states_image2 = outputs2.last_hidden_state
    array_image2 = last_hidden_states_image2.detach().numpy()
    array_image2_flat = array_image2.flatten()
    
    # Calculate the cosine similarity between the embeddings of 'image1' and the current 'claim1' image
    similarity_score = cosine_similarity(array_image1_flat, array_image2_flat)
    
    # Store the similarity score along with the image path
    similarity_scores.append((claim1_image_path, similarity_score))


# Print or log the similarity scores
for image_path, score in similarity_scores:
    logging.info('Similarity score between {} and {}: {}'.format(image1_path, image_path, score))
