import torch
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image
import timm # PyTorch Image Models library

# --- Configuration ---
TRAIN_IMAGE_DIR = r'C:\Users\llegi\OneDrive\Desktop\Amazon\Amazon\train_images'
TEST_IMAGE_DIR = r'C:\Users\llegi\OneDrive\Desktop\Amazon\Amazon\test_images'
MODEL_NAME = 'resnet50'  # Using a standard ResNet50 from timm
BATCH_SIZE = 64 # You can adjust this based on your GPU memory
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- 1. Load DataFrames ---
# These DataFrames are used to get the list of sample_ids
train_df = pd.read_csv("Amazon/Preprocessed_data/preprocessed_data_train.csv")
test_df = pd.read_csv("Amazon/Preprocessed_data/preprocessed_data_test.csv")
print(f"Data loaded. Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# --- 2. Load Model and Transformations ---
# Load the pre-trained model once
print(f"Loading model '{MODEL_NAME}'...")
# num_classes=0 removes the final classification layer, making it a feature extractor
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(DEVICE)
model.eval()

# Get the model-specific transformations
# This ensures images are resized, cropped, and normalized correctly
# Pass an empty dictionary for args and the model as a keyword argument
data_config = timm.data.resolve_data_config({}, model=model)
transforms = timm.data.create_transform(**data_config)
print("Model and transformations loaded successfully.")

def generate_image_embeddings(df, image_dir, output_filename, mode="Train"):
    """
    Generates image embeddings for a DataFrame of sample_ids.
    """
    sample_ids = df['sample_id'].tolist()
    embeddings_list = []
    
    print(f"[{mode}] Starting {MODEL_NAME} embedding generation on {len(sample_ids)} images...")

    # Process in batches
    for i in tqdm(range(0, len(sample_ids), BATCH_SIZE), desc=f"[{mode}] Image Embeddings"):
        batch_ids = sample_ids[i:i + BATCH_SIZE]
        batch_tensors = []
        
        for sample_id in batch_ids:
            img_path = os.path.join(image_dir, f"{sample_id}.jpg")
            try:
                # Load image and apply transformations
                img = Image.open(img_path).convert('RGB')
                tensor = transforms(img)
                batch_tensors.append(tensor)
            except Exception:
                # If an image is missing or corrupt, append a zero tensor
                print(f"Warning: Could not load image {img_path}. Using a zero tensor.")
                batch_tensors.append(torch.zeros((3, 224, 224))) # C, H, W format
        
        # Stack list of tensors into a single batch tensor and move to GPU
        batch = torch.stack(batch_tensors).to(DEVICE)
        
        # Get model output
        with torch.no_grad():
            features = model(batch)
        
        # Move features to CPU and append to the list
        embeddings_list.append(features.cpu().numpy())
    
    # Concatenate all batches and save
    image_features = np.concatenate(embeddings_list, axis=0)
    
    np.save(output_filename, image_features)
    
    print(f"[{mode}] ✅ Image Features generated (Shape: {image_features.shape}) and saved to {output_filename}")
    return image_features

# --- Execution for Train and Test ---
IMAGE_FEAT_TRAIN = 'Amazon/Preprocessed_data/image_features_train.npy'
IMAGE_FEAT_TEST = 'Amazon/Preprocessed_data/image_features_test.npy'

# Generate embeddings for the training set
X_img_train = generate_image_embeddings(train_df, TRAIN_IMAGE_DIR, IMAGE_FEAT_TRAIN, mode="Train")

# Generate embeddings for the test set
X_img_test = generate_image_embeddings(test_df, TEST_IMAGE_DIR, IMAGE_FEAT_TEST, mode="Test")