import torch
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image
import timm # PyTorch Image Models library

# --- Configuration ---
TRAIN_IMAGE_DIR = r'C:\\Amazon\\train_images' # <-- IMPORTANT: Use your correct local path
TEST_IMAGE_DIR = r'C:\\Amazon\\test_images'   # <-- IMPORTANT: Use your correct local path
MODEL_NAME = 'resnet50'
BATCH_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- 1. Load DataFrames ---
train_df = pd.read_csv(r"C:\\Amazon\\Preprocessed_data\\preprocessed_data_train.csv") # Use your correct path
test_df = pd.read_csv(r"C:\\Amazon\\Preprocessed_data\\preprocessed_data_test.csv")   # Use your correct path
print(f"Data loaded. Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# --- 2. Load Model and Transformations ---
print(f"Loading model '{MODEL_NAME}'...")
# num_classes=0 removes the final classification layer, making it a feature extractor
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(DEVICE)
model.eval() # Set the model to evaluation mode

# Get the model-specific transformations (resize, crop, normalize)
data_config = timm.data.resolve_data_config({}, model=model)
transforms = timm.data.create_transform(**data_config)
print("Model and transformations loaded successfully.")

def generate_image_embeddings(df, image_dir, output_filename, mode="Train"):
    """Generates image embeddings for a DataFrame of sample_ids using PyTorch."""
    sample_ids = df['sample_id'].tolist()
    embeddings_list = []

    print(f"[{mode}] Starting {MODEL_NAME} embedding generation...")

    for i in tqdm(range(0, len(sample_ids), BATCH_SIZE), desc=f"[{mode}] Image Embeddings"):
        batch_ids = sample_ids[i:i + BATCH_SIZE]
        batch_tensors = []

        for sample_id in batch_ids:
            img_path = os.path.join(image_dir, f"{sample_id}.jpg")
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = transforms(img)
                batch_tensors.append(tensor)
            except Exception:
                # If an image is missing or corrupt, append a zero tensor
                batch_tensors.append(torch.zeros((3, data_config['input_size'][1], data_config['input_size'][2])))

        batch = torch.stack(batch_tensors).to(DEVICE)

        with torch.no_grad(): # Disable gradient calculation for inference
            features = model(batch)

        embeddings_list.append(features.cpu().numpy())

    image_features = np.concatenate(embeddings_list, axis=0)
    np.save(output_filename, image_features)
    print(f"[{mode}] ✅ Image Features generated (Shape: {image_features.shape}) and saved to {output_filename}")
    return image_features

# --- Execution ---
# IMAGE_FEAT_TRAIN = r'C:\\Amazon\\Preprocessed_data\\image_features_train.npy' # Use your correct path
IMAGE_FEAT_TEST = r'C:\\Amazon\\Preprocessed_data\\image_features_test.npy'   # Use your correct path
# generate_image_embeddings(train_df, TRAIN_IMAGE_DIR, IMAGE_FEAT_TRAIN, mode="Train")
generate_image_embeddings(test_df, TEST_IMAGE_DIR, IMAGE_FEAT_TEST, mode="Test")
