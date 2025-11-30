import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class TriModalDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, mode='prod'):
        """
        Args:
            dataframe (pd.DataFrame): Data with 'ID', 'Knee_Side', 'event', 'time', etc.
            image_dir (str): Root directory of images.
            transform (callable, optional): PyTorch transforms (resize, normalize).
            mode (str): 'prod' (match IDs) or 'sandbox' (random images).
        """
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        
        # Pre-fetch all available image paths for Sandbox mode
        if self.mode == 'sandbox':
            self.all_image_paths = glob.glob(f"{image_dir}/**/*.png", recursive=True) + \
                                   glob.glob(f"{image_dir}/**/*.jpg", recursive=True)
            print(f"Dataset initialized in SANDBOX mode. Found {len(self.all_image_paths)} images to sample from.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get Tabular Data (The "Wide" Input)
        # We need the 8 features: Age, BMI, WOMAC, KL(one-hot), Sex
        # Note: Ensure your DF has these columns processed or process them here.
        # For now, we assume the DF passed in is the PROCESSED one (numeric).
        
        # Extract row
        row = self.df.iloc[idx]
        
        # 2. Get Image Data (The "Deep" Input)
        if self.mode == 'sandbox':
            # Pick a RANDOM image to test the pipeline
            img_path = np.random.choice(self.all_image_paths)
        else:
            # PRODUCTION LOGIC (Strict ID matching)
            # OAI filenames usually follow a pattern like: {ID}_{Side}_Visit.png
            # You will update this logic in Phase 4 when you get real images.
            patient_id = str(row['ID'])
            side = 'R' if row['Knee_Side'] == 1 else 'L'
            img_path = os.path.join(self.image_dir, f"{patient_id}_{side}.png")

        # Load and Transform Image
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except (FileNotFoundError, OSError):
            # Fallback if image is missing (shouldn't happen in sandbox mode)
            print(f"Warning: Image not found {img_path}. Using black image.")
            image = torch.zeros(3, 224, 224)

        # 3. Prepare Tabular Tensor
        # Select only the features your model expects (8 columns)
        # Adjust these column names if they differ in your final parquet
        feature_cols = ['Age', 'BMI', 'WOMAC_Score', 
                        'KL_Grade_1.0', 'KL_Grade_2.0', 'KL_Grade_3.0', 'KL_Grade_4.0', 'Sex_2']
        
        # Handle case where one-hot cols might be missing or named differently
        # For safety, we convert available cols or pad
        clinical_features = row[feature_cols].values.astype(np.float32)
        clinical_tensor = torch.tensor(clinical_features)

        # 4. Get Targets
        event = torch.tensor(row['event'], dtype=torch.float32)
        time = torch.tensor(row['time_to_event'], dtype=torch.float32)

        return image, clinical_tensor, event, time