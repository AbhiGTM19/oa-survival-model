import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple

class TriModalDataset(Dataset):
    """
    Dataset for Tri-Modal Osteoarthritis Survival Analysis.
    Handles Image, Clinical, and Biomarker data.
    """
    def __init__(self, dataframe: pd.DataFrame, image_dir: str, transform: Optional[Callable] = None, mode: str = 'prod'):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing patient data.
            image_dir (str): Directory containing X-Ray images.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'prod' for production (strict ID matching), 'sandbox' for testing (random images).
        """
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        
        if self.mode == 'sandbox':
            self.all_image_paths = glob.glob(f"{image_dir}/**/*.png", recursive=True) + \
                                   glob.glob(f"{image_dir}/**/*.jpg", recursive=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # --- 1. Image Modality ---
        if self.mode == 'sandbox':
            img_path = np.random.choice(self.all_image_paths)
        else:
            patient_id = str(row['ID'])
            side = 'R' if row['Knee_Side'] == 'R' else 'L' # Handle 'R'/'L' string if present
            # Fallback for numeric side if your parquet uses 1/2
            if row['Knee_Side'] == 1: side = 'R'
            if row['Knee_Side'] == 2: side = 'L'
            
            img_path = os.path.join(self.image_dir, f"{patient_id}_{side}.png")

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception:
            # Black image fallback
            image = torch.zeros(3, 224, 224)

        # --- 2. Clinical Modality (PLATINUM CONFIG) ---
        # 11 Features: 8 Original + 3 New High-Value
        clin_cols = [
            # Original Basic
            'Age', 'BMI', 'WOMAC_Score', 
            'KL_Grade_1.0', 'KL_Grade_2.0', 'KL_Grade_3.0', 'KL_Grade_4.0', 'Sex_2',
            # New Platinum Features
            'V00KOOSQOL',   # Quality of Life (Strong predictor)
            'V00PASE',      # Physical Activity
            'MRI_BML_Score', # Bone Marrow Lesion Score (The strongest MRI feature)
            'Medial_Tibial_Thickness',
            'Lateral_Tibial_Thickness',
            'Education',
            'Income'
        ]
        
        clin_features = []
        for col in clin_cols:
            # Safety: If column missing (e.g. one-hot dropped), use 0
            val = row[col] if col in row else 0.0
            clin_features.append(val)
            
        clin_tensor = torch.tensor(clin_features, dtype=torch.float32)

        # --- 3. Biomarker Modality ---
        bio_cols = ['Bio_COMP', 'Bio_CTXI', 'Bio_HA', 'Bio_C2C', 'Bio_CPII']
        bio_features = []
        for col in bio_cols:
            val = row[col] if col in row else 0.0
            bio_features.append(val)
            
        bio_tensor = torch.tensor(bio_features, dtype=torch.float32)

        # --- 4. Targets ---
        event = torch.tensor(row['event'], dtype=torch.float32)
        time = torch.tensor(row['time_to_event'], dtype=torch.float32)

        return image, clin_tensor, bio_tensor, event, time

# --- Added for Phase 4 (Generative Training) ---
class FullScaleImageDataset(Dataset):
    """
    Dataset for Full-Scale Image Training (Generative Models).
    """
    def __init__(self, image_dir: str, transform: Optional[Callable] = None):
        """
        Args:
            image_dir (str): Directory containing images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        # Recursive glob to find ALL images
        self.image_paths = glob.glob(f"{image_dir}/**/*.png", recursive=True) + \
                           glob.glob(f"{image_dir}/**/*.jpg", recursive=True)
        print(f"🚀 Found {len(self.image_paths)} images for Full-Scale Training.")
        
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        try:
            # Load as Grayscale (1 channel) for the Medical Diffusion Model
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception:
            # Return black image if file is corrupt
            return torch.zeros(1, 64, 64)