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
        Tri-Modal Dataset: Image + Clinical + Biomarkers
        """
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        
        # Pre-fetch images for sandbox mode
        if self.mode == 'sandbox':
            self.all_image_paths = glob.glob(f"{image_dir}/**/*.png", recursive=True) + \
                                   glob.glob(f"{image_dir}/**/*.jpg", recursive=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- 1. Image Modality (Deep) ---
        if self.mode == 'sandbox':
            img_path = np.random.choice(self.all_image_paths)
        else:
            # Update this pattern based on your real filenames later
            patient_id = str(row['ID'])
            side = 'R' if row['Knee_Side'] == 1 else 'L'
            img_path = os.path.join(self.image_dir, f"{patient_id}_{side}.png")

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except:
            image = torch.zeros(3, 224, 224)

        # --- 2. Clinical Modality (Wide) ---
        clin_cols = ['Age', 'BMI', 'WOMAC_Score', 
                     'KL_Grade_1.0', 'KL_Grade_2.0', 'KL_Grade_3.0', 'KL_Grade_4.0', 'Sex_2']
        
        # Safety check for missing columns (e.g. if one-hot encoding dropped a column)
        clin_features = []
        for col in clin_cols:
            val = row[col] if col in row else 0.0
            clin_features.append(val)
            
        clin_tensor = torch.tensor(clin_features, dtype=torch.float32)

        # --- 3. Biomarker Modality (New Track 3) ---
        # The 5 real biomarkers we integrated
        bio_cols = ['Bio_COMP', 'Bio_CTXI', 'Bio_HA', 'Bio_C2C', 'Bio_CPII']
        
        bio_features = []
        for col in bio_cols:
            val = row[col] if col in row else 0.0
            bio_features.append(val)
            
        bio_tensor = torch.tensor(bio_features, dtype=torch.float32)

        # --- 4. Targets ---
        event = torch.tensor(row['event'], dtype=torch.float32)
        time = torch.tensor(row['time_to_event'], dtype=torch.float32)

        # Return tuple: (Image, Clinical, Bio, Event, Time)
        return image, clin_tensor, bio_tensor, event, time