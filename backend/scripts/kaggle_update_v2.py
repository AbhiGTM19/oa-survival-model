"""
Kaggle Training Script for Tri-Modal Survival Model (v2)

UPDATED FOR NEW COHORT:
- Uses OAI_mega_cohort_v2.parquet 
- 17 clinical features
- 15 biomarkers (expanded from 5)

Copy this code into your Kaggle notebook to replace the old version.
"""

# ============================================================
# CELL 1: IMPORTS AND CONFIGURATION
# ============================================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import pandas as pd
import numpy as np
import os
import glob
from PIL import Image
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')

try:
    from torchsurv.loss.cox import neg_partial_log_likelihood as base_cox
except (ImportError, AttributeError):
    base_cox = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_COUNT = torch.cuda.device_count()
print(f"🚀 Device: {DEVICE} | GPUs Available: {GPU_COUNT}")

# *** UPDATED PATHS FOR V2 COHORT ***
PARQUET_PATH = '/kaggle/input/oai-preprocessed-data/OAI_mega_cohort_v2.parquet'
IMAGE_ROOT = '/kaggle/input/knee-osteoarthritis-dataset-with-severity'
PRETRAINED_PATH = '/kaggle/input/tpu-diffusion-autoencoder/enc_swin_ep300.pth'

# *** UPDATED DIMENSIONS (V2 Cohort) ***
CLINICAL_INPUT_DIM = 17   # Expanded from 15
BIOMARKER_INPUT_DIM = 15  # Expanded from 5
IMG_SIZE = 320

BATCH_SIZE = 32 if GPU_COUNT > 0 else 8
EPOCHS = 60           
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
AUX_WEIGHT = 0.5      
PATIENCE = 15
"""

# ============================================================
# CELL 2: DATASET (UPDATED COLUMN NAMES)
# ============================================================
DATASET_CODE = '''
class TriModalDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, mode='prod'):
        self.df = dataframe.copy()
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        
        if 'KL_Grade' in self.df.columns:
            self.df['kl_label'] = self.df['KL_Grade'].astype(int)
        else:
            self.df['kl_label'] = 0 

        if self.mode == 'sandbox':
            self.all_image_paths = glob.glob(f"{image_dir}/**/*.png", recursive=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if self.mode == 'sandbox':
            img_path = np.random.choice(self.all_image_paths) if self.all_image_paths else "none.png"
        else:
            patient_id = str(row['ID'])
            side = 'R' if row['Knee_Side'] in ['R', 1] else 'L'
            img_path = os.path.join(self.image_dir, f"{patient_id}_{side}.png")

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform: image = self.transform(image)
        except:
            image = torch.zeros(3, IMG_SIZE, IMG_SIZE)

        # *** UPDATED V2 CLINICAL COLUMNS (17 features) ***
        clin_cols = [
            'V00AGE', 'V00EDCV', 'V00INCOME', 'V00PASE', 'V00KOOSQOL', 'V00NSAIDRX',
            'WOMAC_Score', 'WOMAC_Stiffness', 'KOOS_Symptoms',
            'V00MACLBML', 'V00WMTMTH', 'V00WLTMTH',
            'KL_Grade_1.0', 'KL_Grade_2.0', 'KL_Grade_3.0', 'KL_Grade_4.0', 'Sex_2'
        ]
        clin_tensor = torch.tensor([row.get(c, 0.0) for c in clin_cols], dtype=torch.float32)

        # *** UPDATED V2 BIOMARKER COLUMNS (15 features) ***
        bio_cols = [
            'Bio_C1_2C', 'Bio_C2C', 'Bio_CPII', 'Bio_COMP', 'Bio_CS846',
            'Bio_COLL2_1_NO2', 'Bio_CTXI', 'Bio_NTXI', 'Bio_PIIANP',
            'Bio_HA', 'Bio_MMP3', 'Bio_uCTXII', 'Bio_uC1_2C', 'Bio_uC2C', 'Bio_uNTXI'
        ]
        bio_tensor = torch.tensor([row.get(c, 0.0) for c in bio_cols], dtype=torch.float32)

        event = torch.tensor(row['event'], dtype=torch.float32)
        time = torch.tensor(row['time_to_event'], dtype=torch.float32)
        kl_label = torch.tensor(row.get('kl_label', 0), dtype=torch.long)

        return image, clin_tensor, bio_tensor, event, time, kl_label
'''

# ============================================================
# KEY CHANGES SUMMARY
# ============================================================
"""
WHAT CHANGED FROM v1 TO v2:

1. PARQUET FILE:
   - OLD: OAI_mega_cohort_imputed.parquet
   - NEW: OAI_mega_cohort_v2.parquet

2. CLINICAL FEATURES (17 vs 15):
   Old: Age, BMI, WOMAC_Score, KL_Grade_*, Sex_2, V00KOOSQOL, V00PASE, MRI_BML_Score, *_Thickness, Education, Income
   New: V00AGE, V00EDCV, V00INCOME, V00PASE, V00KOOSQOL, V00NSAIDRX, WOMAC_Score, WOMAC_Stiffness, KOOS_Symptoms, V00MACLBML, V00WMTMTH, V00WLTMTH, KL_Grade_*, Sex_2

3. BIOMARKERS (15 vs 5):
   Old: Bio_COMP, Bio_CTXI, Bio_HA, Bio_C2C, Bio_CPII
   New: Bio_C1_2C, Bio_C2C, Bio_CPII, Bio_COMP, Bio_CS846, Bio_COLL2_1_NO2, Bio_CTXI, Bio_NTXI, Bio_PIIANP, Bio_HA, Bio_MMP3, Bio_uCTXII, Bio_uC1_2C, Bio_uC2C, Bio_uNTXI

4. MODEL DIMENSIONS:
   - CLINICAL_INPUT_DIM: 15 -> 17
   - BIOMARKER_INPUT_DIM: 5 -> 15

TO UPDATE YOUR NOTEBOOK:
1. Upload OAI_mega_cohort_v2.parquet to Kaggle datasets
2. Change PARQUET_PATH
3. Change CLINICAL_INPUT_DIM = 17
4. Change BIOMARKER_INPUT_DIM = 15
5. Replace TriModalDataset class with the updated version above
6. Update SwinMultiTaskModel: bio_encoder first layer should be nn.Linear(15, 32)
"""

print(__doc__)
