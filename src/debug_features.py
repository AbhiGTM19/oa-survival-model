import pandas as pd
import numpy as np
import torch
from dataset import TriModalDataset
from torch.utils.data import DataLoader

# Config
PARQUET_PATH = 'data/processed/OAI_mega_cohort.parquet'
IMAGE_ROOT = 'data/sandbox'

def debug_dataloader():
    print("🕵️‍♂️ Starting Forensic Data Audit...")
    
    if not os.path.exists(PARQUET_PATH):
        print("❌ Parquet not found.")
        return

    # 1. Load Raw Data
    df = pd.read_parquet(PARQUET_PATH)
    print(f"   Raw Data Shape: {df.shape}")
    
    # Check Column Names created by dummies
    df_dum = pd.get_dummies(df, columns=['KL_Grade', 'Sex'], drop_first=True)
    print(f"   Dummy Columns Present: {[c for c in df_dum.columns if 'KL' in c]}")
    
    # 2. Initialize Dataset (Mock Transform)
    ds = TriModalDataset(df_dum, IMAGE_ROOT, mode='sandbox')
    
    # 3. Pull one sample
    print("\n   --- Inspecting Sample Tensor ---")
    img, clin, bio, evt, time = ds[0]
    
    # Check Clinical Tensor (Age, BMI, WOMAC, KL1, KL2, KL3, KL4, Sex)
    print(f"   Clinical Tensor: {clin}")
    
    # CRITICAL CHECK: Are the KL Grade slots (indices 3,4,5,6) all zero?
    kl_sum = clin[3:7].sum().item()
    print(f"   Sum of KL_Grade Features: {kl_sum}")
    
    if kl_sum == 0:
        print("   🚨 CRITICAL FAILURE: KL_Grade features are all ZERO. Model is blind to X-ray severity!")
    else:
        print("   ✅ KL_Grade features are active.")

    # Check Bio Tensor
    print(f"   Biomarker Tensor: {bio}")
    if torch.all(bio.eq(0)):
        print("   ⚠️ WARNING: Biomarkers are all zero (Imputation issue?)")

if __name__ == "__main__":
    import os
    debug_dataloader()