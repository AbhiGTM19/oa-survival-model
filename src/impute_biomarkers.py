import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os

# Paths
INPUT_PATH = 'data/processed/OAI_mega_cohort.parquet'
OUTPUT_PATH = 'data/processed/OAI_mega_cohort_imputed.parquet'

# Features to use for imputation context (Clinical + MRI)
# These features help predict the missing biomarkers
CONTEXT_FEATS = [
    'Age', 'BMI', 'Sex', 'KL_Grade', 'WOMAC_Score', 
    'KOOS_Score', 'Stiffness', 'V00PASE', 'V00NSAIDRX',
    'MRI_BML_Score', 'Medial_Tibial_Thickness', 'Lateral_Tibial_Thickness'
]

# Biomarkers to impute
BIO_FEATS = ['Bio_COMP', 'Bio_CTXI', 'Bio_HA', 'Bio_C2C', 'Bio_CPII']

def impute_data():
    print(f"🚀 Starting Biomarker Imputation...")
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Input file not found: {INPUT_PATH}")
        return

    df = pd.read_parquet(INPUT_PATH)
    print(f"   Loaded Mega Cohort: {df.shape}")
    
    # 1. Preprocessing
    # Standardize Sex
    if df['Sex'].dtype == 'O':
        df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})
        
    # Standardize Categoricals
    for col in ['Education', 'Income']:
        if col in df.columns and df[col].dtype == 'O':
            df[col] = df[col].astype('category').cat.codes

    # Clean Biomarkers (Handle '< 200' strings etc.)
    print("   Cleaning non-numeric biomarker values...")
    for col in BIO_FEATS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 2. Prepare Imputation Matrix
    # We combine Context Features + Biomarkers for the imputer
    # The imputer uses relationships between ALL variables to fill gaps
    impute_cols = CONTEXT_FEATS + BIO_FEATS
    
    # Ensure all context feats exist (fill missing context with median first if needed)
    # For this step, we just select the columns that exist
    existing_cols = [c for c in impute_cols if c in df.columns]
    
    data_to_impute = df[existing_cols].copy()
    
    # 3. Run KNN Imputer (Preserves Variance better than MICE)
    print("   Running KNN Imputer (k=5)...")
    # KNN picks values from the 5 most similar patients based on Clinical/MRI features
    # This ensures imputed values are 'real' observed values, preserving the natural distribution
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    data_imputed = imputer.fit_transform(data_to_impute)
    
    # 4. Update DataFrame
    df_imputed = df.copy()
    df_imputed[existing_cols] = data_imputed
    
    # 5. Save
    df_imputed.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ Imputed Cohort Saved: {OUTPUT_PATH}")
    print(f"   New Shape: {df_imputed.shape}")
    
    # Verify completeness
    missing = df_imputed[BIO_FEATS].isnull().sum().sum()
    print(f"   Remaining Missing Biomarkers: {missing}")

if __name__ == "__main__":
    impute_data()
