import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
DATA_DIR = 'data/OAICompleteData_ASCII'
BASE_PARQUET = 'data/processed/OAI_tri_modal_real.parquet'
OUTPUT_PARQUET = 'data/processed/OAI_mega_cohort.parquet'

def robust_read(fname):
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        print(f"⚠️ Missing: {fname}")
        return None
        
    # Try encodings
    for enc in ['utf-8', 'latin-1']:
        try:
            with open(fpath, 'r', encoding=enc) as f:
                line = f.readline()
                sep = ',' if ',' in line and '|' not in line else '|'
            return pd.read_csv(fpath, sep=sep, on_bad_lines='skip', encoding=enc, low_memory=False)
        except:
            continue
    return None

def build_mega_cohort():
    print("🚀 Starting Mega-Merge...")
    
    # 1. Load Base Cohort (3526 Knees)
    if not os.path.exists(BASE_PARQUET):
        print(f"❌ Base file not found: {BASE_PARQUET}")
        return

    df_base = pd.read_parquet(BASE_PARQUET)
    print(f"Base Cohort: {df_base.shape}")

    # 2. Merge Clinical (Patient-Level)
    print("   Merging Clinical Data...")
    df_clin = robust_read('AllClinical00.txt')
    
    if df_clin is not None:
        # CORRECTED COLUMN NAMES based on your audit
        # V00KOOSYMR = KOOS Symptoms Right
        # V00KOOSYML = KOOS Symptoms Left
        # V00WOMSTFR = WOMAC Stiffness Right
        # V00WOMSTFL = WOMAC Stiffness Left
        # V00PASE    = Physical Activity Scale
        # V00NSAIDRX = NSAID Medication Use
        
        cols_to_clean = ['V00PASE', 'V00KOOSYMR', 'V00KOOSYML', 'V00WOMSTFR', 'V00WOMSTFL', 'V00NSAIDRX']
        
        # Ensure they exist before processing
        existing_cols = [c for c in cols_to_clean if c in df_clin.columns]
        
        for c in existing_cols:
            df_clin[c] = pd.to_numeric(df_clin[c], errors='coerce')
        
        # Merge Patient-Level vars (PASE, Meds)
        # Note: V00NSAIDRX is patient-level
        df_merged = pd.merge(df_base, df_clin[['ID', 'V00PASE', 'V00NSAIDRX']], on='ID', how='left')
        
        # Map Side-Specific Columns (KOOS & Stiffness)
        # Create lookup dicts
        koos_r = dict(zip(df_clin.ID, df_clin.V00KOOSYMR)) if 'V00KOOSYMR' in df_clin.columns else {}
        koos_l = dict(zip(df_clin.ID, df_clin.V00KOOSYML)) if 'V00KOOSYML' in df_clin.columns else {}
        
        stiff_r = dict(zip(df_clin.ID, df_clin.V00WOMSTFR)) if 'V00WOMSTFR' in df_clin.columns else {}
        stiff_l = dict(zip(df_clin.ID, df_clin.V00WOMSTFL)) if 'V00WOMSTFL' in df_clin.columns else {}
        
        # Vectorized map for KOOS
        df_merged['KOOS_Score'] = np.where(
            df_merged['Knee_Side'] == 1,
            df_merged['ID'].map(koos_r),
            df_merged['ID'].map(koos_l)
        )
        
        # Vectorized map for Stiffness
        df_merged['Stiffness'] = np.where(
            df_merged['Knee_Side'] == 1,
            df_merged['ID'].map(stiff_r),
            df_merged['ID'].map(stiff_l)
        )
        
        # Impute NaNs (Standard median/mode filling)
        df_merged['KOOS_Score'] = df_merged['KOOS_Score'].fillna(df_merged['KOOS_Score'].median())
        df_merged['Stiffness'] = df_merged['Stiffness'].fillna(df_merged['Stiffness'].median())
        df_merged['V00PASE'] = df_merged['V00PASE'].fillna(df_merged['V00PASE'].median())
        df_merged['V00NSAIDRX'] = df_merged['V00NSAIDRX'].fillna(0) # Assume 0 (No) if missing
        
        df_base = df_merged
        print(f"   ✅ Added PASE, KOOS, Stiffness, NSAIDs. New Shape: {df_base.shape}")

    # 3. Merge MRI (Knee-Level)
    print("   Merging MRI Data (MOAKS)...")
    df_mri = robust_read('kMRI_FNIH_SQ_MOAKS_BICL00.txt')
    
    if df_mri is not None:
        # Clean Side
        if df_mri['SIDE'].dtype == 'O': 
            df_mri['SIDE'] = df_mri['SIDE'].map({'Right': 1, 'Left': 2})
            
        # Rename columns
        # V00MACLBML = Medial Tibial BML (Bone Marrow Lesion) - Strong predictor
        # V00MACLCYS = Medial Tibial Cysts
        df_mri = df_mri.rename(columns={
            'SIDE': 'Knee_Side', 
            'V00MACLBML': 'MRI_BML_Score',
            'V00MACLCYS': 'MRI_Cyst_Score'
        })
        
        # Check if columns actually exist (MOAKS file columns can vary)
        mri_cols_to_keep = ['ID', 'Knee_Side']
        if 'MRI_BML_Score' in df_mri.columns: mri_cols_to_keep.append('MRI_BML_Score')
        if 'MRI_Cyst_Score' in df_mri.columns: mri_cols_to_keep.append('MRI_Cyst_Score')
        
        # Merge
        df_base = pd.merge(
            df_base, 
            df_mri[mri_cols_to_keep], 
            on=['ID', 'Knee_Side'], 
            how='left'
        )
        
        # Impute MRI
        if 'MRI_BML_Score' in df_base.columns:
            df_base['MRI_BML_Score'] = df_base['MRI_BML_Score'].fillna(0)
        else:
            df_base['MRI_BML_Score'] = 0 # Create dummy if missing
            
        if 'MRI_Cyst_Score' in df_base.columns:
            df_base['MRI_Cyst_Score'] = df_base['MRI_Cyst_Score'].fillna(0)
        else:
            df_base['MRI_Cyst_Score'] = 0
        
        print(f"   ✅ Added MRI Scores. New Shape: {df_base.shape}")

    # 4. Save
    df_base.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\n🎉 Mega-Cohort Saved: {OUTPUT_PARQUET}")
    print(f"Columns: {df_base.columns.tolist()}")

if __name__ == "__main__":
    build_mega_cohort()