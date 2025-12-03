import pandas as pd
import numpy as np
import os
import re

# --- CONFIGURATION ---
DATA_DIR = 'data/OAICompleteData_ASCII'
BASE_PARQUET = 'data/processed/OAI_tri_modal_real.parquet' 
OUTPUT_PLATINUM = 'data/processed/OAI_platinum_cohort.parquet'

def robust_read(fname):
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath): return None
    for enc in ['utf-8', 'latin-1']:
        try:
            return pd.read_csv(fpath, sep='|', on_bad_lines='skip', encoding=enc, low_memory=False)
        except: continue
    return None

def normalize_side(val):
    """Forces Knee Side to 'R' or 'L'"""
    s = str(val).upper().strip()
    if '1' in s or 'RIGHT' in s or s == 'R': return 'R'
    if '2' in s or 'LEFT' in s or s == 'L': return 'L'
    return None

def clean_oai_format(val):
    """
    Converts OAI text formats like '0: No', '1: Yes', '2: Moderate' -> integer 0, 1, 2.
    Also handles pure numbers.
    """
    if pd.isna(val): return np.nan
    
    s = str(val).strip()
    # If it matches "Number: Text", take the number
    if ':' in s:
        try:
            return float(s.split(':')[0])
        except:
            return np.nan
            
    # Try direct conversion
    try:
        return float(s)
    except:
        return np.nan

def build_cohorts():
    print("🚀 Building Platinum Cohort (Cleaning Mode)...")
    
    # 1. Load Base
    if not os.path.exists(BASE_PARQUET):
        print(f"❌ Base file missing: {BASE_PARQUET}")
        return
    
    df_base = pd.read_parquet(BASE_PARQUET)
    df_base['Merge_Side'] = df_base['Knee_Side'].apply(normalize_side)

    # 2. Load MRI Data
    df_mri = robust_read('kMRI_FNIH_SQ_MOAKS_BICL00.txt')
    
    if df_mri is not None:
        df_mri['Merge_Side'] = df_mri['SIDE'].apply(normalize_side)
        
        # Map and Rename
        rename_map = {'V00MACLBML': 'MRI_BML_Score'}
        if 'V00MACLCYS' in df_mri.columns:
            rename_map['V00MACLCYS'] = 'MRI_Cyst_Score'
            
        cols_to_keep = ['ID', 'Merge_Side'] + list(rename_map.keys())
        df_mri = df_mri[cols_to_keep].rename(columns=rename_map)
        
        # --- CRITICAL STEP: CLEAN THE DATA ---
        # Convert "0: No" -> 0.0
        for col in ['MRI_BML_Score', 'MRI_Cyst_Score']:
            if col in df_mri.columns:
                df_mri[col] = df_mri[col].apply(clean_oai_format)
                
        # Merge
        df_merged = pd.merge(df_base, df_mri, on=['ID', 'Merge_Side'], how='left')
        
        # Check success
        mri_matches = df_merged['MRI_BML_Score'].notna().sum()
        print(f"   Matches found: {mri_matches}")

    # 3. Load Symptoms
    df_clin = robust_read('AllClinical00.txt')
    if df_clin is not None:
        clin_cols = ['ID', 'V00KOOSQOL', 'V00PASE']
        valid_clin = [c for c in clin_cols if c in df_clin.columns]
        
        for c in valid_clin:
            if c != 'ID': 
                # Clean these too just in case
                df_clin[c] = df_clin[c].apply(clean_oai_format)
            else:
                df_clin[c] = pd.to_numeric(df_clin[c], errors='coerce')

        df_merged = pd.merge(df_merged, df_clin[valid_clin], on='ID', how='left')

    # --- PLATINUM FILTER ---
    df_platinum = df_merged.dropna(subset=['MRI_BML_Score'])
    print(f"\n🏆 Platinum Cohort Size: {len(df_platinum)}")
    
    if len(df_platinum) > 100:
        print("   >> Using PLATINUM dataset")
        final_df = df_platinum
    else:
        print("   >> Using GOLD dataset (Imputed)")
        final_df = df_merged
        for c in ['MRI_BML_Score', 'MRI_Cyst_Score']:
            if c in final_df.columns:
                final_df[c] = final_df[c].fillna(0)
            else:
                final_df[c] = 0

    if 'Merge_Side' in final_df.columns:
        final_df = final_df.drop(columns=['Merge_Side'])

    # Normalize
    norm_cols = ['V00KOOSQOL', 'V00PASE', 'MRI_BML_Score', 'MRI_Cyst_Score']
    for c in norm_cols:
        if c in final_df.columns:
            final_df[c] = final_df[c].fillna(final_df[c].median())
            if final_df[c].std() > 0:
                final_df[c] = (final_df[c] - final_df[c].mean()) / final_df[c].std()

    final_df.to_parquet(OUTPUT_PLATINUM, index=False)
    print(f"   Saved to: {OUTPUT_PLATINUM}")
    print(f"   Columns: {final_df.columns.tolist()}")

if __name__ == "__main__":
    build_cohorts()