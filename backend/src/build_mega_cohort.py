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

def build_base_cohort():
    print("   ⚠️ Base file not found. Building from raw OAI files...")
    
    # Define paths
    enrollees_path = 'Enrollees.txt'
    outcomes_path = 'OUTCOMES99.txt'
    xray_path = 'KXR_SQ_BU00.txt'
    
    # Load Data
    df_enrol = robust_read(enrollees_path)
    df_out = robust_read(outcomes_path)
    df_xray = robust_read(xray_path)
    
    if df_enrol is None or df_out is None or df_xray is None:
        print("   ❌ Critical raw files missing (Enrollees, Outcomes, or XRay). Cannot build base.")
        return None

    # 1. Clean Outcomes
    if 'id' in df_out.columns:
        df_out = df_out.rename(columns={'id': 'ID'})
    
    # 2. Clean X-Ray (Baseline KL Grade)
    # Filter for READPRJ=15 (Main Reading)
    if 'READPRJ' in df_xray.columns:
        df_xray = df_xray[df_xray['READPRJ'] == 15].copy()
    
    # Clean KL Grade
    if 'V00XRKL' in df_xray.columns:
        # Extract number from "2: Minimal" -> 2
        df_xray['KL_Grade'] = pd.to_numeric(
            df_xray['V00XRKL'].astype(str).str.split(':').str[0], 
            errors='coerce'
        )
    else:
        print("   ⚠️ V00XRKL (KL Grade) not found in XRay file.")
        df_xray['KL_Grade'] = np.nan

    # Clean Side
    if 'SIDE' in df_xray.columns:
        # Extract number from "1: Right" -> 1
        df_xray['Knee_Side'] = pd.to_numeric(
            df_xray['SIDE'].astype(str).str.split(':').str[0], 
            errors='coerce'
        )
    
    # Keep essential X-Ray columns
    df_xray = df_xray[['ID', 'Knee_Side', 'KL_Grade']].dropna(subset=['KL_Grade', 'Knee_Side'])
    
    # Remove duplicates
    df_xray = df_xray.drop_duplicates(subset=['ID', 'Knee_Side'])
    
    # 3. Merge to create Base
    # Start with X-Ray (Knee Level)
    df_base = pd.merge(df_xray, df_enrol, on='ID', how='left')
    df_base = pd.merge(df_base, df_out, on='ID', how='left')
    
    # 4. Define Target (Event/Time)
    # Assuming OUTCOMES99 contains 'V99RNTCNT' (Right Knee TKR Time) and 'V99LNTCNT' (Left Knee TKR Time)
    # And 'V99RKR' / 'V99LKR' (1=Yes, 0=No) or similar.
    # We need to map side-specific outcomes.
    
    # Note: Column names in OUTCOMES99 can vary. 
    # Common OAI: V99ERKR (Right Event), V99ELKR (Left Event), V99RKV (Right Time), V99LKV (Left Time)
    # Let's check columns if possible, or use standard OAI mapping.
    # For now, we'll try standard names.
    
    # Map Event
    # 1=Right, 2=Left
    
    # Initialize
    df_base['event'] = 0
    df_base['time_to_event'] = 0.0
    
    # Check for common outcome columns
    # V99ERKR: Total Knee Replacement Right (1=Yes)
    # V99ELKR: Total Knee Replacement Left (1=Yes)
    # V99RNTCNT: Time to TKR Right (days)
    # V99LNTCNT: Time to TKR Left (days)
    
    has_outcomes = False
    if 'V99ERKR' in df_base.columns and 'V99ELKR' in df_base.columns:
        # Right Knee
        mask_r = (df_base['Knee_Side'] == 1)
        df_base.loc[mask_r, 'event'] = pd.to_numeric(df_base.loc[mask_r, 'V99ERKR'], errors='coerce').fillna(0)
        df_base.loc[mask_r, 'time_to_event'] = pd.to_numeric(df_base.loc[mask_r, 'V99RNTCNT'], errors='coerce').fillna(0)
        
        # Left Knee
        mask_l = (df_base['Knee_Side'] == 2)
        df_base.loc[mask_l, 'event'] = pd.to_numeric(df_base.loc[mask_l, 'V99ELKR'], errors='coerce').fillna(0)
        df_base.loc[mask_l, 'time_to_event'] = pd.to_numeric(df_base.loc[mask_l, 'V99LNTCNT'], errors='coerce').fillna(0)
        
        has_outcomes = True
    else:
        print("   ⚠️ Outcome columns (V99ERKR/V99ELKR) not found. Checking alternatives...")
        # Try finding any column with 'KR' (Knee Replacement)
        kr_cols = [c for c in df_base.columns if 'KR' in c]
        print(f"   Found potential KR columns: {kr_cols[:5]}")
    
    if not has_outcomes:
        print("   ⚠️ Could not map outcomes. Setting event=0 for all.")
    
    print(f"   ✅ Built Base Cohort from raw files. Shape: {df_base.shape}")
    return df_base

def build_mega_cohort():
    print("🚀 Starting Mega-Merge...")
    
    # 1. Load Base Cohort (3526 Knees)
    if os.path.exists(BASE_PARQUET):
        df_base = pd.read_parquet(BASE_PARQUET)
        print(f"Base Cohort Loaded: {df_base.shape}")
    else:
        df_base = build_base_cohort()
        if df_base is None:
            print("❌ Failed to build base cohort. Exiting.")
            return

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

    # 4. Merge Quantitative Cartilage (MRI)
    print("   Merging Quantitative Cartilage Data...")
    df_qcart = robust_read('kMRI_FNIH_QCart_Chondrometrics00.txt')
    
    if df_qcart is not None:
        if df_qcart['side'].dtype == 'O':
            df_qcart['side'] = df_qcart['side'].map({'1: Right': 1, '2: Left': 2})
            
        df_qcart = df_qcart.rename(columns={
            'side': 'Knee_Side',
            'V00WMTMTH': 'Medial_Tibial_Thickness',
            'V00WLTMTH': 'Lateral_Tibial_Thickness'
        })
        
        qcart_cols = ['ID', 'Knee_Side', 'Medial_Tibial_Thickness', 'Lateral_Tibial_Thickness']
        existing_qcart = [c for c in qcart_cols if c in df_qcart.columns]
        
        df_base = pd.merge(df_base, df_qcart[existing_qcart], on=['ID', 'Knee_Side'], how='left')
        
        # Impute
        for col in ['Medial_Tibial_Thickness', 'Lateral_Tibial_Thickness']:
            if col in df_base.columns:
                df_base[col] = df_base[col].fillna(df_base[col].median())
        
        print(f"   ✅ Added Cartilage Thickness. New Shape: {df_base.shape}")

    # 5. Merge Subject Characteristics (Demographics)
    print("   Merging Subject Characteristics...")
    df_char = robust_read('SubjectChar00.txt')
    
    if df_char is not None:
        df_char = df_char.rename(columns={
            'V00EDCV': 'Education',
            'V00INCOME': 'Income'
        })
        
        char_cols = ['ID', 'Education', 'Income']
        existing_char = [c for c in char_cols if c in df_char.columns]
        
        # Merge (Patient Level)
        df_base = pd.merge(df_base, df_char[existing_char], on='ID', how='left')
        
        # Impute
        if 'Education' in df_base.columns: df_base['Education'] = df_base['Education'].fillna('Unknown')
        if 'Income' in df_base.columns: df_base['Income'] = df_base['Income'].fillna('Unknown')
        
        print(f"   ✅ Added Demographics. New Shape: {df_base.shape}")

    # 6. Merge Biomarkers (FNIH Sub-cohort)
    print("   Merging Biomarker Data (FNIH)...")
    # Biospec_FNIH_Labcorp00.txt contains the specific markers needed
    df_bio = robust_read('Biospec_FNIH_Labcorp00.txt')
    
    if df_bio is not None:
        # Rename columns to match App expectations
        # V00Serum_Comp_lc -> Bio_COMP
        # V00Urine_CTXII_lc -> Bio_CTXI
        # V00Serum_HA_lc -> Bio_HA
        # V00Serum_C2C_lc -> Bio_C2C
        # V00Serum_CPII_lc -> Bio_CPII
        
        bio_map = {
            'V00Serum_Comp_lc': 'Bio_COMP',
            'V00Urine_CTXII_lc': 'Bio_CTXI',
            'V00Serum_HA_lc': 'Bio_HA',
            'V00Serum_C2C_lc': 'Bio_C2C',
            'V00Serum_CPII_lc': 'Bio_CPII'
        }
        
        df_bio = df_bio.rename(columns=bio_map)
        
        # Keep only ID and the mapped columns
        bio_cols = ['ID'] + list(bio_map.values())
        existing_bio = [c for c in bio_cols if c in df_bio.columns]
        
        # DROP EXISTING BIOMARKER COLUMNS FROM BASE IF PRESENT
        # This prevents _x/_y suffixes and ensures we use the fresh data
        cols_to_drop = [c for c in list(bio_map.values()) if c in df_base.columns]
        if cols_to_drop:
            print(f"   ⚠️ Dropping existing empty/old columns: {cols_to_drop}")
            df_base = df_base.drop(columns=cols_to_drop)
        
        # Merge (Left Join for Mega Cohort)
        df_mega = pd.merge(df_base, df_bio[existing_bio], on='ID', how='left')
        
        # Create Precision Cohort (Inner Join - Only those with biomarkers)
        # We filter for rows where at least one key biomarker is present
        # Note: Using 'Bio_COMP' as a proxy for "has biomarker data"
        if 'Bio_COMP' in df_mega.columns:
            df_fnih = df_mega.dropna(subset=['Bio_COMP']).copy()
        else:
            df_fnih = pd.DataFrame() # Empty if merge failed
            
        print(f"   ✅ Added Biomarkers. Mega Shape: {df_mega.shape}, FNIH Shape: {df_fnih.shape}")
    else:
        df_mega = df_base
        df_fnih = pd.DataFrame()
        print("   ⚠️ Biomarker file not found. Skipping.")

    # 7. Save Outputs
    # A. Mega Cohort (Full)
    df_mega.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\n🎉 Mega-Cohort Saved: {OUTPUT_PARQUET} (Rows: {len(df_mega)})")
    
    # B. Precision Cohort (FNIH)
    fnih_path = OUTPUT_PARQUET.replace('mega_cohort', 'FNIH_biomarker_cohort')
    if not df_fnih.empty:
        df_fnih.to_parquet(fnih_path, index=False)
        print(f"🎉 Precision Cohort Saved: {fnih_path} (Rows: {len(df_fnih)})")
    else:
        print("⚠️ Precision Cohort empty (no biomarker overlap).")

    print(f"Columns: {df_mega.columns.tolist()}")

if __name__ == "__main__":
    build_mega_cohort()