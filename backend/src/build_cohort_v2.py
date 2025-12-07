"""
OAI Comprehensive Cohort Builder (v2)

Builds the complete tri-modal dataset by:
1. Creating knee-level cohort from X-ray baseline data
2. Merging patient-level demographics and clinical features
3. Adding knee-level imaging features (X-ray, MRI)
4. Adding patient-level biomarkers (FNIH sub-cohort)
5. Computing survival targets (event, time)

Based on comprehensive analysis of 157 OAI data files.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Data directory
DATA_DIR = Path(__file__).parent.parent / 'data' / 'OAICompleteData_ASCII'
OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'processed'


def clean_oai_value(val, to_numeric=True):
    """
    Clean OAI encoded values like '1: Male' -> 1 or 'Male'.
    
    Args:
        val: Value to clean
        to_numeric: If True, extract number. If False, keep string after colon.
    """
    if pd.isna(val):
        return np.nan
    
    val_str = str(val).strip()
    
    # Check for missing indicators
    missing_strings = ['.: Missing Form/Incomplete Workbook', '.: Missing', '.', 'nan', 'NaN', '']
    if val_str in missing_strings:
        return np.nan
    
    # Handle OAI format "X: Label"
    if ':' in val_str:
        parts = val_str.split(':', 1)
        if to_numeric:
            try:
                return float(parts[0].strip())
            except ValueError:
                return np.nan
        else:
            return parts[1].strip() if len(parts) > 1 else val_str
    
    # Try direct numeric conversion
    if to_numeric:
        try:
            return float(val_str)
        except ValueError:
            return np.nan
    
    return val_str


def robust_read(filename: str, encoding: str = 'utf-8') -> Optional[pd.DataFrame]:
    """
    Robustly read an OAI data file with fallback encodings.
    """
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"  ⚠️ File not found: {filename}")
        return None
    
    for enc in [encoding, 'latin-1', 'cp1252']:
        try:
            # Detect separator
            with open(filepath, 'r', encoding=enc, errors='ignore') as f:
                first_line = f.readline()
                sep = '|' if '|' in first_line else ','
            
            df = pd.read_csv(filepath, sep=sep, on_bad_lines='skip', 
                           encoding=enc, low_memory=False)
            return df
        except Exception as e:
            continue
    
    print(f"  ❌ Failed to read: {filename}")
    return None


def build_base_cohort() -> pd.DataFrame:
    """
    Step 1: Build the base knee-level cohort from baseline X-ray data.
    
    Returns:
        DataFrame with one row per knee (ID, Knee_Side, KL_Grade)
    """
    print("📋 Step 1: Building base knee-level cohort...")
    
    # Load X-ray baseline
    df_xray = robust_read('KXR_SQ_BU00.txt')
    if df_xray is None:
        raise FileNotFoundError("KXR_SQ_BU00.txt is required for cohort building")
    
    # Filter for main reading project (READPRJ == 15)
    if 'READPRJ' in df_xray.columns:
        df_xray = df_xray[df_xray['READPRJ'] == 15].copy()
    
    # Clean SIDE
    if 'SIDE' in df_xray.columns:
        df_xray['Knee_Side'] = df_xray['SIDE'].apply(lambda x: clean_oai_value(x, to_numeric=True))
    
    # Clean KL Grade
    if 'V00XRKL' in df_xray.columns:
        df_xray['KL_Grade'] = df_xray['V00XRKL'].apply(lambda x: clean_oai_value(x, to_numeric=True))
    else:
        raise ValueError("V00XRKL column not found in X-ray data")
    
    # Keep essential columns and drop duplicates
    df_base = df_xray[['ID', 'Knee_Side', 'KL_Grade']].dropna(subset=['ID', 'Knee_Side', 'KL_Grade'])
    df_base = df_base.drop_duplicates(subset=['ID', 'Knee_Side'])
    
    print(f"  ✓ Created base cohort: {len(df_base)} knees, {df_base['ID'].nunique()} patients")
    print(f"  ✓ KL Grade distribution: {df_base['KL_Grade'].value_counts().to_dict()}")
    
    return df_base


def merge_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Merge patient-level demographics from SubjectChar00.txt and AllClinical00.txt.
    """
    print("\n👤 Step 2: Merging demographics...")
    
    # Load SubjectChar00
    df_char = robust_read('SubjectChar00.txt')
    if df_char is not None:
        # Extract key demographics
        demo_cols = ['ID', 'V00AGE', 'P02SEX', 'P01BMI', 'V00EDCV', 'V00INCOME', 'V00PASE']
        existing_cols = [c for c in demo_cols if c in df_char.columns]
        
        df_demo = df_char[existing_cols].copy()
        
        # Clean values
        for col in existing_cols:
            if col != 'ID':
                df_demo[col] = df_demo[col].apply(lambda x: clean_oai_value(x, to_numeric=True))
        
        # Merge (patient-level)
        df = pd.merge(df, df_demo, on='ID', how='left')
        print(f"  ✓ Added demographics from SubjectChar00.txt")
    
    # Load Enrollees for additional info
    df_enroll = robust_read('Enrollees.txt')
    if df_enroll is not None and 'P01RACE' in df_enroll.columns:
        df_enroll['Race'] = df_enroll['P01RACE'].apply(lambda x: clean_oai_value(x, to_numeric=True))
        df = pd.merge(df, df_enroll[['ID', 'Race']], on='ID', how='left')
    
    print(f"  ✓ Current shape: {df.shape}")
    return df


def merge_clinical_symptoms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: Merge clinical symptom scores (WOMAC, KOOS).
    Handles knee-side mapping for knee-level features.
    """
    print("\n🩺 Step 3: Merging clinical symptoms...")
    
    df_clin = robust_read('AllClinical00.txt')
    if df_clin is None:
        return df
    
    # ---- Patient-level features ----
    patient_cols = ['ID', 'V00KOOSQOL', 'V00NSAIDRX']
    existing_patient = [c for c in patient_cols if c in df_clin.columns]
    
    if len(existing_patient) > 1:
        df_patient = df_clin[existing_patient].copy()
        for col in existing_patient:
            if col != 'ID':
                df_patient[col] = df_patient[col].apply(lambda x: clean_oai_value(x, to_numeric=True))
        df = pd.merge(df, df_patient, on='ID', how='left')
        print(f"  ✓ Added patient-level features: {existing_patient[1:]}")
    
    # ---- Knee-level features (require side mapping) ----
    # Create lookup dictionaries for side-specific features
    knee_features = {
        # (Right column, Left column, Output name)
        ('V00WOMTSR', 'V00WOMTSL', 'WOMAC_Score'),
        ('V00WOMPNR', 'V00WOMPNL', 'WOMAC_Pain'),
        ('V00WOMSTFR', 'V00WOMSTFL', 'WOMAC_Stiffness'),
        ('V00WOMFNR', 'V00WOMFNL', 'WOMAC_Function'),
        ('V00KOOSYMR', 'V00KOOSYML', 'KOOS_Symptoms'),
    }
    
    for right_col, left_col, output_name in knee_features:
        if right_col in df_clin.columns and left_col in df_clin.columns:
            # Create lookup dicts
            right_dict = dict(zip(df_clin['ID'], 
                                df_clin[right_col].apply(lambda x: clean_oai_value(x, to_numeric=True))))
            left_dict = dict(zip(df_clin['ID'],
                               df_clin[left_col].apply(lambda x: clean_oai_value(x, to_numeric=True))))
            
            # Map based on knee side (1=Right, 2=Left)
            df[output_name] = np.where(
                df['Knee_Side'] == 1,
                df['ID'].map(right_dict),
                df['ID'].map(left_dict)
            )
    
    print(f"  ✓ Added knee-level symptoms")
    print(f"  ✓ Current shape: {df.shape}")
    return df


def merge_imaging_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4: Merge MRI features from FNIH sub-cohort.
    """
    print("\n🔬 Step 4: Merging imaging features (MRI)...")
    
    # ---- MRI MOAKS (Bone Marrow Lesions, etc.) ----
    df_moaks = robust_read('kMRI_FNIH_SQ_MOAKS_BICL00.txt')
    if df_moaks is not None:
        # Clean SIDE
        if 'SIDE' in df_moaks.columns:
            df_moaks['Knee_Side'] = df_moaks['SIDE'].apply(lambda x: clean_oai_value(x, to_numeric=True))
        
        # Select key MRI features
        mri_cols = ['ID', 'Knee_Side']
        potential_mri = ['V00MACLBML', 'V00MACLCYS', 'V00MACLCAR', 'V00MACLMEN', 
                        'V00MACLOSP', 'V00MACLEFF']
        
        for col in potential_mri:
            if col in df_moaks.columns:
                df_moaks[col] = df_moaks[col].apply(lambda x: clean_oai_value(x, to_numeric=True))
                mri_cols.append(col)
        
        if len(mri_cols) > 2:
            df = pd.merge(df, df_moaks[mri_cols], on=['ID', 'Knee_Side'], how='left')
            print(f"  ✓ Added MOAKS features: {mri_cols[2:]}")
    
    # ---- Cartilage Thickness ----
    df_qcart = robust_read('kMRI_FNIH_QCart_Chondrometrics00.txt')
    if df_qcart is not None:
        # Clean side
        side_col = 'SIDE' if 'SIDE' in df_qcart.columns else 'side'
        if side_col in df_qcart.columns:
            df_qcart['Knee_Side'] = df_qcart[side_col].apply(lambda x: clean_oai_value(x, to_numeric=True))
        
        # Select cartilage features
        cart_cols = ['ID', 'Knee_Side']
        potential_cart = ['V00WMTMTH', 'V00WLTMTH', 'V00WMFMTH', 'V00WLFMTH']
        
        for col in potential_cart:
            if col in df_qcart.columns:
                df_qcart[col] = df_qcart[col].apply(lambda x: clean_oai_value(x, to_numeric=True))
                cart_cols.append(col)
        
        if len(cart_cols) > 2:
            df = pd.merge(df, df_qcart[cart_cols], on=['ID', 'Knee_Side'], how='left')
            print(f"  ✓ Added cartilage features: {cart_cols[2:]}")
    
    print(f"  ✓ Current shape: {df.shape}")
    return df


def merge_biomarkers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5: Merge comprehensive biomarkers from FNIH Labcorp data.
    Expands from 5 to 15 biomarkers.
    """
    print("\n🧬 Step 5: Merging biomarkers (FNIH sub-cohort)...")
    
    df_bio = robust_read('Biospec_FNIH_Labcorp00.txt')
    if df_bio is None:
        print("  ⚠️ Biomarker file not found")
        return df
    
    # Define all relevant biomarkers
    biomarker_map = {
        # Serum - Cartilage
        'V00Serum_C1_2C_lc': 'Bio_C1_2C',
        'V00Serum_C2C_lc': 'Bio_C2C',
        'V00Serum_CPII_lc': 'Bio_CPII',
        'V00Serum_Comp_lc': 'Bio_COMP',
        'V00Serum_CS846_lc': 'Bio_CS846',
        'V00Serum_COLL2_1_NO2_lc': 'Bio_COLL2_1_NO2',
        # Serum - Bone
        'V00Serum_CTXI_lc': 'Bio_CTXI',
        'V00Serum_NTXI_lc': 'Bio_NTXI',
        'V00Serum_PIIANP_lc': 'Bio_PIIANP',
        # Serum - Inflammation
        'V00Serum_HA_lc': 'Bio_HA',
        'V00Serum_MMP_3_lc': 'Bio_MMP3',
        # Urine
        'V00Urine_CTXII_lc': 'Bio_uCTXII',
        'V00Urine_C1_2C_lc': 'Bio_uC1_2C',
        'V00Urine_C2C_lc': 'Bio_uC2C',
        'V00Urine_NTXI_lc': 'Bio_uNTXI',
    }
    
    # Select existing columns
    bio_cols = ['ID']
    for src, dst in biomarker_map.items():
        if src in df_bio.columns:
            # Handle "<" values (below detection limit)
            df_bio[dst] = df_bio[src].apply(lambda x: 
                np.nan if pd.isna(x) or str(x).startswith('<') 
                else clean_oai_value(x, to_numeric=True))
            bio_cols.append(dst)
    
    print(f"  ✓ Found {len(bio_cols) - 1} biomarker columns in FNIH data")
    
    # Merge (patient-level)
    df = pd.merge(df, df_bio[bio_cols], on='ID', how='left')
    
    # Report biomarker coverage
    bio_feature_cols = [c for c in df.columns if c.startswith('Bio_')]
    bio_coverage = df[bio_feature_cols[0]].notna().sum() if bio_feature_cols else 0
    print(f"  ✓ Biomarker coverage: {bio_coverage} knees ({bio_coverage/len(df)*100:.1f}%)")
    
    print(f"  ✓ Current shape: {df.shape}")
    return df


def merge_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 6: Merge survival outcomes (TKR event and time).
    """
    print("\n🎯 Step 6: Merging outcomes (TKR)...")
    
    df_out = robust_read('OUTCOMES99.txt')
    if df_out is None:
        raise FileNotFoundError("OUTCOMES99.txt is required for survival analysis")
    
    # Rename id if needed
    if 'id' in df_out.columns and 'ID' not in df_out.columns:
        df_out = df_out.rename(columns={'id': 'ID'})
    
    # Load baseline dates from Clinical data
    df_clin = robust_read('AllClinical00.txt')
    baseline_date_col = 'V00EVDATE'
    
    # TKR dates
    right_tkr_col = 'V99ERKDATE'
    left_tkr_col = 'V99ELKDATE'
    death_date_col = 'V99EDDDATE'
    last_contact_col = 'V99RNTCNT'
    
    # Clean and parse dates
    missing_strings = ['.: Missing Form/Incomplete Workbook', '.: Missing', '.']
    
    def parse_date(val, formats=['%m/%d/%Y', '%m/%d/%y']):
        if pd.isna(val) or str(val) in missing_strings:
            return pd.NaT
        for fmt in formats:
            try:
                return pd.to_datetime(val, format=fmt)
            except:
                continue
        return pd.NaT
    
    # Get baseline date from clinical data
    if df_clin is not None and baseline_date_col in df_clin.columns:
        df_clin['Baseline_Date'] = df_clin[baseline_date_col].apply(parse_date)
        baseline_map = dict(zip(df_clin['ID'], df_clin['Baseline_Date']))
        df['Baseline_Date'] = df['ID'].map(baseline_map)
    
    # Get TKR dates
    if right_tkr_col in df_out.columns:
        df_out['TKR_Date_R'] = df_out[right_tkr_col].apply(parse_date)
    if left_tkr_col in df_out.columns:
        df_out['TKR_Date_L'] = df_out[left_tkr_col].apply(parse_date)
    if death_date_col in df_out.columns:
        df_out['Death_Date'] = df_out[death_date_col].apply(parse_date)
    
    # Get last contact visit
    if last_contact_col in df_out.columns:
        df_out['Last_Visit_Num'] = df_out[last_contact_col].apply(
            lambda x: clean_oai_value(x, to_numeric=True))
    
    # Merge outcomes
    out_cols = ['ID', 'TKR_Date_R', 'TKR_Date_L', 'Death_Date', 'Last_Visit_Num']
    existing_out = [c for c in out_cols if c in df_out.columns]
    df = pd.merge(df, df_out[existing_out], on='ID', how='left')
    
    # ---- Compute survival variables ----
    # TKR date based on knee side
    df['TKR_Date'] = np.where(
        df['Knee_Side'] == 1,
        df['TKR_Date_R'],
        df['TKR_Date_L']
    )
    
    # Event indicator
    df['event'] = df['TKR_Date'].notna().astype(int)
    
    # End date = TKR date if event, else last known contact (approximate)
    # For censored cases, estimate from last visit number (rough approximation)
    # Visit map: 0->0, 1->12mo, 2->18mo, 3->24mo, ..., 14->192mo
    visit_to_days = {
        0: 0, 1: 365, 2: 548, 3: 730, 4: 913, 5: 1095,
        6: 1460, 7: 1643, 8: 1825, 9: 2008, 10: 2190,
        11: 2373, 12: 2555, 13: 2920, 14: 3285
    }
    
    df['Last_Contact_Days'] = df['Last_Visit_Num'].map(visit_to_days).fillna(0)
    
    # Time to event
    df['time_to_event'] = np.where(
        df['event'] == 1,
        (df['TKR_Date'] - df['Baseline_Date']).dt.days,
        df['Last_Contact_Days']
    )
    
    # Clean up invalid times
    df = df[df['time_to_event'] > 0]
    
    # Drop intermediate columns
    drop_cols = ['TKR_Date_R', 'TKR_Date_L', 'TKR_Date', 'Baseline_Date', 
                 'Death_Date', 'Last_Visit_Num', 'Last_Contact_Days']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    print(f"  ✓ Events (TKR): {df['event'].sum()} ({df['event'].mean()*100:.1f}%)")
    print(f"  ✓ Time range: {df['time_to_event'].min():.0f} - {df['time_to_event'].max():.0f} days")
    print(f"  ✓ Final shape: {df.shape}")
    
    return df


def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 7: Impute missing values.
    """
    print("\n🔧 Step 7: Imputing missing values...")
    
    # Identify numeric columns
    exclude_cols = ['ID', 'Knee_Side', 'event', 'time_to_event']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    # Report missing
    missing_pct = df[numeric_cols].isna().mean() * 100
    high_missing = missing_pct[missing_pct > 50]
    if len(high_missing) > 0:
        print(f"  ⚠️ High missing (>50%): {list(high_missing.index)}")
    
    # Impute with median
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    # Z-score normalize biomarkers
    bio_cols = [c for c in df.columns if c.startswith('Bio_')]
    for col in bio_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 0:
            df[col] = (df[col] - mean_val) / std_val
    
    print(f"  ✓ Imputed {len(numeric_cols)} numeric columns")
    print(f"  ✓ Normalized {len(bio_cols)} biomarker columns")
    
    return df


def build_mega_cohort(save_outputs: bool = True) -> pd.DataFrame:
    """
    Main function to build the comprehensive tri-modal cohort.
    
    Returns:
        DataFrame with all features merged and survival outcomes computed.
    """
    print("=" * 60)
    print("🚀 OAI Comprehensive Cohort Builder v2")
    print("=" * 60)
    
    # Build cohort step by step
    df = build_base_cohort()
    df = merge_demographics(df)
    df = merge_clinical_symptoms(df)
    df = merge_imaging_features(df)
    df = merge_biomarkers(df)
    df = merge_outcomes(df)
    df = impute_features(df)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 COHORT SUMMARY")
    print("=" * 60)
    print(f"Total knees: {len(df)}")
    print(f"Total patients: {df['ID'].nunique()}")
    print(f"Total features: {len(df.columns)}")
    print(f"TKR events: {df['event'].sum()} ({df['event'].mean()*100:.1f}%)")
    
    # Feature breakdown
    clinical_cols = [c for c in df.columns if 'WOMAC' in c or 'KOOS' in c or c in ['V00AGE', 'P02SEX', 'P01BMI', 'V00PASE']]
    imaging_cols = [c for c in df.columns if 'KL_Grade' in c or 'V00X' in c or 'V00M' in c or 'V00W' in c]
    biomarker_cols = [c for c in df.columns if c.startswith('Bio_')]
    
    print(f"\nFeature breakdown:")
    print(f"  Clinical: {len(clinical_cols)}")
    print(f"  Imaging: {len(imaging_cols)}")
    print(f"  Biomarkers: {len(biomarker_cols)}")
    
    # Save outputs
    if save_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Full cohort
        output_path = OUTPUT_DIR / 'OAI_mega_cohort_v2.parquet'
        df.to_parquet(output_path, index=False)
        print(f"\n✅ Saved: {output_path}")
        
        # Also save imputed version for training
        imputed_path = OUTPUT_DIR / 'OAI_mega_cohort_imputed.parquet'
        df.to_parquet(imputed_path, index=False)
        print(f"✅ Saved: {imputed_path}")
        
        # Biomarker sub-cohort (patients with biomarker data)
        if biomarker_cols:
            df_bio = df[df[biomarker_cols[0]].notna()].copy()
            bio_path = OUTPUT_DIR / 'OAI_biomarker_subcohort.parquet'
            df_bio.to_parquet(bio_path, index=False)
            print(f"✅ Saved biomarker sub-cohort: {bio_path} ({len(df_bio)} knees)")
    
    print("\n" + "=" * 60)
    print("✅ Cohort building complete!")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    df = build_mega_cohort()
    print(f"\nColumns: {df.columns.tolist()}")
