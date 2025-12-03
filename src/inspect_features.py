import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

# Corrected Path
DATA_DIR = 'data/OAICompleteData_ASCII'

TARGET_FILES = [
    'AllClinical00.txt',
    'kMRI_QCart_Eckstein00.txt',
    'kXR_SQ_BU00.txt'
]

def get_valid_columns(df, keywords):
    valid_cols = []
    for col in df.columns:
        if any(k in col.upper() for k in keywords):
            if df[col].notna().sum() > 1000:
                valid_cols.append(col)
    return valid_cols

def inspect_file(filename):
    fpath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(fpath):
        print(f"⚠️ Skipped (Not Found): {filename}")
        return

    print(f"\n{'='*60}")
    print(f"📂 INSPECTING: {filename}")
    
    try:
        # Robust Read
        with open(fpath, 'r', encoding='latin-1') as f:
            line = f.readline()
            sep = ',' if ',' in line and '|' not in line else '|'
            
        df = pd.read_csv(fpath, sep=sep, on_bad_lines='skip', encoding='latin-1', low_memory=False)
        print(f"   Shape: {df.shape}")
        
        # HUNT FOR FEATURES
        pase_cols = get_valid_columns(df, ['PASE', 'ACT', 'WALK', 'SPORT'])
        if pase_cols: print(f"   🏃 ACTIVITY: {pase_cols[:5]} ...")

        nutri_cols = get_valid_columns(df, ['DIET', 'VIT', 'SUPP', 'MED', 'RX', 'NSAID'])
        if nutri_cols: print(f"   💊 MEDS/DIET: {nutri_cols[:5]} ...")

        bio_cols = get_valid_columns(df, ['SERUM', 'URINE', 'COMP', 'NTX', 'CTX', 'HA'])
        if bio_cols: print(f"   🧪 BIO: {bio_cols[:5]} ...")

        mri_cols = get_valid_columns(df, ['THICK', 'VOL', 'CART', 'FEM', 'TIB'])
        if mri_cols: print(f"   🧲 MRI: {mri_cols[:5]} ...")

    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    inspect_file('AllClinical00.txt')