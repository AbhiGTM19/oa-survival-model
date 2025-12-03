import pandas as pd
import os

# --- CONFIGURATION ---
DATA_DIR = '../data/OAICompleteData_ASCII'
TARGET_FILES = [
    'AllClinical00.txt',            # The "Master File"
    'kMRI_POMA_TKR_MOAKS_BICL.txt'  # The "Gold Standard" MRI Data
]

def inspect_file(filename):
    fpath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(fpath):
        print(f"❌ File not found: {filename}")
        return

    print(f"\n{'='*60}")
    print(f"📂 INSPECTING: {filename}")
    print(f"{'='*60}")

    try:
        # Read header only first to detect separator
        with open(fpath, 'r', encoding='latin-1') as f:
            header = f.readline()
            sep = '|'
            if ',' in header and '|' not in header: sep = ','
        
        # Load Data
        df = pd.read_csv(fpath, sep=sep, on_bad_lines='skip', low_memory=False, encoding='latin-1')
        print(f"Shape: {df.shape} (Rows, Cols)")
        
        # Standardize Column Names (Uppercase)
        df.columns = [c.upper() for c in df.columns]
        
        # 1. Check Identifiers
        ids = [c for c in df.columns if 'ID' in c or 'SIDE' in c]
        print(f"🔑 Identifiers: {ids}")
        
        # 2. Check Visit Versions (V00 = Baseline, V01 = 12 Month, etc.)
        # We want to know if this file is LONGITUDINAL (multiple rows per patient) or WIDE.
        if 'VERSION' in df.columns:
            print(f"📅 Versions found: {df['VERSION'].unique()[:5]} ...")
            
        # 3. HUNT FOR FEATURES (The "Bit Extraction")
        # We look for specific categories of variables
        
        categories = {
            'BIOMARKERS': ['SERUM', 'URINE', 'COMP', 'NTX', 'CTX', 'HA'],
            'MRI_FEATURES': ['BML', 'CART', 'MEN', 'EFF', 'HOFFA', 'SYNOV', 'MOAKS'],
            'SYMPTOMS': ['WOM', 'KOOS', 'PAIN'],
            'MEDS/DIET': ['MED', 'VIT', 'DIET', 'SUPP'],
            'ACTIVITY': ['PASE', 'WALK', 'ACT']
        }
        
        for cat, keywords in categories.items():
            found_cols = []
            for col in df.columns:
                if any(k in col for k in keywords):
                    found_cols.append(col)
            
            if found_cols:
                print(f"\n   🔎 {cat} Columns ({len(found_cols)} found):")
                # Print first 10 examples
                print(f"      {', '.join(found_cols[:10])} ...")
                
                # Check data availability (is it empty?)
                # We check the first valid column found
                test_col = found_cols[0]
                non_null = df[test_col].notna().sum()
                print(f"      DATA HEALTH: Column '{test_col}' has {non_null} non-null values ({(non_null/len(df))*100:.1f}%)")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    for f in TARGET_FILES:
        inspect_file(f)