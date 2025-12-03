import os
import pandas as pd
import glob
import warnings
from tqdm import tqdm

# Suppress DtypeWarnings (we just want to scan headers/rows)
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# Try to find the data folder relative to where the script is run
POSSIBLE_PATHS = [
    '../data/OAICompleteData_ASCII',
    'data/OAICompleteData_ASCII',
    './OAICompleteData_ASCII',
    '/Users/abhi/Downloads/M.tech/Sem 3/Major Project/oa-survival-model/data/OAICompleteData_ASCII'
]

# 1. EXPANDED KEYWORDS (The "Deep" Signal)
KEYWORDS = {
    'OUTCOME': ['TRK', 'Surgery', 'Replace', 'Arthroplasty', 'Revision', 'Fail', 'Progression', 'Time'],
    'SYMPTOMS': ['Pain', 'Ache', 'Stiff', 'Swell', 'WOMAC', 'KOOS', 'SF-12', 'PASE', 'Symptoms'],
    'XRAY': ['XR', 'X-ray', 'Radiograph', 'KL', 'Kellgren', 'JSN', 'Joint Space', 'Osteophyte', 'Sclerosis', 'Alignment', 'Angle'],
    'MRI_SEMILQ': ['MOAKS', 'WORMS', 'BLOKS', 'BML', 'Meniscus', 'Tear', 'Extrusion', 'Effusion', 'Synovitis', 'Hoffa'],
    'MRI_QUANT': ['Volume', 'Thickness', 'Thick', 'Mean_Thick', 'Cartilage', 'QCart', 'Eckstein', 'Chondrometrics'],
    'BIOMARKERS': ['Serum', 'Urine', 'Blood', 'Assay', 'Bio', 'Comp', 'NTX', 'CTX', 'HA', 'MMP', 'C2C', 'PIIANP'],
    'LIFESTYLE': ['Diet', 'Nutrition', 'Vitamin', 'Supplement', 'Activity', 'Exercise', 'Walk', 'Steps', 'Strength', 'Chair', 'Grip'],
    'CLINICAL': ['Medication', 'Analgesic', 'NSAID', 'Injection', 'Steroid', 'Hyaluronic', 'Comorbidity', 'History', 'BMI', 'Weight']
}

# Robust Encodings
ENCODINGS_TO_TRY = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'mac_roman']

def find_data_dir():
    for p in POSSIBLE_PATHS:
        if os.path.exists(p):
            return p
    return None

def try_read_csv(filepath, separator):
    """Tries to read a CSV with multiple encodings to avoid crashes."""
    for enc in ENCODINGS_TO_TRY:
        try:
            # Read header to check cols
            df_head = pd.read_csv(filepath, sep=separator, nrows=5, on_bad_lines='skip', encoding=enc)
            # Read full to get row count (skip low_memory=False to save RAM if huge)
            row_count = sum(1 for _ in open(filepath, encoding=enc)) - 1
            return df_head, row_count
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        except Exception:
            break
    return None, 0

def scan_oai_folder(root_path):
    # Recursive search for all .txt and .csv files
    all_files = glob.glob(os.path.join(root_path, "**", "*.txt"), recursive=True) + \
                glob.glob(os.path.join(root_path, "**", "*.csv"), recursive=True)
    
    print(f"Found {len(all_files)} files in {root_path}. Starting Deep Audit...")
    
    inventory = []
    
    for fpath in tqdm(all_files, desc="Mining Data"):
        fname = os.path.basename(fpath)
        
        try:
            # Detect separator (OAI is usually pipe '|', sometimes comma)
            sep = '|'
            with open(fpath, 'r', encoding='latin-1') as f:
                line = f.readline()
                if ',' in line and '|' not in line: sep = ','
                if '\t' in line: sep = '\t'

            # Robust Read
            df_head, row_count = try_read_csv(fpath, sep)
            
            if df_head is None:
                # print(f"❌ FAILED to decode: {fname}") # Reduce noise
                continue
                
            # 1. ID Check
            cols_upper = [c.upper() for c in df_head.columns]
            has_id = any(c in ['ID', 'PID', 'P01ID', 'SUBJECTID', 'SIDE', 'KNEE'] for c in cols_upper)
            
            # 2. Keyword Hunt
            col_str = " ".join(cols_upper)
            tags = []
            for category, kws in KEYWORDS.items():
                if any(kw.upper() in col_str for kw in kws):
                    tags.append(category)
            
            # 3. Visit Check
            has_baseline = any('V00' in c for c in cols_upper) or 'VERSION' in cols_upper

            # 4. Score (Prioritize Big Files with IDs and Keywords)
            score = (1000 if has_id else 0) + (500 if row_count > 3000 else 0) + (len(tags) * 100)

            inventory.append({
                'File Name': fname,
                'Rows': row_count,
                'Has ID': has_id,
                'Tags': ", ".join(tags),
                'Score': score,
                'Path': fpath
            })
            
        except Exception as e:
            pass # Skip unreadable files silently to keep output clean
            
    return pd.DataFrame(inventory)

if __name__ == "__main__":
    data_dir = find_data_dir()
    
    if not data_dir:
        print("❌ ERROR: Could not find OAI data directory. Check paths.")
    else:
        print(f"✅ Data Directory Found: {data_dir}")
        df_results = scan_oai_folder(data_dir)
        
        if not df_results.empty:
            # Sort by Score
            df_results = df_results.sort_values('Score', ascending=False)
            
            # Save
            df_results.to_csv('oai_comprehensive_inventory.csv', index=False)
            
            print(f"\n✅ Audit Complete. Inventory saved to oai_comprehensive_inventory.csv")
            print("\n--- TOP 20 HIGH-POTENTIAL FILES ---")
            print(df_results[['File Name', 'Rows', 'Tags']].head(20).to_string())
        else:
            print("No valid files found.")