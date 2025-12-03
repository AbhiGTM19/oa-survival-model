import pandas as pd
import os
import glob
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_DIR = 'data/OAICompleteData_ASCII'
OUTPUT_REPORT = 'baseline_feature_report.csv'

# Feature Categories we want to extract
CATEGORIES = {
    'MRI_QUANT': ['THICK', 'VOL', 'CART', 'ECM'],   # Cartilage Thickness/Volume (High Value)
    'MRI_SEMI': ['MOAKS', 'WORMS', 'BML', 'MEN'],   # Structural Scores
    'XRAY': ['XR', 'JSN', 'OSTEO', 'ANGLE'],        # X-Ray alignment/scores
    'BIOMARKER': ['SERUM', 'URINE', 'COMP', 'CTX'], # Blood/Urine
    'NUTRITION': ['DIET', 'VIT', 'SUPP', 'CALC'],   # Intake
    'MEDS': ['MED', 'RX', 'NSAID', 'STEROID'],      # Medications
    'ACTIVITY': ['PASE', 'ACT', 'WALK', 'STR'],     # Physical Activity
    'SYMPTOMS': ['WOM', 'KOOS', 'PAIN']             # Patient Reported Outcomes
}

def get_encoding(fpath):
    """Helper to handle OAI's messy encodings"""
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            with open(fpath, 'r', encoding=enc) as f:
                f.readline()
            return enc
        except:
            continue
    return 'latin-1' # Fallback

def scan_for_baseline():
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data dir not found: {DATA_DIR}")
        return

    all_files = glob.glob(os.path.join(DATA_DIR, "*.txt")) + glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"🔍 Scanning {len(all_files)} files for Baseline Data...")

    report = []

    for fpath in tqdm(all_files):
        fname = os.path.basename(fpath)
        
        # 1. Filename Heuristic: Skip non-baseline files immediately to save time
        # We look for '00', 'BL', 'Base', or files like 'Enrollees'
        is_likely_baseline = any(x in fname.upper() for x in ['00', 'BL', 'BASE', 'ENROL', 'SUBJECT'])
        
        if not is_likely_baseline:
            continue

        try:
            # Read header to check columns
            enc = get_encoding(fpath)
            with open(fpath, 'r', encoding=enc) as f:
                header = f.readline().strip()
                sep = '|'
                if ',' in header and '|' not in header: sep = ','
            
            # Load Data (Low memory mode)
            df = pd.read_csv(fpath, sep=sep, on_bad_lines='skip', encoding=enc, low_memory=False)
            
            # Standardize columns
            df.columns = [c.upper() for c in df.columns]
            
            # 2. ID Check (Must be linkable)
            if not any(c in ['ID', 'PID', 'P01ID', 'SUBJECTID'] for c in df.columns):
                continue

            # 3. Column Scan
            file_hits = 0
            found_cats = []
            
            for cat, keywords in CATEGORIES.items():
                # Find columns matching keywords AND having 'V00' or no time prefix (static)
                relevant_cols = [
                    c for c in df.columns 
                    if any(k in c for k in keywords) 
                    and ('V00' in c or not any(f"V{i:02d}" in c for i in range(1, 10)))
                ]
                
                if relevant_cols:
                    # Verify data quality (are they just empty?)
                    non_null = df[relevant_cols[0]].notna().sum()
                    if non_null > 500: # Must have some data
                        file_hits += 1
                        found_cats.append(cat)
                        
                        # Add to report
                        report.append({
                            'File': fname,
                            'Category': cat,
                            'col_count': len(relevant_cols),
                            'example_col': relevant_cols[0],
                            'rows': len(df),
                            'path': fpath
                        })

        except Exception:
            pass # Skip broken files

    # Convert to DataFrame
    if report:
        df_rep = pd.DataFrame(report)
        df_rep = df_rep.sort_values(['Category', 'rows'], ascending=[True, False])
        
        print(f"\n✅ Found {len(df_rep['File'].unique())} useful Baseline files!")
        print(df_rep[['Category', 'File', 'rows', 'example_col']].to_string())
        
        df_rep.to_csv(OUTPUT_REPORT, index=False)
    else:
        print("No valid baseline feature files found.")

if __name__ == "__main__":
    scan_for_baseline()