import pandas as pd
import os

# --- CONFIGURATION ---
DATA_DIR = 'data/OAICompleteData_ASCII'
TARGET_FILE = 'AllClinical00.txt'

def find_columns():
    fpath = os.path.join(DATA_DIR, TARGET_FILE)
    if not os.path.exists(fpath):
        print(f"❌ File not found: {fpath}")
        return

    try:
        # Read header
        with open(fpath, 'r', encoding='latin-1') as f:
            header = f.readline().strip()
            sep = ',' if ',' in header and '|' not in header else '|'
        
        # Load just the columns
        df = pd.read_csv(fpath, sep=sep, on_bad_lines='skip', encoding='latin-1', nrows=0)
        cols = [c.upper() for c in df.columns]
        
        print(f"📂 Scanning {TARGET_FILE} for variables...")
        
        # 1. Find KOOS (Knee Injury and Osteoarthritis Outcome Score)
        koos = [c for c in cols if 'KOOS' in c and 'V00' in c]
        print(f"\nfound {len(koos)} KOOS columns. Examples:")
        print(f"   {', '.join(koos[:10])}")
        
        # 2. Find WOMAC (Western Ontario and McMaster Universities Arthritis Index)
        womac = [c for c in cols if 'WOM' in c and 'V00' in c]
        print(f"\nfound {len(womac)} WOMAC columns. Examples:")
        print(f"   {', '.join(womac[:10])}")
        
        # 3. Find PASE (Physical Activity Scale for the Elderly)
        pase = [c for c in cols if 'PASE' in c and 'V00' in c]
        print(f"\nfound {len(pase)} PASE columns. Examples:")
        print(f"   {', '.join(pase[:10])}")
        
        # 4. Find Meds (NSAIDs)
        meds = [c for c in cols if ('MED' in c or 'RX' in c) and 'V00' in c]
        print(f"\nfound {len(meds)} Meds columns. Examples:")
        print(f"   {', '.join(meds[:10])}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_columns()