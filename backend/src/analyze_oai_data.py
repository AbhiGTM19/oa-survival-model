"""
OAI Dataset Analysis Script
Comprehensive analysis of all data files in OAICompleteData_ASCII
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json

DATA_DIR = Path(__file__).parent.parent / 'data' / 'OAICompleteData_ASCII'
OUTPUT_FILE = Path(__file__).parent.parent / 'data' / 'processed' / 'oai_data_inventory.json'

def analyze_file(filepath):
    """Analyze a single data file and return its metadata."""
    try:
        # Determine separator
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            sep = '|' if '|' in first_line else ','
        
        # Load entire file
        df = pd.read_csv(filepath, sep=sep, on_bad_lines='skip', low_memory=False)
        
        # Basic stats
        info = {
            'filename': filepath.name,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'has_ID': 'ID' in df.columns,
            'has_SIDE': 'SIDE' in df.columns or 'side' in df.columns,
            'unique_patients': df['ID'].nunique() if 'ID' in df.columns else None,
            'missing_pct': (df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100) if df.shape[0] > 0 else 0,
            'size_mb': filepath.stat().st_size / (1024 * 1024)
        }
        
        # Identify column types by prefix
        visit_cols = [c for c in df.columns if c.startswith('V') and len(c) > 2 and c[1:3].isdigit()]
        info['visit_columns'] = len(visit_cols)
        info['visits_detected'] = sorted(list(set([c[1:3] for c in visit_cols if c[1:3].isdigit()])))
        
        return info
        
    except Exception as e:
        return {
            'filename': filepath.name,
            'error': str(e),
            'size_mb': filepath.stat().st_size / (1024 * 1024) if filepath.exists() else 0
        }

def categorize_files():
    """Categorize all OAI data files by type."""
    categories = {
        'clinical': [],      # AllClinical*, SubjectChar*
        'outcomes': [],      # OUTCOMES*
        'xray': [],          # KXR_*, XRay*
        'mri': [],           # MRI*, kMRI*
        'biomarkers': [],    # Biomarkers*, Biospec*
        'accelerometry': [], # Accel*
        'other': []
    }
    
    all_files = list(DATA_DIR.glob('*.txt')) + list(DATA_DIR.glob('*.csv'))
    
    for f in all_files:
        name_lower = f.name.lower()
        if 'clinical' in name_lower or 'subjectchar' in name_lower or 'enrollee' in name_lower:
            categories['clinical'].append(f)
        elif 'outcome' in name_lower:
            categories['outcomes'].append(f)
        elif 'xray' in name_lower or 'kxr_' in name_lower or 'flxr' in name_lower or 'hxr' in name_lower:
            categories['xray'].append(f)
        elif 'mri' in name_lower or 'kmri' in name_lower:
            categories['mri'].append(f)
        elif 'biomarker' in name_lower or 'biospec' in name_lower or 'labcorp' in name_lower.replace(' ', ''):
            categories['biomarkers'].append(f)
        elif 'accel' in name_lower:
            categories['accelerometry'].append(f)
        else:
            categories['other'].append(f)
    
    return categories

def main():
    print("=" * 60)
    print("OAI Dataset Comprehensive Analysis")
    print("=" * 60)
    
    # Categorize files
    categories = categorize_files()
    
    # Summary
    print("\n📊 FILE CATEGORIES:")
    for cat, files in categories.items():
        print(f"  {cat.upper()}: {len(files)} files")
    
    # Analyze each category
    all_analysis = {}
    
    for category, files in categories.items():
        print(f"\n{'=' * 40}")
        print(f"Analyzing {category.upper()} files...")
        print(f"{'=' * 40}")
        
        category_analysis = []
        for f in sorted(files, key=lambda x: x.name):
            print(f"  Analyzing: {f.name}...", end=" ")
            info = analyze_file(f)
            category_analysis.append(info)
            if 'error' in info:
                print(f"ERROR: {info['error']}")
            else:
                print(f"✓ {info['rows']} rows, {info['columns']} cols")
        
        all_analysis[category] = category_analysis
    
    # Generate summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    total_files = 0
    total_rows = 0
    all_patients = set()
    
    for category, analyses in all_analysis.items():
        cat_rows = sum(a.get('rows', 0) for a in analyses if 'rows' in a)
        total_files += len(analyses)
        total_rows += cat_rows
        
        for a in analyses:
            if a.get('unique_patients'):
                # We can't really merge sets here, but we note the max
                pass
        
        print(f"\n{category.upper()}:")
        print(f"  Files: {len(analyses)}")
        print(f"  Total Rows: {cat_rows:,}")
        
        if category == 'biomarkers':
            print("\n  BIOMARKER DETAILS:")
            for a in analyses:
                if 'error' not in a:
                    cols = [c for c in a['column_names'] if 'serum' in c.lower() or 'urine' in c.lower()]
                    if cols:
                        print(f"    {a['filename']}: {len(cols)} biomarker columns, {a.get('unique_patients', 0)} patients")
    
    print(f"\n{'=' * 60}")
    print(f"TOTAL FILES: {total_files}")
    print(f"TOTAL ROWS: {total_rows:,}")
    print("=" * 60)
    
    # Save analysis
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_analysis, f, indent=2, default=str)
    
    print(f"\n✅ Full analysis saved to: {OUTPUT_FILE}")
    
    # Key recommendations
    print("\n" + "=" * 60)
    print("KEY DATA FILES FOR TRI-MODAL MODEL")
    print("=" * 60)
    
    print("""
RECOMMENDED PRIMARY FILES:
1. Enrollees.txt - Master patient list (ID key)
2. OUTCOMES99.txt - TKR outcomes (event/time)
3. AllClinical00.txt - Baseline clinical features
4. KXR_SQ_BU00.txt - Baseline KL Grade
5. Biospec_FNIH_Labcorp00.txt - Comprehensive biomarkers (~600 patients)
6. SubjectChar00.txt - Demographics (Age, Sex, BMI)
7. kMRI_FNIH_SQ_MOAKS_BICL00.txt - MRI features (BML, etc.)
8. kMRI_FNIH_QCart_Chondrometrics00.txt - Cartilage thickness

BIOMARKER COLUMNS TO PRIORITIZE:
- V00Serum_C1_2C_lc (Collagen degradation)
- V00Serum_C2C_lc (Collagen II cleavage)
- V00Serum_CPII_lc (Cartilage synthesis)
- V00Serum_CTXI_lc (Bone resorption)
- V00Serum_Comp_lc (COMP - cartilage breakdown)
- V00Serum_HA_lc (Hyaluronic acid - inflammation)
- V00Serum_MMP_3_lc (Matrix metalloproteinase)
- V00Serum_NTXI_lc (Bone turnover)
- V00Serum_PIIANP_lc (Type II procollagen)
- V00Urine_CTXII_lc (Urine cartilage marker)
    """)

if __name__ == "__main__":
    main()
