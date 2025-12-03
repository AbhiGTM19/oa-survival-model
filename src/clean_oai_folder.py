import os
import shutil
import glob

# --- CONFIGURATION ---
# Correct path relative to your project root (oa-survival-model)
DATA_DIR = 'data/OAICompleteData_ASCII'
DOCS_DIR = os.path.join(DATA_DIR, '_DOCS')

# Keywords that identify a "Helper File" (Not Data)
HELPER_KEYWORDS = ['Descrip', 'Comments', 'Contents', 'Stats', 'Formats', 'Legend']
EXTENSIONS_TO_MOVE = ['.pdf', '.doc', '.docx', '.ppt', '.pptx']

def clean_folder():
    # 1. Verify Directory Exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Data directory not found at: {os.path.abspath(DATA_DIR)}")
        print("   Make sure you are running this from the 'oa-survival-model' root folder.")
        return

    # 2. Create Docs Folder
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"📁 Created documentation folder: {DOCS_DIR}")

    # 3. Scan and Move
    all_files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    moved_count = 0
    
    print(f"🔍 Scanning {len(all_files)} files...")
    
    for filename in all_files:
        file_path = os.path.join(DATA_DIR, filename)
        
        # Logic: Move if PDF/Doc OR if filename implies documentation
        is_doc_ext = any(filename.lower().endswith(ext) for ext in EXTENSIONS_TO_MOVE)
        is_helper_text = any(kw in filename for kw in HELPER_KEYWORDS)
        
        if is_doc_ext or is_helper_text:
            shutil.move(file_path, os.path.join(DOCS_DIR, filename))
            moved_count += 1
            
    print(f"✅ Cleanup Complete!")
    print(f"📦 Moved {moved_count} helper files into '{DOCS_DIR}'")
    
    # Count remaining
    remaining = len([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))])
    print(f"✨ Remaining Data Files: {remaining}")

if __name__ == "__main__":
    clean_folder()