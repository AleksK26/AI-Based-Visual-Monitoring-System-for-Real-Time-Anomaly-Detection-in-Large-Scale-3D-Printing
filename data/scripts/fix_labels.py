# Fixing the labels in processed/ labels for them to be without .jpg.txt and only .txt
import os
import glob

# POINT THIS TO YOUR LABELS FOLDER
LABEL_DIR = r"C:\Users\hp pc\Desktop\3-D Prining Anomalies\data\processed\labels\train"

def fix_extensions():
    # Find all files ending in .jpg.txt
    bad_files = glob.glob(os.path.join(LABEL_DIR, "*.jpg.txt"))
    
    print(f"Found {len(bad_files)} files with double extensions.")
    
    if len(bad_files) == 0:
        print("No fixes needed! Checking for other issues...")
        # Check if they are already fixed
        txt_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))
        if len(txt_files) > 0:
            print(f"Folder contains {len(txt_files)} correct .txt files.")
            print("If training still fails, check your YAML path.")
        return

    print("Renaming files...")
    for old_path in bad_files:
        # Create new path: replace .jpg.txt with .txt
        new_path = old_path.replace(".jpg.txt", ".txt")
        os.rename(old_path, new_path)
        
    print("Success! All labels are now named correctly (e.g., image.txt)")

if __name__ == "__main__":
    fix_extensions()