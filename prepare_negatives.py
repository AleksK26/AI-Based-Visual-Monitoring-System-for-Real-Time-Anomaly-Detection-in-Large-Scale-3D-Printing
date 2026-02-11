import os
import glob

# --- CONFIGURATION ---
# The folder where you put your downloaded perfect prints
SOURCE_FOLDER = r"C:\Users\hp pc\Desktop\3-D Prining Anomalies\data\processed\newperfprint"
PREFIX = "perfect_complex" # The new name for the files

def process_negatives():
    # 1. Check if folder exists
    if not os.path.exists(SOURCE_FOLDER):
        print(f"Error: Folder not found: {SOURCE_FOLDER}")
        return

    # 2. Get all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(SOURCE_FOLDER, ext)))
    
    if not image_files:
        print("No images found! Make sure they are jpg, png, or jpeg.")
        return

    print(f"Found {len(image_files)} images. Processing...")

    # 3. Rename and Create Labels
    for index, old_path in enumerate(image_files):
        # Get the file extension (e.g., .jpg)
        ext = os.path.splitext(old_path)[1]
        
        # Define new names
        new_filename = f"{PREFIX}_{index}{ext}"
        new_image_path = os.path.join(SOURCE_FOLDER, new_filename)
        new_txt_path = os.path.join(SOURCE_FOLDER, f"{PREFIX}_{index}.txt")

        # Rename the image
        try:
            os.rename(old_path, new_image_path)
        except OSError as e:
            print(f"Error renaming {old_path}: {e}")
            continue

        # Create the EMPTY text file (0 bytes)
        # This tells YOLO: "There are NO defects in this image"
        with open(new_txt_path, 'w') as f:
            pass # Write nothing

    print("Success! Images renamed and empty labels created.")
    print(f"Go check: {SOURCE_FOLDER}")

if __name__ == "__main__":
    process_negatives();