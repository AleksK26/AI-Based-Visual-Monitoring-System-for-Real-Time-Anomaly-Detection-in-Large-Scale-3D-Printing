import cv2
import os
import glob

# --- CONFIGURATION ---
INPUT_FOLDER = "data\printer_bed"   # Put your original phone photos here
OUTPUT_FOLDER = "data/clean_printerbed"      # Where the clean, cropped JPGs go
TARGET_SIZE = (640, 640)                # Optional: Resize to YOLO standard

def crop_images():
    # Create folders
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Created folder '{INPUT_FOLDER}'. Please put your raw photos there and run again!")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Get all images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.HEIC']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    print(f"Found {len(files)} images. Instructions:")
    print("1. Draw a box around the printer bed using your mouse.")
    print("2. Press ENTER or SPACE to confirm the crop.")
    print("3. Press 'c' to cancel/skip an image.")
    print("-" * 30)

    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        print(f"Processing [{i+1}/{len(files)}]: {filename}")

        # Load image
        img = cv2.imread(filepath)
        
        if img is None:
            print(f"Error reading {filename}. Skipping.")
            continue

        # Resize huge images (like 4000px from phone) so they fit on screen
        # This is just for display; we will crop the original resolution if needed
        # But for simplicity, let's just work with a resized view
        scale_percent = 0.3 # Adjust if images are too big/small on your screen
        width = int(img.shape[1] * scale_percent)
        height = int(img.shape[0] * scale_percent)
        dim = (width, height)
        
        display_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        # OPEN SELECTOR WINDOW
        # Draw the box, then hit SPACE or ENTER
        roi = cv2.selectROI("Select Bed Area (Press Enter)", display_img, showCrosshair=True, fromCenter=False)
        
        # roi is (x, y, w, h)
        x, y, w, h = roi
        
        # Check if user cancelled (all zeros)
        if w == 0 or h == 0:
            print("Skipped.")
            cv2.destroyAllWindows()
            continue

        # Map the crop back to the ORIGINAL image size
        real_x = int(x / scale_percent)
        real_y = int(y / scale_percent)
        real_w = int(w / scale_percent)
        real_h = int(h / scale_percent)

        cropped_img = img[real_y:real_y+real_h, real_x:real_x+real_w]

        # Save as JPG
        save_name = os.path.splitext(filename)[0] + ".jpg"
        save_path = os.path.join(OUTPUT_FOLDER, save_name)
        
        cv2.imwrite(save_path, cropped_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"Saved: {save_path}")

        # Close window to clean up
        cv2.destroyAllWindows()

    print("All done! Your backgrounds are ready.")

if __name__ == "__main__":
    crop_images()