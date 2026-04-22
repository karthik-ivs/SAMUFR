import cv2
import os

# ================================
# INPUT & OUTPUT FOLDERS
# ================================
input_folder = "dataset"          # your current images folder
output_folder = "dataset_fixed"   # new folder for corrected images

os.makedirs(output_folder, exist_ok=True)

# ================================
# ROTATION TYPE (CHANGE IF NEEDED)
# ================================
ROTATION = cv2.ROTATE_180
# Try alternatives if needed:
# cv2.ROTATE_90_COUNTERCLOCKWISE
# cv2.ROTATE_180

# ================================
# PROCESS IMAGES
# ================================
for root, dirs, files in os.walk(input_folder):

    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            input_path = os.path.join(root, file)

            # Maintain folder structure
            relative_path = os.path.relpath(root, input_folder)
            save_dir = os.path.join(output_folder, relative_path)
            os.makedirs(save_dir, exist_ok=True)

            output_path = os.path.join(save_dir, file)

            # Read image
            img = cv2.imread(input_path)

            if img is None:
                print(f"Skipping: {input_path}")
                continue

            # Rotate image
            rotated = cv2.rotate(img, ROTATION)

            # Save corrected image
            cv2.imwrite(output_path, rotated)

            print(f"Fixed: {output_path}")

print("✅ All images fixed and saved!")