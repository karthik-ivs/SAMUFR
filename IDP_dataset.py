import cv2
import os
from PIL import Image
from torchvision import transforms

DATASET_PATH = "dataset"
AUG_PER_IMAGE = 10

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3,
                           contrast=0.3,
                           saturation=0.2,
                           hue=0.05),
    transforms.RandomAffine(degrees=0,
                            translate=(0.1,0.1),
                            scale=(0.9,1.1)),
])

# ========================
# ADDING NEW PERSONS (MULTIPLE)
# ========================
while True:

    person_name = input("\nEnter person name (or type 'exit'): ").strip()
    
    if person_name.lower() == "exit":
        break

    person_dir = os.path.join(DATASET_PATH, person_name)
    os.makedirs(person_dir, exist_ok=True)

    # Get next index
    existing_images = [f for f in os.listdir(person_dir)
                       if f.endswith(".jpg") and "_aug" not in f]

    img_index = len(existing_images) + 1

    cap = cv2.VideoCapture(0)

    print(f"\nCapturing for {person_name}")
    print("Press 's' to capture")
    print("Press 'q' to finish this person\n")

    new_images = []

    # ========================
    # CAPTURE LOOP
    # ========================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(f"Capture - {person_name}", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            img_name = f"{person_name}_{img_index}.jpg"
            img_path = os.path.join(person_dir, img_name)

            cv2.imwrite(img_path, frame)
            new_images.append(img_path)

            print(f"Saved: {img_name}")

            img_index += 1

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ========================
    # AUGMENT ONLY NEW IMAGES
    # ========================
    if new_images:
        print(f"\nAugmenting images for {person_name}...")

        for img_path in new_images:
            image = Image.open(img_path).convert("RGB")

            name, ext = os.path.splitext(os.path.basename(img_path))

            for i in range(AUG_PER_IMAGE):
                aug_img = transform(image)
                aug_img = transforms(aug_img)

                save_name = f"{name}_aug{i}{ext}"
                save_path = os.path.join(person_dir, save_name)

                aug_img.save(save_path)

        print(f"Augmentation completed for {person_name}!")
    else:
        print("No images captured.")
    
    import subprocess

print("\nStarting training...")

subprocess.run(["python", "IDP_train_mbnetv3_s.py"])

print("Training completed!")

print("\nAll done!")