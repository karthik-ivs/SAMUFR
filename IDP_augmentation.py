import os
import random
from PIL import Image
from torchvision import transforms

DATASET_PATH = "dataset"
AUG_PER_IMAGE = 10   # number of augmented images per original

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

for person in os.listdir(DATASET_PATH):

    person_dir = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_dir):
        continue

    print(f"Processing {person}")

    images = [img for img in os.listdir(person_dir)
              if img.lower().endswith(('.jpg','.jpeg','.png'))]

    for img_name in images:

        img_path = os.path.join(person_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        name, ext = os.path.splitext(img_name)

        for i in range(AUG_PER_IMAGE):

            aug_img = transform(image)

            save_name = f"{name}_aug{i}{ext}"
            save_path = os.path.join(person_dir, save_name)

            aug_img.save(save_path)

print("Augmentation completed.")