import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# =========================
# SETTINGS
# =========================
DATASET_PATH = "dataset"
SAVE_PATH = "embeddings.npy"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =========================
# LOAD FACENET
# =========================
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

database = {}

print("Generating embeddings...")

for person in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_dir):
        continue

    embeddings = []

    for img_name in os.listdir(person_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(person_dir, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model(img)

            embeddings.append(emb.cpu().numpy()[0])

        except:
            print(f"Skipping {img_name}")

    if embeddings:
        database[person] = embeddings
        print(f"{person}: {len(embeddings)} embeddings")

# SAVE
np.save(SAVE_PATH, database)

print("✅ Embeddings saved to embeddings.npy")