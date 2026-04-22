import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50

# =========================================================
# DEVICE (MacBook M4 GPU - MPS)
# =========================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# =========================================================
# PATHS
# =========================================================
DATASET_PATH = "dataset"        # <-- CHANGE IF NEEDED
TEST_IMAGE_PATH = "test_images"
test_image = os.path.join(TEST_IMAGE_PATH, "test1.jpeg")

# =========================================================
# IMAGE PREPROCESSING (VGG-Face style)
# =========================================================
#vgg_transform = transforms.Compose([
#    transforms.Resize((224, 224)),
#   transforms.ToTensor(),
#    transforms.Normalize(
#        mean=[91.4953 / 255, 103.8827 / 255, 131.0912 / 255],
#        std=[1 / 255, 1 / 255, 1 / 255]
#    )
#])

## With Data Augmentation

print("Augmentation enabled")

vgg_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 🔥 DATA AUGMENTATION (TRAINING ONLY)
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1
    ),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[91.4953 / 255, 103.8827 / 255, 131.0912 / 255],
        std=[1 / 255, 1 / 255, 1 / 255]
    )
])


# =========================================================
# DATASET CLASS
# =========================================================
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.samples = []
        self.transform = transform

        self.class_names = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        for person in self.class_names:
            person_dir = os.path.join(root_dir, person)
            for img in os.listdir(person_dir):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append(
                        (os.path.join(person_dir, img), self.class_to_idx[person])
                    )

        print("Classes:", self.class_names)
        print("Total images:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

# =========================================================
# VGG-FACE RESNET50 BACKBONE
# =========================================================
class VGGFaceResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Identity()  # remove classifier

    def forward(self, x):
        return self.model(x)

# =========================================================
# CLASSIFIER MODEL (TRAINABLE)
# =========================================================
class FaceClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# =========================================================
# LOAD DATA
# =========================================================
dataset = FaceDataset(DATASET_PATH, vgg_transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

num_classes = len(dataset.class_names)

# =========================================================
# MODEL SETUP
# =========================================================
backbone = VGGFaceResNet50().to(device)

# Freeze backbone
for param in backbone.parameters():
    param.requires_grad = False

model = FaceClassifier(backbone, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

# =========================================================
# TRAINING LOOP
# =========================================================
epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f}")

print("Training completed!")

# =========================================================
# SAVE MODEL
# =========================================================
torch.save(model.state_dict(), "face_classifier.pth")
print("Model saved as face_classifier.pth")

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_face(image_path, model, class_names):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img = vgg_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        idx = torch.argmax(probs).item()

    return class_names[idx], probs[0, idx].item() * 100

# =========================================================
# TEST ON IMAGE
# =========================================================
if os.path.exists(test_image):
    name, confidence = predict_face(
        test_image, model, dataset.class_names
    )

    img = plt.imread(test_image)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prediction: {name}\nConfidence: {confidence:.1f}%")
    plt.show()

    print(f"Predicted: {name} ({confidence:.1f}%)")
else:
    print("Test image not found!")
