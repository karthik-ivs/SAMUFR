import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# =========================================================
# HYPERPARAMETERS
# =========================================================
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
EPOCHS = 50
PATIENCE = 3

DATASET_PATH = "dataset"
MODEL_PATH = "best_mbnet_model.pth"

# =========================================================
# TRANSFORMS
# =========================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================================================
# DATASET
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
                if img.lower().endswith(('.jpg','.jpeg','.png')):
                    self.samples.append(
                        (os.path.join(person_dir, img),
                         self.class_to_idx[person])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# =========================================================
# LOAD DATA
# =========================================================
full_dataset = FaceDataset(DATASET_PATH, train_transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

new_class_names = full_dataset.class_names
new_num_classes = len(new_class_names)

# =========================================================
# LOAD OLD MODEL (IF EXISTS)
# =========================================================
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    old_class_names = checkpoint["class_names"]
else:
    checkpoint = None
    old_class_names = []

# =========================================================
# MODEL
# =========================================================
model = mobilenet_v3_small(pretrained=True)

# Freeze most layers
for param in model.parameters():
    param.requires_grad = False

for param in model.features[-3:].parameters():
    param.requires_grad = True

# Replace classifier
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    new_num_classes
)

# =========================================================
# MERGE OLD KNOWLEDGE
# =========================================================
if checkpoint is not None:
    print("Merging old knowledge...")

    old_weights = checkpoint["model_state"]

    # Load backbone
    # Remove classifier weights before loading
    old_weights_filtered = { k: v for k, v in old_weights.items() if not k.startswith("classifier.3")}

    model.load_state_dict(old_weights_filtered, strict=False)

    old_num_classes = len(old_class_names)

    old_w = old_weights["classifier.3.weight"]
    old_b = old_weights["classifier.3.bias"]

    with torch.no_grad():
        model.classifier[3].weight[:old_num_classes] = old_w
        model.classifier[3].bias[:old_num_classes] = old_b

model = model.to(device)

# =========================================================
# TRAINING SETUP
# =========================================================
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# =========================================================
# TRAIN LOOP
# =========================================================
best_val_loss = float("inf")
early_stop_counter = 0

for epoch in range(EPOCHS):

    model.train()
    train_correct, train_total = 0, 0
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_total

    # ================= VALIDATION =================
    model.eval()
    val_loss = 0
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # ================= SAVE BEST =================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0

        torch.save({
            "model_state": model.state_dict(),
            "class_names": new_class_names
        }, MODEL_PATH)

        print("✅ Model updated & saved")

    else:
        early_stop_counter += 1

    if early_stop_counter >= PATIENCE:
        print("⛔ Early stopping")
        break

print("🎯 Training complete!")