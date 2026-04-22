import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image
import matplotlib.pyplot as plt

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
MODEL_PATH = "best_mbnet_model.pth"
TEST_IMAGE = "test_images/test1.jpeg"

# =========================================================
# IMAGE PREPROCESSING (NO AUGMENTATION FOR TESTING)
# =========================================================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# LOAD SAVED MODEL
# =========================================================
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]

model = mobilenet_v3_small(pretrained=False)

# Replace classifier layer to match number of classes
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(class_names)
)

model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

print("Model loaded successfully.")

# =========================================================
# PREDICTION
# =========================================================
if os.path.exists(TEST_IMAGE):
    img = Image.open(TEST_IMAGE).convert("RGB")
    img_tensor = test_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        idx = torch.argmax(probs).item()
        confidence = probs[0, idx].item() * 100

    name = class_names[idx]

    # Display result
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{name} ({confidence:.1f}%)")
    plt.show()

    print(f"Predicted: {name} ({confidence:.1f}%)")

else:
    print("Test image not found!")
