import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import matplotlib.pyplot as plt

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ================= PATHS =================
MODEL_PATH = "best_resnet_model.pth"
TEST_IMAGE = "test_images/test1.jpeg"   # change if needed

# ================= TRANSFORM (NO AUGMENTATION) =================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= LOAD MODEL =================
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]

# Rebuild backbone
backbone = resnet50(pretrained=False)
backbone.fc = nn.Identity()

class FaceClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

model = FaceClassifier(backbone, len(class_names))
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

print("Model loaded successfully.")

# ================= PREDICTION =================
if os.path.exists(TEST_IMAGE):

    img = Image.open(TEST_IMAGE).convert("RGB")
    img_tensor = test_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        idx = torch.argmax(probs).item()
        confidence = probs[0, idx].item() * 100

    predicted_name = class_names[idx]

    # Display image
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{predicted_name} ({confidence:.1f}%)")
    plt.show()

    print(f"Predicted: {predicted_name}")
    print(f"Confidence: {confidence:.2f}%")

else:
    print("Test image not found!")
