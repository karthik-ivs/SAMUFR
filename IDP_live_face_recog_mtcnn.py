import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
import cv2
from PIL import Image
import numpy as np
from mtcnn import MTCNN

# DEVICE
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MODEL_PATH = "best_mbnet_model.pth"

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]

model = mobilenet_v3_small(pretrained=False)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(class_names)
)

model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

print("Model loaded")

# Applyng image transformations
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# loading haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start video capture
cap = cv2.VideoCapture(0)


detector = MTCNN()
# Real time detection loop
while True:

    ret, frame = cap.read()
    if not ret:
        break

    
    faces = detector.detect_faces(frame)

    for face_data in faces:

        x, y, w, h = face_data['box']

        face = frame[y:y+h, x:x+w]

        # Resize face for better recognition
        face = cv2.resize(face, (224,224))

        img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            idx = torch.argmax(probs).item()
            confidence = probs[0,idx].item()*100

            if confidence < 60:
                name = "Unknown"
            else:
                name = class_names[idx]

            label = f"{name} {confidence:.1f}%"

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,255,0),2)

    cv2.imshow("Face Recognition",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()