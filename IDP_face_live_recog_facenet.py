import torch
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os

# ================================
# DEVICE
# ================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================================
# LOAD FACENET
# ================================
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ================================
# LOAD DATABASE
# ================================
database = np.load("embeddings.npy", allow_pickle=True).item()

print("Loaded embeddings")

# ================================
# YOLO
# ================================
yolo_model = YOLO("yolov8n-face.pt")

# ================================
# TRANSFORM
# ================================
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# ================================
# RECOGNITION FUNCTION
# ================================
def recognize(embedding, db, threshold=0.7):
    best_name = "Unknown"
    best_score = -1

    for name, embeds in db.items():
        for db_emb in embeds:
            db_emb = torch.tensor(db_emb).to(device)

            score = F.cosine_similarity(embedding, db_emb.unsqueeze(0)).item()

            if score > best_score:
                best_score = score
                best_name = name

    if best_score > threshold:
        return best_name, best_score
    else:
        return "Unknown", best_score

# ================================
# CAMERA
# ================================
cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # PREPROCESS
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(face_rgb)
        img_tensor = transform(img).unsqueeze(0).to(device)

        # EMBEDDING
        with torch.no_grad():
            embedding = model(img_tensor)

        # RECOGNIZE
        name, score = recognize(embedding, database)

        label = f"{name} ({score:.2f})"

        # DRAW
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    color, 2)

    cv2.imshow("FaceNet Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()